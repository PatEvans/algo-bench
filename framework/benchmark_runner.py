"""
Core benchmark runner logic.
Handles Docker container setup, execution of the wrapper script,
and parsing results. Configured by the specific benchmark type.
"""

import time
import json
import tempfile
import os
import docker
from docker.errors import APIError, ImageNotFound, DockerException
from requests.exceptions import ConnectionError as DockerConnectionError
from contextlib import ExitStack
import tarfile
import io
import traceback
import ctypes # Needed for serialization helper
from typing import Callable, Optional, Any

# Constants from wrapper script (must match)
WRAPPER_SCRIPT_FILENAME_CONT = "exec_runner.py" # Name inside container
WRAPPER_STDOUT_MARKER_BEFORE = "---WRAPPER_STDOUT_MARKER_BEFORE---"
WRAPPER_STDOUT_MARKER_AFTER = "---WRAPPER_STDOUT_MARKER_AFTER---"

# --- CTypes Definitions & Helpers ---
# Define Buffer struct commonly used in compression, might be needed by others
class CBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)),
                ("size", ctypes.c_size_t)]

# Map string names to ctypes types (add more as needed)
# This map is used by the serialization helper below
CTYPES_MAP_FOR_SERIALIZATION = {
    "POINTER_ubyte": ctypes.POINTER(ctypes.c_ubyte),
    "POINTER_int": ctypes.POINTER(ctypes.c_int),
    "size_t": ctypes.c_size_t,
    "void": None,
    "int": ctypes.c_int,
    "Buffer": CBuffer,
    # Add other types like float, double, char*, etc. if required by benchmarks
}

def get_ctype_name(ctype_obj):
    """Inverse of get_ctype - gets string name from ctypes object for serialization."""
    if ctype_obj is None: return "void"
    # Reverse lookup in CTYPES_MAP_FOR_SERIALIZATION
    for name, ctype in CTYPES_MAP_FOR_SERIALIZATION.items():
        # Need careful comparison for complex types like POINTER
        if isinstance(ctype_obj, type(ctype)) and name != "Buffer": # Check type match for non-structs
             # Special handling for POINTER types
             if name.startswith("POINTER_"):
                  # Ensure the pointee types match
                  if hasattr(ctype_obj, '_type_') and hasattr(ctype, '_type_') and ctype_obj._type_ == ctype._type_:
                       return name
             elif ctype == ctype_obj: # Direct comparison for simple types
                  return name
        elif name == "Buffer" and isinstance(ctype_obj, type) and issubclass(ctype_obj, CBuffer): # Check for Buffer struct type
             return name


    # Fallback or raise error
    print(f"Warning: Could not find string name for ctype: {ctype_obj}")
    return str(ctype_obj) # Or raise error

def make_signatures_serializable(signatures_dict):
    """Converts a dictionary of C function signatures with ctypes objects to a JSON-serializable format."""
    serializable = {}
    for name, sig_info in signatures_dict.items():
        serializable_info = {}
        if "argtypes" in sig_info:
            serializable_info["argtypes"] = [get_ctype_name(t) for t in sig_info["argtypes"]]
        if "restype" in sig_info:
            serializable_info["restype"] = get_ctype_name(sig_info["restype"])
        # Handle struct definitions
        if sig_info.get("is_struct", False):
             serializable_info["is_struct"] = True
             serializable_info["fields"] = [[fname, get_ctype_name(ftype)] for fname, ftype in sig_info.get("fields", [])]
        serializable[name] = serializable_info
    return serializable


class BenchmarkRunner:
    """Manages the execution of benchmarks within Docker."""

    def __init__(self, config):
        """
        Initializes the BenchmarkRunner.

        Args:
            config: Configuration object/module for the specific benchmark.
                    Expected attributes: BENCHMARK_TYPE, DOCKER_IMAGE,
                                         LLM_CODE_FILENAME, TEST_SUITE_FILENAME,
                                         C_FUNCTION_NAMES (dict), C_FUNCTION_SIGNATURES (dict with ctypes),
                                         CALCULATE_RATIO (bool), TIME_SECONDARY_FUNCTION (bool),
                                         WRAPPER_SCRIPT_PATH (path to framework/docker_exec_wrapper.py)
                                         Optional: CONTAINER_MEM_LIMIT, CONTAINER_CPU_SHARES, EXEC_TIMEOUT_SECONDS
        """
        self.config = config
        self.docker_client = None
        self._connect_docker()

        # Make signatures serializable for passing via environment variable
        self.serializable_signatures = make_signatures_serializable(config.C_FUNCTION_SIGNATURES)

        # Read wrapper script content once
        try:
            wrapper_path = getattr(config, 'WRAPPER_SCRIPT_PATH', None)
            if not wrapper_path or not os.path.exists(wrapper_path):
                 raise FileNotFoundError(f"Wrapper script path not configured or not found: {wrapper_path}")
            with open(wrapper_path, 'r', encoding='utf-8') as f:
                self.exec_wrapper_code = f.read()
            print(f"Framework Runner: Wrapper script loaded from {wrapper_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read Docker wrapper script: {e}")


    def _connect_docker(self):
        """Initializes Docker client and checks connection."""
        if self.docker_client:
            try:
                self.docker_client.ping()
                return # Already connected
            except (APIError, DockerConnectionError, DockerException):
                print("Framework Runner: Docker connection lost, attempting to reconnect...")
                self.docker_client = None # Force reconnect

        try:
            print("Framework Runner: Connecting to Docker daemon...")
            # Increase timeouts
            self.docker_client = docker.from_env(timeout=120)
            # Set API timeout for long operations like exec
            self.docker_client.api.timeout = getattr(self.config, 'EXEC_TIMEOUT_SECONDS', 600)
            self.docker_client.ping()
            print(f"Framework Runner: Docker client connected. API timeout: {self.docker_client.api.timeout}s")
        except (DockerConnectionError, APIError, DockerException, Exception) as e:
            self.docker_client = None # Ensure client is None on failure
            raise RuntimeError(f"Docker connection failed: {e}. Is Docker running?")

    def _ensure_image_exists(self):
        """Pulls the required Docker image if not found locally."""
        image_name = self.config.DOCKER_IMAGE
        try:
            self.docker_client.images.get(image_name)
            print(f"Framework Runner: Docker image '{image_name}' found locally.")
        except ImageNotFound:
            print(f"Framework Runner: Pulling Docker image: {image_name}...")
            try:
                self.docker_client.images.pull(image_name)
                print(f"Framework Runner: Docker image '{image_name}' pulled.")
            except APIError as e:
                raise RuntimeError(f"Failed to pull Docker image '{image_name}': {e}")
        except (APIError, DockerConnectionError) as e:
             raise RuntimeError(f"Error checking Docker image '{image_name}': {e}")


    def run_evaluation(self, generated_code: str, test_suite_data: Any, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Runs the full benchmark evaluation in Docker for the given code and test suite.

        Args:
            generated_code: The C code string generated by the LLM or baseline.
            test_suite_data: The loaded test suite data (format depends on benchmark).
            progress_callback: Optional function for progress updates.

        Returns:
            A dictionary containing evaluation results from the Docker wrapper.
        """
        results = {
            'correctness': 0,
            'avg_time_ms': None,
            'avg_secondary_time_ms': None,
            'avg_ratio': None,
            'error': None,
            'performance_details': {}
        }

        if not generated_code:
            results['error'] = "No generated code provided."
            return results
        if not test_suite_data:
            results['error'] = "No test suite data provided."
            return results

        # Ensure Docker connection is active
        try:
            self._connect_docker()
            self._ensure_image_exists()
        except RuntimeError as e:
            results['error'] = str(e)
            if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
            return results

        container = None
        # Use ExitStack for robust cleanup of tempdir and container
        with ExitStack() as stack:
            try:
                # --- 1. Prepare Host Files ---
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Preparing Files', 'message': 'Preparing files on host...'})
                temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                sandbox_dir = "/sandbox" # Standard sandbox dir in container

                # Paths on host
                llm_code_path_host = os.path.join(temp_dir, self.config.LLM_CODE_FILENAME)
                runner_script_path_host = os.path.join(temp_dir, WRAPPER_SCRIPT_FILENAME_CONT) # Use fixed name for runner script
                test_suite_path_host = os.path.join(temp_dir, self.config.TEST_SUITE_FILENAME) # Use configured test suite name

                # Paths in container
                llm_code_path_cont = f"{sandbox_dir}/{self.config.LLM_CODE_FILENAME}"
                runner_script_path_cont = f"{sandbox_dir}/{WRAPPER_SCRIPT_FILENAME_CONT}"
                test_suite_path_cont = f"{sandbox_dir}/{self.config.TEST_SUITE_FILENAME}"

                # Write generated C code
                with open(llm_code_path_host, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                # Write wrapper script content
                with open(runner_script_path_host, 'w', encoding='utf-8') as f:
                    f.write(self.exec_wrapper_code)
                # Write test suite data (as JSON)
                try:
                    with open(test_suite_path_host, 'w', encoding='utf-8') as f:
                        json.dump(test_suite_data, f)
                except TypeError as e:
                    raise RuntimeError(f"Failed to serialize test suite data to JSON: {e}")

                print(f"Framework Runner: Host files prepared in {temp_dir}")

                # --- 2. Start Container ---
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Starting Container', 'message': 'Starting Docker container...'})
                container = self.docker_client.containers.run(
                    image=self.config.DOCKER_IMAGE,
                    command=["sleep", "infinity"],
                    working_dir=sandbox_dir,
                    mem_limit=getattr(self.config, 'CONTAINER_MEM_LIMIT', "1g"),
                    cpu_shares=getattr(self.config, 'CONTAINER_CPU_SHARES', 512),
                    detach=True,
                    auto_remove=False, # Manage removal manually
                    # Consider security options like network_mode='none' if needed
                )
                stack.callback(lambda c: (c.stop(timeout=10), c.remove(force=True)), container) # Ensure cleanup
                print(f"Framework Runner: Container {container.short_id} started.")
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Container Started', 'message': f'Container ready ({container.short_id}).'})

                # --- 3. Copy Files to Container ---
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Copying Files', 'message': 'Copying files to container...'})
                # Create sandbox directory inside container first
                exit_code_mkdir, _ = container.exec_run(cmd=f"mkdir -p {sandbox_dir}")
                if exit_code_mkdir != 0:
                     raise RuntimeError(f"Failed to create {sandbox_dir} in container (exit code {exit_code_mkdir}).")

                # Create tar stream and copy
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                    tar.add(llm_code_path_host, arcname=self.config.LLM_CODE_FILENAME)
                    tar.add(runner_script_path_host, arcname=WRAPPER_SCRIPT_FILENAME_CONT)
                    tar.add(test_suite_path_host, arcname=self.config.TEST_SUITE_FILENAME)
                tar_stream.seek(0)
                container.put_archive(path=sandbox_dir, data=tar_stream)
                print("Framework Runner: Files copied to container.")
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Files Copied', 'message': 'Benchmark files ready in container.'})

                # --- 4. Execute Wrapper Script ---
                if progress_callback: progress_callback({'status': 'Running', 'category': 'Executing Benchmark', 'message': 'Executing benchmark wrapper script...'})
                exec_command = ["python", runner_script_path_cont]

                # Prepare environment variables for the wrapper script
                wrapper_env = {
                    "LLM_CODE_SOURCE_FILE": llm_code_path_cont,
                    "TEST_SUITE_FILE": test_suite_path_cont,
                    "C_FUNCTION_NAMES": json.dumps(self.config.C_FUNCTION_NAMES),
                    "C_FUNCTION_SIGNATURES": json.dumps(self.serializable_signatures), # Use serializable version
                    "BENCHMARK_TYPE": self.config.BENCHMARK_TYPE,
                    "CALCULATE_RATIO": "1" if getattr(self.config, 'CALCULATE_RATIO', False) else "0",
                    "TIME_SECONDARY_FUNCTION": "1" if getattr(self.config, 'TIME_SECONDARY_FUNCTION', False) else "0",
                    # Add PYTHONUNBUFFERED for immediate output flushing in wrapper
                    "PYTHONUNBUFFERED": "1",
                }

                exec_instance = self.docker_client.api.exec_create(
                    container=container.id,
                    cmd=exec_command,
                    environment=wrapper_env,
                    workdir=sandbox_dir,
                    stdout=True, stderr=True, tty=False
                )
                exec_id = exec_instance['Id']

                exec_stream = self.docker_client.api.exec_start(exec_id=exec_id, stream=True, demux=True)

                stdout_acc = b""
                stderr_buffer = b""
                exit_code = None

                print(f"Framework Runner: Streaming output from container exec_id: {exec_id}...")
                for stdout_chunk, stderr_chunk in exec_stream:
                    if stdout_chunk:
                        # print(f"DEBUG RUNNER STDOUT: {stdout_chunk!r}") # Verbose
                        stdout_acc += stdout_chunk
                    if stderr_chunk:
                        # print(f"DEBUG RUNNER STDERR: {stderr_chunk!r}") # Verbose
                        stderr_buffer += stderr_chunk
                        # Process complete lines from stderr for progress updates
                        lines = stderr_buffer.split(b'\n')
                        stderr_buffer = lines[-1] # Keep incomplete line
                        for line_bytes in lines[:-1]:
                            line_str = line_bytes.decode('utf-8', errors='replace').strip()
                            if not line_str: continue
                            try:
                                progress_json = json.loads(line_str)
                                if progress_json.get("type") == "progress" and progress_callback:
                                    progress_callback(progress_json.get("data", {}))
                                else:
                                    print(f"RUNNER (stderr JSON): {line_str}") # Log other JSON
                            except json.JSONDecodeError:
                                print(f"RUNNER (stderr raw): {line_str}") # Log raw stderr
                            except Exception as cb_err:
                                print(f"ERROR: Progress callback failed: {cb_err}")

                # --- 5. Process Results ---
                print(f"Framework Runner: Stream finished for exec_id: {exec_id}.")
                try:
                    exec_inspect = self.docker_client.api.exec_inspect(exec_id)
                    exit_code = exec_inspect.get('ExitCode')
                    print(f"Framework Runner: Exec exit code: {exit_code}")
                except APIError as inspect_err:
                    print(f"Framework Runner: Error inspecting exec {exec_id}: {inspect_err}")
                    results['error'] = f"Failed to inspect container execution: {inspect_err}"
                    exit_code = -1 # Assume failure

                if exit_code == 0:
                    output_str_raw = stdout_acc.decode('utf-8', errors='replace')
                    try:
                        start_index = output_str_raw.find(WRAPPER_STDOUT_MARKER_BEFORE)
                        end_index = output_str_raw.find(WRAPPER_STDOUT_MARKER_AFTER)

                        if start_index == -1 or end_index == -1 or end_index <= start_index:
                            raise ValueError(f"Could not find expected markers in container stdout.")

                        json_content_str = output_str_raw[start_index + len(WRAPPER_STDOUT_MARKER_BEFORE):end_index].strip()
                        if not json_content_str:
                             raise ValueError("Container stdout between markers was empty.")

                        final_message = json.loads(json_content_str)
                        if final_message.get("type") == "result":
                            container_results = final_message.get("data", {})
                            print("Framework Runner: Successfully parsed final result from container.")
                            results.update(container_results) # Update host results dict
                            if container_results.get('error'):
                                results['error'] = f"Wrapper script error: {container_results['error']}"
                        else:
                            raise ValueError("Unexpected JSON format from wrapper stdout.")
                    except (json.JSONDecodeError, ValueError) as json_e:
                        output_snippet = output_str_raw[:1000]
                        results['error'] = f"Failed to parse final JSON from container (Exit Code 0): {json_e}. Output:\n---\n{output_snippet}\n---"
                    except Exception as parse_e:
                         output_snippet = output_str_raw[:1000]
                         results['error'] = f"Unexpected error parsing container stdout (Exit Code 0): {parse_e}. Output:\n---\n{output_snippet}\n---"
                elif exit_code is not None:
                    stderr_final = stderr_buffer.decode('utf-8', errors='replace') # Process remaining buffer
                    stdout_snippet = stdout_acc.decode('utf-8', errors='replace')[:500]
                    stderr_snippet = stderr_final[:500]
                    results['error'] = f"Container wrapper exited with code {exit_code}. Stderr:\n---\n{stderr_snippet}\n---\nStdout:\n---\n{stdout_snippet}\n---"
                else: # Exit code is None
                     results['error'] = f"Failed to determine container exec exit code. Inspect result: {exec_inspect}"


            # Catch errors during setup/execution within the ExitStack context
            except (APIError, DockerConnectionError, DockerException) as docker_err:
                error_str = f"Docker error during benchmark execution: {docker_err}\n{traceback.format_exc()}"
                results['error'] = error_str
                print(f"ERROR: {error_str}")
            except FileNotFoundError as fnf_err:
                 error_str = f"File not found during benchmark execution: {fnf_err}\n{traceback.format_exc()}"
                 results['error'] = error_str
                 print(f"ERROR: {error_str}")
            except RuntimeError as rt_err:
                 error_str = f"Runtime error during benchmark execution: {rt_err}\n{traceback.format_exc()}"
                 results['error'] = error_str
                 print(f"ERROR: {error_str}")
            except Exception as e:
                error_str = f"Unexpected error during benchmark execution: {e}\n{traceback.format_exc()}"
                results['error'] = error_str
                print(f"ERROR: {error_str}")
            finally:
                # ExitStack handles container stop/remove and tempdir cleanup
                if container:
                    print(f"Framework Runner: Container {container.short_id} cleanup handled by ExitStack.")
                else:
                    print("Framework Runner: No container was started or setup failed.")
                # Final progress update
                if progress_callback:
                    final_status = 'Completed' if not results.get('error') else 'Error'
                    progress_callback({
                        'status': final_status,
                        'category': 'Finished',
                        'error': results.get('error'),
                        'message': 'Benchmark execution finished.'
                    })

        return results
