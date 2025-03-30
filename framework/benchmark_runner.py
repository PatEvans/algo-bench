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
# import ctypes # No longer needed here
from typing import Callable, Optional, Any, Dict

# Constants from wrapper script (must match)
WRAPPER_SCRIPT_FILENAME_CONT = "exec_runner.py" # Name inside container
WRAPPER_STDOUT_MARKER_BEFORE = "---WRAPPER_STDOUT_MARKER_BEFORE---"
WRAPPER_STDOUT_MARKER_AFTER = "---WRAPPER_STDOUT_MARKER_AFTER---"
# Default sandbox directory inside the container
DEFAULT_SANDBOX_DIR = "/sandbox"
# Define the standard Docker image name built by the root Dockerfile
UNIFIED_DOCKER_IMAGE = "llm-benchmark-env:latest" # Or your preferred image name/tag

# --- CTypes Definitions & Helpers (REMOVED) ---
# Serialization logic is no longer needed here.
# The config files should provide string-based signatures directly.


class BenchmarkRunner:
    """Manages the execution of benchmarks within Docker."""

    def __init__(self, config):
        """
        Initializes the BenchmarkRunner.

        Args:
            config: Configuration object/module for the specific benchmark.
                    Expected attributes: BENCHMARK_TYPE, LLM_CODE_FILENAME,
                                         TEST_SUITE_FILENAME, FUNCTION_NAMES (dict),
                                         FUNCTION_SIGNATURES (dict with string types),
                                         CALCULATE_RATIO (bool), TIME_SECONDARY_FUNCTION (bool),
                                         WRAPPER_SCRIPT_PATH (path to framework/docker_exec_wrapper.py)
                                         Optional: CONTAINER_MEM_LIMIT, CONTAINER_CPU_SHARES, EXEC_TIMEOUT_SECONDS
                    NOTE: DOCKER_IMAGE is no longer read from config.
        """
        self.config = config
        self.docker_client = None
        # Note: _connect_docker is now called lazily before first use

        # Validate required config attributes (DOCKER_IMAGE removed)
        required_attrs = [
            'BENCHMARK_TYPE', 'LLM_CODE_FILENAME', 'TEST_SUITE_FILENAME',
            'FUNCTION_NAMES', 'FUNCTION_SIGNATURES', 'WRAPPER_SCRIPT_PATH'
        ]
        missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing_attrs:
            raise ValueError(f"Benchmark config is missing required attributes: {', '.join(missing_attrs)}")

        # Config should already have serializable signatures (using string types)
        # Validate that the required keys exist in the config dictionaries
        if not isinstance(config.FUNCTION_NAMES, dict):
             raise ValueError("Config attribute 'FUNCTION_NAMES' must be a dictionary.")
        if not isinstance(config.FUNCTION_SIGNATURES, dict):
             raise ValueError("Config attribute 'FUNCTION_SIGNATURES' must be a dictionary.")

        # Read wrapper script content once
        try:
            # Use WRAPPER_SCRIPT_PATH from config
            wrapper_path = getattr(config, 'WRAPPER_SCRIPT_PATH', None)
            if not wrapper_path or not os.path.exists(wrapper_path):
                 raise FileNotFoundError(f"Wrapper script path not configured or not found: {wrapper_path}")
            with open(wrapper_path, 'r', encoding='utf-8') as f:
                self.exec_wrapper_code = f.read()
            print(f"Framework Runner: Wrapper script loaded from {wrapper_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read Docker wrapper script from '{wrapper_path}': {e}")


    def _connect_docker(self):
        """Initializes Docker client and checks connection."""
        # Use configured timeout, default to 600s (10 min)
        api_timeout = getattr(self.config, 'EXEC_TIMEOUT_SECONDS', 600)
        if self.docker_client:
            try:
                self.docker_client.ping()
                return # Already connected
            except (APIError, DockerConnectionError, DockerException):
                print("Framework Runner: Docker connection lost, attempting to reconnect...")
                self.docker_client = None # Force reconnect

        try:
            print(f"Framework Runner: Connecting to Docker daemon (API Timeout: {api_timeout}s)...")
            # Use standard timeout for client connection, API timeout for operations
            self.docker_client = docker.from_env(timeout=120)
            self.docker_client.api.timeout = api_timeout
            self.docker_client.ping() # Verify connection
            print(f"Framework Runner: Docker client connected.")
            if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Docker', 'message': 'Docker connection successful.'})
        except (DockerConnectionError, APIError, DockerException, Exception) as e:
            self.docker_client = None # Ensure client is None on failure
            err_msg = f"Docker connection failed: {e}. Is Docker running?"
            if progress_callback: progress_callback({'status': 'Error', 'category': 'Setup: Docker', 'error': err_msg})
            raise RuntimeError(err_msg) from e

    def _ensure_image_exists(self, progress_callback: Optional[Callable[[dict], None]] = None):
        """Checks if the unified Docker image exists locally."""
        image_name = UNIFIED_DOCKER_IMAGE # Use the hardcoded image name
        if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Docker Image', 'message': f"Checking for image '{image_name}'..."})
        try:
            self.docker_client.images.get(image_name)
            print(f"Framework Runner: Unified Docker image '{image_name}' found locally.")
            if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Docker Image', 'message': f"Image '{image_name}' found."})
        except ImageNotFound:
            # Don't pull automatically, user should build it from the Dockerfile
            error_msg = (f"Unified Docker image '{image_name}' not found locally. "
                         f"Please build it using the Dockerfile in the project root:\n"
                         f"docker build -t {image_name} .")
            print(f"ERROR: {error_msg}")
            if progress_callback: progress_callback({'status': 'Error', 'category': 'Setup: Docker Image', 'error': error_msg})
            raise RuntimeError(error_msg)
        except (APIError, DockerConnectionError) as e:
             err_msg = f"Error checking Docker image '{image_name}': {e}"
             if progress_callback: progress_callback({'status': 'Error', 'category': 'Setup: Docker Image', 'error': err_msg})
             raise RuntimeError(err_msg) from e


    def run_evaluation(self, generated_code: str, test_suite_data: Any, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Runs the full benchmark evaluation in Docker for the given code and test suite,
        using settings from the config object provided during initialization.

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

        # Ensure Docker connection is active and image exists
        try:
            self._connect_docker(progress_callback) # Pass callback
            self._ensure_image_exists(progress_callback) # Pass callback
        except RuntimeError as e:
            results['error'] = str(e)
            # Progress callback already called by helper functions on error
            return results

        container = None
        # Use ExitStack for robust cleanup of tempdir and container
        with ExitStack() as stack:
            try:
                # --- 1. Prepare Host Files ---
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Preparing Files', 'message': 'Preparing files on host...'})
                temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                sandbox_dir = getattr(self.config, 'CONTAINER_SANDBOX_DIR', DEFAULT_SANDBOX_DIR)

                # Get filenames from config
                llm_code_filename = self.config.LLM_CODE_FILENAME
                test_suite_filename = self.config.TEST_SUITE_FILENAME

                # Paths on host
                llm_code_path_host = os.path.join(temp_dir, llm_code_filename)
                runner_script_path_host = os.path.join(temp_dir, WRAPPER_SCRIPT_FILENAME_CONT)
                test_suite_path_host = os.path.join(temp_dir, test_suite_filename)

                # Paths in container
                llm_code_path_cont = f"{sandbox_dir}/{llm_code_filename}"
                runner_script_path_cont = f"{sandbox_dir}/{WRAPPER_SCRIPT_FILENAME_CONT}"
                test_suite_path_cont = f"{sandbox_dir}/{test_suite_filename}"

                # Write generated C code
                with open(llm_code_path_host, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                # Write wrapper script content
                with open(runner_script_path_host, 'w', encoding='utf-8') as f:
                    f.write(self.exec_wrapper_code)
                # Write test suite data (as JSON)
                try:
                    with open(test_suite_path_host, 'w', encoding='utf-8') as f:
                        json.dump(test_suite_data, f, indent=2) # Add indent for readability if opened manually
                except TypeError as e:
                    raise RuntimeError(f"Failed to serialize test suite data to JSON: {e}") from e

                print(f"Framework Runner: Host files prepared in {temp_dir}: {llm_code_filename}, {WRAPPER_SCRIPT_FILENAME_CONT}, {test_suite_filename}")

                # --- 2. Start Container ---
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Starting Container', 'message': 'Starting Docker container...'})
                container = self.docker_client.containers.run(
                    image=UNIFIED_DOCKER_IMAGE, # Use the unified image name
                    command=["sleep", "infinity"], # Keep container running
                    working_dir=sandbox_dir, # Set working dir
                    mem_limit=getattr(self.config, 'CONTAINER_MEM_LIMIT', "1g"), # Use config value or default
                    cpu_shares=getattr(self.config, 'CONTAINER_CPU_SHARES', 512), # Use config value or default
                    detach=True, # Run in background
                    auto_remove=False, # Manage removal manually via ExitStack
                    # Security: Consider network_mode='none' if benchmark doesn't need network
                    # network_mode='none',
                    # Security: Consider read_only=True for root filesystem if possible
                    # read_only=True,
                    # Security: Drop capabilities if not needed
                    # cap_drop=['ALL'],
                )
                # Ensure container is stopped and removed even if errors occur
                stack.callback(self._cleanup_container, container)
                print(f"Framework Runner: Container {container.short_id} started using image '{UNIFIED_DOCKER_IMAGE}'.")
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Container Started', 'message': f'Container ready ({container.short_id}).'})

                # --- 3. Copy Files to Container ---
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Copying Files', 'message': 'Copying files to container...'})
                # Create sandbox directory inside container first
                exit_code_mkdir, _ = container.exec_run(cmd=f"mkdir -p {sandbox_dir}")
                if exit_code_mkdir != 0:
                     raise RuntimeError(f"Failed to create {sandbox_dir} in container (exit code {exit_code_mkdir}).")

                # Create tar stream and copy files to sandbox_dir
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                    # Use the filenames defined earlier for arcname
                    tar.add(llm_code_path_host, arcname=llm_code_filename)
                    tar.add(runner_script_path_host, arcname=WRAPPER_SCRIPT_FILENAME_CONT)
                    tar.add(test_suite_path_host, arcname=test_suite_filename)
                tar_stream.seek(0)
                if not container.put_archive(path=sandbox_dir, data=tar_stream):
                     raise RuntimeError("Failed to copy files to container via put_archive.")
                print(f"Framework Runner: Files copied to {sandbox_dir} in container.")
                if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Files Copied', 'message': 'Benchmark files ready in container.'})

                # --- 4. Execute Wrapper Script ---
                if progress_callback: progress_callback({'status': 'Running', 'category': 'Executing Benchmark', 'message': 'Executing benchmark wrapper script...'})
                exec_command = ["python", runner_script_path_cont]

                # Prepare environment variables for the wrapper script
                wrapper_env = {
                    # LLM_CODE_SOURCE_FILE is determined inside the wrapper based on BENCHMARK_TYPE
                    "TEST_SUITE_FILE": test_suite_path_cont,
                    # Pass the dictionaries directly from config as JSON strings
                    "FUNCTION_NAMES": json.dumps(self.config.FUNCTION_NAMES),
                    "FUNCTION_SIGNATURES": json.dumps(self.config.FUNCTION_SIGNATURES),
                    "BENCHMARK_TYPE": self.config.BENCHMARK_TYPE,
                    # Use getattr for optional config values, defaulting to False -> "0"
                    "CALCULATE_RATIO": "1" if getattr(self.config, 'CALCULATE_RATIO', False) else "0",
                    "TIME_SECONDARY_FUNCTION": "1" if getattr(self.config, 'TIME_SECONDARY_FUNCTION', False) else "0",
                    # Add PYTHONUNBUFFERED for immediate output flushing in wrapper script
                    "PYTHONUNBUFFERED": "1",
                    # Pass the actual LLM code filename from config
                    "LLM_CODE_FILENAME_ENV": self.config.LLM_CODE_FILENAME,
                }
                print(f"Framework Runner: Environment for wrapper: {wrapper_env}")

                exec_instance = self.docker_client.api.exec_create(
                    container=container.id,
                    cmd=exec_command,
                    environment=wrapper_env,
                    workdir=sandbox_dir,
                    stdout=True, stderr=True, tty=False
                )
                exec_id = exec_instance['Id']

                exec_stream = self.docker_client.api.exec_start(exec_id=exec_id, stream=True, demux=True)

                stdout_acc = bytearray() # Use bytearray for mutable accumulation
                stderr_buffer = bytearray() # Use bytearray for mutable accumulation
                exit_code = None

                print(f"Framework Runner: Streaming output from container exec_id: {exec_id}...")
                # Process stream, handling potential decoding errors and calling progress callback
                self._process_exec_stream(exec_stream, stdout_acc, stderr_buffer, progress_callback)

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

                # Process final output based on exit code
                self._process_final_output(exit_code, stdout_acc, stderr_buffer, results)


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
                # ExitStack handles container stop/remove and tempdir cleanup automatically
                print("Framework Runner: Benchmark execution attempt finished. Cleanup handled by ExitStack.")
                # Final progress update (ensure it happens even on error)
                if progress_callback:
                    final_status = 'Completed' if not results.get('error') else 'Error'
                    progress_callback({
                        'status': final_status,
                        'category': 'Finished',
                        'error': results.get('error'),
                        'message': f'Benchmark execution finished ({final_status}).'
                    })

        return results

    def _process_exec_stream(self, exec_stream, stdout_acc: bytearray, stderr_buffer: bytearray, progress_callback: Optional[Callable[[dict], None]]):
        """Helper to process the Docker exec stream."""
        for stdout_chunk, stderr_chunk in exec_stream:
            if stdout_chunk:
                stdout_acc.extend(stdout_chunk)
            if stderr_chunk:
                stderr_buffer.extend(stderr_chunk)
                # Process complete lines from stderr for progress updates
                lines = stderr_buffer.split(b'\n')
                stderr_buffer[:] = lines[-1] # Keep incomplete line (modify in-place)
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

    def _process_final_output(self, exit_code: Optional[int], stdout_acc: bytearray, stderr_buffer: bytearray, results: dict):
         """Helper to parse final output based on exit code."""
         # Decode bytearrays to strings for processing
         output_str_raw = stdout_acc.decode('utf-8', errors='replace')
         stderr_final = stderr_buffer.decode('utf-8', errors='replace') # Process remaining buffer

         if exit_code == 0:
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
                     # If wrapper reported an error, keep it
                     if container_results.get('error') and not results.get('error'):
                         results['error'] = f"Wrapper script error: {container_results['error']}"
                 else:
                     raise ValueError("Unexpected JSON format from wrapper stdout.")
             except (json.JSONDecodeError, ValueError) as json_e:
                 output_snippet = output_str_raw[:1000]
                 stderr_snippet = stderr_final[:500]
                 results['error'] = f"Failed to parse final JSON from container (Exit Code 0): {json_e}. Stdout:\n---\n{output_snippet}\n---\nStderr:\n---\n{stderr_snippet}\n---"
             except Exception as parse_e:
                  output_snippet = output_str_raw[:1000]
                  stderr_snippet = stderr_final[:500]
                  results['error'] = f"Unexpected error parsing container stdout (Exit Code 0): {parse_e}. Stdout:\n---\n{output_snippet}\n---\nStderr:\n---\n{stderr_snippet}\n---"
         elif exit_code is not None:
             stdout_snippet = output_str_raw[:500]
             stderr_snippet = stderr_final[:500]
             results['error'] = f"Container wrapper exited with code {exit_code}. Stderr:\n---\n{stderr_snippet}\n---\nStdout:\n---\n{stdout_snippet}\n---"
         else: # Exit code is None (shouldn't happen if inspect worked)
              results['error'] = f"Failed to determine container exec exit code."

    def _cleanup_container(self, container):
        """Safely stops and removes a Docker container."""
        if not container:
            return
        container_id = container.short_id
        try:
            print(f"Framework Runner: Stopping container {container_id}...")
            container.stop(timeout=10) # Give 10 seconds to stop gracefully
            print(f"Framework Runner: Removing container {container_id}...")
            container.remove(force=True) # Force remove if stop failed
            print(f"Framework Runner: Container {container_id} cleaned up.")
        except (APIError, DockerException) as e:
            # Log error but don't crash the main process
            print(f"Framework Runner WARNING: Error cleaning up container {container_id}: {e}. It might need manual removal.")
        except Exception as e:
             print(f"Framework Runner WARNING: Unexpected error cleaning up container {container_id}: {e}")
