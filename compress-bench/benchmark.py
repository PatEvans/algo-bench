"""
Module for benchmarking LLM-generated lossless compression algorithms.

Handles:
- Generating prompts for LLMs.
- Safely executing the generated code (compress/decompress) within Docker.
- Evaluating correctness, performance (speed), and compression ratio.
"""

import time
import json
import tempfile # For creating temporary files AND directories
import base64 # For decoding test suite data
import os # For file path operations
import random # Moved import to top level
from collections import defaultdict
from typing import Callable, Optional # For type hinting the callback (Removed Any)
import docker # For Docker interaction
from docker.errors import APIError, ImageNotFound # Specific Docker errors (Removed ContainerError)
from requests.exceptions import ConnectionError as DockerConnectionError # For Docker daemon connection issues
from contextlib import ExitStack # To manage multiple context managers (tempdir, container)
import tarfile # To create tar archives for copying into container
import io # To handle in-memory tar archive
# Socket import no longer needed here as we don't interact with exec stream directly
# import socket

# Import functions from the new test suite generator module
import test_suite_generator

# Using Docker containers provides better isolation than subprocess.
# Moved import random to top level

# Constant for the label used for all benchmarks run via evaluate_algorithm
BENCHMARKED_ALGORITHM_LABEL = "LLM C Compression [Docker]" # Updated label for C code
# Constant for the filename of the docker wrapper script
DOCKER_WRAPPER_SCRIPT_NAME = "docker_exec_wrapper.py"
# Constant for the filename of the test suite data inside the container
# Use the new default from the generator module
TEST_SUITE_DATA_FILENAME = test_suite_generator.DEFAULT_TEST_SUITE_FILE # e.g., "compression_test_suite.json"


# --- Prompt Generation ---

def generate_prompt_examples(num_examples: int = 2) -> list[tuple[bytes, bytes, bytes]]:
    """Generates small input/compressed/output examples for the compression prompt."""
    examples = []
    # Example 1: Simple text
    original1 = b"hello hello world"
    # Example compressed data (could be anything, just needs to decompress correctly)
    # This is illustrative; a real LLM might produce different compressed forms.
    compressed1 = b"h\x01\x01w\x01" # Simplistic example representation
    examples.append((original1, compressed1, original1))

    # Example 2: Repeating data
    original2 = b"AAAAABBBBB"
    compressed2 = b"A\x05B\x05" # Simplistic example representation
    examples.append((original2, compressed2, original2))

    # Add more complex examples if needed, ensuring compressed data is plausible
    # and can be decompressed back to original by a hypothetical function.

    return examples[:num_examples] # Return exactly num_examples

def create_compression_prompt(examples: Optional[list[tuple[bytes, bytes, bytes]]] = None) -> str:
    """
    Creates a prompt to ask an LLM for lossless compression and decompression functions.
    """
    base_prompt = """
Generate C code implementing a lossless data compression algorithm.
Start your response *directly* with the necessary `#include` directives or the `typedef` statement. Do not include any introductory text, language identifiers (like 'c' or '```c'), or markdown formatting before the actual code begins.

The code MUST include the following three functions with the exact signatures specified:

1.  `typedef struct { unsigned char* data; size_t size; } Buffer;`
    (You MUST include this struct definition.)

2.  `Buffer compress(const unsigned char* input_data, size_t input_size);`
    - Takes a pointer `input_data` to the raw input bytes and its `input_size`.
    - Compresses the data.
    - Allocates memory for the compressed data using `malloc`.
    - Returns a `Buffer` struct containing the pointer to the allocated compressed data and its size.
    - If compression fails or input is invalid, it should return a Buffer with data=NULL and size=0.

3.  `Buffer decompress(const unsigned char* compressed_data, size_t compressed_size);`
    - Takes a pointer `compressed_data` to the compressed bytes and its `compressed_size`.
    - Decompresses the data.
    - Allocates memory for the original data using `malloc`.
    - Returns a `Buffer` struct containing the pointer to the allocated original data and its size.
    - If decompression fails or input is invalid, it should return a Buffer with data=NULL and size=0.

4.  `void free_buffer(Buffer buffer);`
    - Takes a `Buffer` struct (as returned by `compress` or `decompress`).
    - Frees the memory pointed to by `buffer.data` using `free`.
    - This function is crucial for memory management by the caller.

Constraints and Requirements:
- The code MUST be self-contained standard C (C99 or C11 recommended).
- Include necessary headers (like `<stdlib.h>`, `<string.h>`, `<stddef.h>`).
- DO NOT use external compression libraries (like zlib, zstd, etc.). Implement the algorithm directly.
- The functions MUST handle arbitrary byte sequences, not just text.
- The functions MUST be thread-safe if they use global state (prefer avoiding global state).
- The goal is low latency and good compression ratio. Aim for a balance.
- DO NOT include a `main` function. Provide only the struct definition and the three required functions.
- Ensure `malloc` return values are checked for NULL.
"""
    # Examples are less critical for C prompt structure but can be kept for conceptual illustration
    if examples:
        example_str = "\n\nConceptual Examples (Illustrative - actual C implementation details will differ):\n"
        for original, compressed, expected_decompressed in examples:
            example_str += f"Original Data (bytes): {repr(original)}\n"
            example_str += f"Conceptually Compressed To (bytes): {repr(compressed)}\n"
            example_str += f"Decompressed Back To (bytes): {repr(expected_decompressed)}\n\n"
        return base_prompt + example_str.strip()
    else:
        return base_prompt

# Test suite generation/loading functions are now in test_suite_generator.py

# --- Evaluation Logic ---

def evaluate_algorithm(generated_code: str, categorized_test_cases_b64: dict, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Evaluates the generated compression/decompression code using provided test cases (base64 encoded),
    running the evaluation inside a Docker container and optionally reporting progress.

    Args:
        generated_code: The Python code string generated by the LLM (containing compress and decompress).
        categorized_test_cases_b64: A dictionary containing the base64 encoded test cases, keyed by category.
        progress_callback: An optional function to call with progress updates from the Docker wrapper.

    Returns:
        A dictionary containing evaluation results from the Docker container:
        - 'correctness': 1 if all cases passed, 0 otherwise.
        - 'avg_compression_time_ms': Average compression time (ms).
        - 'avg_decompression_time_ms': Average decompression time (ms).
        - 'avg_compression_ratio': Average compression ratio.
        - 'error': Error message if the wrapper script encountered a critical error.
    """
    # Initialize results structure matching the expected output from the wrapper
    results = {
        'correctness': 0, # Default to failure
        'avg_compression_time_ms': None,
        'avg_decompression_time_ms': None,
        'avg_compression_ratio': None,
        'error': None
    }

    if not categorized_test_cases_b64:
        results['error'] = "No test cases provided for evaluation."
        return results

    # Calculate total cases for initial progress reporting
    # The test cases are now base64 strings, but the structure is the same
    total_overall_cases_calculated = sum(len(cases) for cases in categorized_test_cases_b64.values())

    # Define the Docker image to use (MUST match the tag used in 'docker build')
    DOCKER_IMAGE = "compression-benchmark" # Use the locally built image
    # Define container resource limits (adjust as needed)
    # Timeout for the *entire* exec call (adjust as needed, compression might take longer)
    EXEC_TIMEOUT_SECONDS = 600 # Increased timeout to 10 minutes
    CONTAINER_MEM_LIMIT = "1g" # Keep 1 GB memory limit, adjust if needed for large cases
    CONTAINER_CPU_SHARES = 512 # Relative CPU weight (default 1024)

    # Initialize Docker client
    docker_client = None
    try:
        # Signal start of evaluation process
        if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Initializing', 'message': 'Initializing evaluation environment...'})

        # Increase timeout for initial connection to Docker daemon
        docker_client = docker.from_env(timeout=120) # Increased initial connection timeout to 120s
        docker_client.ping() # Verify connection after increasing timeout
        print("Successfully connected to Docker daemon.")
        if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Connecting Docker', 'message': 'Docker client connected.'})

        # Increase the default timeout for API calls (including streaming reads)
        # Default is often 60s, but read timeouts might be shorter or hit socket defaults.
        # Let's set it explicitly higher for potentially long benchmark runs.
        docker_client.api.timeout = 600 # Set timeout to 600 seconds (10 minutes)
        print(f"Docker client API timeout set to {docker_client.api.timeout} seconds.")

        # Ensure image exists
        try:
            docker_client.images.get(DOCKER_IMAGE)
            print(f"Docker image {DOCKER_IMAGE} found locally.")
            if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Checking Image', 'message': f'Docker image ready ({DOCKER_IMAGE}).'})
        except ImageNotFound:
            print(f"Pulling Docker image: {DOCKER_IMAGE}...")
            # Keep this message as pulling can take time
            if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Pulling Image', 'message': f'Pulling Docker image {DOCKER_IMAGE}...'})
            docker_client.images.pull(DOCKER_IMAGE)
            print(f"Docker image {DOCKER_IMAGE} pulled.")
            if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Image Pulled', 'message': f'Docker image pulled ({DOCKER_IMAGE}).'})
        print(f"Using Docker image: {DOCKER_IMAGE}")
    except (DockerConnectionError, APIError, Exception) as e:
        results['error'] = f"Docker initialization failed: {e}. Is Docker running?"
        print(f"Error: {results['error']}")
        if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
        return results

    # Construct the absolute path to the Docker wrapper script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_script_path = os.path.join(script_dir, DOCKER_WRAPPER_SCRIPT_NAME)

    # Read the Docker execution wrapper script content
    try:
        print(f"Attempting to read wrapper script from: {wrapper_script_path}") # Debug print
        with open(wrapper_script_path, 'r', encoding='utf-8') as f:
            exec_wrapper_code = f.read()
        print(f"Successfully read wrapper script: {DOCKER_WRAPPER_SCRIPT_NAME}") # Debug print
    except FileNotFoundError:
        results['error'] = f"Critical Error: Docker wrapper script '{DOCKER_WRAPPER_SCRIPT_NAME}' not found at expected path: {wrapper_script_path}."
        print(results['error'])
        if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
        return results
    except Exception as e:
        results['error'] = f"Critical Error: Failed to read Docker wrapper script '{DOCKER_WRAPPER_SCRIPT_NAME}' from path '{wrapper_script_path}': {e}"
        print(results['error'])
        if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
        return results


    container = None
    # Use ExitStack for robust cleanup of tempdir and container
    with ExitStack() as stack:
        try:
           # Create TemporaryDirectory for the LLM code and runner script
           temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
           llm_code_filename = "llm_compress.c" # Changed filename for C code
           # Use the constant for the wrapper script filename on the host
           runner_script_filename_host = DOCKER_WRAPPER_SCRIPT_NAME
           llm_code_path_host = os.path.join(temp_dir, llm_code_filename) # Path for C source file
           # The runner script is read from its fixed location, but we'll write its content
           # to the temp dir for copying into the container, using a consistent name inside.
           runner_script_filename_cont = "exec_runner.py" # Name inside container
           runner_script_path_host = os.path.join(temp_dir, runner_script_filename_cont) # Path on host temp dir
           # Define path for the test suite data file on the host
           test_suite_data_path_host = os.path.join(temp_dir, TEST_SUITE_DATA_FILENAME)

           sandbox_dir = "/sandbox" # Mount point inside container
           llm_code_path_cont = f"{sandbox_dir}/{llm_code_filename}" # Container path for C source
           runner_script_path_cont = f"{sandbox_dir}/{runner_script_filename_cont}" # Container path for runner
           test_suite_data_path_cont = f"{sandbox_dir}/{TEST_SUITE_DATA_FILENAME}" # Container path for test data

           # --- DEBUG: Verify host files before container start ---
           print(f"DEBUG: Host temporary directory: {temp_dir}")
           print(f"DEBUG: Host LLM code path: {llm_code_path_host}")
           print(f"DEBUG: Host runner script path (in temp): {runner_script_path_host}")
           # --- END DEBUG ---

           # Preprocess generated_code: Remove leading 'c' line if present
           code_to_write = generated_code
           lines = generated_code.split('\n', 1) # Split into first line and the rest
           if len(lines) > 0 and lines[0].strip() == 'c':
               print("DEBUG: Stripping leading 'c' line from generated code.") # Debug print
               code_to_write = lines[1] if len(lines) > 1 else "" # Use the rest, or empty if only 'c' was present

           # Write the potentially modified code and the wrapper script to the host temp directory
           with open(llm_code_path_host, 'w', encoding='utf-8') as f_llm_script:
               f_llm_script.write(code_to_write)
           with open(runner_script_path_host, 'w', encoding='utf-8') as f_runner_script:
               f_runner_script.write(exec_wrapper_code) # Write the content read earlier

           # --- DEBUG: Check if files exist on host before copy ---
           llm_exists = os.path.exists(llm_code_path_host)
           runner_exists = os.path.exists(runner_script_path_host)
           print(f"DEBUG: Host LLM code exists before run?: {llm_exists}")
           print(f"DEBUG: Host runner script exists before run?: {runner_exists}")
           if not llm_exists or not runner_exists:
                print("ERROR: Script files not found on host before starting container!")
                # Consider raising an error here if needed
           # --- END DEBUG ---

           # Write the base64 encoded test suite data to the host temp directory as JSON
           try:
               with open(test_suite_data_path_host, 'w', encoding='utf-8') as f_suite:
                   # Dump the already base64 encoded dictionary
                   json.dump(categorized_test_cases_b64, f_suite)
               print(f"DEBUG: Base64 encoded test suite data written to {test_suite_data_path_host}")
           except Exception as e:
               raise RuntimeError(f"Failed to write base64 test suite data JSON to host: {e}")


           # Start the container once, keep it running
           print("Starting persistent Docker container...")
           container = docker_client.containers.run(
               image=DOCKER_IMAGE,
               command=["sleep", "infinity"], # Keep container alive
               # --- Volume mount removed, will copy files instead ---
               # volumes={temp_dir: {'bind': sandbox_dir}},
               working_dir=sandbox_dir, # Still useful for exec_run context
               mem_limit=CONTAINER_MEM_LIMIT,
               cpu_shares=CONTAINER_CPU_SHARES,
                detach=True,
                auto_remove=False, # We need to manage removal manually after loop
                # Security options
                # read_only=True, # Root filesystem read-only (volume is separate)
               # network_mode='none', # Disable networking
           )
           # Ensure container is stopped and removed at the end
           stack.callback(lambda c: (c.stop(timeout=5), c.remove(force=True)), container)
           print(f"Container {container.short_id} started.")
           if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Starting Container', 'message': f'Container started ({container.short_id}).'})

           # --- Create sandbox directory inside container ---
           # Although working_dir is set, explicitly create it for clarity and put_archive target
           print(f"Creating {sandbox_dir} inside container...")
           container.exec_run(cmd=f"mkdir -p {sandbox_dir}")

           # --- Prepare and copy files into the container ---
           print(f"Copying {llm_code_filename}, {runner_script_filename_cont}, and {TEST_SUITE_DATA_FILENAME} to container:{sandbox_dir}...")
           # Create an in-memory tar archive
           tar_stream = io.BytesIO()
           with tarfile.open(fileobj=tar_stream, mode='w') as tar:
               # Add llm_sort.py
               tar.add(llm_code_path_host, arcname=llm_code_filename)
               # Add the runner script using its container-internal name
               tar.add(runner_script_path_host, arcname=runner_script_filename_cont)
               # Add the test suite data file
               tar.add(test_suite_data_path_host, arcname=TEST_SUITE_DATA_FILENAME)

           tar_stream.seek(0) # Rewind the stream
           # Copy the archive to the container
           container.put_archive(path=sandbox_dir, data=tar_stream)
           print("Files copied.")
           if progress_callback: progress_callback({'status': 'Setup', 'category': 'Setup: Copying Files', 'message': 'Benchmark files copied to container.'})

           # Optional: Short delay after copy might still be beneficial? Unlikely needed now.
           # time.sleep(0.1)

           # --- Execute the wrapper script ONCE inside the container ---
           print(f"Executing wrapper script {runner_script_path_cont} in container...")
           # This is the final setup step before execution starts
           if progress_callback:
               progress_callback({
                   'status': 'Running', # Change status to Running here
                   'category': 'Executing All Cases', # Keep category general for the main run
                   'total_cases': total_overall_cases_calculated,
                   'message': 'Starting execution of all test cases in container...' # Updated message
               })

           exec_command = ["python", runner_script_path_cont]
           container_results = None
           exec_error = None

           try:
               # Use lower-level API to get exec_id before streaming
               exec_create_resp = docker_client.api.exec_create(
                   container=container.id,
                   cmd=exec_command,
                   workdir=sandbox_dir,
                   stdout=True,
                   stderr=True,
                   tty=False # Important: TTY should be False for demux=True to work correctly
               )
               exec_id = exec_create_resp['Id']

               # Start the execution and get the stream
               exec_stream_generator = docker_client.api.exec_start(
                   exec_id=exec_id,
                   stream=True,
                   demux=True # Separate stdout (1) and stderr (2)
               )

               stdout_acc = b""
               stderr_buffer = b"" # Buffer for potentially incomplete stderr lines
               exit_code = None # Will be determined after stream finishes

               print(f"Streaming output from container exec_id: {exec_id}...")
               # Iterate safely, checking for None from the generator
               # Note: exec_start returns a generator directly
               print(f"DEBUG BENCHMARK: Starting stream read loop for exec_id: {exec_id}") # DEBUG ADDED
               # According to docker-py docs for exec_start(stream=True, demux=True),
               # it yields tuples of (stdout_chunk, stderr_chunk).
               for stdout_chunk, stderr_chunk in exec_stream_generator:
                   # Log the raw chunks received
                   # print(f"DEBUG BENCHMARK: Raw chunks received - stdout: {repr(stdout_chunk)}, stderr: {repr(stderr_chunk)}") # Optional: Very verbose

                   if stdout_chunk:
                       chunk_repr = repr(stdout_chunk)
                       print(f"DEBUG BENCHMARK: Received STDOUT chunk - Size: {len(stdout_chunk)}, Content (repr): {chunk_repr[:200]}{'...' if len(stdout_chunk) > 200 else ''}") # DEBUG ADDED
                       stdout_acc += stdout_chunk

                   if stderr_chunk:
                       chunk_repr = repr(stderr_chunk)
                       print(f"DEBUG BENCHMARK: Received STDERR chunk - Size: {len(stderr_chunk)}, Content (repr): {chunk_repr[:200]}{'...' if len(stderr_chunk) > 200 else ''}") # DEBUG ADDED
                       stderr_buffer += stderr_chunk
                       # Process complete lines from stderr buffer
                       lines = stderr_buffer.split(b'\n')
                       stderr_buffer = lines[-1] # Keep incomplete line in buffer
                       for line_bytes in lines[:-1]:
                           line_str = line_bytes.decode('utf-8', errors='replace').strip()
                           if not line_str: continue # Skip empty lines
                           try:
                               progress_json = json.loads(line_str)
                               if progress_json.get("type") == "progress" and progress_callback:
                                   # Pass the inner 'data' dict to the callback
                                   progress_callback(progress_json.get("data", {}))
                               else:
                                   # Log unexpected JSON or non-progress messages from stderr
                                   print(f"DEBUG (stderr JSON): {line_str}")
                           except json.JSONDecodeError:
                               # Log non-JSON lines from stderr for debugging
                               print(f"DEBUG (stderr raw): {line_str}")
                           except Exception as cb_err:
                               print(f"ERROR: Progress callback failed: {cb_err}")

                   # Check if both are None, which might indicate the end, although the loop should terminate naturally.
                   if stdout_chunk is None and stderr_chunk is None:
                       print(f"DEBUG BENCHMARK: Received (None, None) from stream generator (exec_id: {exec_id}). Assuming stream ended.") # DEBUG ADDED
                       break # Exit loop if we get (None, None) explicitly

               # --- Stream finished, now inspect exit code using the saved exec_id ---
               print(f"DEBUG BENCHMARK: Stream finished for exec_id: {exec_id}. Inspecting exit code...") # DEBUG ADDED
               exec_inspect = docker_client.api.exec_inspect(exec_id) # Use the saved exec_id
               exit_code = exec_inspect.get('ExitCode')
               print(f"DEBUG BENCHMARK: exec_inspect result: {exec_inspect}") # DEBUG ADDED
               print(f"DEBUG BENCHMARK: Determined Exit Code: {exit_code}") # DEBUG ADDED

               # Handle potential None exit code
               if exit_code is None:
                   print(f"WARNING/DEBUG: exec_inspect returned ExitCode=None immediately after stream finished. Inspect: {exec_inspect}") # DEBUG ADDED
                   # Optionally add a small delay and retry inspect here if needed

               # Process final stdout result
               print(f"DEBUG BENCHMARK: Final accumulated stdout size: {len(stdout_acc)} bytes.") # DEBUG ADDED
               stdout_content_for_debug = stdout_acc.decode('utf-8', errors='replace') # Decode for logging
               print(f"DEBUG BENCHMARK: Final accumulated stdout content (first 1000 chars):\n---\n{stdout_content_for_debug[:1000]}\n---") # DEBUG ADDED

               if exit_code == 0:
                   print("DEBUG BENCHMARK: Exit code is 0. Processing stdout...") # DEBUG ADDED
                   output_str_raw = stdout_acc.decode('utf-8', errors='replace')
                   # Define markers
                   start_marker = "---WRAPPER_STDOUT_MARKER_BEFORE---"
                   end_marker = "---WRAPPER_STDOUT_MARKER_AFTER---"
                   try:
                       # Find markers
                       start_index = output_str_raw.find(start_marker)
                       end_index = output_str_raw.find(end_marker)

                       if start_index == -1 or end_index == -1 or end_index <= start_index:
                           print(f"DEBUG BENCHMARK: Markers not found or in wrong order in stdout.") # DEBUG ADDED
                           raise ValueError(f"Could not find expected markers '{start_marker}' and '{end_marker}' in container stdout.")

                       # Extract content between markers
                       json_content_str = output_str_raw[start_index + len(start_marker):end_index].strip()
                       print(f"DEBUG BENCHMARK: Extracted JSON content string (first 500 chars): {json_content_str[:500]}") # DEBUG ADDED

                       if not json_content_str:
                           print("DEBUG BENCHMARK: Extracted content between markers is empty.") # DEBUG ADDED
                           raise ValueError("Container stdout between markers was empty (Exit Code 0).")

                       # Parse the extracted JSON content
                       final_message = json.loads(json_content_str)
                       if final_message.get("type") == "result":
                           container_results = final_message.get("data", {})
                           print("Successfully received and parsed final result from container stdout.")
                           results.update(container_results) # Update host results
                           if container_results.get('error'):
                               exec_error = f"Container wrapper script reported an internal error: {container_results['error']}"
                               results['error'] = exec_error # Ensure host result reflects this
                       else:
                           raise ValueError(f"Unexpected JSON format in stdout: Missing 'type' or not 'result'. Content: {json_content_str[:200]}...")
                   except (json.JSONDecodeError, ValueError) as json_e:
                       output_snippet = output_str_raw[:1000] # Show raw output in error
                       exec_error = f"Failed to decode/parse final JSON result from container stdout (Exit Code 0): {json_e}. Output:\n---\n{output_snippet}\n---"
                       results['error'] = exec_error
                   except Exception as parse_e:
                       output_snippet = output_str_raw[:1000] # Show raw output in error
                       exec_error = f"Unexpected error parsing container stdout (Exit Code 0): {parse_e}. Output:\n---\n{output_snippet}\n---"
                       results['error'] = exec_error
               elif exit_code is not None: # Execution failed (non-zero exit code)
                   stderr_snippet = stderr_buffer.decode('utf-8', errors='replace')[:500] # Include final stderr buffer content
                   stdout_snippet = stdout_acc.decode('utf-8', errors='replace')[:500] # Use the raw stdout here
                   exec_error = f"Container exec wrapper exited with code {exit_code}. Stderr:\n---\n{stderr_snippet}\n---\nStdout:\n---\n{stdout_snippet}\n---"
                   results['error'] = exec_error
               else: # Exit code remained None (problem inspecting exec)
                    exec_error = f"Failed to determine container exec exit code. Inspect result: {exec_inspect}"
                    results['error'] = exec_error


           except (APIError, DockerConnectionError) as docker_exec_err:
               exec_error = f"Docker API/Connection error during exec stream: {docker_exec_err}"
               results['error'] = exec_error
           except Exception as host_exec_e:
               exec_error = f"Host error during container exec_run call: {host_exec_e}"
               results['error'] = exec_error

           # --- Final Progress Update ---
           if progress_callback:
               final_status = 'Completed' if not results.get('error') else 'Error'
               progress_callback({
                   'status': final_status,
                   'category': 'Finished',
                   'total_cases': total_overall_cases_calculated,
                   'error': results.get('error'), # Report the final error status
                   'message': 'Container execution finished.'
               })

           if exec_error:
               print(f"Error during container execution: {exec_error}")
           # No more loops or per-case logic needed here

        # Catch errors during the initial container setup phase (remains the same)
        except (APIError, DockerConnectionError) as docker_setup_err:
            llm_error_str = f"Docker API/Connection error during container setup: {docker_setup_err}"
            results['error'] = llm_error_str
            print(f"Aborting evaluation due to Docker setup error: {llm_error_str}")
            if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
            # ExitStack will handle cleanup if container was partially started
            return results
        except Exception as setup_exec_e:
            llm_error_str = f"Error setting up container or temporary directory: {setup_exec_e}"
            results['error'] = llm_error_str
            print(f"Aborting evaluation due to setup error: {llm_error_str}")
            if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
            # ExitStack will handle cleanup
            return results
        finally:
            # ExitStack automatically handles stopping/removing the container and deleting the temp dir
            if container:
                 print(f"Container {container.short_id} stopped and removed.")
            else:
                 print("No container was started or setup failed.")


    # --- Final Results ---
    # The 'results' dictionary is now populated directly from the container's output
    # or contains an error message if the execution failed.
    # No further calculation needed here.

    return results

# Now accepts pre-generated code and base64 encoded test cases.
def run_single_benchmark(llm_name: str, generated_code: str, categorized_test_cases_b64: dict, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Runs the evaluation part of a compression benchmark for a single LLM, using pre-generated code
    and provided base64 encoded test cases, optionally reporting progress. Evaluation runs inside Docker.

    Args:
        llm_name: The identifier for the LLM used (for reporting).
        generated_code: The Python code string (containing compress and decompress functions).
        categorized_test_cases_b64: A dictionary containing the base64 encoded test cases, keyed by category.
        progress_callback: An optional function to call with progress updates during evaluation.

    Returns:
        A dictionary containing benchmark evaluation results based on the Docker wrapper's output.
    """
    algorithm_name = BENCHMARKED_ALGORITHM_LABEL # Use the constant label
    print(f"Evaluating compression benchmark for {llm_name} ({algorithm_name}) using provided code...")

    # Code generation is done *before* calling this function.

    # --- Run Evaluation ---
    # Pass the base64 encoded test cases directly to evaluate_algorithm
    evaluation_results = evaluate_algorithm(
        generated_code=generated_code,
        categorized_test_cases_b64=categorized_test_cases_b64,
        progress_callback=progress_callback # Pass callback for evaluation steps
    )

    # --- Callback for completion of evaluation ---
    final_status = 'Completed' if not evaluation_results.get('error') else 'Error'
    if progress_callback:
        progress_callback({
            'status': final_status, 'category': 'Finished',
            'error': evaluation_results.get('error')
        })

    # Structure the final result based on the evaluation output
    # The keys should match the database schema
    return {
        'llm': llm_name,
        'algorithm': algorithm_name,
        'correctness': evaluation_results.get('correctness'), # 0 or 1
        'avg_compression_time_ms': evaluation_results.get('avg_compression_time_ms'),
        'avg_decompression_time_ms': evaluation_results.get('avg_decompression_time_ms'),
        'avg_compression_ratio': evaluation_results.get('avg_compression_ratio'),
        'error': evaluation_results.get('error'),
        'generated_code': generated_code # Include the code in the final result dict
    }


# Note: The __main__ block below is for standalone testing/example usage.
if __name__ == '__main__':
    import argparse # Keep import here
    import traceback # Import traceback for better error reporting in main

    parser = argparse.ArgumentParser(description="Compression Benchmark Execution Examples")
    # Use the default from the generator module
    parser.add_argument('--suite-file', default=test_suite_generator.DEFAULT_TEST_SUITE_FILE, help="Specify the compression test suite JSON file path")

    args = parser.parse_args()

    # Example of running benchmarks using a loaded suite
    print("Running example compression benchmarks with loaded suite...")
    try:
        # Load the base64 encoded test suite
        test_suite_b64 = test_suite_generator.load_test_suite(args.suite_file)

        # --- Run example with fixed (dummy) compression code ---
        print("\nRunning example benchmark with fixed dummy compression code...")

        # IMPORTANT: This is a placeholder C code example.
        # This dummy C implementation does NOT actually compress.
        EXAMPLE_DUMMY_C_COMPRESSION_CODE = """
#include <stdlib.h>
#include <string.h>
#include <stddef.h> // For size_t

// Required struct definition
typedef struct {
    unsigned char* data;
    size_t size;
} Buffer;

// Dummy compress: Allocates memory and copies input data (no compression)
Buffer compress(const unsigned char* input_data, size_t input_size) {
    Buffer output_buffer;
    output_buffer.data = NULL;
    output_buffer.size = 0;

    if (input_data == NULL && input_size > 0) {
        return output_buffer; // Invalid input
    }
    if (input_size == 0) {
         // Handle zero-size input: return empty buffer allocated
         output_buffer.data = (unsigned char*)malloc(0); // Allocate 0 bytes is implementation-defined but often works
         output_buffer.size = 0;
         // No need to check malloc(0) for NULL usually, but safe practice varies
         return output_buffer;
    }


    // Allocate memory for the "compressed" data (just a copy)
    output_buffer.data = (unsigned char*)malloc(input_size);
    if (output_buffer.data == NULL) {
        // Allocation failed
        return output_buffer; // Return empty buffer
    }

    // Copy input data to output buffer
    memcpy(output_buffer.data, input_data, input_size);
    output_buffer.size = input_size;

    return output_buffer;
}

// Dummy decompress: Allocates memory and copies input data (no decompression)
Buffer decompress(const unsigned char* compressed_data, size_t compressed_size) {
    Buffer output_buffer;
    output_buffer.data = NULL;
    output_buffer.size = 0;

     if (compressed_data == NULL && compressed_size > 0) {
        return output_buffer; // Invalid input
    }
     if (compressed_size == 0) {
         // Handle zero-size input: return empty buffer allocated
         output_buffer.data = (unsigned char*)malloc(0);
         output_buffer.size = 0;
         return output_buffer;
    }

    // Allocate memory for the "decompressed" data (just a copy)
    output_buffer.data = (unsigned char*)malloc(compressed_size);
    if (output_buffer.data == NULL) {
        // Allocation failed
        return output_buffer; // Return empty buffer
    }

    // Copy compressed data to output buffer
    memcpy(output_buffer.data, compressed_data, compressed_size);
    output_buffer.size = compressed_size;

    return output_buffer;
}

// Function to free the allocated buffer data
void free_buffer(Buffer buffer) {
    if (buffer.data != NULL) {
        free(buffer.data);
        // Optional: Set to NULL after free to prevent double-free issues if struct is reused
        // buffer.data = NULL;
        // buffer.size = 0;
    }
}
"""
        # Run the benchmark using the fixed dummy C code
        result_example = run_single_benchmark(
            llm_name="Example Dummy C Compression", # Use a descriptive name
            generated_code=EXAMPLE_DUMMY_C_COMPRESSION_CODE,
            categorized_test_cases_b64=test_suite_b64,
            progress_callback=lambda p: print(f"Progress: {p}") # Simple print callback
        )
        print("\nExample Dummy Compression Benchmark Result:\n", json.dumps(result_example, indent=2))

    except FileNotFoundError:
        print(f"Test suite file '{args.suite_file}' not found. Generate it first using: python test_suite_generator.py --generate-suite")
    except Exception as e:
        print(f"An error occurred during example benchmark run: {e}\n{traceback.format_exc()}") # Add traceback
