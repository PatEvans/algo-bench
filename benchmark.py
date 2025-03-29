"""
Module for benchmarking LLM-generated sorting algorithms.

Handles:
- Generating prompts for LLMs.
- Safely executing the generated code.
- Evaluating correctness and performance.
"""

import time
import random
import llm_interface
import json
# import subprocess # No longer needed for execution
import sys # To get the current Python executable path
import tempfile # For creating temporary files AND directories
import os # For file path operations
import shutil # For removing temp dirs if needed (though tempfile handles it)
from collections import defaultdict
import time # Re-importing for clarity, used within functions
import random # Re-importing for clarity, used within functions
import llm_interface # Re-importing for clarity, used within functions
from typing import Callable, Optional, Any # For type hinting the callback
import io # To capture stdout within the subprocess script
# import contextlib # No longer needed for host-side redirection
import docker # For Docker interaction
from docker.errors import APIError, ImageNotFound, ContainerError # Specific Docker errors
from requests.exceptions import ConnectionError as DockerConnectionError # For Docker daemon connection issues
from contextlib import ExitStack # To manage multiple context managers (tempdir, container)

# Using Docker containers provides better isolation than subprocess.

# Constant for the generic algorithm name
GENERAL_ALGORITHM_NAME = "LLM General Sort"

def generate_prompt_examples(num_examples: int = 3, max_size: int = 8, min_val: int = -10, max_val: int = 10) -> list[tuple[list[int], list[int]]]:
    """Generates small input/output examples for the LLM prompt."""
    examples = []
    # Ensure basic cases are covered
    if num_examples >= 1:
        examples.append(([], [])) # Empty list
    if num_examples >= 2:
        examples.append(([5], [5])) # Single element
    if num_examples >= 3:
        examples.append(([3, 1, 4, 1, 5, 9], [1, 1, 3, 4, 5, 9])) # Basic unsorted with duplicate

    # Add more random examples if needed
    current_examples = len(examples)
    for _ in range(max(0, num_examples - current_examples)):
        size = random.randint(2, max_size)
        input_list = [random.randint(min_val, max_val) for _ in range(size)]
        output_list = sorted(input_list)
        examples.append((input_list, output_list))

    return examples[:num_examples] # Return exactly num_examples

def create_sort_prompt(examples: Optional[list[tuple[list[int], list[int]]]] = None) -> str:
    """
    Creates a prompt to ask an LLM for an efficient general-purpose sorting algorithm,
    optionally including examples.
    """
    base_prompt = (
        "Generate a Python function named `sort_algorithm` that implements an efficient sorting algorithm "
        "suitable for general use cases (handling various data distributions like random, sorted, reversed, duplicates, etc.).\n"
        "The function MUST take a list of numbers as input and return a new sorted list.\n"
        "IMPORTANT: The function MUST NOT use the built-in sorted() function or the .sort() method.\n"
        "IMPORTANT: The function MUST NOT print anything to standard output. It should ONLY return the sorted list."
    )

    if examples:
        example_str = "\n\nHere are some examples of how the function should behave:\n"
        for input_list, output_list in examples:
            example_str += f"Input: {input_list}\nOutput: {output_list}\n\n"
        return base_prompt + example_str.strip() # Add examples and remove trailing newline
    else:
        return base_prompt

# Increased default sizes for more meaningful timing
def generate_test_cases(size_small=10, size_medium=10000, size_large=1000000, num_cases_per_type=2) -> dict[str, list[list[int]]]: # Corrected return type hint
    """
    Generates integer test cases based on specified patterns:
    - Randomized (within a range)
    - Duplicates (many repeating elements)
    - Sorted (ascending)
    - Reversed (descending)
    - Nearly Sorted (a few elements swapped)

    Args:
        size_small: Size for small test cases.
        size_medium: Size for medium test cases.
        size_large: Size for large test cases.
        num_cases_per_type: Number of random/duplicate/nearly sorted cases per size.

    Returns:
        A dictionary where keys are category names (e.g., "random_small", "sorted_large")
        and values are lists containing the test case lists for that category.
    """
    cases_by_category = defaultdict(list)
    cases_by_category['special_empty'] = [[]]
    cases_by_category['special_single'] = [[5]]

    sizes = {'small': size_small, 'medium': size_medium, 'large': size_large}
    min_val, max_val = -10000, 10000

    total_cases = 2 # Start with empty and single

    for name, size in sizes.items():
        if size == 0: continue

        print(f"Generating cases for size: {name} ({size})...")

        # 1. Randomized
        cat_random = f"random_{name}"
        for i in range(num_cases_per_type):
            random_case = [random.randint(min_val, max_val) for _ in range(size)]
            cases_by_category[cat_random].append(random_case)
            print(f"  - Added {cat_random} case {i+1}")
            total_cases += 1

        # 2. With Duplicates
        cat_duplicates = f"duplicates_{name}"
        duplicate_range_max = max(1, size // 10)
        for i in range(num_cases_per_type):
            duplicate_case = [random.randint(0, duplicate_range_max) for _ in range(size)]
            cases_by_category[cat_duplicates].append(duplicate_case)
            print(f"  - Added {cat_duplicates} case {i+1}")
            total_cases += 1

        # 3. Sorted (Ascending)
        cat_sorted = f"sorted_{name}"
        sorted_case = list(range(size))
        cases_by_category[cat_sorted].append(sorted_case)
        print(f"  - Added {cat_sorted} case")
        total_cases += 1

        # 4. Reversed (Descending)
        cat_reversed = f"reversed_{name}"
        reversed_case = list(range(size, 0, -1))
        cases_by_category[cat_reversed].append(reversed_case)
        print(f"  - Added {cat_reversed} case")
        total_cases += 1

        # 5. Nearly Sorted (e.g., swap a few pairs) - Optional but good
        cat_nearly_sorted = f"nearly_sorted_{name}"
        num_swaps = max(1, size // 20) # Swap ~5% of elements
        for i in range(num_cases_per_type):
            nearly_sorted_case = list(range(size))
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(size), 2)
                nearly_sorted_case[idx1], nearly_sorted_case[idx2] = nearly_sorted_case[idx2], nearly_sorted_case[idx1]
            cases_by_category[cat_nearly_sorted].append(nearly_sorted_case)
            print(f"  - Added {cat_nearly_sorted} case {i+1}")
            total_cases += 1


    # Example of adding a very large case (use with caution!)
    # size_xl = 10_000_000
    # cat_xl = "random_xl"
    # print(f"Generating XL random case ({size_xl})...")
    # cases_by_category[cat_xl].append([random.randint(min_val, max_val) for _ in range(size_xl)])
    # total_cases += 1

    print(f"Generated a total of {total_cases} test cases across {len(cases_by_category)} categories.")
    return dict(cases_by_category) # Convert back to regular dict

def evaluate_algorithm(generated_code: str, categorized_test_cases: dict, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Evaluates the generated sorting algorithm code using provided test cases, optionally reporting progress.

    Args:
        generated_code: The Python code string generated by the LLM.
        categorized_test_cases: A dictionary containing the test cases, keyed by category.
        progress_callback: An optional function to call with progress updates.
                           The dictionary passed to the callback might contain keys like:
                           'current_case', 'total_cases', 'category', 'status',
                           'input_snippet', 'output_snippet', 'error'.

    Returns:
        A dictionary containing evaluation results:
        - 'correctness': Overall correctness percentage.
        - 'avg_time_ms': Overall average LLM time (ms) for correct runs.
        - 'baseline_avg_time_ms': Overall average baseline time (ms).
        - 'performance_details': A dict mapping category names to {'correctness': float, 'avg_time_ms': float, 'baseline_avg_time_ms': float, 'count': int}.
        - 'error': Error message if execution or validation fails.
    """
    results = {
        'correctness': 0.0,
        'avg_time_ms': None,
        'baseline_avg_time_ms': None,
        'performance_details': {}, # To store per-category results
        'error': None
    }

    if not categorized_test_cases:
        results['error'] = "No test cases provided for evaluation."
        return results

    # Aggregators for overall results
    overall_correct_count = 0
    overall_llm_time = 0
    overall_baseline_time = 0
    overall_total_cases = 0
    overall_llm_runs_timed = 0 # Count only successful, correct LLM runs for averaging

    # Aggregators for per-category results
    category_results = defaultdict(lambda: {'correct_count': 0, 'llm_time': 0, 'baseline_time': 0, 'case_count': 0, 'llm_runs_timed': 0})

    # Define the boilerplate code for the temporary script
    # This code reads JSON from stdin, calls sort_algorithm, and prints JSON to stdout
    script_boilerplate_header = """
import sys
import json

# --- Start of LLM Generated Code ---
"""
    # Modified footer to read input from a file inside the container (/sandbox/input.json)
    # and write JSON result to stdout, errors/prints to stderr.
    script_boilerplate_footer = """
# --- End of LLM Generated Code ---

import sys
import json
import io
import traceback # For better error reporting
import os # Need os to check for input file

INPUT_FILE = '/sandbox/input.json' # Standardized input file path within container
OUTPUT_FILE = '/sandbox/output.json' # Standardized output file path within container
ERROR_FILE = '/sandbox/error.txt' # Standardized error file path within container

if __name__ == "__main__":
    # Keep original streams FIRST
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    print("--- Script started (Original stderr) ---", file=original_stderr, flush=True) # DEBUG

    exit_code = 0

    # Redirect Python's sys.stdout/stderr to capture potential prints/errors from LLM code
    print("--- Redirecting streams (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    final_output_json = None
    error_message = None

    try:
        # Read input from the mounted file instead of stdin
        if not os.path.exists(INPUT_FILE):
             raise FileNotFoundError(f"Input file {INPUT_FILE} not found in container.")

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
             input_data_json = f.read()

        input_list = json.loads(input_data_json)

        if 'sort_algorithm' in locals() and callable(sort_algorithm):
            # Call the LLM function. Any prints/errors inside it will go to captured streams.
            output_list = sort_algorithm(input_list)
            # --- DEBUG ---
            # Note: This print goes to the *captured* stderr
            print(f"--- sort_algorithm returned: {repr(output_list)} (Redirected stderr) ---", file=sys.stderr, flush=True)
            # --- END DEBUG ---

            # Serialize the actual result to JSON
            final_output_json = json.dumps(output_list)

        else:
             # This error should be captured
             raise NameError("Function 'sort_algorithm' not found or not callable in generated code.")

    except Exception as e:
        # --- DEBUG ---
        # Note: This print goes to the *captured* stderr, which gets added to error_message later
        print(f"--- EXCEPTION CAUGHT in main try block! Type: {type(e).__name__} (Redirected stderr) ---", file=sys.stderr, flush=True)
        # --- END DEBUG ---
        # Capture any exception during execution
        error_message = f"Error in generated script execution: {type(e).__name__}: {e}\\n"
        error_message += traceback.format_exc()
        exit_code = 1 # Indicate failure

    finally:
        print(f"--- Entering finally block. Initial exit_code: {exit_code} (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
        # Restore original streams FIRST in finally block
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Combine captured output/error with any explicit error message
        stray_prints = captured_stdout.getvalue()
        internal_errors = captured_stderr.getvalue()
        if error_message is None: # If no major exception occurred
             error_message = ""
        if stray_prints:
            error_message += "\\n--- Captured Stdout (potential stray prints) ---\\n" + stray_prints
        if internal_errors:
             error_message += "\\n--- Captured Stderr (potential internal errors) ---\\n" + internal_errors

        # Write the final JSON output to the designated file
        if final_output_json is not None:
             try:
                 print(f"--- Attempting to write output file: {OUTPUT_FILE} (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
                 with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
                     f_out.write(final_output_json)
                 print(f"--- Successfully wrote output file (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
             except Exception as write_e:
                 error_message += f"\\nCRITICAL: Failed to write output JSON to {OUTPUT_FILE}: {write_e}"
                 error_message += f"\\nCRITICAL: Failed to write output JSON to {OUTPUT_FILE}: {write_e}"
                 exit_code = 2 # Different exit code for I/O failure

        # Write any accumulated errors/prints to the error file if an error occurred
        if exit_code != 0:
             final_error_content = error_message.strip()
             # If exit code is non-zero but the message is empty, provide a default
             if not final_error_content:
                  final_error_content = f"(Script exited with code {exit_code} but produced no specific error message or captured output)"

             try:
                 print(f"--- Attempting to write error file: {ERROR_FILE} (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
                 with open(ERROR_FILE, 'w', encoding='utf-8') as f_err:
                     f_err.write(final_error_content)
                 print(f"--- Successfully wrote error file (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
             except Exception as write_e:
                 # If we can't even write the error, print to original stderr as last resort
                 # Avoid f-string for the error_message itself to prevent syntax errors
                 print(f"CRITICAL: Failed to write error log to {ERROR_FILE}: {write_e}", file=original_stderr, flush=True) # Use original_stderr
                 print("Original error was:", file=original_stderr, flush=True) # Use original_stderr
                 print(error_message, file=original_stderr, flush=True) # Print raw message separately
                 exit_code = 3 # Yet another exit code

        # Close StringIO objects
        captured_stdout.close()
        captured_stderr.close()

    print(f"--- Exiting script with code: {exit_code} (Original stderr) ---", file=original_stderr, flush=True) # DEBUG
    sys.exit(exit_code)
"""

    # Calculate total cases for progress reporting
    total_overall_cases_calculated = sum(len(cases) for cases in categorized_test_cases.values())
    current_overall_case_num = 0
    # Define a timeout for each Docker container run (increase for larger inputs/slower algorithms)
    DOCKER_TIMEOUT = 120.0 # Timeout in seconds (Increased from 30)
    # Define the Docker image to use
    DOCKER_IMAGE = "python:3.10-slim" # Or choose another appropriate Python image
    # Define container resource limits (adjust as needed)
    CONTAINER_MEM_LIMIT = "256m" # e.g., 256 MB memory limit
    CONTAINER_CPU_SHARES = 512 # Relative CPU weight (default 1024)

    # Initialize Docker client (do this once outside the loop)
    try:
        docker_client = docker.from_env(timeout=10) # Add timeout for client connection
        # Test connection
        docker_client.ping()
        print(f"Successfully connected to Docker daemon.")
        # Optionally pull the image explicitly beforehand
        try:
            print(f"Checking for Docker image: {DOCKER_IMAGE}...")
            docker_client.images.get(DOCKER_IMAGE)
            print("Image found locally.")
        except ImageNotFound:
            print(f"Image '{DOCKER_IMAGE}' not found locally, pulling...")
            try:
                docker_client.images.pull(DOCKER_IMAGE)
                print("Image pulled successfully.")
            except APIError as pull_err:
                 results['error'] = f"Failed to pull Docker image '{DOCKER_IMAGE}': {pull_err}. Check image name and registry access."
                 print(f"Error: {results['error']}")
                 return results # Cannot proceed without image
    except DockerConnectionError as e:
        results['error'] = f"Failed to connect to Docker daemon: {e}. Is Docker running and accessible?"
        print(f"Error: {results['error']}")
        return results # Cannot proceed without Docker
    except Exception as e: # Catch other potential docker.from_env errors
        results['error'] = f"Failed to initialize Docker client: {e}."
        print(f"Error: {results['error']}")
        return results


    # Iterate through each category and its test cases
    for category, test_cases_in_category in categorized_test_cases.items():
        print(f"  Evaluating category: {category} ({len(test_cases_in_category)} cases)")
        cat_stats = category_results[category] # Get stats dict for this category
        num_cases_in_category = len(test_cases_in_category)

        for i, test_case in enumerate(test_cases_in_category):
            current_overall_case_num += 1
            cat_stats['case_count'] += 1 # Increment here to match overall count logic

            # --- Prepare data for callback ---
            progress_data = {
                'current_case': current_overall_case_num,
                'total_cases': total_overall_cases_calculated,
                'category': category,
                'category_case_num': i + 1,
                'category_total_cases': num_cases_in_category,
                'status': 'Running',
                'input_snippet': repr(test_case[:10]) + ('...' if len(test_case) > 10 else ''),
                'output_snippet': None,
                'error': None
            }
            if progress_callback:
                progress_callback(progress_data) # Report start of case processing

            # Prepare inputs
            baseline_input = list(test_case) # For baseline timing
            expected_output = sorted(test_case) # Ground truth

            # --- Execute and Time LLM's sort_algorithm via Subprocess ---
            actual_output = None
            llm_error_str = None
            current_llm_time = None
            is_correct = False
            container = None # Ensure container variable exists for cleanup
            llm_error_str = None # Reset error for each case

            # Use ExitStack for robust cleanup of tempdir and container
            with ExitStack() as stack:
                try:
                    # Create TemporaryDirectory (automatically cleaned up by ExitStack)
                    temp_dir = stack.enter_context(tempfile.TemporaryDirectory())

                    script_filename = "script.py"
                    input_filename = "input.json"
                    output_filename = "output.json" # Expected output file from container
                    error_filename = "error.txt" # Expected error file from container

                    script_path_host = os.path.join(temp_dir, script_filename)
                    input_path_host = os.path.join(temp_dir, input_filename)
                    output_path_host = os.path.join(temp_dir, output_filename)
                    error_path_host = os.path.join(temp_dir, error_filename)

                    # Define paths inside the container's sandbox
                    sandbox_dir = "/sandbox"
                    script_path_cont = f"{sandbox_dir}/{script_filename}"
                    # input_path_cont = f"{sandbox_dir}/{input_filename}" # Defined in footer
                    # output_path_cont = f"{sandbox_dir}/{output_filename}" # Defined in footer
                    # error_path_cont = f"{sandbox_dir}/{error_filename}" # Defined in footer


                    # 1. Write script and input to temp directory on host
                    with open(script_path_host, 'w', encoding='utf-8') as f_script:
                        f_script.write(script_boilerplate_header)
                        f_script.write(generated_code)
                        f_script.write(script_boilerplate_footer)

                    input_json = json.dumps(test_case)
                    with open(input_path_host, 'w', encoding='utf-8') as f_input:
                        f_input.write(input_json)

                    # Add a small delay before starting the container (potential workaround for FS propagation)
                    time.sleep(0.1) # Sleep for 100ms

                    # 2. Run Docker container
                    llm_start_time = time.perf_counter()

                    # Command to check for script existence before running
                    container_command = [
                        "sh",
                        "-c",
                        f"test -f {script_path_cont} && python {script_path_cont} || (echo '--- ERROR: {script_path_cont} not found or not a file in container! ---' >&2; exit 2)"
                    ]

                    container = docker_client.containers.run(
                        image=DOCKER_IMAGE,
                        # command=["python", script_path_cont], # Original command
                        command=container_command, # Use the wrapper command
                        volumes={temp_dir: {'bind': sandbox_dir, 'mode': 'rw'}}, # Mount RW for output/error files
                        working_dir=sandbox_dir,
                        stdout=False, # Don't capture stdout directly, use files
                        stderr=True, # CAPTURE stderr directly
                        mem_limit=CONTAINER_MEM_LIMIT,
                        cpu_shares=CONTAINER_CPU_SHARES,
                        detach=True, # Run detached to manage timeout manually
                        # Security options (consider adding more if needed)
                        # read_only=True, # Filesystem read-only (except mounted volume)
                        # network_mode='none', # Disable networking
                    )
                    # Ensure container is removed even if errors occur below
                    stack.callback(lambda c: c.remove(force=True), container)

                    # Wait for container to finish with timeout
                    try:
                        # wait() returns dict with 'StatusCode' and 'Error'
                        result = container.wait(timeout=DOCKER_TIMEOUT)
                        exit_code = result.get('StatusCode', -1)
                        container_error = result.get('Error') # Docker-level error message
                        if container_error:
                             llm_error_str = f"Container reported error: {container_error}"
                    except (DockerConnectionError, APIError) as wait_err:
                        # Error communicating with Docker *during* wait
                        llm_error_str = f"Docker API error during wait: {wait_err}"
                        exit_code = -2 # Indicate API error during wait
                    except Exception as wait_timeout_err: # Catch potential timeout errors from requests lib used by docker-py
                        # Check if the error message indicates a timeout (likely container execution timeout)
                        if 'read timed out' in str(wait_timeout_err).lower():
                             llm_error_str = f"Container execution likely exceeded timeout of {DOCKER_TIMEOUT} seconds (wait operation timed out)."
                             exit_code = -3 # Indicate timeout
                             # Attempt to kill the container if it timed out
                             try: container.kill()
                             except APIError: pass # Ignore if already stopped or gone
                        else:
                             # Different unexpected error during wait
                             llm_error_str = f"Unexpected error during container wait: {wait_timeout_err}"
                             exit_code = -4
                    finally:
                         llm_end_time = time.perf_counter()
                         current_llm_time = llm_end_time - llm_start_time


                    # 3. Process results: Get logs first, then check files
                    container_stderr_logs = ""
                    try:
                        # Retrieve logs regardless of exit code, might contain warnings/prints
                        container_stderr_logs = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace').strip()
                    except APIError as log_err:
                        print(f"    Warning: Failed to retrieve container stderr logs: {log_err}")
                        # Optionally add this warning to llm_error_str later?
                    except Exception as log_generic_err:
                         print(f"    Warning: Unexpected error retrieving container stderr logs: {log_generic_err}")


                    container_error_log_content = "" # Content from the error file
                    error_log_read_error = None
                    if os.path.exists(error_path_host):
                        try:
                            with open(error_path_host, 'r', encoding='utf-8') as f_err:
                                container_error_log_content = f_err.read().strip()
                        except Exception as read_err:
                            error_log_read_error = f"(Host failed to read error log: {read_err})"
                            print(f"    Warning: {error_log_read_error}")


                    # Consolidate error reporting
                    if llm_error_str: # Docker API error or timeout takes precedence
                        # Append stderr logs if they exist and might provide more context
                        if container_stderr_logs:
                             llm_error_str += f"\nContainer stderr:\n---\n{container_stderr_logs}\n---"
                    elif exit_code != 0:
                        llm_error_str = f"Container exited with code {exit_code}."
                        # Append error log content if found
                        if container_error_log_content:
                            llm_error_str += f" Error log file content:\n---\n{container_error_log_content}\n---"
                        # Append stderr logs if found (might contain the actual error if file write failed)
                        if container_stderr_logs:
                            llm_error_str += f"\nContainer stderr:\n---\n{container_stderr_logs}\n---"
                        # If both file and stderr are empty, add a specific message
                        if not container_error_log_content and not container_stderr_logs:
                             llm_error_str += " (No error log file content found and container stderr was empty)"
                        # Report if reading the error log file itself failed
                        if error_log_read_error:
                             llm_error_str += f"\n{error_log_read_error}"

                    elif container_error_log_content: # Exit code 0, but error log has content
                         # Treat this as a potential issue (e.g., stray prints written to error file)
                         print(f"    Warning: Container exited code 0 but error log file has content:\n---\n{container_error_log_content}\n---")
                         # Optionally, capture this warning in the results? For now, just print.
                    elif container_stderr_logs: # Exit code 0, no error log file, but stderr has content
                         # Also potentially stray prints/warnings
                         print(f"    Warning: Container exited code 0, no error log file, but stderr has content:\n---\n{container_stderr_logs}\n---")


                    # If no critical error reported so far (llm_error_str is still None), try to read output
                    if llm_error_str is None:
                        if not os.path.exists(output_path_host):
                            llm_error_str = "Container finished successfully but output file was not created."
                            # Add context if available
                            if container_error_log_content: llm_error_str += f"\nError log file content:\n---\n{container_error_log_content}\n---"
                            if container_stderr_logs: llm_error_str += f"\nContainer stderr:\n---\n{container_stderr_logs}\n---"
                        else:
                            try:
                                with open(output_path_host, 'r', encoding='utf-8') as f_out:
                                    stdout_output = f_out.read()
                                actual_output = json.loads(stdout_output)

                                # Validate correctness
                                if actual_output == expected_output:
                                    is_correct = True
                                    overall_correct_count += 1
                                    cat_stats['correct_count'] += 1
                                    # Only add time for correct runs
                                    overall_llm_time += current_llm_time
                                    cat_stats['llm_time'] += current_llm_time
                                    overall_llm_runs_timed += 1
                                    cat_stats['llm_runs_timed'] += 1
                                    progress_data['status'] = 'Correct'
                                else:
                                    # Log incorrect sort
                                    actual_repr = repr(actual_output[:20]) + '...' if isinstance(actual_output, list) and len(actual_output) > 20 else repr(actual_output)
                                    expected_repr = repr(expected_output[:20]) + '...' if len(expected_output) > 20 else repr(expected_output)
                                    test_repr = repr(test_case[:20]) + '...' if len(test_case) > 20 else repr(test_case)
                                    print(f"    Incorrect sort: Input={test_repr}, Expected={expected_repr}, Got={actual_repr}")
                                    progress_data['status'] = 'Incorrect'
                                    progress_data['output_snippet'] = actual_repr
                                    # Optionally capture the incorrect output in llm_error_str for reporting?
                                    # llm_error_str = f"Incorrect output. Expected: {expected_repr}, Got: {actual_repr}"

                            except json.JSONDecodeError as json_e:
                                llm_error_str = f"Failed to decode JSON from output file: {json_e}. Raw content: '{stdout_output[:200]}...'"
                                if container_error_log_content: llm_error_str += f"\nError log file content:\n---\n{container_error_log_content}\n---"
                                if container_stderr_logs: llm_error_str += f"\nContainer stderr:\n---\n{container_stderr_logs}\n---"
                            except Exception as parse_e:
                                 llm_error_str = f"Error reading or parsing output file: {parse_e}"
                                 if container_error_log_content: llm_error_str += f"\nError log file content:\n---\n{container_error_log_content}\n---"
                                 if container_stderr_logs: llm_error_str += f"\nContainer stderr:\n---\n{container_stderr_logs}\n---"

                # Catch errors during the setup/execution phase *outside* the container run itself
                except (APIError, DockerConnectionError) as docker_setup_err:
                    # Catch errors like image not found *during run*, or API errors before wait
                    llm_error_str = f"Docker API/Connection error during setup/run: {docker_setup_err}"
                    # Abort evaluation if Docker itself fails fundamentally
                    results['error'] = llm_error_str
                    print(f"Aborting evaluation due to Docker error: {llm_error_str}")
                    return results
                except Exception as setup_exec_e:
                    llm_error_str = f"Error setting up or initiating Docker container run: {setup_exec_e}"
                    # Record time if possible, though it might be minimal
                    current_llm_time = time.perf_counter() - llm_start_time if 'llm_start_time' in locals() else 0

                # --- Handle LLM Run Outcome ---
                # llm_error_str should now contain the most relevant error message if any occurred
                if llm_error_str:
                    test_repr = repr(test_case[:20]) + '...' if len(test_case) > 20 else repr(test_case)
                    print(f"    Error during LLM sort subprocess: Input={test_repr}, Error={llm_error_str}")
                    # Report error via callback
                    if progress_callback:
                        progress_data['status'] = 'Error'
                        progress_data['error'] = llm_error_str
                        progress_callback(progress_data)
                elif progress_callback: # If no error, update progress (Correct/Incorrect status set above)
                     progress_callback(progress_data)


            # --- Time Python's built-in sorted() ---
            baseline_start_time = time.perf_counter()
            current_baseline_time = None
            try:
                _ = sorted(baseline_input) # Execute baseline sort
                baseline_end_time = time.perf_counter()
                current_baseline_time = baseline_end_time - baseline_start_time
                overall_baseline_time += current_baseline_time
                cat_stats['baseline_time'] += current_baseline_time
            except Exception as e:
                test_repr = repr(test_case[:20]) + '...' if len(test_case) > 20 else repr(test_case)
                print(f"    Error during baseline sort execution: Input={test_repr}, Error={e}")
                # Decide how to handle baseline errors (e.g., skip timing for this case?)

    # --- Calculate Final Results ---
    # (Error handling for overall execution moved outside the loop)
    # Check if any fundamental error occurred before processing cases (e.g., invalid generated_code structure)
    # This part is less relevant now as syntax errors are caught per-case by subprocess

    # Overall results
        if overall_total_cases > 0:
            results['correctness'] = (overall_correct_count / overall_total_cases) * 100
            results['baseline_avg_time_ms'] = (overall_baseline_time / overall_total_cases) * 1000
        if overall_llm_runs_timed > 0: # Average LLM time only over correctly sorted cases
            results['avg_time_ms'] = (overall_llm_time / overall_llm_runs_timed) * 1000

        # Per-category results
        for category, stats in category_results.items():
            cat_correctness = (stats['correct_count'] / stats['case_count']) * 100 if stats['case_count'] > 0 else 0.0
            cat_avg_llm_time = (stats['llm_time'] / stats['llm_runs_timed']) * 1000 if stats['llm_runs_timed'] > 0 else None
            cat_avg_baseline_time = (stats['baseline_time'] / stats['case_count']) * 1000 if stats['case_count'] > 0 else None
            results['performance_details'][category] = {
                'correctness': cat_correctness,
                'avg_time_ms': cat_avg_llm_time, # Note: Includes subprocess overhead
                'baseline_avg_time_ms': cat_avg_baseline_time,
                'count': stats['case_count']
            }

    # No top-level try-except for SyntaxError needed anymore, handled per case.
    # General errors during setup might still occur but are less likely.

    return results


# Note: algorithm_name parameter is removed. We use the constant GENERAL_ALGORITHM_NAME internally.
# Now accepts pre-generated code.
def run_single_benchmark(llm_name: str, generated_code: str, categorized_test_cases: dict, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Runs the evaluation part of a benchmark for a single LLM, using pre-generated code
    and provided test cases, optionally reporting progress.

    Args:
        llm_name: The identifier for the LLM used (for reporting).
        generated_code: The Python code string generated by the LLM.
        categorized_test_cases: A dictionary containing the test cases, keyed by category.
        progress_callback: An optional function to call with progress updates during evaluation.

    Returns:
        A dictionary containing benchmark evaluation results.
    """
    algorithm_name = GENERAL_ALGORITHM_NAME # Use the constant label
    print(f"Evaluating benchmark for {llm_name} ({algorithm_name}) using provided code...")

    # Code generation is now done *before* calling this function.
    # The initial progress callback indicating evaluation start is also handled before this call.

    # --- Run Evaluation ---
    evaluation = evaluate_algorithm(
        generated_code=generated_code, # Use the passed-in code
        categorized_test_cases=categorized_test_cases,
        progress_callback=progress_callback # Pass callback for evaluation steps
    )

    # --- Callback for completion of evaluation ---
    final_status = 'Completed' if not evaluation.get('error') else 'Error'
    if progress_callback:
        progress_callback({
            'status': final_status, 'category': 'Finished',
            'error': evaluation.get('error')
        })


    return {
        'llm': llm_name,
        'algorithm': algorithm_name,
        'correctness': evaluation.get('correctness'),
        'avg_time_ms': evaluation.get('avg_time_ms'),
        'baseline_avg_time_ms': evaluation.get('baseline_avg_time_ms'),
        'performance_details': evaluation.get('performance_details'),
        'error': evaluation.get('error'),
        # 'generated_code' is no longer added here, it's handled in app.py
    }

# Constant for the baseline benchmark label
BASELINE_ALGORITHM_NAME = "Python sorted() Baseline"

# Note: algorithm_name parameter removed, using constant label instead.
def run_python_sorted_benchmark(categorized_test_cases: dict, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Runs a benchmark using Python's built-in sorted() function as a baseline,
    using provided test cases and optionally reporting progress.

    Args:
        categorized_test_cases: A dictionary containing the test cases, keyed by category.
        progress_callback: An optional function to call with progress updates.

    Returns:
        A dictionary containing benchmark results.
    """
    algorithm_name = BASELINE_ALGORITHM_NAME # Use the constant label
    print(f"Running {algorithm_name} benchmark...")
    results = {
        'llm': PYTHON_SORTED_BENCHMARK, # Note: This constant needs to be defined/imported if run standalone
        'algorithm': algorithm_name, # Use the constant label
        'correctness': 100.0, # sorted() is assumed correct
        'avg_time_ms': None, # This is the baseline itself
        'baseline_avg_time_ms': None, # Will be calculated
        'performance_details': {}, # To store per-category results
        'error': None,
        'generated_code': "N/A - Python sorted()"
    }

    if not categorized_test_cases:
        results['error'] = "No test cases provided for baseline benchmark."
        return results

    # Aggregators for overall results
    overall_baseline_time = 0
    overall_total_cases = 0

    # Aggregators for per-category results
    category_results = defaultdict(lambda: {'baseline_time': 0, 'case_count': 0})

    try:
        # Calculate total cases for progress reporting
        total_overall_cases_calculated = sum(len(cases) for cases in categorized_test_cases.values())
        current_overall_case_num = 0

        # Iterate through each category and its test cases
        for category, test_cases_in_category in categorized_test_cases.items():
            print(f"  Benchmarking category: {category} ({len(test_cases_in_category)} cases)")
            cat_stats = category_results[category] # Get stats dict for this category
            num_cases_in_category = len(test_cases_in_category)

            for i, test_case in enumerate(test_cases_in_category):
                current_overall_case_num += 1
                overall_total_cases += 1 # Increment the overall counter
                cat_stats['case_count'] += 1 # Increment here

                # --- Prepare data for callback ---
                progress_data = {
                    'current_case': current_overall_case_num,
                    'total_cases': total_overall_cases_calculated,
                    'category': category,
                    'category_case_num': i + 1,
                    'category_total_cases': num_cases_in_category,
                    'status': 'Running',
                    'input_snippet': repr(test_case[:10]) + ('...' if len(test_case) > 10 else ''),
                    # Calculate the expected output snippet for baseline
                    'output_snippet': repr(sorted(test_case)[:10]) + ('...' if len(test_case) > 10 else ''),
                    'error': None
                }
                # Report start of case processing (optional for baseline, but consistent)
                # if progress_callback: progress_callback(progress_data)

                baseline_input = list(test_case)
                start_time = time.perf_counter()
                _ = sorted(baseline_input) # Execute and time sorted()
                end_time = time.perf_counter()
                current_baseline_time = end_time - start_time

                overall_baseline_time += current_baseline_time
                cat_stats['baseline_time'] += current_baseline_time

                # Report completion of case for baseline
                if progress_callback:
                    progress_data['status'] = 'Correct (Baseline)' # Baseline is assumed correct
                    progress_callback(progress_data)


        # --- Calculate Final Results ---

        # Overall results
        if overall_total_cases > 0:
            overall_avg_time = (overall_baseline_time / overall_total_cases) * 1000
            # Since this *is* the baseline, set both avg_time and baseline_avg_time
            results['avg_time_ms'] = overall_avg_time
            results['baseline_avg_time_ms'] = overall_avg_time
        else:
             results['avg_time_ms'] = 0.0
             results['baseline_avg_time_ms'] = 0.0

        # Per-category results
        for category, stats in category_results.items():
            cat_avg_baseline_time = (stats['baseline_time'] / stats['case_count']) * 1000 if stats['case_count'] > 0 else None
            results['performance_details'][category] = {
                'correctness': 100.0, # Assumed correct
                'avg_time_ms': cat_avg_baseline_time, # This is the baseline time for this category
                'baseline_avg_time_ms': cat_avg_baseline_time,
                'count': stats['case_count']
            }

    except Exception as e:
        results['error'] = f"Error during Python sorted() execution: {e}"
        results['correctness'] = None # Mark correctness as unknown if execution fails
        # Ensure performance_details is still present, even if empty
        results['performance_details'] = results.get('performance_details', {})

    return results


# Example usage needs the constant defined in app.py or locally
PYTHON_SORTED_BENCHMARK = "Python sorted()" # Define locally for example usage
DEFAULT_TEST_SUITE_FILE = "test_suite.json"

# --- Placeholder functions for standalone execution ---
def generate_and_save_test_suite(filename, **kwargs):
    """Placeholder: Generates and saves test suite."""
    print(f"Placeholder: Generating test suite with params {kwargs} and saving to {filename}")
    # Actual implementation would call generate_test_cases and save to JSON
    test_cases = generate_test_cases(**kwargs)
    try:
        with open(filename, 'w') as f:
            json.dump(test_cases, f, indent=2)
        print(f"Successfully generated and saved test suite to {filename}")
    except Exception as e:
        print(f"Error saving test suite to {filename}: {e}")

def load_test_suite(filename):
    """Placeholder: Loads test suite from file."""
    print(f"Placeholder: Loading test suite from {filename}")
    # Actual implementation would load from JSON
    try:
        with open(filename, 'r') as f:
            test_suite = json.load(f)
        print(f"Successfully loaded test suite from {filename}")
        return test_suite
    except FileNotFoundError:
        print(f"Error: Test suite file '{filename}' not found.")
        raise
    except Exception as e:
        print(f"Error loading test suite from {filename}: {e}")
        raise
# --- End Placeholder functions ---

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Utility Functions")
    parser.add_argument('--generate-suite', action='store_true', help=f"Generate and save a new test suite to {DEFAULT_TEST_SUITE_FILE}")
    parser.add_argument('--suite-file', default=DEFAULT_TEST_SUITE_FILE, help="Specify the test suite JSON file path")
    # Add arguments to control generation parameters if needed, e.g.:
    parser.add_argument('--num-cases', type=int, default=5, help="Number of cases per type/size for suite generation")
    parser.add_argument('--size-s', type=int, default=20)
    parser.add_argument('--size-m', type=int, default=20000)
    parser.add_argument('--size-l', type=int, default=2000000)


    args = parser.parse_args()

    if args.generate_suite:
        gen_params = {
            'size_small': args.size_s,
            'size_medium': args.size_m,
            'size_large': args.size_l,
            'num_cases_per_type': args.num_cases
        }
        generate_and_save_test_suite(args.suite_file, **gen_params)
    else:
        # Example of running benchmarks using a loaded suite (adjust as needed)
        print("Running example benchmarks with loaded suite...")
        try:
            test_suite = load_test_suite(args.suite_file)

            # Example usage - Load suite first, then run benchmarks
            print("\nRunning example benchmarks with loaded suite...")

            # Run baseline benchmark
            print("\nRunning baseline benchmark example...")
            result_baseline = run_python_sorted_benchmark(categorized_test_cases=test_suite) # Run baseline
            print("\nPython sorted() Benchmark Result:\n", json.dumps(result_baseline, indent=2))

            # --- Run example with fixed Merge Sort code ---
            print("\nRunning example benchmark with fixed Merge Sort code...")

            EXAMPLE_MERGE_SORT_CODE = """
import sys
from typing import List, TypeVar

# Increase recursion depth limit for potentially deep recursion with large lists
try:
    sys.setrecursionlimit(2000)
except Exception:
    pass

T = TypeVar('T')

def sort_algorithm(arr: List[T]) -> List[T]:
    # Base Case
    if len(arr) <= 1:
        return arr[:]

    # Divide
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Conquer
    sorted_left = sort_algorithm(left_half)
    sorted_right = sort_algorithm(right_half)

    # Combine (Merge)
    merged = []
    left_idx, right_idx = 0, 0
    while left_idx < len(sorted_left) and right_idx < len(sorted_right):
        if sorted_left[left_idx] <= sorted_right[right_idx]:
            merged.append(sorted_left[left_idx])
            left_idx += 1
        else:
            merged.append(sorted_right[right_idx])
            right_idx += 1

    # Append Remaining
    while left_idx < len(sorted_left):
        merged.append(sorted_left[left_idx])
        left_idx += 1
    while right_idx < len(sorted_right):
        merged.append(sorted_right[right_idx])
        right_idx += 1

    return merged
"""
            # Run the benchmark using the fixed code
            result_example = run_single_benchmark(
                llm_name="Example Merge Sort", # Use a descriptive name
                generated_code=EXAMPLE_MERGE_SORT_CODE,
                categorized_test_cases=test_suite
                # Add progress_callback=print if you want to see progress updates
            )
            print("\nExample Merge Sort Benchmark Result:\n", json.dumps(result_example, indent=2))


        except FileNotFoundError:
            print(f"Test suite file '{args.suite_file}' not found. Generate it first using --generate-suite.")
        except Exception as e:
            print(f"An error occurred during example benchmark run: {e}")
