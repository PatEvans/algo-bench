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

    # Calculate total cases for progress reporting
    total_overall_cases_calculated = sum(len(cases) for cases in categorized_test_cases.values())
    current_overall_case_num = 0
    # Define a timeout for each *individual execution* within the container
    EXEC_TIMEOUT = 120.0 # Timeout in seconds for container.exec_run
    # Define the Docker image to use
    DOCKER_IMAGE = "python:3.10-slim" # Or choose another appropriate Python image
    # Define container resource limits (adjust as needed)
    CONTAINER_MEM_LIMIT = "256m" # e.g., 256 MB memory limit
    CONTAINER_CPU_SHARES = 512 # Relative CPU weight (default 1024)

    # Initialize Docker client
    docker_client = None
    try:
        docker_client = docker.from_env(timeout=10)
        docker_client.ping()
        print("Successfully connected to Docker daemon.")
        # Ensure image exists
        try:
            docker_client.images.get(DOCKER_IMAGE)
        except ImageNotFound:
            print(f"Pulling Docker image: {DOCKER_IMAGE}...")
            docker_client.images.pull(DOCKER_IMAGE)
        print(f"Using Docker image: {DOCKER_IMAGE}")
    except (DockerConnectionError, APIError, Exception) as e:
        results['error'] = f"Docker initialization failed: {e}. Is Docker running?"
        print(f"Error: {results['error']}")
        if progress_callback: progress_callback({'status': 'Error', 'error': results['error']})
        return results

    # Define the Python code to be executed inside the container for each test case
    # This code loads the LLM function, runs it with input from stdin, and prints JSON result to stdout
    exec_wrapper_code = """
import sys
import json
import time
import traceback
import importlib.util
import io
import os # Need os module for file existence check

# --- Define the function to load and run ---
def load_and_run_sort():
    result = {'output': None, 'error': None, 'stdout': None, 'stderr': None, 'exec_time_ms': None}
    llm_module = None
    sort_algorithm = None

    # Capture stdout/stderr during the entire process
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    start_time = time.perf_counter()

    try:
        # 1. Add sandbox to path and import llm_sort directly
        file_path = "/sandbox/llm_sort.py" # Still useful for error messages
        # --- Check if file exists before attempting import ---
        if not os.path.exists(file_path):
             # List directory contents for debugging if file not found
             try:
                 sandbox_contents = os.listdir('/sandbox')
                 dir_listing_str = f"Contents of /sandbox: {sandbox_contents}"
             except Exception as list_e:
                 dir_listing_str = f"(Could not list /sandbox contents: {list_e})"
             raise FileNotFoundError(f"[Errno 2] No such file or directory: '{file_path}'. {dir_listing_str}")
       # --- End check ---

       # Import the module directly (should work if workdir=/sandbox)
       try:
            import llm_sort
        except ModuleNotFoundError:
             # This shouldn't happen if the file exists and path is set, but catch defensively
             raise ImportError(f"Could not import 'llm_sort' even after adding /sandbox to sys.path.")
        except Exception as import_err: # Catch other potential import errors (e.g., syntax errors in llm_sort.py)
             raise ImportError(f"Error importing 'llm_sort': {type(import_err).__name__}: {import_err}")

        # 2. Get the sort_algorithm function
        if not hasattr(llm_sort, 'sort_algorithm') or not callable(llm_sort.sort_algorithm):
             raise NameError("Function 'sort_algorithm' not found or not callable in imported llm_sort module.")
        sort_algorithm = llm_sort.sort_algorithm

        # 3. Read input JSON from stdin
        input_data_json = sys.stdin.read()
        if not input_data_json:
             raise ValueError("No input data received via stdin.")
        input_list = json.loads(input_data_json)

        # 4. Execute the sort_algorithm
        output_list = sort_algorithm(input_list)
        result['output'] = output_list # Store the actual Python object/list

    except Exception as e:
        # Capture any exception during loading or execution
        result['error'] = f"{type(e).__name__}: {e}\\n{traceback.format_exc()}"
    finally:
        end_time = time.perf_counter()
        result['exec_time_ms'] = (end_time - start_time) * 1000

        # Restore streams and get captured content
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        result['stdout'] = captured_stdout.getvalue()
        result['stderr'] = captured_stderr.getvalue()
        captured_stdout.close()
        captured_stderr.close()

        # Combine captured stdout/stderr into error if an error occurred or if they contain data
        # Append safely to avoid syntax errors if captured output contains problematic characters
        combined_error = result['error'] if result['error'] else ""
        captured_stdout_val = result.get('stdout', '')
        captured_stderr_val = result.get('stderr', '')

        if combined_error:
            if captured_stdout_val:
                combined_error += "\\n--- Captured Stdout ---\\n" + captured_stdout_val
            if captured_stderr_val:
                combined_error += "\\n--- Captured Stderr ---\\n" + captured_stderr_val
        elif captured_stdout_val or captured_stderr_val: # No primary error, but stray output/errors
             combined_error = "(No primary exception, but captured output found)"
             if captured_stdout_val:
                 combined_error += "\\n--- Captured Stdout ---\\n" + captured_stdout_val
             if captured_stderr_val:
                 combined_error += "\\n--- Captured Stderr ---\\n" + captured_stderr_val

        result['error'] = combined_error # Update the result dict with the combined error string

        # Print the final result dictionary as JSON to the original stdout
        # Ensure output is serializable (it should be if sort_algorithm returns a list)
        try:
            print(json.dumps(result))
        except TypeError as json_err:
            # Fallback if the output itself is not JSON serializable
            fallback_result = {
                'output': repr(result.get('output')), # Use repr as fallback
                'error': (result.get('error') or "") + f"\\nJSON Serialization Error: {json_err}",
                'stdout': result.get('stdout'),
                'stderr': result.get('stderr'),
                'exec_time_ms': result.get('exec_time_ms')
            }
            print(json.dumps(fallback_result))

# --- Run the function ---
if __name__ == "__main__":
    load_and_run_sort()
"""

    container = None
    # Use ExitStack for robust cleanup of tempdir and container
    with ExitStack() as stack:
        try:
           # Create TemporaryDirectory for the LLM code and runner script
           temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
           llm_code_filename = "llm_sort.py"
           runner_script_filename = "exec_runner.py" # Filename for the wrapper script
           llm_code_path_host = os.path.join(temp_dir, llm_code_filename)
           runner_script_path_host = os.path.join(temp_dir, runner_script_filename) # Host path for runner
           sandbox_dir = "/sandbox" # Mount point inside container
           llm_code_path_cont = f"{sandbox_dir}/{llm_code_filename}"
           runner_script_path_cont = f"{sandbox_dir}/{runner_script_filename}" # Container path for runner

           # Write the generated code and the runner script to the host temp directory
           with open(llm_code_path_host, 'w', encoding='utf-8') as f_llm_script:
               f_llm_script.write(generated_code)
           with open(runner_script_path_host, 'w', encoding='utf-8') as f_runner_script:
               f_runner_script.write(exec_wrapper_code) # Write the wrapper code to its own file

           # Start the container once, keep it running
           print("Starting persistent Docker container...")
           container = docker_client.containers.run(
               image=DOCKER_IMAGE,
               command=["sleep", "infinity"], # Keep container alive
                volumes={temp_dir: {'bind': sandbox_dir, 'mode': 'ro'}}, # Mount code read-only
                working_dir=sandbox_dir,
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

           # Give container a moment to stabilize (optional, usually not needed for sleep infinity)
           # time.sleep(1)

           # Iterate through each category and its test cases
           for category, test_cases_in_category in categorized_test_cases.items():
               print(f"  Evaluating category: {category} ({len(test_cases_in_category)} cases)")
               cat_stats = category_results[category] # Get stats dict for this category
               num_cases_in_category = len(test_cases_in_category)

               for i, test_case in enumerate(test_cases_in_category):
                   current_overall_case_num += 1
                   overall_total_cases += 1 # Increment overall counter here
                   cat_stats['case_count'] += 1

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
                   input_json = json.dumps(test_case)

                   # --- Execute code inside the running container using exec_run ---
                   actual_output = None
                   llm_error_str = None
                   current_llm_time_ms = None # Time reported by the container script
                   is_correct = False

                   host_exec_start_time = time.perf_counter() # Host-side timing for exec_run call itself

                   try:
                       # Command to execute the runner script file directly inside the container
                       # Pass input via stdin=True
                       exec_command = ["python", runner_script_path_cont] # e.g., ["python", "/sandbox/exec_runner.py"]

                        # Use stream=False, demux=False to get combined output as bytes
                        exec_result = container.exec_run(
                            cmd=exec_command,
                            stdin=True,
                            stdout=True, # Capture stdout from exec_run
                            stderr=True, # Capture stderr from exec_run (though wrapper redirects)
                            socket=False, # Use simple exec, not socket
                            stream=False, # Get result after completion
                            demux=False, # Get interleaved stdout/stderr
                            workdir=sandbox_dir,
                            # Note: Timeout for exec_run itself, not the whole container
                            # timeout=EXEC_TIMEOUT # Timeout seems unreliable/buggy with stdin=True? Manage externally if needed.
                        )

                        exit_code = exec_result.exit_code
                        output_bytes = exec_result.output or b"" # Combined stdout/stderr from exec_run

                        host_exec_end_time = time.perf_counter()

                        # Decode the output bytes (potential stdout/stderr from python -c itself, plus our JSON)
                        try:
                             output_str = output_bytes.decode('utf-8', errors='replace').strip()
                        except Exception as decode_err:
                             llm_error_str = f"Host failed to decode exec_run output: {decode_err}. Raw bytes: {repr(output_bytes[:200])}"
                             output_str = "" # Cannot parse JSON

                        # Attempt to parse the JSON result from the *last line* of the output
                        parsed_result = None
                        if output_str and llm_error_str is None:
                            try:
                                # Find the last line that looks like JSON
                                last_line = output_str.splitlines()[-1]
                                parsed_result = json.loads(last_line)
                            except (json.JSONDecodeError, IndexError) as json_e:
                                llm_error_str = f"Failed to decode JSON result from exec_run output: {json_e}. Full output:\n---\n{output_str[:1000]}\n---"
                            except Exception as parse_e:
                                llm_error_str = f"Unexpected error parsing exec_run output: {parse_e}. Full output:\n---\n{output_str[:1000]}\n---"

                        # --- Process the parsed result ---
                        if llm_error_str: # If host-level parsing failed
                            pass # Error already set
                        elif exit_code is None:
                             llm_error_str = f"exec_run did not return an exit code (may indicate timeout or Docker issue)."
                        elif exit_code != 0:
                             llm_error_str = f"Exec wrapper exited with code {exit_code}."
                             # Append JSON error if available, otherwise append raw output
                             if parsed_result and parsed_result.get('error'):
                                 llm_error_str += f" Internal error:\n---\n{parsed_result['error']}\n---"
                             elif output_str:
                                 llm_error_str += f" Raw output:\n---\n{output_str[:1000]}\n---"
                        elif parsed_result is None:
                             # Should not happen if exit code is 0 and no parsing error occurred, but check defensively
                             llm_error_str = "Exec wrapper exited code 0 but host failed to parse JSON result."
                             if output_str: llm_error_str += f" Raw output:\n---\n{output_str[:1000]}\n---"
                        elif parsed_result.get('error'):
                             # Exit code 0, but the script internally caught an error
                             llm_error_str = f"Exec wrapper reported internal error:\n---\n{parsed_result['error']}\n---"
                        else:
                             # --- Success Case ---
                             actual_output = parsed_result.get('output')
                             current_llm_time_ms = parsed_result.get('exec_time_ms') # Use time measured inside container

                             if actual_output == expected_output:
                                 is_correct = True
                                 overall_correct_count += 1
                                 cat_stats['correct_count'] += 1
                                 if current_llm_time_ms is not None:
                                     overall_llm_time += (current_llm_time_ms / 1000.0) # Convert ms to seconds for aggregation
                                     cat_stats['llm_time'] += (current_llm_time_ms / 1000.0)
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


                    # Catch host-side errors during exec_run call itself
                    except (APIError, DockerConnectionError) as docker_exec_err:
                        llm_error_str = f"Docker API/Connection error during exec_run: {docker_exec_err}"
                        # This might be fatal for subsequent calls, consider aborting? For now, report per case.
                    except Exception as host_exec_e:
                        llm_error_str = f"Host error during container exec_run setup or call: {host_exec_e}"


                    # --- Handle LLM Run Outcome for this case ---
                    if llm_error_str:
                        test_repr = repr(test_case[:20]) + '...' if len(test_case) > 20 else repr(test_case)
                        print(f"    Error during LLM sort execution: Input={test_repr}, Error={llm_error_str}")
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

        # --- End of loop over categories/cases ---

        # Catch errors during the initial container setup phase
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
