# This file contains the Python script executed inside the Docker container
# by benchmark.py's evaluate_algorithm function.
# It reads input data (list to sort) from stdin as JSON,
# imports and runs the 'llm_sort.sort_algorithm' function from /sandbox/llm_sort.py,
# and prints the result (or error details) as JSON to stdout.

import sys
import json
import sys
import json
import time
import traceback
import io
import os # Need os module for file existence check

# --- Define the function to load and run ---
def load_and_run_sort():
    # Initialize result structure FIRST
    result = {'output': None, 'error': None, 'stdout': None, 'stderr': None, 'exec_time_ms': None}
    start_time = None # Initialize start_time

    # Capture stdout/stderr using context managers for robustness
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    final_output_json = None # Ensure this is defined in the outer scope

    try:
        # Redirect streams
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        start_time = time.perf_counter() # Start timer after setup

        # --- Inner try for core logic: reading input, importing, executing ---
        try:
            # 1. Read input JSON from stdin FIRST
            input_data_json = sys.stdin.read()
            if not input_data_json:
                raise ValueError("No input data received via stdin.")

            # 2. Check for llm_sort.py existence
            file_path = "/sandbox/llm_sort.py"
            if not os.path.exists(file_path):
                try:
                    sandbox_contents = os.listdir('/sandbox')
                    dir_listing_str = f"Contents of /sandbox: {sandbox_contents}"
                except Exception as list_e:
                    dir_listing_str = f"(Could not list /sandbox contents: {list_e})"
                raise FileNotFoundError(
                    f"[Errno 2] No such file or directory: '{file_path}'. {dir_listing_str}"
                )

            # 3. Import the module
            try:
                # Ensure the sandbox is in the path for the import
                # sys.path.insert(0, '/sandbox') # Not strictly needed if WORKDIR=/sandbox
                import llm_sort
            except ModuleNotFoundError:
                raise ImportError("Could not import 'llm_sort'. Check WORKDIR or sys.path.")
            except Exception as import_err:
                raise ImportError(f"Error importing 'llm_sort': {type(import_err).__name__}: {import_err}")

            # 4. Get the sort_algorithm function
            if not hasattr(llm_sort, 'sort_algorithm') or not callable(llm_sort.sort_algorithm):
                raise NameError("Function 'sort_algorithm' not found or not callable in llm_sort module.")
            sort_algorithm = llm_sort.sort_algorithm

            # 5. Parse the input JSON
            input_list = json.loads(input_data_json)

            # 6. Execute the sort_algorithm
            output_list = sort_algorithm(input_list)
            result['output'] = output_list # Store the actual Python object/list

        except Exception as e:
            # Capture any exception during the core logic
            result['error'] = f"{type(e).__name__}: {e}\\n{traceback.format_exc()}"
        # --- End of inner try ---

    finally:
        # This block *always* runs, even if errors occurred above

        # Calculate execution time if start_time was set
        if start_time is not None:
            end_time = time.perf_counter()
            result['exec_time_ms'] = (end_time - start_time) * 1000

        # Restore streams and get captured content
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        result['stdout'] = captured_stdout.getvalue()
        result['stderr'] = captured_stderr.getvalue()
        captured_stdout.close()
        captured_stderr.close()

        # Combine captured stdout/stderr into the error field if necessary
        combined_error = result['error'] if result['error'] else ""
        captured_stdout_val = result.get('stdout', '')
        captured_stderr_val = result.get('stderr', '')

        # Append captured streams to error message for context
        if combined_error:
            if captured_stdout_val:
                # Escape backslashes and quotes in captured output for JSON safety
                safe_stdout = json.dumps(captured_stdout_val)[1:-1] # Get string content without outer quotes
                combined_error += "\\n--- Captured Stdout ---\\n" + safe_stdout
            if captured_stderr_val:
                safe_stderr = json.dumps(captured_stderr_val)[1:-1]
                combined_error += "\\n--- Captured Stderr ---\\n" + safe_stderr
        elif captured_stdout_val or captured_stderr_val: # No primary error, but stray output
            combined_error = "(No primary exception, but captured output found)"
            if captured_stdout_val:
                safe_stdout = json.dumps(captured_stdout_val)[1:-1]
                combined_error += "\\n--- Captured Stdout ---\\n" + safe_stdout
            if captured_stderr_val:
                safe_stderr = json.dumps(captured_stderr_val)[1:-1]
                combined_error += "\\n--- Captured Stderr ---\\n" + safe_stderr

        result['error'] = combined_error # Update result dict

        # --- Serialize the final result dictionary to JSON ---
        # This section MUST succeed or produce a fallback JSON string
        try:
            # Attempt to serialize the primary result (potentially updated error field)
            final_output_json = json.dumps(result)
        except TypeError as json_err:
            # Fallback if output or other fields are not JSON serializable
            print(f"Warning: JSON serialization failed for primary result: {json_err}", file=sys.stderr)
            fallback_result = {
                'output': repr(result.get('output')), # Use repr as fallback
                'error': (result.get('error') or "") + f"\\nJSON Serialization Error: {json_err}",
                'stdout': result.get('stdout'), # Keep original captured strings here
                'stderr': result.get('stderr'),
                'exec_time_ms': result.get('exec_time_ms')
            }
            try:
                final_output_json = json.dumps(fallback_result)
            except Exception as fallback_json_err:
                print(f"ERROR: JSON serialization failed even for fallback result: {fallback_json_err}", file=sys.stderr)
                # Construct minimal error JSON manually, escaping the errors themselves
                error_hint = json.dumps(repr(result.get("error")))[1:-1]
                ser_error = json.dumps(repr(str(fallback_json_err)))[1:-1]
                exec_time = result.get("exec_time_ms", "null") # Use "null" if missing
                final_output_json = (
                    f'{{"output": null, "error": "FATAL: Could not serialize execution results. '
                    f'Original error hint: {error_hint}. Serialization error: {ser_error}", '
                    f'"stdout": null, "stderr": null, "exec_time_ms": {exec_time}}}'
                )

        # --- Print the final JSON to the original stdout ---
        # This print is the crucial communication back to the host
        print(final_output_json)


# --- Run the function ---
if __name__ == "__main__":
    load_and_run_sort()
