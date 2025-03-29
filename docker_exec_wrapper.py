# This file contains the Python script executed inside the Docker container
# by benchmark.py's evaluate_algorithm function.
# It reads input data (list to sort) from stdin as JSON,
# imports and runs the 'llm_sort.sort_algorithm' function from /sandbox/llm_sort.py,
# and prints the result (or error details) as JSON to stdout.

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
        # 1. Read input JSON from stdin FIRST to avoid broken pipes if import fails
        input_data_json = sys.stdin.read()
        if not input_data_json:
             raise ValueError("No input data received via stdin.")

        # 2. Add sandbox to path and import llm_sort directly
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

        # 3. Get the sort_algorithm function
        if not hasattr(llm_sort, 'sort_algorithm') or not callable(llm_sort.sort_algorithm):
            raise NameError("Function 'sort_algorithm' not found or not callable in imported llm_sort module.")
        sort_algorithm = llm_sort.sort_algorithm

        # 4. Parse the input JSON (already read)
        input_list = json.loads(input_data_json)

        # 5. Execute the sort_algorithm
        output_list = sort_algorithm(input_list)
        result['output'] = output_list # Store the actual Python object/list

    except Exception as e:
        # Capture any exception during reading, loading, parsing, or execution
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
        final_output_json = None
        try:
            # Attempt to serialize the primary result
            final_output_json = json.dumps(result)
        except TypeError as json_err:
            # Fallback if the output or other fields are not JSON serializable
            print(f"Warning: JSON serialization failed for primary result: {json_err}", file=sys.stderr) # Log warning to stderr
            fallback_result = {
                'output': repr(result.get('output')), # Use repr as fallback
                'error': (result.get('error') or "") + f"\\nJSON Serialization Error: {json_err}",
                'stdout': result.get('stdout'),
                'stderr': result.get('stderr'),
                'exec_time_ms': result.get('exec_time_ms')
            }
            try:
                # Attempt to serialize the fallback result
                final_output_json = json.dumps(fallback_result)
            except Exception as fallback_json_err:
                # Very unlikely, but catch errors serializing the fallback itself
                print(f"ERROR: JSON serialization failed even for fallback result: {fallback_json_err}", file=sys.stderr)
                # Construct a minimal error JSON string manually
                final_output_json = f'{{"output": null, "error": "FATAL: Could not serialize execution results. Original error hint: {repr(result.get(\\"error\\"))}. Serialization error: {repr(str(fallback_json_err))}", "stdout": null, "stderr": null, "exec_time_ms": {result.get("exec_time_ms", "null")}}}'

        # Print the determined JSON output (either primary, fallback, or minimal error)
        print(final_output_json)


# --- Run the function ---
if __name__ == "__main__":
    load_and_run_sort()
```
