# This script runs inside the Docker container for BENCHMARKING C code.
# It expects configuration via environment variables.
# It loads the test suite (format depends on config) from /sandbox/test_suite.json,
# prepares test data for C functions (depends on config),
# compiles the LLM's C code from /sandbox/llm_code.c using gcc,
# loads the compiled shared library (.so),
# retrieves function pointers using ctypes based on configured names,
# runs all test cases:
#   - Times the primary C function (e.g., compress, sort)
#   - Times the secondary C function if configured (e.g., decompress)
#   - Checks correctness using a configured method
#   - Calculates ratio if configured
# Aggregates results (avg times, avg ratio, overall correctness),
# and prints a single JSON object containing all results to stdout, framed by markers.

import sys
import json
import time
import traceback
import os
import base64
from collections import defaultdict
import subprocess # For calling gcc
import ctypes # For loading shared library and calling C functions
import platform # To determine shared library extension

# --- Configuration via Environment Variables ---
# These MUST be set by the calling process (framework/benchmark_runner.py)

# File paths
LLM_CODE_SOURCE_FILE = os.environ.get("LLM_CODE_SOURCE_FILE", "/sandbox/llm_code.c")
TEST_SUITE_FILE = os.environ.get("TEST_SUITE_FILE", "/sandbox/test_suite.json")

# C Function Names & Signatures (as JSON string)
# Example for Compression: '{"primary": "compress", "secondary": "decompress", "free": "free_buffer"}'
# Example for Sort: '{"primary": "sort_int_array", "free": null}'
C_FUNCTION_NAMES_JSON = os.environ.get("C_FUNCTION_NAMES", '{}')
# Example for Compression: '{"Buffer": {"fields": [["data", "POINTER_ubyte"], ["size", "size_t"]]}, "primary": {"argtypes": ["POINTER_ubyte", "size_t"], "restype": "Buffer"}, ...}'
# Example for Sort: '{"primary": {"argtypes": ["POINTER_int", "size_t"], "restype": null}}'
C_FUNCTION_SIGNATURES_JSON = os.environ.get("C_FUNCTION_SIGNATURES", '{}')

# Benchmark Type & Configuration
# Determines how test suite is loaded and correctness checked
# e.g., "compression", "sort_int_array"
BENCHMARK_TYPE = os.environ.get("BENCHMARK_TYPE", "compression")
# e.g., "1" or "0", indicates if a ratio (input_size / output_size) should be calculated
CALCULATE_RATIO = os.environ.get("CALCULATE_RATIO", "1") == "1"
# e.g., "1" or "0", indicates if a secondary function timing is needed
TIME_SECONDARY_FUNCTION = os.environ.get("TIME_SECONDARY_FUNCTION", "1") == "1"


# --- Constants ---
# Determine shared library extension based on platform (inside container, likely .so)
lib_ext = ".so" if platform.system() == "Linux" else ".dylib" if platform.system() == "Darwin" else ".dll"
LLM_SHARED_LIB_FILE = f"/sandbox/llm_compiled_code{lib_ext}" # Compiled shared library path

# --- CTypes Definitions ---
# Define Buffer struct commonly used in compression, might be needed by others
class CBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)),
                ("size", ctypes.c_size_t)]

# Map string names to ctypes types (add more as needed)
CTYPES_MAP = {
    "POINTER_ubyte": ctypes.POINTER(ctypes.c_ubyte),
    "POINTER_int": ctypes.POINTER(ctypes.c_int),
    "size_t": ctypes.c_size_t,
    "void": None,
    "int": ctypes.c_int,
    "Buffer": CBuffer, # Special case for the struct
    # Add other types like float, double, char*, etc. if required by benchmarks
}

# --- Helper Functions ---

def send_progress(data: dict):
    """Formats data as JSON and prints to stderr."""
    try:
        progress_message = {"type": "progress", "data": data}
        print(json.dumps(progress_message), file=sys.stderr, flush=True)
    except Exception as e:
        print(json.dumps({"type": "progress", "data": {"error": f"Failed to serialize progress: {e}"}}), file=sys.stderr, flush=True)

def parse_json_env_var(var_name: str, json_string: str) -> dict:
    """Parses a JSON string from an environment variable."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in environment variable {var_name}: {e}. Value: '{json_string}'")

def get_ctype(type_name: str):
    """Gets a ctypes type from its string name using CTYPES_MAP."""
    if type_name is None or type_name.lower() == "void":
        return None
    ctype = CTYPES_MAP.get(type_name)
    if ctype is None:
        raise TypeError(f"Unsupported ctypes type name: {type_name}")
    return ctype

# --- Test Suite Loading ---

def load_test_suite_compression(filename: str) -> dict:
    """Loads base64 encoded byte strings for compression benchmarks."""
    send_progress({'status': 'Setup', 'message': f"Loading BASE64 test suite: {filename}"})
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Test suite file not found: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        test_suite_b64 = json.load(f)
    if not test_suite_b64:
        raise ValueError("Test suite file is empty or invalid.")

    send_progress({'status': 'Setup', 'message': "Decoding test cases..."})
    categorized_test_cases_bytes = defaultdict(list)
    total_overall_cases = 0
    decode_errors = 0
    for category, cases_b64 in test_suite_b64.items():
        for i, case_b64 in enumerate(cases_b64):
            try:
                missing_padding = len(case_b64) % 4
                if missing_padding:
                    case_b64 += '=' * (4 - missing_padding)
                decoded_bytes = base64.b64decode(case_b64, validate=True)
                categorized_test_cases_bytes[category].append(decoded_bytes)
                total_overall_cases += 1
            except (base64.binascii.Error, ValueError, TypeError) as decode_err:
                decode_errors += 1
                err_msg = f"Failed to decode base64 case {i+1} in category '{category}': {decode_err}. Skipping case."
                print(f"Warning: {err_msg}", file=sys.stderr)
                send_progress({'status': 'Warning', 'category': category, 'message': err_msg})

    if decode_errors > 0:
         send_progress({'status': 'Warning', 'message': f"Encountered {decode_errors} base64 decoding errors."})
    if total_overall_cases == 0:
        raise ValueError("Test suite contains no valid test cases after decoding.")

    send_progress({'status': 'Setup', 'message': f"Decoded {total_overall_cases} test cases."})
    return dict(categorized_test_cases_bytes)

def load_test_suite_sort_int_array(filename: str) -> dict:
    """Loads arrays of integers for sorting benchmarks."""
    send_progress({'status': 'Setup', 'message': f"Loading INT ARRAY test suite: {filename}"})
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Test suite file not found: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        categorized_test_cases_lists = json.load(f)
    if not categorized_test_cases_lists:
        raise ValueError("Test suite file is empty or invalid.")

    # Basic validation: ensure values are lists of integers
    total_overall_cases = 0
    validation_errors = 0
    validated_cases = defaultdict(list)
    for category, cases in categorized_test_cases_lists.items():
        if not isinstance(cases, list):
             print(f"Warning: Invalid format for category '{category}'. Expected list, got {type(cases)}. Skipping category.", file=sys.stderr)
             validation_errors += 1
             continue
        valid_cases_in_cat = []
        for i, case in enumerate(cases):
            if isinstance(case, list) and all(isinstance(x, int) for x in case):
                valid_cases_in_cat.append(case)
                total_overall_cases += 1
            else:
                print(f"Warning: Invalid data in case {i+1} for category '{category}'. Expected list of ints. Skipping case.", file=sys.stderr)
                validation_errors += 1
        if valid_cases_in_cat:
             validated_cases[category] = valid_cases_in_cat

    if validation_errors > 0:
         send_progress({'status': 'Warning', 'message': f"Encountered {validation_errors} test case validation errors."})
    if total_overall_cases == 0:
        raise ValueError("Test suite contains no valid test cases after validation.")

    send_progress({'status': 'Setup', 'message': f"Loaded {total_overall_cases} valid test cases."})
    return dict(validated_cases)

# --- Main Benchmark Execution ---

def run_all_benchmarks():
    """Loads suite, compiles C code, runs tests, aggregates, and returns results dict."""
    results = {
        'correctness': 0, # Default to failure (0)
        'avg_time_ms': None,
        'avg_secondary_time_ms': None,
        'avg_ratio': None,
        'error': None,
        'performance_details': {} # For potential future use
    }
    c_func_pointers = {}
    llm_lib = None
    categorized_test_cases = None
    c_function_names = {}
    c_function_signatures = {}

    try:
        # --- 1. Parse Configuration ---
        send_progress({'status': 'Setup', 'message': "Parsing configuration..."})
        c_function_names = parse_json_env_var("C_FUNCTION_NAMES", C_FUNCTION_NAMES_JSON)
        c_function_signatures = parse_json_env_var("C_FUNCTION_SIGNATURES", C_FUNCTION_SIGNATURES_JSON)

        # Validate required function names
        if 'primary' not in c_function_names or not c_function_names['primary']:
             raise ValueError("Missing 'primary' function name in C_FUNCTION_NAMES")
        if TIME_SECONDARY_FUNCTION and ('secondary' not in c_function_names or not c_function_names['secondary']):
             raise ValueError("Missing 'secondary' function name in C_FUNCTION_NAMES when TIME_SECONDARY_FUNCTION is true")
        # 'free' function is optional depending on whether C code allocates memory to be returned
        free_func_name = c_function_names.get('free')

        send_progress({'status': 'Setup', 'message': f"Benchmark Type: {BENCHMARK_TYPE}"})

        # --- 2. Load Test Suite (Based on Benchmark Type) ---
        if BENCHMARK_TYPE == "compression":
            categorized_test_cases = load_test_suite_compression(TEST_SUITE_FILE)
            # Define expected output type for correctness check
            get_expected_output = lambda original_input, primary_output: original_input # Decompressed should match original
            prepare_primary_input = lambda data: (ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte)), len(data)) if data else (None, 0)
            prepare_secondary_input = lambda primary_output_buffer: (primary_output_buffer.data, primary_output_buffer.size)
            get_output_bytes_and_size = lambda output_buffer: (ctypes.string_at(output_buffer.data, output_buffer.size), output_buffer.size) if output_buffer.data else (b"", 0)

        elif BENCHMARK_TYPE == "sort_int_array":
            categorized_test_cases = load_test_suite_sort_int_array(TEST_SUITE_FILE)
            # Define expected output: Python's sorted list
            get_expected_output = lambda original_input, primary_output: sorted(original_input)
            # Prepare input: Convert Python list to C int array
            def prepare_primary_input(data_list):
                count = len(data_list)
                c_array = (ctypes.c_int * count)(*data_list)
                return (c_array, count) # Pass the array and count
            # No secondary function for sort
            prepare_secondary_input = lambda primary_output_buffer: (None, 0) # Not applicable
            # Get output: Convert C int array back to Python list (assuming primary modifies in-place or returns pointer)
            # This depends heavily on the C function signature defined in C_FUNCTION_SIGNATURES
            # Assuming sort modifies in-place and primary_output is the input array pointer
            def get_output_bytes_and_size(primary_output_c_array, input_count):
                 # primary_output_c_array is likely the same pointer passed as input
                 # We need the count passed during input prep
                 py_list = list(primary_output_c_array[:input_count])
                 # Return list and its size (count)
                 # We don't have 'bytes' here, so return the list itself and count
                 return (py_list, input_count)

        else:
            raise ValueError(f"Unsupported BENCHMARK_TYPE: {BENCHMARK_TYPE}")

        total_overall_cases = sum(len(cases) for cases in categorized_test_cases.values())
        if total_overall_cases == 0:
             raise ValueError("Test suite is empty after loading/validation.")


        # --- 3. Compile LLM C Code ---
        send_progress({'status': 'Setup', 'message': f"Compiling LLM C code: {LLM_CODE_SOURCE_FILE} -> {LLM_SHARED_LIB_FILE}"})
        if not os.path.exists(LLM_CODE_SOURCE_FILE):
            raise FileNotFoundError(f"LLM C source code file not found: {LLM_CODE_SOURCE_FILE}")

        compile_command = [
            "gcc", "-shared", "-fPIC", "-O2", "-std=c11", # Or -std=c99
            LLM_CODE_SOURCE_FILE, "-o", LLM_SHARED_LIB_FILE
        ]
        try:
            compile_process = subprocess.run(compile_command, check=True, capture_output=True, text=True, timeout=60)
            print(f"GCC stdout:\n{compile_process.stdout}", file=sys.stderr)
            print(f"GCC stderr:\n{compile_process.stderr}", file=sys.stderr)
            send_progress({'status': 'Setup', 'message': "C code compiled successfully."})
        except FileNotFoundError:
             raise RuntimeError("`gcc` compiler not found in container PATH.")
        except subprocess.CalledProcessError as compile_err:
            error_message = f"C code compilation failed (Exit code {compile_err.returncode}).\n" \
                            f"Command: {' '.join(compile_command)}\nStderr:\n{compile_err.stderr}\nStdout:\n{compile_err.stdout}"
            raise RuntimeError(error_message)
        except subprocess.TimeoutExpired as timeout_err:
             raise RuntimeError(f"C code compilation timed out after {timeout_err.timeout} seconds.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during C compilation: {e}")


        # --- 4. Load Shared Library and Define Function Signatures ---
        send_progress({'status': 'Setup', 'message': f"Loading shared library: {LLM_SHARED_LIB_FILE}"})
        try:
            llm_lib = ctypes.CDLL(LLM_SHARED_LIB_FILE)
        except OSError as load_err:
            raise OSError(f"Failed to load shared library {LLM_SHARED_LIB_FILE}: {load_err}")

        try:
            # Define signatures based on JSON config
            defined_structs = {}
            # First define any structs specified
            for name, sig_info in c_function_signatures.items():
                 if sig_info.get("is_struct", False):
                      fields = []
                      for field_name, field_type_str in sig_info.get("fields", []):
                           fields.append((field_name, get_ctype(field_type_str)))
                      # Dynamically create struct class
                      struct_class = type(name, (ctypes.Structure,), {'_fields_': fields})
                      defined_structs[name] = struct_class
                      CTYPES_MAP[name] = struct_class # Add to map for use in function signatures

            # Now define function signatures
            for func_type, func_name in c_function_names.items(): # func_type is 'primary', 'secondary', 'free'
                if not func_name: continue # Skip if function name is null/empty (e.g., no free func)

                sig_info = c_function_signatures.get(func_name)
                if not sig_info:
                     raise ValueError(f"Signature information missing for function '{func_name}' in C_FUNCTION_SIGNATURES_JSON")

                c_func = getattr(llm_lib, func_name) # Get function pointer

                # Set argtypes
                arg_type_names = sig_info.get("argtypes", [])
                c_func.argtypes = [get_ctype(name) for name in arg_type_names]

                # Set restype
                res_type_name = sig_info.get("restype")
                c_func.restype = get_ctype(res_type_name)

                c_func_pointers[func_type] = c_func # Store pointer (e.g., c_func_pointers['primary'])

        except AttributeError as func_err:
             raise NameError(f"Function not found in shared library {LLM_SHARED_LIB_FILE}: {func_err}. Ensure C code defines required functions.")
        except (TypeError, ValueError) as sig_err:
             raise TypeError(f"Error defining C function signatures: {sig_err}")

        send_progress({'status': 'Setup', 'message': "C functions loaded and signatures defined."})


        # --- 5. Iterate and Evaluate Test Cases ---
        total_categories = len(categorized_test_cases)
        current_category_num = 0
        current_overall_case_num = 0
        send_progress({'status': 'Running', 'message': "Starting test case evaluation..."})

        # Aggregators
        total_primary_time_sec = 0.0
        total_secondary_time_sec = 0.0
        total_ratio = 0.0
        total_cases_processed = 0 # Count cases successfully processed through primary/secondary(if applicable)/check
        total_cases_correct = 0
        any_case_failed = False # Flag if any case fails correctness check or errors

        for category, test_cases_in_category in categorized_test_cases.items():
            current_category_num += 1
            num_cases_in_cat = len(test_cases_in_category)
            send_progress({
                'status': 'Running', 'category': category,
                'category_num': current_category_num, 'total_categories': total_categories,
                'category_total_cases': num_cases_in_cat,
                'message': f"Starting category '{category}'..."
            })

            for i, original_input_data in enumerate(test_cases_in_category):
                case_num_in_cat = i + 1
                current_overall_case_num += 1
                # Represent input size/type based on benchmark
                input_size = len(original_input_data) if isinstance(original_input_data, (bytes, list)) else None
                input_repr = f"{type(original_input_data).__name__}[{input_size}]" if input_size is not None else type(original_input_data).__name__

                send_progress({
                    'status': 'Running', 'category': category,
                    'category_num': current_category_num, 'total_categories': total_categories,
                    'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                    'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                    'input_info': input_repr,
                    'message': f"Running Case {current_overall_case_num}/{total_overall_cases}..."
                })

                primary_time_sec = None
                secondary_time_sec = None
                output_size = None
                ratio = None
                is_correct = False
                case_error = None
                primary_output = None # Result from primary C function
                secondary_output = None # Result from secondary C function (if used)
                final_output_for_check = None # Data to compare against expected

                try:
                    # --- Prepare Input for C Primary Function ---
                    c_primary_args = prepare_primary_input(original_input_data)

                    # --- Primary Function Timing & Execution ---
                    start_primary = time.perf_counter()
                    primary_output = c_func_pointers['primary'](*c_primary_args)
                    end_primary = time.perf_counter()
                    primary_time_sec = end_primary - start_primary

                    # --- Get Primary Output Size & Data (for ratio/secondary input) ---
                    # This logic depends on the benchmark type and C function's return
                    primary_output_data, primary_output_size = get_output_bytes_and_size(primary_output, c_primary_args[1] if BENCHMARK_TYPE == "sort_int_array" else None) # Pass input count for sort

                    # --- Calculate Ratio (if applicable) ---
                    if CALCULATE_RATIO:
                        original_size = input_size # Size of the initial input
                        output_size = primary_output_size # Size of the primary output
                        if original_size is not None and output_size is not None:
                            if original_size > 0 and output_size > 0:
                                ratio = original_size / output_size
                            elif original_size == 0 and output_size == 0: ratio = 1.0
                            elif original_size > 0 and output_size == 0: ratio = float('inf')
                            elif original_size == 0 and output_size > 0: ratio = 0.0
                            else: ratio = None
                        else: ratio = None


                    # --- Secondary Function Timing & Execution (if applicable) ---
                    if TIME_SECONDARY_FUNCTION:
                        c_secondary_args = prepare_secondary_input(primary_output) # Use primary output as input
                        start_secondary = time.perf_counter()
                        secondary_output = c_func_pointers['secondary'](*c_secondary_args)
                        end_secondary = time.perf_counter()
                        secondary_time_sec = end_secondary - start_secondary
                        # Get the final data for correctness check from secondary output
                        final_output_for_check, _ = get_output_bytes_and_size(secondary_output, None) # Size not needed here
                    else:
                        # If no secondary function, the primary output is used for check
                        final_output_for_check = primary_output_data


                    # --- Correctness Check ---
                    expected_output = get_expected_output(original_input_data, primary_output)
                    if final_output_for_check == expected_output:
                        is_correct = True
                        total_cases_correct += 1
                        # Aggregate metrics only for fully successful cases
                        total_primary_time_sec += primary_time_sec
                        if TIME_SECONDARY_FUNCTION and secondary_time_sec is not None:
                            total_secondary_time_sec += secondary_time_sec
                        if CALCULATE_RATIO and ratio is not None and ratio != float('inf'):
                             total_ratio += ratio
                        total_cases_processed += 1

                        send_progress({
                            'status': 'Correct', 'category': category,
                            'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                            'primary_time_ms': primary_time_sec * 1000,
                            'secondary_time_ms': secondary_time_sec * 1000 if secondary_time_sec is not None else None,
                            'input_size': input_size,
                            'output_size': output_size,
                            'ratio': ratio,
                            'message': f"Case {current_overall_case_num} Correct."
                        })
                    else:
                        any_case_failed = True
                        # Log details about the mismatch
                        try:
                             mismatch_details = f"Output type: {type(final_output_for_check)}, Expected type: {type(expected_output)}. "
                             mismatch_details += f"Output (first 50): {repr(final_output_for_check[:50])}, Expected (first 50): {repr(expected_output[:50])}"
                        except Exception:
                             mismatch_details = "Could not represent mismatch details."
                        case_error = f"Correctness mismatch: {mismatch_details}"
                        send_progress({
                            'status': 'Incorrect', 'category': category,
                            'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                            'primary_time_ms': primary_time_sec * 1000 if primary_time_sec is not None else None,
                            'secondary_time_ms': secondary_time_sec * 1000 if secondary_time_sec is not None else None,
                            'input_size': input_size,
                            'output_size': output_size,
                            'error': case_error,
                            'message': f"Case {current_overall_case_num} Incorrect."
                        })

                except Exception as exec_err:
                    any_case_failed = True
                    tb_str = traceback.format_exc()
                    print(f"Error during case {current_overall_case_num} ({category}):\n{tb_str}", file=sys.stderr)
                    case_error = f"{type(exec_err).__name__}: {exec_err}"
                    send_progress({
                        'status': 'Error', 'category': category,
                        'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                        'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                        'input_info': input_repr,
                        'error': case_error,
                        'message': f"Case {current_overall_case_num} Error during execution."
                    })

                finally:
                    # --- IMPORTANT: Free C memory if a free function is provided ---
                    if free_func_name and 'free' in c_func_pointers:
                        # Free memory returned by primary and secondary functions if applicable
                        # This assumes they return structs/pointers that need freeing.
                        # Adjust based on actual C function behavior defined by signatures.
                        if primary_output and c_function_signatures[c_function_names['primary']].get('restype'): # Only free if primary returns something
                             try:
                                 # Check if primary_output itself needs freeing or if it contains a pointer
                                 # This logic might need refinement based on signature details
                                 # Assuming Buffer struct for now:
                                 if isinstance(primary_output, CBuffer) and primary_output.data:
                                     c_func_pointers['free'](primary_output)
                                 # Add elif for other freeable types if needed
                             except Exception as free_err:
                                 print(f"Warning: Error calling free function for primary output: {free_err}", file=sys.stderr)
                        if secondary_output and c_function_signatures[c_function_names['secondary']].get('restype'): # Only free if secondary returns something
                             try:
                                 if isinstance(secondary_output, CBuffer) and secondary_output.data:
                                     c_func_pointers['free'](secondary_output)
                                 # Add elif for other freeable types if needed
                             except Exception as free_err:
                                 print(f"Warning: Error calling free function for secondary output: {free_err}", file=sys.stderr)


        # --- 6. Calculate Final Aggregated Results ---
        send_progress({'status': 'Aggregating', 'message': "Calculating final results..."})
        if total_overall_cases > 0:
            results['correctness'] = 1 if not any_case_failed and total_cases_processed == total_overall_cases else 0

            if total_cases_processed > 0:
                results['avg_time_ms'] = (total_primary_time_sec / total_cases_processed) * 1000
                if TIME_SECONDARY_FUNCTION:
                    results['avg_secondary_time_ms'] = (total_secondary_time_sec / total_cases_processed) * 1000
                if CALCULATE_RATIO:
                    # Avoid division by zero if total_ratio is 0 but cases were processed
                    results['avg_ratio'] = total_ratio / total_cases_processed if total_cases_processed > 0 else 0
            else:
                 results['error'] = results.get('error') or "No test cases were successfully processed."
                 results['correctness'] = 0

        else:
             results['error'] = "No test cases found to process."
             results['correctness'] = 0


    except Exception as top_level_err:
        tb_str = traceback.format_exc()
        print(f"Critical error in wrapper script:\n{tb_str}", file=sys.stderr)
        results['error'] = f"Critical error in wrapper: {type(top_level_err).__name__}: {top_level_err}"
        results['correctness'] = 0
        send_progress({'status': 'Error', 'message': results['error']})


    # --- 7. Return the final results dictionary ---
    send_progress({'status': 'Finished', 'message': "Benchmark evaluation complete."})
    return results

# --- Main execution block ---
if __name__ == "__main__":
    final_results = run_all_benchmarks()

    # --- Print the final aggregated results as JSON to stdout ---
    final_message = {"type": "result", "data": final_results}
    print("DEBUG WRAPPER: Preparing to print final JSON to stdout.", file=sys.stderr, flush=True)
    print("---WRAPPER_STDOUT_MARKER_BEFORE---", flush=True)
    try:
        # Handle potential non-finite floats (inf, nan) before JSON dump
        def handle_non_finite(obj):
             if isinstance(obj, float):
                 if obj == float('inf'): return "Infinity"
                 if obj == float('-inf'): return "-Infinity"
                 if obj != obj: return "NaN" # Check for NaN
             # Let default JSON encoder handle the rest
             raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # Use default=handle_non_finite only if necessary, json handles inf/nan by default
        # but target might not. Let's convert them to strings.
        final_json_string = json.dumps(final_message, default=lambda o: handle_non_finite(o) if isinstance(o, float) and not o.is_integer() and (o == float('inf') or o == float('-inf') or o != o) else o)

        print(f"DEBUG WRAPPER: Final JSON string length: {len(final_json_string)}", file=sys.stderr, flush=True)
        print(final_json_string, flush=True)
        print("---WRAPPER_STDOUT_MARKER_AFTER---", flush=True)
        print("DEBUG WRAPPER: Successfully printed final JSON to stdout.", file=sys.stderr, flush=True)
    except Exception as final_print_err:
         print(f"DEBUG WRAPPER: Caught Exception ({type(final_print_err).__name__}) during final JSON serialization/print: {final_print_err}", file=sys.stderr, flush=True)
         # Fallback structure matching generic results
         fallback_error = f"FATAL: Error printing final JSON: {final_print_err}. Original error: {final_results.get('error')}"
         final_message = {"type": "result", "data": {
             'error': fallback_error, 'correctness': 0,
             'avg_time_ms': None, 'avg_secondary_time_ms': None, 'avg_ratio': None,
             'performance_details': {}
         }}
         print(f"DEBUG WRAPPER: Printing fallback JSON.", file=sys.stderr, flush=True)
         # Use standard json.dumps for fallback, should be safe
         print(json.dumps(final_message), flush=True)
         print("---WRAPPER_STDOUT_MARKER_AFTER---", flush=True) # Still need markers
         print("DEBUG WRAPPER: Successfully printed fallback JSON to stdout.", file=sys.stderr, flush=True)

    # Exit code: 0 if correctness is 1 AND no critical error occurred, 1 otherwise.
    exit_code_to_use = 1
    if final_results.get('correctness') == 1 and not final_results.get('error'):
        exit_code_to_use = 0

    print(f"DEBUG WRAPPER: Exiting with code {exit_code_to_use}.", file=sys.stderr, flush=True)
    sys.exit(exit_code_to_use)
