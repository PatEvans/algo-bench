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
import importlib.util # For importing Python modules dynamically

# --- Configuration via Environment Variables ---
# These MUST be set by the calling process (framework/benchmark_runner.py)

# File paths - Determine source file based on benchmark type later
LLM_CODE_SOURCE_FILE_C = "/sandbox/llm_code.c"
LLM_CODE_SOURCE_FILE_PY = "/sandbox/llm_sort.py" # Example for Python sort
TEST_SUITE_FILE = os.environ.get("TEST_SUITE_FILE", "/sandbox/test_suite.json")

# Function Names & Signatures (as JSON string) - Could be C or Python
# Example C: '{"primary": "compress", "secondary": "decompress", "free": "free_buffer"}'
# Example Py: '{"primary": "sort_algorithm"}'
FUNCTION_NAMES_JSON = os.environ.get("FUNCTION_NAMES", '{}')
# Example C: '{"Buffer": ..., "compress": {"argtypes": ["POINTER_ubyte", "size_t"], "restype": "Buffer"}, ...}'
# Example Py: '{"sort_algorithm": {"argtypes": ["list_int"], "restype": "list_int"}}' # Less formal for Python
FUNCTION_SIGNATURES_JSON = os.environ.get("FUNCTION_SIGNATURES", '{}')

# Benchmark Type & Configuration
# Determines how test suite is loaded, data prepared, correctness checked, etc.
# e.g., "c_compression", "c_sort_int_array"
BENCHMARK_TYPE = os.environ.get("BENCHMARK_TYPE")
if not BENCHMARK_TYPE:
    raise ValueError("BENCHMARK_TYPE environment variable not set.")

# --- Optional Configuration ---
# Default to False/0 if not explicitly set to "1"
CALCULATE_RATIO = os.environ.get("CALCULATE_RATIO", "0") == "1"
TIME_SECONDARY_FUNCTION = os.environ.get("TIME_SECONDARY_FUNCTION", "0") == "1"

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
    # Add other types like float, double, char*, etc. if required by C benchmarks
    # Python types are handled differently (passed directly or via JSON)
    "list_int": list, # Placeholder for Python list of ints
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
    """Gets a ctypes type from its string name using CTYPES_MAP. Returns None for non-ctypes."""
    if type_name is None or type_name.lower() == "void":
        return None
    # Handle Python types separately if needed, otherwise assume ctypes
    if type_name in ["list_int"]: # Add other Python type hints here
        return CTYPES_MAP.get(type_name) # Return the Python type itself

    # Assume ctypes otherwise
    ctype = CTYPES_MAP.get(type_name)
    if ctype is None:
        # Only raise error if it wasn't a known Python type hint
        raise TypeError(f"Unsupported type name: {type_name}")
    return ctype

# --- Data Preparation & Correctness Check Dispatch ---
# These functions will be selected based on BENCHMARK_TYPE

def prepare_input_compression(data: bytes) -> tuple:
    """Prepare bytes for C compression function."""
    return (ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte)), len(data)) if data else (None, 0)

def prepare_secondary_input_compression(primary_output_buffer: CBuffer) -> tuple:
    """Prepare CBuffer output from compress for decompress input."""
    return (primary_output_buffer.data, primary_output_buffer.size)

def get_output_data_compression(output_buffer: CBuffer, input_args: tuple) -> tuple[bytes, int]:
    """Extract bytes and size from CBuffer."""
    return (ctypes.string_at(output_buffer.data, output_buffer.size), output_buffer.size) if output_buffer.data else (b"", 0)

def check_correctness_compression(original_input: bytes, final_output_data: bytes) -> bool:
    """Compare original bytes with final decompressed bytes."""
    return original_input == final_output_data

def get_input_size_compression(original_input: bytes) -> int:
    """Get size of original byte input."""
    return len(original_input)

# --- Python Sort Helpers ---

def prepare_input_python_sort(data_list: list) -> tuple:
    """Prepare list for Python sort function (just pass it)."""
    # The function expects the list directly. Return as tuple for consistency.
    return (list(data_list),) # Pass a copy to prevent modification of original test case

def prepare_secondary_input_python_sort(primary_output: list) -> tuple:
    """No secondary input needed for sort."""
    return () # Empty tuple

def get_output_data_python_sort(output_list: list, input_args: tuple) -> tuple[list, int]:
    """Extract list and size from Python sort output."""
    # output_list is the direct result from the sort function
    return (output_list, len(output_list) if isinstance(output_list, list) else 0)

def check_correctness_python_sort(original_input: list, final_output_data: list) -> bool:
    """Compare output list with Python's built-in sorted()."""
    if not isinstance(final_output_data, list): # Check if LLM returned a list
        return False
    # Ensure original_input is treated as a list for comparison
    if not isinstance(original_input, list):
         return False # Or handle appropriately if non-list inputs are possible
    return sorted(original_input) == final_output_data

def get_input_size_python_sort(original_input: list) -> int:
    """Get size of input list."""
    return len(original_input) if isinstance(original_input, list) else 0
# --- C Sort Helpers ---

def prepare_input_c_sort(data_list: list) -> tuple:
    """Prepare list for C sort function (convert to ctypes array)."""
    if not data_list:
        return (None, 0)
    n = len(data_list)
    # Create a ctypes array of integers from the Python list
    c_array = (ctypes.c_int * n)(*data_list)
    # Return pointer to the array and its size
    return (ctypes.cast(c_array, ctypes.POINTER(ctypes.c_int)), n)

def prepare_secondary_input_c_sort(primary_output: None) -> tuple:
    """No secondary input needed for sort."""
    return () # Empty tuple

def get_output_data_c_sort(primary_output: None, input_args: tuple) -> tuple[list, int]:
    """Extract list and size from C sort output (which modified the input array)."""
    # The C function sorts in-place, so the 'output' is the modified input array.
    # We need the original pointer and size passed in input_args.
    c_array_ptr, n = input_args
    if c_array_ptr is None or n == 0:
        return ([], 0)
    # Convert the (potentially modified) ctypes array back to a Python list
    # Dereference the pointer back to the array type to access elements
    c_array = ctypes.cast(c_array_ptr, ctypes.POINTER(ctypes.c_int * n)).contents
    output_list = list(c_array)
    return (output_list, n)

def check_correctness_c_sort(original_input: list, final_output_data: list) -> bool:
    """Compare output list (from modified C array) with Python's sorted() of original."""
    if not isinstance(final_output_data, list): # Check if conversion worked
        return False
    # Ensure original_input is treated as a list for comparison
    if not isinstance(original_input, list):
         return False # Should always be a list from test suite
    # Compare the list derived from the C array with the sorted version of the *original* list
    return sorted(original_input) == final_output_data

def get_input_size_c_sort(original_input: list) -> int:
    """Get size of input list."""
    return len(original_input) if isinstance(original_input, list) else 0


# --- Dispatch Dictionary ---
# Maps BENCHMARK_TYPE to the relevant helper functions
BENCHMARK_HELPERS = {
    "c_compression": {
        "is_c_benchmark": True, # Flag for C-specific steps
        "source_file": LLM_CODE_SOURCE_FILE_C,
        "load_suite": lambda f: load_test_suite_generic(f, decode_base64=True, expected_type=bytes),
        "prepare_primary_input": prepare_input_compression,
        "prepare_secondary_input": prepare_secondary_input_compression,
        "get_output_data": get_output_data_compression,
        "check_correctness": check_correctness_compression,
        "get_input_size": get_input_size_compression,
        "free_func_name_config": "free", # Name from FUNCTION_NAMES config
        "free_arg_type": "Buffer",       # Type expected by the free function
    },
    "python_sort": {
        "is_c_benchmark": False, # Flag for Python-specific steps
        "source_file": LLM_CODE_SOURCE_FILE_PY,
        "load_suite": lambda f: load_test_suite_generic(f, decode_base64=False, expected_type=list),
        "prepare_primary_input": prepare_input_python_sort,
        "prepare_secondary_input": prepare_secondary_input_python_sort, # No-op
        "get_output_data": get_output_data_python_sort,
        "check_correctness": check_correctness_python_sort,
        "get_input_size": get_input_size_python_sort,
        "free_func_name_config": None, # No free function for Python sort
        "free_arg_type": None,
    },
    "c_sort_int_array": {
        "is_c_benchmark": True, # Flag for C-specific steps
        "source_file": LLM_CODE_SOURCE_FILE_C, # Assumes LLM code is in llm_code.c
        "load_suite": lambda f: load_test_suite_generic(f, decode_base64=False, expected_type=list), # Loads lists of ints
        "prepare_primary_input": prepare_input_c_sort,
        "prepare_secondary_input": prepare_secondary_input_c_sort, # No-op
        "get_output_data": get_output_data_c_sort, # Gets data from modified input array
        "check_correctness": check_correctness_c_sort,
        "get_input_size": get_input_size_c_sort,
        "free_func_name_config": None, # No specific free function needed if sort is in-place
        "free_arg_type": None,
    },
    # Add entries for other benchmark types here
}


# --- Generic Test Suite Loading ---

def load_test_suite_generic(filename: str, decode_base64: bool = False, expected_type: type = bytes) -> dict:
    """Loads a test suite from JSON, optionally decoding base64."""
    send_progress({'status': 'Setup', 'message': f"Loading test suite: {filename} (Base64: {decode_base64}, Type: {expected_type.__name__})"})
    if not os.path.exists(filename): # Check if file exists before opening
        raise FileNotFoundError(f"Test suite file not found: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        test_suite_raw = json.load(f)
    if not test_suite_raw:
        raise ValueError("Test suite file is empty or invalid.")

    send_progress({'status': 'Setup', 'message': "Processing/Validating test cases..."})
    categorized_test_cases_processed = defaultdict(list)
    total_overall_cases = 0
    processing_errors = 0

    for category, cases_raw in test_suite_raw.items():
        if not isinstance(cases_raw, list):
            processing_errors += 1
            err_msg = f"Invalid format for category '{category}'. Expected list, got {type(cases_raw)}. Skipping category."
            print(f"Warning: {err_msg}", file=sys.stderr)
            send_progress({'status': 'Warning', 'category': category, 'message': err_msg})
            continue

        valid_cases_in_cat = []
        for i, case_raw in enumerate(cases_raw):
            processed_case = None
            try:
                if decode_base64:
                    # Handle potential padding issues during decode
                    missing_padding = len(case_raw) % 4
                    if missing_padding:
                        case_raw += '=' * (4 - missing_padding)
                    processed_case = base64.b64decode(case_raw, validate=True)
                else:
                    processed_case = case_raw # Assume already in correct format (e.g., list for sort)

                # Validate type if expected_type is provided
                if expected_type and not isinstance(processed_case, expected_type):
                     # Special check for list of ints if expected_type is list
                     if expected_type is list and isinstance(processed_case, list) and all(isinstance(x, int) for x in processed_case):
                          pass # It's a list of ints, which is valid
                     else:
                          raise TypeError(f"Expected type {expected_type.__name__}, got {type(processed_case).__name__}")

                valid_cases_in_cat.append(processed_case)
                total_overall_cases += 1

            except (base64.binascii.Error, ValueError, TypeError) as proc_err:
                processing_errors += 1
                err_msg = f"Failed to process/validate case {i+1} in category '{category}': {proc_err}. Skipping case."
                print(f"Warning: {err_msg}", file=sys.stderr)
                send_progress({'status': 'Warning', 'category': category, 'message': err_msg})

        if valid_cases_in_cat:
            categorized_test_cases_processed[category] = valid_cases_in_cat

    if processing_errors > 0:
         send_progress({'status': 'Warning', 'message': f"Encountered {processing_errors} test case processing/validation errors."})
    if total_overall_cases == 0:
        raise ValueError("Test suite contains no valid test cases after processing/validation.")

    send_progress({'status': 'Setup', 'message': f"Loaded and processed {total_overall_cases} valid test cases."})
    return dict(categorized_test_cases_processed)


# --- Main Benchmark Execution ---

def run_all_benchmarks():
    """Loads suite, compiles C code, runs tests, aggregates, and returns results dict."""
    results = {
        'correctness': 0, # Default to failure (0)
        'avg_time_ms': None,
        'avg_secondary_time_ms': None,
        'avg_ratio': None,
        'error': None,
        'performance_details': {} # Stores per-category results if needed later
    }
    # Function pointers/references (can be C or Python)
    func_pointers = {}
    llm_module_or_lib = None # Can be ctypes.CDLL or Python module
    categorized_test_cases = None
    function_names = {}
    function_signatures = {}
    helpers = {}

    try:
        # --- 1. Parse Configuration & Select Helpers ---
        send_progress({'status': 'Setup', 'message': "Parsing configuration..."})
        function_names = parse_json_env_var("FUNCTION_NAMES", FUNCTION_NAMES_JSON)
        function_signatures = parse_json_env_var("FUNCTION_SIGNATURES", FUNCTION_SIGNATURES_JSON)

        # Validate required function names
        primary_func_name = function_names.get('primary')
        secondary_func_name = function_names.get('secondary') # Optional
        free_func_name = function_names.get('free') # Optional (mainly for C)

        if not primary_func_name:
             raise ValueError("Missing 'primary' function name in FUNCTION_NAMES")
        # Validation for secondary/free depends on benchmark type and flags
        # Assign env var value to local variable for use in this function
        should_time_secondary = TIME_SECONDARY_FUNCTION

        send_progress({'status': 'Setup', 'message': f"Benchmark Type: {BENCHMARK_TYPE}"})

        # Select the appropriate helper functions based on BENCHMARK_TYPE
        helpers = BENCHMARK_HELPERS.get(BENCHMARK_TYPE)
        if not helpers:
            raise ValueError(f"Unsupported BENCHMARK_TYPE: {BENCHMARK_TYPE}. No helpers defined.")

        # Determine if it's a C benchmark for conditional steps
        is_c_benchmark = helpers.get("is_c_benchmark", False)
        llm_source_file = helpers.get("source_file")
        if not llm_source_file:
             raise ValueError(f"Missing 'source_file' definition for BENCHMARK_TYPE: {BENCHMARK_TYPE}")

        # Validate secondary/free function requirements based on type and flags
        if is_c_benchmark:
             if should_time_secondary and not secondary_func_name:
                  raise ValueError("Missing 'secondary' function name in FUNCTION_NAMES when secondary timing is enabled for a C benchmark")
             # Free function is optional for C, handled later
        else: # Python benchmark
             if should_time_secondary:
                  print("Warning: Secondary timing was enabled, but benchmark type is Python. Ignoring secondary timing.", file=sys.stderr)
                  should_time_secondary = False # Override local flag for Python
             if free_func_name:
                  print("Warning: 'free' function specified in FUNCTION_NAMES, but benchmark type is Python. Ignoring.", file=sys.stderr)
                  free_func_name = None # Override for Python


        # --- 2. Load Test Suite (Using selected helper) ---
        load_suite_func = helpers['load_suite']
        categorized_test_cases = load_suite_func(TEST_SUITE_FILE)

        total_overall_cases = sum(len(cases) for cases in categorized_test_cases.values())
        if total_overall_cases == 0:
             raise ValueError("Test suite is empty after loading/processing.")

        # --- 3. Compile C Code OR Load Python Module ---
        if is_c_benchmark:
            # --- Compile LLM C Code ---
            send_progress({'status': 'Setup', 'message': f"Compiling LLM C code: {llm_source_file} -> {LLM_SHARED_LIB_FILE}"})
            if not os.path.exists(llm_source_file):
                raise FileNotFoundError(f"LLM C source code file not found: {llm_source_file}")

            compile_command = [
                "gcc", "-shared", "-fPIC", "-O2", "-std=c11", # Or -std=c99
                llm_source_file, "-o", LLM_SHARED_LIB_FILE
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

            # --- Load Shared Library and Define C Function Signatures ---
            send_progress({'status': 'Setup', 'message': f"Loading shared library: {LLM_SHARED_LIB_FILE}"})
            try:
                llm_module_or_lib = ctypes.CDLL(LLM_SHARED_LIB_FILE)
            except OSError as load_err:
                raise OSError(f"Failed to load shared library {LLM_SHARED_LIB_FILE}: {load_err}")

            try:
                # Define C signatures based on JSON config
                defined_structs = {}
                # First define any structs specified
                for name, sig_info in function_signatures.items():
                    # Ensure consistent 4-space indentation
                    if sig_info.get("is_struct", False):
                        fields = []
                        for field_name, field_type_str in sig_info.get("fields", []):
                            ctype = get_ctype(field_type_str)
                            if ctype is None: # Should not happen if validation passed
                                raise TypeError(f"Invalid type '{field_type_str}' for struct field '{field_name}'")
                            fields.append((field_name, ctype))
                        # Dynamically create struct class
                        struct_class = type(name, (ctypes.Structure,), {'_fields_': fields})
                        defined_structs[name] = struct_class
                        CTYPES_MAP[name] = struct_class # Add to map for use in function signatures

                # Now define function signatures using the actual function names
                for func_type, func_name in function_names.items(): # func_type is 'primary', 'secondary', 'free'
                    if not func_name: continue # Skip if function name is null/empty

                    sig_info = function_signatures.get(func_name)
                    if not sig_info:
                         # Allow missing signature for 'free' if it wasn't required/provided
                         if func_type == 'free':
                              print(f"Warning: Signature info missing for optional 'free' function '{func_name}'. Will not be callable.", file=sys.stderr)
                              free_func_name = None # Ensure we don't try to call it later
                              continue
                         else:
                              raise ValueError(f"Signature information missing for function '{func_name}' in FUNCTION_SIGNATURES_JSON")

                    try:
                        c_func = getattr(llm_module_or_lib, func_name) # Get function pointer
                    except AttributeError:
                         # Allow missing 'free' function if it wasn't required/provided
                         if func_type == 'free':
                              print(f"Warning: Optional 'free' function '{func_name}' not found in shared library.", file=sys.stderr)
                              free_func_name = None # Ensure we don't try to call it later
                              continue
                         else:
                              raise NameError(f"Function '{func_name}' not found in shared library {LLM_SHARED_LIB_FILE}.")

                    # Set argtypes
                    arg_type_names = sig_info.get("argtypes", [])
                    c_func.argtypes = [get_ctype(name) for name in arg_type_names if get_ctype(name) is not None] # Filter out non-ctypes

                    # Set restype
                    res_type_name = sig_info.get("restype")
                    c_func.restype = get_ctype(res_type_name)

                    func_pointers[func_type] = c_func # Store pointer (e.g., func_pointers['primary'])

            except AttributeError as func_err:
                 raise NameError(f"Function not found in shared library {LLM_SHARED_LIB_FILE}: {func_err}. Ensure C code defines required functions.")
            except (TypeError, ValueError) as sig_err:
                 raise TypeError(f"Error defining C function signatures: {sig_err}")

            send_progress({'status': 'Setup', 'message': "C functions loaded and signatures defined."})

        else: # --- Load Python Module ---
            send_progress({'status': 'Setup', 'message': f"Loading LLM Python module: {llm_source_file}"})
            if not os.path.exists(llm_source_file):
                raise FileNotFoundError(f"LLM Python source code file not found: {llm_source_file}")

            module_name = os.path.splitext(os.path.basename(llm_source_file))[0] # e.g., "llm_sort"
            try:
                spec = importlib.util.spec_from_file_location(module_name, llm_source_file)
                if spec is None or spec.loader is None:
                     raise ImportError(f"Could not create module spec for {llm_source_file}")
                llm_module_or_lib = importlib.util.module_from_spec(spec)
                # Add to sys.modules BEFORE exec_module to handle relative imports within the loaded module
                sys.modules[module_name] = llm_module_or_lib
                spec.loader.exec_module(llm_module_or_lib)
                send_progress({'status': 'Setup', 'message': "Python module loaded successfully."})

                # Get function references
                for func_type, func_name in function_names.items():
                    if not func_name: continue
                    try:
                        py_func = getattr(llm_module_or_lib, func_name)
                        if not callable(py_func):
                             raise TypeError(f"Attribute '{func_name}' in module {module_name} is not callable.")
                        func_pointers[func_type] = py_func # Store Python function reference
                    except AttributeError:
                         # Only raise error if the function was mandatory (e.g., primary)
                         if func_type == 'primary':
                              raise NameError(f"Primary function '{func_name}' not found in Python module {llm_source_file}.")
                         else:
                              print(f"Warning: Optional function '{func_name}' not found in Python module {llm_source_file}.", file=sys.stderr)

            except Exception as import_err:
                raise ImportError(f"Error importing Python module '{module_name}' from {llm_source_file}: {type(import_err).__name__}: {import_err}\n{traceback.format_exc()}")

            send_progress({'status': 'Setup', 'message': "Python functions loaded."})


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
                # Get input size/representation using helper
                input_size = helpers['get_input_size'](original_input_data)
                input_repr = f"{type(original_input_data).__name__}[{input_size}]"

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
                    # --- Prepare Input for Primary Function (using helper) ---
                    primary_args = helpers['prepare_primary_input'](original_input_data)

                    # --- Primary Function Timing & Execution ---
                    start_primary = time.perf_counter()
                    primary_output = func_pointers['primary'](*primary_args)
                    end_primary = time.perf_counter()
                    primary_time_sec = end_primary - start_primary

                    # --- Get Primary Output Size & Data (using helper) ---
                    # Pass original args as they might be needed (e.g., input count for sort)
                    primary_output_data, primary_output_size = helpers['get_output_data'](primary_output, primary_args)

                    # --- Calculate Ratio (if applicable, typically only for C compression) ---
                    if CALCULATE_RATIO and is_c_benchmark: # Only calculate ratio for C benchmarks configured to do so
                        original_size = input_size # Size of the initial input
                        output_size = primary_output_size # Size of the primary output
                        if original_size is not None and output_size is not None:
                            # Handle division by zero or zero input/output
                            if output_size > 0:
                                ratio = original_size / output_size
                            elif original_size == 0 and output_size == 0: ratio = 1.0 # Empty input to empty output
                            elif original_size > 0 and output_size == 0: ratio = float('inf') # Non-empty input compressed to nothing
                            else: ratio = 0.0 # Empty input to non-empty output (shouldn't happen?)
                        else: ratio = None


                    # --- Secondary Function Timing & Execution (if applicable) ---
                    # Use the locally adjusted should_time_secondary flag
                    if should_time_secondary and 'secondary' in func_pointers:
                        # Prepare secondary input using helper (takes primary output)
                        secondary_args = helpers['prepare_secondary_input'](primary_output)
                        start_secondary = time.perf_counter()
                        secondary_output = func_pointers['secondary'](*secondary_args)
                        end_secondary = time.perf_counter()
                        secondary_time_sec = end_secondary - start_secondary
                        # Get the final data for correctness check from secondary output (using helper)
                        final_output_for_check, _ = helpers['get_output_data'](secondary_output, secondary_args)
                    else:
                        # If no secondary function, the primary output data is used for check
                        final_output_for_check = primary_output_data


                    # --- Correctness Check (using helper) ---
                    # For in-place sorts (like c_sort_int_array), the 'final_output_for_check'
                    # is derived from the modified input array. The helper needs the *original*
                    # input data for comparison.
                    is_correct = helpers['check_correctness'](original_input_data, final_output_for_check)

                    if is_correct:
                        total_cases_correct += 1
                        # Aggregate metrics only for fully successful cases (correctness check passed)
                        total_primary_time_sec += primary_time_sec
                        # Note: TIME_SECONDARY_FUNCTION is likely False for sort, but check anyway
                        if TIME_SECONDARY_FUNCTION and secondary_time_sec is not None:
                            total_secondary_time_sec += secondary_time_sec
                        if CALCULATE_RATIO and ratio is not None and ratio != float('inf'):
                             total_ratio += ratio
                       # The following duplicate aggregation lines seem redundant and were removed in a previous step.
                       # Keeping the SEARCH block structure correct, but the REPLACE block reflects the intended state
                       # after removing the duplicate lines and applying the variable name change.
                        if CALCULATE_RATIO and ratio is not None and ratio != float('inf') and ratio != float('-inf') and ratio == ratio: # Check for inf/nan
                             total_ratio += ratio
                        total_cases_processed += 1 # Increment only if correct

                        send_progress({
                            'status': 'Correct', 'category': category,
                            'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                            'primary_time_ms': primary_time_sec * 1000,
                            'secondary_time_ms': secondary_time_sec * 1000 if secondary_time_sec is not None else None,
                            'input_size': input_size,
                            'output_size': primary_output_size, # Use primary output size for reporting
                            'ratio': ratio,
                            'message': f"Case {current_overall_case_num} Correct."
                        })
                    else:
                        any_case_failed = True
                        # Log details about the mismatch (use repr for clarity)
                        try:
                             # Generate expected output using the same helper logic as the check
                             # Note: This assumes check_correctness helper doesn't modify input/output
                             expected_output_for_log = "N/A"
                             if BENCHMARK_TYPE == "c_compression":
                                 expected_output_for_log = original_input_data
                             elif BENCHMARK_TYPE == "c_sort_int_array":
                                 # Ensure original_input is a list before sorting for logging
                                 expected_output_for_log = sorted(original_input_data) if isinstance(original_input_data, list) else "N/A (Input not list)"
                             # Add other types as needed

                             mismatch_details = f"Output (type {type(final_output_for_check).__name__}, first 100): {repr(final_output_for_check[:100])}. Expected (type {type(expected_output_for_log).__name__}, first 100): {repr(expected_output_for_log[:100])}"
                        except Exception as log_err:
                             mismatch_details = f"Could not represent mismatch details: {log_err}"
                        case_error = f"Correctness mismatch. {mismatch_details}"
                        send_progress({
                            'status': 'Incorrect', 'category': category,
                            'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                            'primary_time_ms': primary_time_sec * 1000 if primary_time_sec is not None else None,
                            'secondary_time_ms': secondary_time_sec * 1000 if secondary_time_sec is not None else None,
                            'input_size': input_size,
                            'output_size': primary_output_size, # Use primary output size
                            'error': case_error,
                            'message': f"Case {current_overall_case_num} Incorrect."
                        })
                        # Do not increment total_cases_processed for incorrect cases

                except Exception as exec_err:
                    any_case_failed = True # Mark run as failed
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
                    # --- IMPORTANT: Free C memory ONLY for C benchmarks that require it ---
                    # Get the configured free function name and expected type from helpers
                    free_func_name_config = helpers.get("free_func_name_config") # e.g., "free"
                    free_arg_type_name = helpers.get("free_arg_type") # e.g., "Buffer"

                    if is_c_benchmark and free_func_name_config and free_arg_type_name and 'free' in func_pointers and func_pointers['free']:
                        free_func = func_pointers['free']
                        free_arg_ctype = CTYPES_MAP.get(free_arg_type_name) # Get the actual ctype (e.g., CBuffer)

                        if not free_arg_ctype:
                             print(f"Warning: Could not find ctype for configured free argument type '{free_arg_type_name}'. Skipping free.", file=sys.stderr)
                             continue # Skip freeing for this case

                        # Determine what needs freeing based on function signatures
                        primary_sig = function_signatures.get(primary_func_name, {})
                        secondary_sig = function_signatures.get(secondary_func_name, {}) if secondary_func_name else {}
                        primary_restype_name = primary_sig.get('restype')
                        secondary_restype_name = secondary_sig.get('restype')

                        # Free primary output if its return type matches the configured free_arg_type
                        if primary_output is not None and primary_restype_name == free_arg_type_name:
                            if isinstance(primary_output, free_arg_ctype):
                                try:
                                    # Special check for Buffer type to see if data is valid
                                    should_free = True
                                    if free_arg_type_name == "Buffer" and not getattr(primary_output, 'data', True):
                                         should_free = False

                                    if should_free:
                                         free_func(primary_output)
                                         # Optional: Nullify fields after free
                                         # if free_arg_type_name == "Buffer":
                                         #     primary_output.data = None
                                         #     primary_output.size = 0
                                except Exception as free_err:
                                    print(f"Warning: Error calling free function for primary output ({primary_restype_name}): {free_err}", file=sys.stderr)
                            else:
                                 print(f"Warning: Expected {free_arg_type_name} for primary output but got {type(primary_output)}. Cannot free.", file=sys.stderr)
                        # Add elif for other freeable C primary return types here (if needed)

                        # Free secondary output if its return type matches the configured free_arg_type
                        if secondary_output is not None and secondary_restype_name == free_arg_type_name:
                             if isinstance(secondary_output, free_arg_ctype):
                                 try:
                                     should_free = True
                                     if free_arg_type_name == "Buffer" and not getattr(secondary_output, 'data', True):
                                          should_free = False

                                     if should_free:
                                          free_func(secondary_output)
                                          # Optional: Nullify fields after free
                                          # if free_arg_type_name == "Buffer":
                                          #     secondary_output.data = None
                                          #     secondary_output.size = 0
                                 except Exception as free_err:
                                     print(f"Warning: Error calling free function for secondary output ({secondary_restype_name}): {free_err}", file=sys.stderr)
                             else:
                                 print(f"Warning: Expected {free_arg_type_name} for secondary output but got {type(secondary_output)}. Cannot free.", file=sys.stderr)
                        # Add elif for other freeable C secondary return types here (if needed)
                    # --- End of Freeing Logic ---


        # --- 6. Calculate Final Aggregated Results ---
        send_progress({'status': 'Aggregating', 'message': "Calculating final results..."})
        if total_overall_cases > 0:
             # Correctness is 1 only if NO cases failed AND at least one case was processed successfully
             # (Handles case where all cases error out before correctness check)
            results['correctness'] = 1 if not any_case_failed and total_cases_processed > 0 else 0

            if total_cases_processed > 0: # Calculate averages only based on successfully processed cases
                results['avg_time_ms'] = (total_primary_time_sec / total_cases_processed) * 1000
                if TIME_SECONDARY_FUNCTION and total_secondary_time_sec > 0: # Check if secondary time was actually recorded
                    results['avg_secondary_time_ms'] = (total_secondary_time_sec / total_cases_processed) * 1000
                else:
                     results['avg_secondary_time_ms'] = None # Explicitly set to None if not timed or no successful cases

                if CALCULATE_RATIO and total_ratio > 0: # Check if ratio was calculated and > 0
                    results['avg_ratio'] = total_ratio / total_cases_processed
                else:
                     results['avg_ratio'] = None # Explicitly set to None

                # If correctness is 0 but some cases were processed, the error is likely the first failure encountered
                if results['correctness'] == 0 and not results.get('error'):
                     results['error'] = results.get('error') or "One or more test cases failed correctness check."

            else: # No cases processed successfully
                 results['error'] = results.get('error') or "No test cases were successfully processed (all failed or errored)."
                 results['correctness'] = 0
                 results['avg_time_ms'] = None
                 results['avg_secondary_time_ms'] = None
                 results['avg_ratio'] = None

        else: # No test cases loaded initially
             results['error'] = "No test cases found to process."
             results['correctness'] = 0
             results['avg_time_ms'] = None
             results['avg_secondary_time_ms'] = None
             results['avg_ratio'] = None


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
