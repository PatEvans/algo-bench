# This script runs inside the Docker container for COMPRESSION benchmarks.
# It loads the base64 encoded test suite from /sandbox/compression_test_suite.json,
# decodes the test data to bytes,
# imports the LLM's compress and decompress functions from /sandbox/llm_compress.py,
# runs all test cases:
#   - Times compress()
#   - Times decompress()
#   - Checks correctness (decompressed == original)
#   - Calculates compression ratio
# Aggregates results (avg times, avg ratio, overall correctness),
# and prints a single JSON object containing all results to stdout.

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

# Constants - Updated for C Compression
LLM_CODE_SOURCE_FILE = "/sandbox/llm_compress.c" # C source file path
# Determine shared library extension based on platform (inside container, likely .so)
lib_ext = ".so" if platform.system() == "Linux" else ".dylib" if platform.system() == "Darwin" else ".dll"
LLM_SHARED_LIB_FILE = f"/sandbox/llm_compress{lib_ext}" # Compiled shared library path
COMPRESS_FUNCTION_NAME = "compress"
DECOMPRESS_FUNCTION_NAME = "decompress"
FREE_BUFFER_FUNCTION_NAME = "free_buffer" # Function to free C-allocated memory
# Use the default from the generator module (ensure consistency)
# Assuming test_suite_generator.DEFAULT_TEST_SUITE_FILE is accessible or hardcoded here
# Let's hardcode it for simplicity within the container script:
TEST_SUITE_FILE = "/sandbox/compression_test_suite.json" # Changed test suite filename

# Helper function to send progress updates as JSON to stderr
def send_progress(data: dict):
    """Formats data as JSON and prints to stderr."""
    try:
        # Add a type field to distinguish progress messages easily
        progress_message = {"type": "progress", "data": data}
        print(json.dumps(progress_message), file=sys.stderr, flush=True)
    except Exception as e:
        # Fallback if JSON serialization fails for progress
        print(json.dumps({"type": "progress", "data": {"error": f"Failed to serialize progress: {e}"}}), file=sys.stderr, flush=True)


def run_all_compression_benchmarks():
    """Loads suite, runs compression/decompression tests, aggregates, and returns results dict."""
    # Initialize final results structure for compression
    results = {
        'correctness': 0, # Default to failure (0)
        'avg_compression_time_ms': None,
        'avg_decompression_time_ms': None,
        'avg_compression_ratio': None,
        'error': None # For critical errors preventing full run
    }

    # Aggregators for overall results
    total_compression_time_sec = 0.0
    total_decompression_time_sec = 0.0
    total_compression_ratio = 0.0
    total_cases_processed = 0 # Count cases successfully processed (compressed & decompressed)
    total_cases_correct = 0 # Count cases where decompression was successful and matched original
    any_case_failed = False # Flag if any case fails correctness check or errors during execution

    # Per-category results are not stored in the final dict for now, but could be added
    # category_results = defaultdict(lambda: {...}) # Example structure if needed later

    # Ctypes function pointers
    c_compress_func = None
    c_decompress_func = None
    c_free_buffer_func = None
    llm_lib = None # Handle for the loaded shared library

    categorized_test_cases_bytes = None # Will hold the decoded byte data

    # Define the Buffer struct in ctypes
    class CBuffer(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)),
                    ("size", ctypes.c_size_t)]

    try:
        # --- 1. Load Test Suite (Base64 Encoded JSON) ---
        send_progress({'status': 'Setup', 'message': f"Loading test suite: {TEST_SUITE_FILE}"})
        if not os.path.exists(TEST_SUITE_FILE):
            raise FileNotFoundError(f"Test suite file not found: {TEST_SUITE_FILE}")
        with open(TEST_SUITE_FILE, 'r', encoding='utf-8') as f:
            test_suite_b64 = json.load(f)
        if not test_suite_b64:
            raise ValueError("Test suite file is empty or invalid.")
        send_progress({'status': 'Setup', 'message': "Test suite loaded."})

        # --- 2. Decode Base64 Test Cases ---
        send_progress({'status': 'Setup', 'message': "Decoding test cases..."})
        categorized_test_cases_bytes = defaultdict(list)
        total_overall_cases = 0
        for category, cases_b64 in test_suite_b64.items():
            for i, case_b64 in enumerate(cases_b64):
                try:
                    # Ensure the base64 string is correctly padded if needed
                    # Python's b64decode usually handles padding, but explicit check can help debug
                    missing_padding = len(case_b64) % 4
                    if missing_padding:
                        case_b64 += '=' * (4 - missing_padding)
                    decoded_bytes = base64.b64decode(case_b64, validate=True) # Add validate=True
                    categorized_test_cases_bytes[category].append(decoded_bytes)
                    total_overall_cases += 1
                except (base64.binascii.Error, ValueError, TypeError) as decode_err:
                    # Log warning but continue if possible, mark run as failed later
                    any_case_failed = True # Mark failure due to decode error
                    err_msg = f"Failed to decode base64 case {i+1} in category '{category}': {decode_err}. Skipping case."
                    print(f"Warning: {err_msg}", file=sys.stderr)
                    send_progress({'status': 'Error', 'category': category, 'message': err_msg}) # Send error progress
        categorized_test_cases_bytes = dict(categorized_test_cases_bytes) # Convert back to dict

        if total_overall_cases == 0:
            raise ValueError("Test suite contains no valid test cases after decoding.")
        send_progress({'status': 'Setup', 'message': f"Decoded {total_overall_cases} test cases."})

        # --- 3. Compile LLM C Code ---
        send_progress({'status': 'Setup', 'message': f"Compiling LLM C code: {LLM_CODE_SOURCE_FILE} -> {LLM_SHARED_LIB_FILE}"})
        if not os.path.exists(LLM_CODE_SOURCE_FILE):
            raise FileNotFoundError(f"LLM C source code file not found: {LLM_CODE_SOURCE_FILE}")

        # Compile the C code into a shared library
        # -shared: Create a shared library
        # -fPIC: Generate position-independent code (required for shared libraries)
        # -O2: Optimization level (can adjust, e.g., -O3, -Os)
        # -o: Output file name
        # -std=c11: Specify C standard
        compile_command = [
            "gcc", "-shared", "-fPIC", "-O2", "-std=c11",
            LLM_CODE_SOURCE_FILE, "-o", LLM_SHARED_LIB_FILE
        ]
        try:
            # Use check=True to raise CalledProcessError on failure
            compile_process = subprocess.run(compile_command, check=True, capture_output=True, text=True, timeout=60) # Added timeout
            print(f"GCC stdout:\n{compile_process.stdout}", file=sys.stderr) # Log stdout to stderr
            print(f"GCC stderr:\n{compile_process.stderr}", file=sys.stderr) # Log stderr (warnings etc.) to stderr
            send_progress({'status': 'Setup', 'message': "C code compiled successfully."})
        except FileNotFoundError:
             raise RuntimeError("`gcc` compiler not found in container PATH. Ensure it's installed in the Dockerfile.")
        except subprocess.CalledProcessError as compile_err:
            # Compilation failed, include compiler output in the error
            error_message = f"C code compilation failed (Exit code {compile_err.returncode}).\n" \
                            f"Command: {' '.join(compile_command)}\n" \
                            f"Stderr:\n{compile_err.stderr}\n" \
                            f"Stdout:\n{compile_err.stdout}"
            raise RuntimeError(error_message)
        except subprocess.TimeoutExpired as timeout_err:
             raise RuntimeError(f"C code compilation timed out after {timeout_err.timeout} seconds. Command: {' '.join(compile_command)}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during C compilation: {e}")


        # --- 4. Load Shared Library and Define Function Signatures ---
        send_progress({'status': 'Setup', 'message': f"Loading shared library: {LLM_SHARED_LIB_FILE}"})
        try:
            llm_lib = ctypes.CDLL(LLM_SHARED_LIB_FILE)
        except OSError as load_err:
            raise OSError(f"Failed to load shared library {LLM_SHARED_LIB_FILE}: {load_err}")

        try:
            # Define compress function signature
            c_compress_func = llm_lib[COMPRESS_FUNCTION_NAME]
            c_compress_func.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
            c_compress_func.restype = CBuffer

            # Define decompress function signature
            c_decompress_func = llm_lib[DECOMPRESS_FUNCTION_NAME]
            c_decompress_func.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
            c_decompress_func.restype = CBuffer

            # Define free_buffer function signature
            c_free_buffer_func = llm_lib[FREE_BUFFER_FUNCTION_NAME]
            c_free_buffer_func.argtypes = [CBuffer]
            c_free_buffer_func.restype = None # void return type
        except AttributeError as func_err:
             raise NameError(f"Function not found in shared library {LLM_SHARED_LIB_FILE}: {func_err}. Ensure C code defines {COMPRESS_FUNCTION_NAME}, {DECOMPRESS_FUNCTION_NAME}, and {FREE_BUFFER_FUNCTION_NAME}.")

        send_progress({'status': 'Setup', 'message': "C functions loaded and signatures defined."})


        # --- 5. Iterate and Evaluate Test Cases ---
        total_categories = len(categorized_test_cases_bytes)
        current_category_num = 0
        current_overall_case_num = 0
        send_progress({'status': 'Running', 'message': "Starting test case evaluation..."})

        for category, test_cases_in_category in categorized_test_cases_bytes.items():
            current_category_num += 1
            num_cases_in_cat = len(test_cases_in_category)
            send_progress({
                'status': 'Running', 'category': category,
                'category_num': current_category_num, 'total_categories': total_categories,
                'category_total_cases': num_cases_in_cat,
                'message': f"Starting category '{category}'..."
            })

            for i, original_data_bytes in enumerate(test_cases_in_category):
                case_num_in_cat = i + 1
                current_overall_case_num += 1
                original_size = len(original_data_bytes)
                # input_repr = f"bytes[{original_size}]" # Represent input by size

                send_progress({
                    'status': 'Running', 'category': category,
                    'category_num': current_category_num, 'total_categories': total_categories,
                    'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                    'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                    'original_size': original_size,
                    'message': f"Running Case {current_overall_case_num}/{total_overall_cases}..."
                })

                compress_time_sec = None
                decompress_time_sec = None
                compressed_size = None
                compression_ratio = None
                is_correct = False
                case_error = None
                compressed_buffer = None # C Buffer struct for compressed data
                decompressed_buffer = None # C Buffer struct for decompressed data
                compressed_data_bytes = None # Python bytes version of compressed data
                decompressed_data_bytes = None # Python bytes version of decompressed data

                try:
                    # --- Prepare Input Buffer for C ---
                    # Create a ctypes buffer from the Python bytes object
                    # Handle empty input case carefully
                    c_input_buffer = None
                    if original_size > 0:
                        c_input_buffer = (ctypes.c_ubyte * original_size).from_buffer_copy(original_data_bytes)
                    else:
                        # Pass NULL pointer if size is 0, or a valid pointer to 0 bytes if C function expects that
                        # Assuming NULL pointer for 0 size is acceptable based on prompt description
                        c_input_buffer = None # Representing NULL pointer

                    # --- Compression Step (Call C function) ---
                    start_compress = time.perf_counter()
                    compressed_buffer = c_compress_func(c_input_buffer, original_size)
                    end_compress = time.perf_counter()
                    compress_time_sec = end_compress - start_compress

                    # Check if C function indicated failure (NULL data pointer)
                    if not compressed_buffer.data and compressed_buffer.size > 0: # Check size too, NULL with size 0 might be valid for empty output
                         raise RuntimeError(f"{COMPRESS_FUNCTION_NAME} returned NULL data pointer with non-zero size ({compressed_buffer.size}). Compression failed.")
                    # Handle case where NULL/0 is returned for valid empty output
                    if not compressed_buffer.data and compressed_buffer.size == 0:
                         compressed_size = 0
                         compressed_data_bytes = b"" # Empty bytes
                    else:
                        compressed_size = compressed_buffer.size
                        # Create Python bytes from the C buffer for ratio calculation and decompression input
                        # Important: This copies the data. The original C buffer must be freed later.
                        compressed_data_bytes = ctypes.string_at(compressed_buffer.data, compressed_size)


                    # Calculate compression ratio (handle division by zero) - Logic remains similar
                    if original_size > 0 and compressed_size > 0:
                        compression_ratio = original_size / compressed_size
                    elif original_size == 0 and compressed_size == 0:
                        compression_ratio = 1.0
                    elif original_size > 0 and compressed_size == 0:
                        compression_ratio = float('inf')
                    elif original_size == 0 and compressed_size > 0:
                        compression_ratio = 0.0
                    else:
                        compression_ratio = None

                    # --- Log individual case details ---
                    # Print to stderr for debugging, won't interfere with final JSON result on stdout
                    print(f"DEBUG CASE {current_overall_case_num}: "
                          f"original_size={original_size}, "
                          f"compressed_size={compressed_size}, "
                          f"ratio={compression_ratio}", file=sys.stderr, flush=True)

                    # --- Decompression Step (Call C function) ---
                    # Prepare input buffer for decompress (using the data from compressed_buffer)
                    # Pass pointer and size directly from the compressed_buffer struct
                    start_decompress = time.perf_counter()
                    decompressed_buffer = c_decompress_func(compressed_buffer.data, compressed_size)
                    end_decompress = time.perf_counter()
                    decompress_time_sec = end_decompress - start_decompress

                    # Check if C function indicated failure
                    if not decompressed_buffer.data and decompressed_buffer.size > 0:
                         raise RuntimeError(f"{DECOMPRESS_FUNCTION_NAME} returned NULL data pointer with non-zero size ({decompressed_buffer.size}). Decompression failed.")
                    # Handle case where NULL/0 is returned for valid empty output
                    if not decompressed_buffer.data and decompressed_buffer.size == 0:
                         decompressed_size = 0
                         decompressed_data_bytes = b""
                    else:
                        decompressed_size = decompressed_buffer.size
                        # Create Python bytes from the C buffer for correctness check
                        decompressed_data_bytes = ctypes.string_at(decompressed_buffer.data, decompressed_size)


                    # --- Correctness Check ---
                    if decompressed_data_bytes == original_data_bytes:
                        is_correct = True
                        total_cases_correct += 1
                        # Aggregate metrics only for fully successful cases
                        total_compression_time_sec += compress_time_sec
                        total_decompression_time_sec += decompress_time_sec
                        # Avoid division by zero or inf ratio in average calculation
                        if compression_ratio is not None and compression_ratio != float('inf'):
                             total_compression_ratio += compression_ratio
                        total_cases_processed += 1 # Increment only if compress/decompress/check succeeded

                        send_progress({
                            'status': 'Correct', 'category': category,
                            'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                            'compression_time_ms': compress_time_sec * 1000,
                            'decompression_time_ms': decompress_time_sec * 1000,
                            'original_size': original_size,
                            'compressed_size': compressed_size,
                            'compression_ratio': compression_ratio,
                            'message': f"Case {current_overall_case_num} Correct."
                        })
                    else:
                        any_case_failed = True # Mark failure
                        # Log details about the mismatch using sizes from C buffers/Python copies
                        mismatch_details = f"Size mismatch (orig={original_size}, decomp={decompressed_size})"
                        # Optionally compare first/last few bytes if sizes match but content differs
                        if original_size == decompressed_size:
                             # Avoid printing potentially huge byte strings
                             diff_preview = f"Orig starts: {repr(original_data_bytes[:20])}, Decomp starts: {repr(decompressed_data_bytes[:20])}"
                             mismatch_details = f"Content mismatch (size={original_size}). {diff_preview}"
                        case_error = f"Decompression mismatch: {mismatch_details}"
                        send_progress({
                            'status': 'Incorrect', 'category': category,
                            'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                            'compression_time_ms': compress_time_sec * 1000 if compress_time_sec is not None else None,
                            'decompression_time_ms': decompress_time_sec * 1000 if decompress_time_sec is not None else None,
                            'original_size': original_size,
                            'compressed_size': compressed_size,
                            'error': case_error, # Report the mismatch
                            'message': f"Case {current_overall_case_num} Incorrect."
                        })
                        # Do not increment total_cases_processed or add times/ratio if incorrect

                except Exception as exec_err:
                    any_case_failed = True # Mark failure
                    # Format error nicely, include traceback in stderr but maybe not in progress message
                    tb_str = traceback.format_exc()
                    print(f"Error during case {current_overall_case_num} ({category}):\n{tb_str}", file=sys.stderr)
                    case_error = f"{type(exec_err).__name__}: {exec_err}" # Shorter error for progress
                    send_progress({
                        'status': 'Error', 'category': category,
                        'category_case_num': case_num_in_cat, 'category_total_cases': num_cases_in_cat,
                        'current_case': current_overall_case_num, 'total_cases': total_overall_cases,
                        'original_size': original_size,
                        'error': case_error,
                        'message': f"Case {current_overall_case_num} Error during execution."
                    })
                    # Do not increment total_cases_processed or add times/ratio if error occurred

                finally:
                    # --- IMPORTANT: Free C memory ---
                    # Free the buffers returned by compress and decompress, regardless of success/failure
                    if compressed_buffer and compressed_buffer.data:
                        try:
                            c_free_buffer_func(compressed_buffer)
                        except Exception as free_err:
                             print(f"Warning: Error calling free_buffer for compressed data: {free_err}", file=sys.stderr)
                    if decompressed_buffer and decompressed_buffer.data:
                         try:
                            c_free_buffer_func(decompressed_buffer)
                         except Exception as free_err:
                             print(f"Warning: Error calling free_buffer for decompressed data: {free_err}", file=sys.stderr)


        # --- 6. Calculate Final Aggregated Results ---
        send_progress({'status': 'Aggregating', 'message': "Calculating final results..."})
        if total_overall_cases > 0:
            # Correctness is 1 only if ALL cases passed without error or mismatch
            results['correctness'] = 1 if not any_case_failed else 0

            if total_cases_processed > 0:
                # Calculate averages based on successfully processed cases
                results['avg_compression_time_ms'] = (total_compression_time_sec / total_cases_processed) * 1000
                results['avg_decompression_time_ms'] = (total_decompression_time_sec / total_cases_processed) * 1000
                # Ensure total_compression_ratio was accumulated correctly
                results['avg_compression_ratio'] = total_compression_ratio / total_cases_processed
            else:
                 # Handle case where no cases could be processed (e.g., all errored early or failed decode)
                 results['avg_compression_time_ms'] = None
                 results['avg_decompression_time_ms'] = None
                 results['avg_compression_ratio'] = None
                 if not results['error']: # If no critical error set yet
                      results['error'] = "No test cases were successfully processed (check for decode/execution errors)."
                 results['correctness'] = 0 # Ensure correctness is 0 if nothing processed

        else: # Should not happen if initial checks pass
             results['error'] = "No test cases found to process."
             results['correctness'] = 0


    except Exception as top_level_err:
        # Catch critical errors (loading suite, importing code, decoding issues)
        tb_str = traceback.format_exc()
        print(f"Critical error in wrapper script:\n{tb_str}", file=sys.stderr)
        results['error'] = f"Critical error in wrapper: {type(top_level_err).__name__}: {top_level_err}"
        results['correctness'] = 0 # Mark as incorrect due to critical failure
        # Send final error progress update if possible
        send_progress({'status': 'Error', 'message': results['error']})


    # --- 6. Return the final results dictionary ---
    send_progress({'status': 'Finished', 'message': "Benchmark evaluation complete."})
    return results

# --- Main execution block ---
if __name__ == "__main__":
    # Run the compression benchmark evaluation
    final_results = run_all_compression_benchmarks()

    # --- Print the final aggregated results as JSON to stdout ---
    # Add a type field to distinguish the final result message
    final_message = {"type": "result", "data": final_results}
    print("DEBUG WRAPPER: Preparing to print final compression JSON to stdout.", file=sys.stderr, flush=True)
    print("---WRAPPER_STDOUT_MARKER_BEFORE---", flush=True) # Marker for benchmark.py to find JSON start
    try:
        # Ensure floats are handled correctly, prevent NaN/Infinity issues if possible
        # json.dumps handles float('inf') as Infinity, -float('inf') as -Infinity, float('nan') as NaN by default
        # which is standard JSON behavior but might need handling on the receiving end if strict parsing is used.
        # Python's json module handles them; ensure the JS side does too or replace them here.
        # Example replacement (optional):
        # def handle_non_finite(obj):
        #     if isinstance(obj, float):
        #         if obj == float('inf') or obj == float('-inf') or obj != obj: # Check for inf, -inf, nan
        #             return None # Or str(obj)
        #     return obj
        # final_json_string = json.dumps(final_message, default=handle_non_finite)

        final_json_string = json.dumps(final_message)
        print(f"DEBUG WRAPPER: Final JSON string length: {len(final_json_string)}", file=sys.stderr, flush=True)
        # print(f"DEBUG WRAPPER: Final JSON to print (first 500 chars): {final_json_string[:500]}", file=sys.stderr, flush=True) # Keep commented unless debugging serialization
        print(final_json_string, flush=True)
        print("---WRAPPER_STDOUT_MARKER_AFTER---", flush=True) # Marker for benchmark.py to find JSON end
        print("DEBUG WRAPPER: Successfully printed final compression JSON to stdout.", file=sys.stderr, flush=True)
    except TypeError as json_err:
        print("DEBUG WRAPPER: Caught TypeError during final JSON serialization/print.", file=sys.stderr, flush=True)
        # Ensure the fallback structure matches the expected compression results format
        fallback_error = f"FATAL: Could not serialize final results dictionary: {json_err}. Original error: {final_results.get('error')}"
        final_message = {"type": "result", "data": {
            'error': fallback_error, 'correctness': 0,
            'avg_compression_time_ms': None, 'avg_decompression_time_ms': None,
            'avg_compression_ratio': None
        }}
        print(f"DEBUG WRAPPER: Printing fallback JSON due to TypeError.", file=sys.stderr, flush=True)
        print(json.dumps(final_message), flush=True) # Still need the markers for the fallback
        print("---WRAPPER_STDOUT_MARKER_AFTER---", flush=True)
        print("DEBUG WRAPPER: Successfully printed fallback JSON to stdout (TypeError).", file=sys.stderr, flush=True)
    except Exception as final_print_err:
         print(f"DEBUG WRAPPER: Caught Exception ({type(final_print_err).__name__}) during final JSON serialization/print.", file=sys.stderr, flush=True)
         # Ensure the fallback structure matches the expected compression results format
         final_message = {"type": "result", "data": {
             'error': f"FATAL: Error printing final JSON: {final_print_err}", 'correctness': 0,
             'avg_compression_time_ms': None, 'avg_decompression_time_ms': None,
             'avg_compression_ratio': None
         }}
         print(f"DEBUG WRAPPER: Printing fallback JSON due to other Exception.", file=sys.stderr, flush=True)
         print(json.dumps(final_message), flush=True) # Still need the markers for the fallback
         print("---WRAPPER_STDOUT_MARKER_AFTER---", flush=True)
         print("DEBUG WRAPPER: Successfully printed fallback JSON to stdout (Exception).", file=sys.stderr, flush=True)

    # Exit code: 0 if correctness is 1 (True) AND no critical error occurred, 1 otherwise.
    # Use .get() with default values to handle potential missing keys in final_results
    exit_code_to_use = 1
    if final_results.get('correctness') == 1 and not final_results.get('error'):
        exit_code_to_use = 0

    print(f"DEBUG WRAPPER: Exiting with code {exit_code_to_use}.", file=sys.stderr, flush=True)
    sys.exit(exit_code_to_use)
