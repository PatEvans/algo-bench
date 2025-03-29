# This script runs inside the Docker container.
# It loads the test suite from /sandbox/test_suite_data.json,
# imports the LLM's sort function from /sandbox/llm_sort.py,
# runs all test cases, times them, checks correctness,
# calculates baseline times using sorted(), aggregates results,
# and prints a single JSON object containing all results to stdout.

import sys
import json
import time
import traceback
import os
from collections import defaultdict
import importlib.util # Moved import here, needed earlier

# Constants
LLM_CODE_MODULE = "llm_sort"
LLM_CODE_FILE = f"/sandbox/{LLM_CODE_MODULE}.py"
SORT_FUNCTION_NAME = "sort_algorithm"
TEST_SUITE_FILE = "/sandbox/test_suite_data.json"

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


def run_all_benchmarks():
    """Loads suite, runs tests, aggregates, and returns results dict."""
    # Initialize final results structure
    results = {
        'correctness': 0.0,
        'avg_time_ms': None,
        'baseline_avg_time_ms': None,
        'performance_details': {},
        'error': None # Will be populated on critical failure
    }

    # Aggregators for overall results
    overall_correct_count = 0
    overall_llm_time_sec = 0.0 # Use seconds for aggregation
    overall_baseline_time_sec = 0.0
    overall_total_cases = 0
    overall_llm_runs_timed = 0 # Count only successful, correct LLM runs

    # Aggregators for per-category results
    category_results = defaultdict(lambda: {
        'correct_count': 0,
        'llm_time_sec': 0.0,
        'baseline_time_sec': 0.0,
        'case_count': 0,
        'llm_runs_timed': 0,
        'errors': [] # Store errors per category if needed
    })

    sort_algorithm = None
    categorized_test_cases = None

    try:
        # --- 1. Load Test Suite ---
        if not os.path.exists(TEST_SUITE_FILE):
            raise FileNotFoundError(f"Test suite file not found: {TEST_SUITE_FILE}")
        with open(TEST_SUITE_FILE, 'r', encoding='utf-8') as f:
            categorized_test_cases = json.load(f)
        if not categorized_test_cases:
            raise ValueError("Test suite file is empty or invalid.")

        # Calculate total overall cases beforehand
        total_overall_cases = sum(len(cases) for cases in categorized_test_cases.values())
        if total_overall_cases == 0:
            raise ValueError("Test suite contains no test cases.")

        # --- 2. Load LLM Code ---
        if not os.path.exists(LLM_CODE_FILE):
            raise FileNotFoundError(f"LLM code file not found: {LLM_CODE_FILE}")
        try:
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(LLM_CODE_MODULE, LLM_CODE_FILE)
            if spec is None or spec.loader is None:
                 raise ImportError(f"Could not create module spec for {LLM_CODE_FILE}")
            llm_module = importlib.util.module_from_spec(spec)
            sys.modules[LLM_CODE_MODULE] = llm_module # Add to sys.modules for potential relative imports within the module
            spec.loader.exec_module(llm_module)
        except Exception as import_err:
            raise ImportError(f"Error importing '{LLM_CODE_MODULE}' from {LLM_CODE_FILE}: {type(import_err).__name__}: {import_err}\n{traceback.format_exc()}")

        if not hasattr(llm_module, SORT_FUNCTION_NAME) or not callable(getattr(llm_module, SORT_FUNCTION_NAME)):
            raise NameError(f"Function '{SORT_FUNCTION_NAME}' not found or not callable in {LLM_CODE_FILE}.")
        sort_algorithm = getattr(llm_module, SORT_FUNCTION_NAME)

        # --- 3. Iterate and Evaluate Test Cases ---
        total_categories = len(categorized_test_cases)
        current_category_num = 0
        current_overall_case_num = 0 # Initialize overall counter before the loop
        for category, test_cases_in_category in categorized_test_cases.items():
            current_category_num += 1
            cat_stats = category_results[category]
            num_cases_in_cat = len(test_cases_in_category) # Get total cases for this category
            # Send progress update for category start
            send_progress({
                'status': 'Running',
                'category': category,
                'category_num': current_category_num,
                'total_categories': total_categories,
                'category_total_cases': num_cases_in_cat,
                'message': f"Starting category '{category}'..."
            })
            # print(f"--- Starting Category {current_category_num}/{total_categories}: '{category}' ({num_cases_in_cat} cases) ---", file=sys.stderr) # Keep simple stderr log too

            # Use enumerate to get case number within category
            for i, test_case in enumerate(test_cases_in_category):
                case_num_in_cat = i + 1
                overall_total_cases += 1
                cat_stats['case_count'] += 1
                input_repr = repr(test_case[:15]) + ('...' if len(test_case) > 15 else '')
                # Send progress update for case start
                send_progress({
                    'status': 'Running',
                    'category': category,
                    'category_num': current_category_num,
                    'total_categories': total_categories,
                    'category_case_num': case_num_in_cat,
                    'category_total_cases': num_cases_in_cat,
                    'current_case': current_overall_case_num, # Use the incremented overall counter
                    'total_cases': total_overall_cases, # Add overall total
                    'input_snippet': input_repr,
                    'message': f"Running Case {current_overall_case_num}/{total_overall_cases} (Category: {category} {case_num_in_cat}/{num_cases_in_cat})..."
                })
                # print(f"  Case {case_num_in_cat}/{num_cases_in_cat} (Overall {overall_total_cases}): Input={input_repr}", file=sys.stderr, end='') # Keep simple stderr log too
                llm_time_sec = None
                baseline_time_sec = None
                is_correct = False
                case_error = None

                try:
                    # --- Baseline Timing ---
                    baseline_input = list(test_case) # Copy for safety
                    start_baseline = time.perf_counter()
                    expected_output = sorted(baseline_input)
                    end_baseline = time.perf_counter()
                    baseline_time_sec = end_baseline - start_baseline
                    overall_baseline_time_sec += baseline_time_sec
                    cat_stats['baseline_time_sec'] += baseline_time_sec

                    # --- LLM Execution and Timing ---
                    llm_input = list(test_case) # Copy for safety
                    start_llm = time.perf_counter()
                    actual_output = sort_algorithm(llm_input)
                    end_llm = time.perf_counter()
                    llm_time_sec = end_llm - start_llm

                    # --- Correctness Check ---
                    if actual_output == expected_output:
                        is_correct = True
                        overall_correct_count += 1
                        cat_stats['correct_count'] += 1
                        # Only aggregate time for correct runs
                        overall_llm_time_sec += llm_time_sec
                        cat_stats['llm_time_sec'] += llm_time_sec
                        overall_llm_runs_timed += 1
                        cat_stats['llm_runs_timed'] += 1
                        # print(" -> Correct", file=sys.stderr) # Keep simple stderr log too
                        # Send progress update for case completion (Correct)
                        send_progress({
                            'status': 'Correct',
                            'category': category,
                            'category_case_num': case_num_in_cat, # Keep category info if needed downstream
                            'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num,
                            'total_cases': total_overall_cases,
                            'llm_time_ms': llm_time_sec * 1000 if llm_time_sec is not None else None,
                            'baseline_time_ms': baseline_time_sec * 1000 if baseline_time_sec is not None else None,
                            'output_snippet': repr(actual_output[:15]) + ('...' if isinstance(actual_output, list) and len(actual_output) > 15 else ''),
                            'message': f"Case {current_overall_case_num}/{total_overall_cases} Correct."
                        })
                    else:
                        actual_repr = repr(actual_output[:15]) + ('...' if isinstance(actual_output, list) and len(actual_output) > 15 else '')
                        expected_repr = repr(expected_output[:15]) + ('...' if len(expected_output) > 15 else '')
                        # print(f" -> INCORRECT (Expected: {expected_repr}, Got: {actual_repr})", file=sys.stderr) # Keep simple stderr log too
                        # Send progress update for case completion (Incorrect)
                        send_progress({
                            'status': 'Incorrect',
                            'category': category,
                            'category_case_num': case_num_in_cat,
                            'category_total_cases': num_cases_in_cat,
                            'current_case': current_overall_case_num,
                            'total_cases': total_overall_cases,
                            'llm_time_ms': llm_time_sec * 1000 if llm_time_sec is not None else None,
                            'baseline_time_ms': baseline_time_sec * 1000 if baseline_time_sec is not None else None,
                            'output_snippet': actual_repr,
                            'expected_snippet': expected_repr,
                            'message': f"Case {current_overall_case_num}/{total_overall_cases} Incorrect."
                        })

                except Exception as exec_err:
                    # Error during LLM execution or baseline sort for this case
                    # print(f" -> ERROR", file=sys.stderr) # Keep simple stderr log too
                    case_error = f"{type(exec_err).__name__}: {exec_err}\n{traceback.format_exc()}"
                    input_snippet_for_error = repr(test_case[:20]) + ('...' if len(test_case) > 20 else '')
                    # Send progress update for case completion (Error)
                    send_progress({
                        'status': 'Error',
                        'category': category,
                        'category_case_num': case_num_in_cat,
                        'category_total_cases': num_cases_in_cat,
                        'current_case': current_overall_case_num,
                        'total_cases': total_overall_cases,
                        'input_snippet': input_snippet_for_error,
                        'error': case_error,
                        'message': f"Case {current_overall_case_num}/{total_overall_cases} Error."
                    })
                    cat_stats['errors'].append({
                        'input_snippet': input_snippet_for_error,
                        'error': case_error
                    })
                    # Optional: Log error details if needed for debugging wrapper
                    # print(f"DEBUG WRAPPER: Error in category '{category}'. Input: {test_case[:10]}... Error: {case_error}", file=sys.stderr)


        # --- 4. Calculate Final Aggregated Results ---
        if overall_total_cases > 0:
            results['correctness'] = (overall_correct_count / overall_total_cases) * 100
            results['baseline_avg_time_ms'] = (overall_baseline_time_sec / overall_total_cases) * 1000
        if overall_llm_runs_timed > 0:
            results['avg_time_ms'] = (overall_llm_time_sec / overall_llm_runs_timed) * 1000

        for category, stats in category_results.items():
            cat_correctness = 0.0
            if stats['case_count'] > 0:
                cat_correctness = (stats['correct_count'] / stats['case_count']) * 100

            cat_avg_llm_time_ms = None
            if stats['llm_runs_timed'] > 0:
                cat_avg_llm_time_ms = (stats['llm_time_sec'] / stats['llm_runs_timed']) * 1000

            cat_avg_baseline_time_ms = None
            if stats['case_count'] > 0:
                cat_avg_baseline_time_ms = (stats['baseline_time_sec'] / stats['case_count']) * 1000

            results['performance_details'][category] = {
                'correctness': cat_correctness,
                'avg_time_ms': cat_avg_llm_time_ms,
                'baseline_avg_time_ms': cat_avg_baseline_time_ms,
                'count': stats['case_count'],
                'error_count': len(stats['errors']) # Report count of errors in this category
                # Optionally include 'errors': stats['errors'] if detailed errors are needed by host
            }

    except Exception as top_level_err:
        # Catch critical errors (loading suite, importing code)
        results['error'] = f"Critical error in wrapper: {type(top_level_err).__name__}: {top_level_err}\n{traceback.format_exc()}"
        # Ensure performance_details exists even on error
        results['performance_details'] = results.get('performance_details', {})


    # --- 5. Return the final results dictionary ---
    return results

# --- Main execution block ---
if __name__ == "__main__":
    # Moved importlib.util to top as it's needed by run_all_benchmarks

    final_results = run_all_benchmarks()

    # --- Print the final aggregated results as JSON to stdout ---
    # Add a type field to distinguish the final result message
    final_message = {"type": "result", "data": final_results}
    print("DEBUG WRAPPER: Preparing to print final JSON to stdout.", file=sys.stderr, flush=True)
    print("---WRAPPER_STDOUT_MARKER_BEFORE---", flush=True) # MARKER ADDED to stdout
    try:
        final_json_string = json.dumps(final_message) # Serialize once
        print(f"DEBUG WRAPPER: Final JSON string length: {len(final_json_string)}", file=sys.stderr, flush=True)
        print(f"DEBUG WRAPPER: Final JSON to print (first 500 chars): {final_json_string[:500]}", file=sys.stderr, flush=True)
        print(final_json_string, flush=True) # Print final result to stdout and flush
        print("---WRAPPER_STDOUT_MARKER_AFTER---", flush=True) # MARKER ADDED to stdout
        print("DEBUG WRAPPER: Successfully printed final JSON to stdout.", file=sys.stderr, flush=True)
    except TypeError as json_err:
        print("DEBUG WRAPPER: Caught TypeError during final JSON serialization/print.", file=sys.stderr, flush=True)
        # Fallback if results contain non-serializable data
        fallback_error = f"FATAL: Could not serialize final results dictionary: {json_err}. Original error: {final_results.get('error')}"
        final_message = {"type": "result", "data": {'error': fallback_error, 'correctness': 0.0, 'avg_time_ms': None, 'baseline_avg_time_ms': None, 'performance_details': {}}}
        print(f"DEBUG WRAPPER: Printing fallback JSON due to TypeError.", file=sys.stderr, flush=True) # DEBUG ADDED
        print(json.dumps(final_message), flush=True) # Print fallback final result to stdout and flush
        print("DEBUG WRAPPER: Successfully printed fallback JSON to stdout (TypeError).", file=sys.stderr, flush=True) # DEBUG ADDED
    except Exception as final_print_err:
         print(f"DEBUG WRAPPER: Caught Exception ({type(final_print_err).__name__}) during final JSON serialization/print.", file=sys.stderr, flush=True) # DEBUG ADDED
         # Ultimate fallback for any other printing error
         final_message = {"type": "result", "data": {'error': f"FATAL: Error printing final JSON: {final_print_err}", 'correctness': 0.0, 'avg_time_ms': None, 'baseline_avg_time_ms': None, 'performance_details': {}}}
         print(f"DEBUG WRAPPER: Printing fallback JSON due to other Exception.", file=sys.stderr, flush=True) # DEBUG ADDED
         print(json.dumps(final_message), flush=True) # Print ultimate fallback final result to stdout and flush
         print("DEBUG WRAPPER: Successfully printed fallback JSON to stdout (Exception).", file=sys.stderr, flush=True) # DEBUG ADDED

    exit_code_to_use = 1 if final_results.get('error') else 0
    print(f"DEBUG WRAPPER: Exiting with code {exit_code_to_use}.", file=sys.stderr, flush=True) # DEBUG ADDED
    # Exit explicitly - 0 if no critical error in the final_results dict, 1 otherwise
    sys.exit(exit_code_to_use)
