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

# Constants
LLM_CODE_MODULE = "llm_sort"
LLM_CODE_FILE = f"/sandbox/{LLM_CODE_MODULE}.py"
SORT_FUNCTION_NAME = "sort_algorithm"
TEST_SUITE_FILE = "/sandbox/test_suite_data.json"

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
        for category, test_cases_in_category in categorized_test_cases.items():
            cat_stats = category_results[category]

            for test_case in test_cases_in_category:
                overall_total_cases += 1
                cat_stats['case_count'] += 1
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
                        print(" -> Correct", file=sys.stderr) # Append outcome to the initial print
                    else:
                        # Log incorrectness details if needed for debugging wrapper
                        actual_repr = repr(actual_output[:15]) + ('...' if isinstance(actual_output, list) and len(actual_output) > 15 else '')
                        expected_repr = repr(expected_output[:15]) + ('...' if len(expected_output) > 15 else '')
                        print(f" -> INCORRECT (Expected: {expected_repr}, Got: {actual_repr})", file=sys.stderr)


                except Exception as exec_err:
                    # Error during LLM execution or baseline sort for this case
                    print(f" -> ERROR", file=sys.stderr) # Append outcome to the initial print
                    case_error = f"{type(exec_err).__name__}: {exec_err}\n{traceback.format_exc()}"
                    cat_stats['errors'].append({
                        'input_snippet': repr(test_case[:20]) + ('...' if len(test_case) > 20 else ''),
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
    # Need importlib for dynamic loading
    import importlib.util

    final_results = run_all_benchmarks()

    # --- Print the final aggregated results as JSON to stdout ---
    try:
        print(json.dumps(final_results))
    except TypeError as json_err:
        # Fallback if results contain non-serializable data (shouldn't happen with current structure)
        fallback_error = f"FATAL: Could not serialize final results dictionary: {json_err}. Original error: {final_results.get('error')}"
        print(json.dumps({'error': fallback_error, 'correctness': 0.0, 'avg_time_ms': None, 'baseline_avg_time_ms': None, 'performance_details': {}}))
    except Exception as final_print_err:
         # Ultimate fallback for any other printing error
         print(json.dumps({'error': f"FATAL: Error printing final JSON: {final_print_err}", 'correctness': 0.0, 'avg_time_ms': None, 'baseline_avg_time_ms': None, 'performance_details': {}}))

    # Exit explicitly - 0 if no critical error, 1 otherwise
    sys.exit(1 if final_results.get('error') else 0)
