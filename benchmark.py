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
from collections import defaultdict
import time # Re-importing for clarity, used within functions
import random # Re-importing for clarity, used within functions
import llm_interface # Re-importing for clarity, used within functions
from typing import Callable, Optional, Any # For type hinting the callback

# Need a secure way to execute code, e.g., using restricted environments or sandboxing.
# This is a critical security consideration. A simple `exec` is DANGEROUS.

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
    base_prompt = f"Generate a Python function named `sort_algorithm` that implements an efficient sorting algorithm suitable for general use cases (handling various data distributions like random, sorted, reversed, duplicates, etc.). The function should take a list of numbers as input and return a new sorted list. Do not use the built-in sorted() function or .sort() method."

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

    try:
        # !!! SECURITY WARNING !!!
        # Executing arbitrary code from LLMs is highly risky.
        # `exec` is used here for simplicity but is NOT SAFE for production.
        # A proper sandboxing mechanism (e.g., Docker containers, restricted Python interpreters)
        # is essential to prevent malicious code execution.
        local_namespace = {}
        exec(generated_code, {}, local_namespace)

        if 'sort_algorithm' not in local_namespace or not callable(local_namespace['sort_algorithm']):
            results['error'] = "Generated code does not contain a callable function named 'sort_algorithm'."
            return results

        sort_func = local_namespace['sort_algorithm']

        # Calculate total cases for progress reporting
        total_overall_cases_calculated = sum(len(cases) for cases in categorized_test_cases.values())
        current_overall_case_num = 0

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

                # Prepare inputs for both sorts
                llm_input = list(test_case)
                baseline_input = list(test_case)
                expected_output = sorted(test_case) # Ground truth

                # --- Time LLM's sort_algorithm ---
                llm_start_time = time.perf_counter()
                actual_output = None
                llm_error = None
                current_llm_time = None
                is_correct = False

                try:
                    actual_output = sort_func(llm_input)
                    llm_end_time = time.perf_counter()
                    current_llm_time = llm_end_time - llm_start_time

                    if actual_output == expected_output:
                        is_correct = True
                        overall_correct_count += 1
                        cat_stats['correct_count'] += 1
                        # Only add time for correct runs to avoid skewing averages
                        overall_llm_time += current_llm_time
                        cat_stats['llm_time'] += current_llm_time
                        overall_llm_runs_timed += 1
                        cat_stats['llm_runs_timed'] += 1
                    else:
                        # Log incorrect sort
                        actual_repr = repr(actual_output[:20]) + '...' if isinstance(actual_output, list) and len(actual_output) > 20 else repr(actual_output)
                        expected_repr = repr(expected_output[:20]) + '...' if len(expected_output) > 20 else repr(expected_output)
                        test_repr = repr(test_case[:20]) + '...' if len(test_case) > 20 else repr(test_case)
                        print(f"    Incorrect sort: Input={test_repr}, Expected={expected_repr}, Got={actual_repr}")
                        progress_data['status'] = 'Incorrect'
                        progress_data['output_snippet'] = actual_repr
                    # Update progress after LLM run (if no exception)
                    if progress_callback:
                        progress_callback(progress_data)

                except Exception as e:
                    llm_error = e
                    test_repr = repr(test_case[:20]) + '...' if len(test_case) > 20 else repr(test_case)
                    print(f"    Error during LLM sort execution: Input={test_repr}, Error={e}")
                    # Do not count time if it errored
                    # Report error via callback
                    if progress_callback:
                        progress_data['status'] = 'Error'
                        progress_data['error'] = str(e)
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
                'avg_time_ms': cat_avg_llm_time,
                'baseline_avg_time_ms': cat_avg_baseline_time,
                'count': stats['case_count']
            }

    except SyntaxError as e:
        results['error'] = f"Syntax error in generated code: {e}"
        # Ensure performance_details is still present, even if empty
        results['performance_details'] = results.get('performance_details', {})
    except Exception as e:
        results['error'] = f"Error executing generated code: {e}"

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

        except FileNotFoundError:
            print(f"Test suite file '{args.suite_file}' not found. Generate it first using --generate-suite.")
        except Exception as e:
            print(f"An error occurred during example benchmark run: {e}")


    # Example if running another LLM test (assuming test_suite is loaded)
    # try:
    #     if 'test_suite' in locals():
    #          result_other = run_single_benchmark('some_other_llm', categorized_test_cases=test_suite)
    #          print("\nOther LLM Benchmark Result:\n", json.dumps(result_other, indent=2))
    # except NameError:
    #      print("Cannot run other LLM example without loaded test suite.")
    # except Exception as e:
    #      print(f"An error occurred during other LLM benchmark run: {e}")
