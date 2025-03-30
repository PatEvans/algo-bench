"""
Module for benchmarking LLM-generated sorting algorithms.

Handles:
- Generating prompts for LLMs.
- Safely executing the generated code within Docker.
- Evaluating correctness and performance.
"""

import json
import os # For file path operations
import random
from typing import Optional, List, Tuple # Adjusted imports

# Import functions from the new test suite generator module
from . import test_suite_generator

# Constants related to Docker execution are removed as it's handled by the framework.
# BENCHMARKED_ALGORITHM_LABEL might still be useful if referenced elsewhere, keep for now.
BENCHMARKED_ALGORITHM_LABEL = "LLM/Baseline Sort [Framework]" # Updated label slightly


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

# Test suite generation/loading functions are now in test_suite_generator.py

# --- Evaluation Logic (REMOVED) ---
# The evaluate_algorithm function, which handled Docker execution,
# is removed. This logic is now handled by the framework's
# BenchmarkRunner and the generic docker_exec_wrapper.py script.


# --- run_single_benchmark (REMOVED/SIMPLIFIED) ---
# This function is also largely superseded by the framework's background task
# execution (`run_benchmark_background_base` called via `sort-bench/app.py`).
# If direct invocation is needed for testing, it would need to be adapted
# to use the framework's BenchmarkRunner. For now, we remove it to avoid confusion.
# def run_single_benchmark(...): -> dict:
#    ... (Removed) ...


# Note: The __main__ block below is for standalone testing/example usage of benchmark.py.
# Test suite generation is now handled by test_suite_generator.py.
if __name__ == '__main__':
    # Import argparse here as it's only needed for CLI execution
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Execution Examples")
    # Use the default from the generator module
    parser.add_argument('--suite-file', default=test_suite_generator.DEFAULT_TEST_SUITE_FILE, help="Specify the test suite JSON file path")

    args = parser.parse_args()

    # Example of running benchmarks using a loaded suite
    print("Running example benchmarks with loaded suite...")
    try:
        # Load the test suite using the function from the generator module
        test_suite = test_suite_generator.load_test_suite(args.suite_file)

        # Example usage - Load suite first, then run benchmarks
        print("\nRunning example benchmarks with loaded suite...")

        # --- Run example with fixed Merge Sort code ---
        # Note: Baseline run is removed from standalone example. Run it via the web UI.
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
            print(f"Test suite file '{args.suite_file}' not found. Generate it first using: python test_suite_generator.py --generate-suite")
    except Exception as e:
            print(f"An error occurred during example benchmark run: {e}")
