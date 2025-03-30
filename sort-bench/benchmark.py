"""
Module for benchmarking LLM-generated sorting algorithms.

Handles:
- Generating prompts for LLMs.
- Safely executing the generated code within Docker.
- Evaluating correctness and performance.
"""

import json
import os # For file path operations
import re # Needed for cleaning C code output
import random
from typing import Optional, List, Tuple # Adjusted imports

# Import functions from the test suite generator module
from . import test_suite_generator

# Constants related to Docker execution are removed as it's handled by the framework.
# BENCHMARKED_ALGORITHM_LABEL might still be useful if referenced elsewhere, keep for now.
BENCHMARKED_ALGORITHM_LABEL = "LLM/Baseline C Sort [Framework]" # Updated label


def generate_prompt_examples(num_examples: int = 3, max_size: int = 8, min_val: int = -10, max_val: int = 10) -> list[tuple[str, str]]:
    """Generates small input/output examples in C array format for the LLM prompt."""
    examples = []

    def format_c_array(data: list[int]) -> str:
        if not data:
            return "{}"
        return "{" + ", ".join(map(str, data)) + "}"

    # Ensure basic cases are covered
    if num_examples >= 1:
        examples.append(("// Input: size=0, arr={}", "// Output: arr={} (no change)")) # Empty array
    if num_examples >= 2:
        examples.append(("// Input: size=1, arr={5}", "// Output: arr={5} (no change)")) # Single element
    if num_examples >= 3:
        input_list = [3, 1, 4, 1, 5, 9]
        output_list = sorted(input_list)
        examples.append((f"// Input: size={len(input_list)}, arr={format_c_array(input_list)}",
                         f"// Output: arr={format_c_array(output_list)}")) # Basic unsorted with duplicate

    # Add more random examples if needed
    current_examples = len(examples)
    for _ in range(max(0, num_examples - current_examples)):
        size = random.randint(2, max_size)
        input_list = [random.randint(min_val, max_val) for _ in range(size)]
        output_list = sorted(input_list)
        examples.append((f"// Input: size={size}, arr={format_c_array(input_list)}",
                         f"// Output: arr={format_c_array(output_list)}"))

    return examples[:num_examples] # Return exactly num_examples


def create_sort_prompt(examples: Optional[list[tuple[str, str]]] = None) -> str:
    """
    Creates a prompt to ask an LLM for an efficient C sorting function,
    optionally including examples.
    """
    base_prompt = (
        "Generate a complete C function named `sort_array` that implements an efficient sorting algorithm "
        "for an array of integers. The function should be suitable for general use cases (handling various data distributions like random, sorted, reversed, duplicates, etc.).\n\n"
        "Function Signature:\n"
        "```c\n"
        "void sort_array(int* arr, size_t n);\n"
        "```\n\n"
        "Requirements:\n"
        "- The function MUST take a pointer to an integer array (`int* arr`) and the number of elements (`size_t n`) as input.\n"
        "- The function MUST sort the array **in-place**.\n"
        "- The function MUST be self-contained or include necessary helper functions within the generated code block.\n"
        "- Include necessary standard library headers (like `<stdlib.h>` for `size_t` or memory allocation if needed, `<stdio.h>` only for debugging if essential but remove before final output).\n"
        "- IMPORTANT: The function MUST NOT use the standard library `qsort` function.\n"
        "- IMPORTANT: The function MUST NOT print anything to standard output (stdout or stderr).\n"
        "- Focus on correctness and reasonable efficiency (e.g., O(n log n) average time complexity is preferred)."
    )

    if examples:
        example_str = "\n\nHere are some examples of how the function should modify the input array:\n"
        for input_comment, output_comment in examples:
            example_str += f"{input_comment}\n{output_comment}\n\n"
        # Add a final code block structure hint
        example_str += (
            "Please provide the complete C code for the `sort_array` function, including any necessary headers and helper functions, within a single C code block.\n"
            "```c\n"
            "// Include headers here\n\n"
            "// Helper functions (if any) here\n\n"
            "void sort_array(int* arr, size_t n) {\n"
            "    // Your implementation here\n"
            "}\n"
            "```"
        )
        return base_prompt + example_str.strip()
    else:
        return base_prompt + "\n\nPlease provide the complete C code for the `sort_array` function within a single C code block."

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


# --- Evaluation Logic & run_single_benchmark (REMOVED) ---
# These are handled by the framework runner and wrapper script.


# Note: The __main__ block below is for standalone testing/example usage of this module.
if __name__ == '__main__':
    print("Generating example C sort prompt:")
    # Generate examples using the updated function
    c_examples = generate_prompt_examples(num_examples=5)
    # Create the prompt using the updated function
    c_prompt = create_sort_prompt(examples=c_examples)
    print(c_prompt)

    print("\n---")
    print("To generate the test suite, run:")
    # Use the updated default filename from the generator module
    print(f"python sort-bench/test_suite_generator.py --generate-suite --suite-file sort-bench/{test_suite_generator.DEFAULT_TEST_SUITE_FILE}")
    print("\nTo run the full benchmark, use the web interface via main_app.py")
