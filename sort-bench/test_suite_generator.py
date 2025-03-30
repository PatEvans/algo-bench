"""
Module for generating and managing benchmark test suites.
"""

import json
import random
from collections import defaultdict

# Default filename, can be overridden
DEFAULT_TEST_SUITE_FILE = "c_sort_test_suite.json" # Updated default filename

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


# --- Test Suite Loading/Saving Functions ---

def load_test_suite(filename: str) -> dict:
    """Loads the test suite from a JSON file."""
    print(f"Loading test suite from {filename}...")
    try:
        with open(filename, 'r') as f:
            test_suite = json.load(f)
        print(f"Successfully loaded test suite from {filename}")
        return test_suite
    except FileNotFoundError:
        print(f"Error: Test suite file '{filename}' not found.")
        raise # Re-raise to be handled by caller
    except Exception as e:
        print(f"Error loading test suite from {filename}: {e}")
        raise # Re-raise to be handled by caller

def generate_and_save_test_suite(filename: str, **kwargs):
    """Generates test cases using generate_test_cases and saves them to a JSON file."""
    print(f"Generating test suite with params {kwargs} and saving to {filename}")
    try:
        test_cases = generate_test_cases(**kwargs)
        with open(filename, 'w') as f:
            json.dump(test_cases, f, indent=2)
        print(f"Successfully generated and saved test suite to {filename}")
    except Exception as e:
        print(f"Error generating or saving test suite to {filename}: {e}")
        raise # Re-raise to be handled by caller


if __name__ == '__main__':
    # Import argparse here as it's only needed for CLI execution
    import argparse

    parser = argparse.ArgumentParser(description="Test Suite Generation Utility")
    parser.add_argument('--generate-suite', action='store_true', help=f"Generate and save a new test suite.")
    parser.add_argument('--suite-file', default=DEFAULT_TEST_SUITE_FILE, help="Specify the test suite JSON file path")
    # Add arguments to control generation parameters
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
        print("Use --generate-suite to create a new test suite file.")
        # Update example command to use the new default filename
        print(f"Example: python sort-bench/test_suite_generator.py --generate-suite --suite-file sort-bench/{DEFAULT_TEST_SUITE_FILE}")
