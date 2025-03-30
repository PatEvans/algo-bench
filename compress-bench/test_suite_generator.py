"""
Module for generating and managing benchmark test suites for compression algorithms.
"""

import json
import random
import string
import base64 # To encode binary data for JSON
import os
from collections import defaultdict

# Default filename, can be overridden
DEFAULT_TEST_SUITE_FILE = "compression_test_suite.json"

# Define typical data sizes (adjust as needed)
DEFAULT_SIZES = {
    'tiny': 10,         # Very small strings
    'small': 1024,      # 1 KB
    'medium': 100 * 1024, # 100 KB
    'large': 1 * 1024 * 1024, # 1 MB
    # 'xlarge': 10 * 1024 * 1024 # 10 MB (Optional, can take time)
}

# --- Data Generation Functions ---

def generate_random_bytes(size: int) -> bytes:
    """Generates a sequence of random bytes."""
    return os.urandom(size)

def generate_text_data(size: int, alphabet=string.ascii_letters + string.digits + string.punctuation + ' \n\t') -> bytes:
    """Generates random text data using a specified alphabet."""
    # Ensure size is at least 1 if alphabet is not empty
    if size == 0: return b""
    if not alphabet: return b""
    text = ''.join(random.choices(alphabet, k=size))
    return text.encode('utf-8', errors='ignore') # Encode to bytes

def generate_highly_compressible_data(size: int, chunk_size: int = 10) -> bytes:
    """Generates data with repeating patterns."""
    if size == 0: return b""
    chunk_size = max(1, chunk_size)
    pattern = os.urandom(chunk_size)
    repetitions = (size + chunk_size - 1) // chunk_size
    data = (pattern * repetitions)[:size]
    return data

def generate_json_like_data(size: int) -> bytes:
    """Generates data resembling JSON structure (highly simplified)."""
    if size == 0: return b""
    data = {}
    current_size = 2 # Account for {}
    key_num = 1
    while current_size < size:
        key = f"key_{key_num}"
        # Generate random value (string, number, bool)
        val_type = random.choice(['str', 'num', 'bool'])
        if val_type == 'str':
            # Estimate remaining size needed, generate string
            remaining = size - current_size - len(key) - 6 # Account for quotes, colon, comma, space
            str_len = max(1, random.randint(1, max(1, remaining // 2))) # Avoid overly long strings
            value = ''.join(random.choices(string.ascii_letters + string.digits, k=str_len))
            data[key] = value
            current_size += len(key) + len(value) + 6
        elif val_type == 'num':
            value = random.randint(-10000, 10000)
            data[key] = value
            current_size += len(key) + len(str(value)) + 4
        else: # bool
            value = random.choice([True, False])
            data[key] = value
            current_size += len(key) + len(str(value).lower()) + 4
        key_num += 1
        if current_size >= size - 10: # Stop if close to target size
            break

    # Convert dict to JSON string, then encode to bytes
    json_string = json.dumps(data)
    # Truncate or pad if necessary (crude way to hit target size)
    json_bytes = json_string.encode('utf-8')
    if len(json_bytes) > size:
        json_bytes = json_bytes[:size]
    elif len(json_bytes) < size:
        padding = b' ' * (size - len(json_bytes))
        json_bytes += padding
    return json_bytes


# --- Main Test Case Generation ---

def generate_test_cases(sizes: dict = None, num_cases_per_type: int = 3) -> dict[str, list[str]]:
    """
    Generates diverse byte test cases for compression and encodes them in base64.

    Args:
        sizes: Dictionary mapping size names (e.g., 'small') to byte counts.
               Defaults to DEFAULT_SIZES.
        num_cases_per_type: Number of random/text cases per size.

    Returns:
        A dictionary where keys are category names (e.g., "random_bytes_small", "text_large")
        and values are lists containing base64 encoded strings of the test case bytes.
    """
    if sizes is None:
        sizes = DEFAULT_SIZES

    cases_by_category_b64 = defaultdict(list)

    # --- Special Cases ---
    print("Generating special cases...")
    # Empty data
    cases_by_category_b64['special_empty'] = [base64.b64encode(b"").decode('ascii')]
    # Single byte
    cases_by_category_b64['special_single_byte'] = [base64.b64encode(b"A").decode('ascii')]
    # Small repeating pattern
    cases_by_category_b64['special_repeating_small'] = [base64.b64encode(b"ABABABABAB").decode('ascii')]
    # Simple text
    cases_by_category_b64['special_simple_text'] = [base64.b64encode(b"Hello world! This is a test.").decode('ascii')]
    total_cases = len(cases_by_category_b64)
    print(f"  - Added {total_cases} special cases.")

    # --- Size-Based Cases ---
    for name, size in sizes.items():
        if size == 0: continue
        print(f"Generating cases for size: {name} ({size} bytes)...")

        # 1. Random Bytes (less compressible)
        cat_random = f"random_bytes_{name}"
        for i in range(num_cases_per_type):
            data_bytes = generate_random_bytes(size)
            data_b64 = base64.b64encode(data_bytes).decode('ascii')
            cases_by_category_b64[cat_random].append(data_b64)
            print(f"  - Added {cat_random} case {i+1}")
            total_cases += 1

        # 2. Random Text (more compressible than random bytes)
        cat_text = f"random_text_{name}"
        for i in range(num_cases_per_type):
            data_bytes = generate_text_data(size)
            data_b64 = base64.b64encode(data_bytes).decode('ascii')
            cases_by_category_b64[cat_text].append(data_b64)
            print(f"  - Added {cat_text} case {i+1}")
            total_cases += 1

        # 3. Highly Compressible (Repeating patterns)
        cat_compressible = f"highly_compressible_{name}"
        # Generate one case for this type per size
        data_bytes = generate_highly_compressible_data(size, chunk_size=max(1, size // 100)) # Vary pattern size slightly
        data_b64 = base64.b64encode(data_bytes).decode('ascii')
        cases_by_category_b64[cat_compressible].append(data_b64)
        print(f"  - Added {cat_compressible} case")
        total_cases += 1

        # 4. JSON-like Data (Structured text)
        cat_json = f"json_like_{name}"
        # Generate one case for this type per size
        data_bytes = generate_json_like_data(size)
        data_b64 = base64.b64encode(data_bytes).decode('ascii')
        cases_by_category_b64[cat_json].append(data_b64)
        print(f"  - Added {cat_json} case")
        total_cases += 1

    # --- Add specific file content (Corrected to sonnets.txt) ---
    sonnets_file_path = "sonnets.txt" # Assume file is in the root directory
    sonnets_category = "literature_sonnets"
    try:
        if os.path.exists(sonnets_file_path):
            with open(sonnets_file_path, 'rb') as f: # Read as bytes
                sonnets_content_bytes = f.read()
            if sonnets_content_bytes: # Only add if file is not empty
                sonnets_content_b64 = base64.b64encode(sonnets_content_bytes).decode('ascii')
                cases_by_category_b64[sonnets_category] = [sonnets_content_b64]
                print(f"  - Added test case from '{sonnets_file_path}' under category '{sonnets_category}'.")
                total_cases += 1 # Increment total case count
            else:
                print(f"Warning: File '{sonnets_file_path}' is empty. Skipping.")
        else:
            print(f"Warning: File '{sonnets_file_path}' not found. Skipping specific file test case.")
    except Exception as e:
        print(f"Warning: Error reading or processing '{sonnets_file_path}': {e}. Skipping specific file test case.")

    # --- Add second specific file content ---
    # sonnets_file_path = "sonnets.txt" # Assume file is in the root directory
    # sonnets_category = "literature_sonnets"
    # try:
    #     if os.path.exists(sonnets_file_path):
    #         with open(sonnets_file_path, 'rb') as f: # Read as bytes
    #             sonnets_content_bytes = f.read()
    #         if sonnets_content_bytes: # Only add if file is not empty
    #             sonnets_content_b64 = base64.b64encode(sonnets_content_bytes).decode('ascii')
    #             cases_by_category_b64[sonnets_category] = [sonnets_content_b64]
    #             print(f"  - Added test case from '{sonnets_file_path}' under category '{sonnets_category}'.")
    #             total_cases += 1 # Increment total case count
    #         else:
    #             print(f"Warning: File '{sonnets_file_path}' is empty. Skipping.")
    #     else:
    #         print(f"Warning: File '{sonnets_file_path}' not found. Skipping specific file test case.")
    # except Exception as e:
    #     print(f"Warning: Error reading or processing '{sonnets_file_path}': {e}. Skipping specific file test case.")


    print(f"Generated a total of {total_cases} test cases (base64 encoded) across {len(cases_by_category_b64)} categories.")
    return dict(cases_by_category_b64) # Convert back to regular dict


# --- Test Suite Loading/Saving Functions ---

def load_test_suite(filename: str) -> dict:
    """Loads the base64 encoded test suite from a JSON file."""
    print(f"Loading compression test suite from {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            test_suite_b64 = json.load(f)
        print(f"Successfully loaded base64 encoded test suite from {filename}")
        # Note: Data remains base64 encoded here. Decoding happens in the benchmark runner.
        return test_suite_b64
    except FileNotFoundError:
        print(f"Error: Test suite file '{filename}' not found.")
        raise # Re-raise to be handled by caller
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        raise
    except Exception as e:
        print(f"Error loading test suite from {filename}: {e}")
        raise # Re-raise to be handled by caller

def generate_and_save_test_suite(filename: str, sizes: dict = None, num_cases_per_type: int = 3):
    """Generates byte test cases, base64 encodes them, and saves to a JSON file."""
    print(f"Generating compression test suite (sizes: {sizes or DEFAULT_SIZES}, cases/type: {num_cases_per_type}) and saving to {filename}")
    try:
        # Pass generation parameters correctly
        test_cases_b64 = generate_test_cases(sizes=sizes, num_cases_per_type=num_cases_per_type)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(test_cases_b64, f, indent=2) # Use indent for readability
        print(f"Successfully generated and saved base64 encoded test suite to {filename}")
    except Exception as e:
        print(f"Error generating or saving test suite to {filename}: {e}")
        raise # Re-raise to be handled by caller


if __name__ == '__main__':
    import argparse # Keep import here

    parser = argparse.ArgumentParser(description="Compression Test Suite Generation Utility")
    parser.add_argument('--generate-suite', action='store_true', help=f"Generate and save a new compression test suite.")
    parser.add_argument('--suite-file', default=DEFAULT_TEST_SUITE_FILE, help="Specify the test suite JSON file path")
    # Add arguments to control generation parameters
    parser.add_argument('--num-cases', type=int, default=3, help="Number of random/text cases per type/size for suite generation")
    # Allow specifying sizes via command line (simple example, could be more complex)
    parser.add_argument('--size-tiny', type=int, default=DEFAULT_SIZES['tiny'])
    parser.add_argument('--size-small', type=int, default=DEFAULT_SIZES['small'])
    parser.add_argument('--size-medium', type=int, default=DEFAULT_SIZES['medium'])
    parser.add_argument('--size-large', type=int, default=DEFAULT_SIZES['large'])
    # parser.add_argument('--size-xlarge', type=int, default=DEFAULT_SIZES.get('xlarge')) # Optional

    args = parser.parse_args()

    if args.generate_suite:
        # Collect sizes from args
        gen_sizes = {
            'tiny': args.size_tiny,
            'small': args.size_small,
            'medium': args.size_medium,
            'large': args.size_large,
        }
        # if args.size_xlarge is not None: gen_sizes['xlarge'] = args.size_xlarge

        generate_and_save_test_suite(
            filename=args.suite_file,
            sizes=gen_sizes,
            num_cases_per_type=args.num_cases
        )
    else:
        print("Use --generate-suite to create a new compression test suite file.")
        print(f"Example: python {__file__} --generate-suite --suite-file {args.suite_file}")
