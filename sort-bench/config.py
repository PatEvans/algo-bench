"""
Configuration specific to the Sort Benchmark.
"""
import os
import sys
import json
import ctypes # Keep ctypes if the framework runner expects C-like signatures, even for Python

# --- Project Setup ---
# Assumes this config file is in the sort-bench directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
framework_dir = os.path.join(project_root, 'framework')
# Ensure framework is importable if needed directly (though app_base handles it)
if framework_dir not in sys.path:
    sys.path.insert(0, framework_dir)


# --- Benchmark Identification ---
BENCHMARK_NAME = "Python List Sort"
# Use a benchmark-specific database file
DATABASE_FILE = "sort-bench/benchmark_results.db"
# Test suite file specific to this benchmark (relative to sort-bench dir)
TEST_SUITE_FILENAME = "test_suite.json" # Matches generator default
# LLM Code filename (used within Docker) - will contain the Python code
LLM_CODE_FILENAME = "llm_sort.py" # Matches docker_exec_wrapper.py expectation

# --- Docker/Execution Configuration (Leveraging Framework Runner) ---
# We'll use a generic Python image or build a simple one if needed.
DOCKER_IMAGE = "python:3.10-slim" # Example: Use a standard Python image
CONTAINER_MEM_LIMIT = "1g"
CONTAINER_CPU_SHARES = 512 # Relative CPU weight (default 1024)
EXEC_TIMEOUT_SECONDS = 600 # Timeout for the entire docker exec run

# Path to the framework's generic wrapper script
WRAPPER_SCRIPT_PATH = os.path.abspath(os.path.join(framework_dir, 'docker_exec_wrapper.py'))

# --- Baseline Configuration ---
PYTHON_SORTED_BENCHMARK_LABEL = "Python sorted() [Baseline]"
# The actual code snippet for the baseline benchmark
PYTHON_SORTED_CODE_SNIPPET = """
# Baseline implementation using Python's built-in sorted()
def sort_algorithm(data: list) -> list:
    # Return a new sorted list, as required by the benchmark structure
    return sorted(data)
"""
# Define baseline snippets dictionary
BASELINE_CODE_SNIPPETS = {
    PYTHON_SORTED_BENCHMARK_LABEL: PYTHON_SORTED_CODE_SNIPPET
}

# --- LLM Configuration ---
# Add real LLM identifiers here. The baseline is now just another option.
AVAILABLE_LLMS = [PYTHON_SORTED_BENCHMARK_LABEL, "Gemini 2.5 Pro Exp"] # Example LLM

# --- Function Names & Signatures (for Framework Runner/Wrapper Interaction) ---
# These names MUST match expectations in framework/docker_exec_wrapper.py
# and the generated Python code structure.
# Passed as FUNCTION_NAMES env var (JSON)
FUNCTION_NAMES = {
    "primary": "sort_algorithm", # The main function to benchmark
    "secondary": None, # No secondary function like decompress
    "free": None # No memory free function needed for Python lists typically
}

# Define Python "signatures" in a way the generic wrapper *might* understand
# or ignore. This part is less critical for pure Python if the wrapper handles it.
# We'll use placeholder types for now. The wrapper script needs to know how
# to serialize/deserialize Python lists for the target function.
# Passed as FUNCTION_SIGNATURES env var (JSON)
# NOTE: The current framework `docker_exec_wrapper.py` uses these hints mainly for C.
# For Python, it primarily relies on the function name and direct calls.
FUNCTION_SIGNATURES = {
     "sort_algorithm": {
         "argtypes": ["list_int"], # Custom type hint for the wrapper (informational)
         "restype": "list_int"     # Custom type hint for the wrapper (informational)
     }
}

# --- Benchmark Type & Runner Flags ---
# Identifier for the wrapper script logic. Needs to match a handler
# in framework/docker_exec_wrapper.py (e.g., "python_sort").
BENCHMARK_TYPE = "python_sort" # Needs corresponding logic in docker_exec_wrapper.py
CALCULATE_RATIO = False # No ratio calculation needed for sort
TIME_SECONDARY_FUNCTION = False # No secondary function to time

# --- Function References for Framework ---
# These functions are defined in the sort-bench modules
# and assigned here for the framework app_base to use.

# Function to generate the prompt for the LLM
# Defined in sort-bench/benchmark.py
def prompt_generator_func():
    from . import benchmark # Local import to avoid circular dependency issues at load time
    # Generate examples dynamically each time or use cached ones if preferred
    prompt_examples = benchmark.generate_prompt_examples(num_examples=5)
    return benchmark.create_sort_prompt(examples=prompt_examples)

# Function to load (and potentially generate) the test suite
# Defined in sort-bench/test_suite_generator.py
def test_suite_loader_func():
    from . import test_suite_generator # Local import
    # Construct path relative to this config file's directory
    test_suite_path = os.path.join(os.path.dirname(__file__), TEST_SUITE_FILENAME)

    # This function should handle checking existence, generating if needed, and loading
    if not os.path.exists(test_suite_path):
         print(f"Config: Test suite '{test_suite_path}' not found. Generating...")
         # Use default generation parameters or define specific ones here
         gen_params = {
             'size_small': 20, 'size_medium': 20000, 'size_large': 2000000,
             'num_cases_per_type': 5
         }
         test_suite_generator.generate_and_save_test_suite(test_suite_path, **gen_params)
         print(f"Config: Test suite generated and saved to '{test_suite_path}'.")
    else:
         print(f"Config: Using existing test suite file: '{test_suite_path}'")

    print("Config: Loading test suite...")
    try:
        loaded_suite = test_suite_generator.load_test_suite(test_suite_path)
        print(f"Config: Test suite loaded successfully ({len(loaded_suite)} categories).")
        return loaded_suite
    except Exception as e:
        print(f"Config CRITICAL ERROR: Failed to load test suite '{test_suite_path}': {e}")
        # Re-raise or return None/empty dict? Re-raising is probably better.
        raise RuntimeError(f"Failed to load test suite '{test_suite_path}': {e}") from e


# --- Placeholders for framework hooks (will be set by create_blueprint) ---
benchmark_runner = None # Will be instance of BenchmarkRunner
run_benchmark_func = None # Will be the background task trigger function
