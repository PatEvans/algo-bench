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
BENCHMARK_NAME = "C Integer Array Sort"
# Use a benchmark-specific database file
DATABASE_FILE = "sort-bench/c_sort_benchmark_results.db" # New DB file
# Test suite file specific to this benchmark (relative to sort-bench dir)
TEST_SUITE_FILENAME = "c_sort_test_suite.json" # New test suite file
# LLM Code filename (used within Docker) - will contain the C code
LLM_CODE_FILENAME = "llm_sort.c" # C source file

# --- Docker/Execution Configuration (Leveraging Framework Runner) ---
# DOCKER_IMAGE setting is removed/commented out.
# The BenchmarkRunner uses the unified image built by the root Dockerfile.
# DOCKER_IMAGE = "python:3.10-slim"
CONTAINER_MEM_LIMIT = "1g"
CONTAINER_CPU_SHARES = 512 # Relative CPU weight (default 1024)
EXEC_TIMEOUT_SECONDS = 600 # Timeout for the entire docker exec run

# Path to the framework's generic wrapper script
WRAPPER_SCRIPT_PATH = os.path.abspath(os.path.join(framework_dir, 'docker_exec_wrapper.py'))

# --- Baseline Configuration ---
C_QSORT_BASELINE_LABEL = "C qsort() [Baseline]"
# Path to the baseline C code file (relative to sort-bench dir)
BASELINE_C_CODE_FILENAME = "baseline_c_sort.c"
# Load the C code snippet from the file
BASELINE_C_CODE_SNIPPET = None # Will be loaded dynamically if needed, or handled by runner
try:
    baseline_c_path = os.path.join(os.path.dirname(__file__), BASELINE_C_CODE_FILENAME)
    if os.path.exists(baseline_c_path):
        with open(baseline_c_path, 'r', encoding='utf-8') as f:
            BASELINE_C_CODE_SNIPPET = f.read()
        print(f"Sort Config: Loaded C baseline code from '{baseline_c_path}'.")
    else:
        print(f"Sort Config WARNING: Baseline C code file '{baseline_c_path}' not found. C baseline will be unavailable.")
except Exception as e:
    print(f"Sort Config WARNING: Failed to read baseline C code file '{baseline_c_path}': {e}. C baseline will be unavailable.")

# Define baseline snippets dictionary
BASELINE_CODE_SNIPPETS = {}
if BASELINE_C_CODE_SNIPPET:
     BASELINE_CODE_SNIPPETS[C_QSORT_BASELINE_LABEL] = BASELINE_C_CODE_SNIPPET
else:
     # Provide a placeholder or handle the error appropriately if baseline is mandatory
     BASELINE_CODE_SNIPPETS[C_QSORT_BASELINE_LABEL] = "// Baseline C code could not be loaded."


# --- LLM Configuration ---
# Add real LLM identifiers here. The baseline is now just another option.
AVAILABLE_LLMS = []
if BASELINE_C_CODE_SNIPPET:
    AVAILABLE_LLMS.append(C_QSORT_BASELINE_LABEL)
AVAILABLE_LLMS.extend(["Gemini 2.5 Pro Exp"]) # Example LLM

# --- Function Names & Signatures (for Framework Runner/Wrapper Interaction) ---
# These names MUST match expectations in framework/docker_exec_wrapper.py
# and the generated Python code structure.
# Passed as FUNCTION_NAMES env var (JSON)
FUNCTION_NAMES = {
    "primary": "sort_array", # The C function to benchmark
    "secondary": None,       # No secondary function
    "free": None             # No specific free function needed if sort is in-place
}

# Define C function signatures using ctypes types for the wrapper script
# Passed as FUNCTION_SIGNATURES env var (JSON)
FUNCTION_SIGNATURES = {
    "sort_array": {
        "argtypes": ["POINTER_int", "size_t"], # int* arr, size_t n
        "restype": "void"                      # Function sorts in-place
    }
    # Add struct definitions here if the C code uses them (unlikely for basic sort)
}

# --- Benchmark Type & Runner Flags ---
# Identifier for the wrapper script logic. Needs to match a handler
# in framework/docker_exec_wrapper.py.
BENCHMARK_TYPE = "c_sort_int_array" # New type for C integer array sorting
CALCULATE_RATIO = False # No ratio calculation needed for sort
TIME_SECONDARY_FUNCTION = False # No secondary function to time

# --- Function References for Framework ---
# These functions are defined in the sort-bench modules
# and assigned here for the framework app_base to use.

# Function to generate the prompt for the LLM
# Defined in sort-bench/benchmark.py
def prompt_generator_func():
    from . import benchmark # Local import
    # Generate C-style examples
    prompt_examples = benchmark.generate_prompt_examples(num_examples=5)
    # Create C-specific prompt
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
        # Adjust sizes if needed for C performance characteristics
        gen_params = {
            'size_small': 50, 'size_medium': 50000, 'size_large': 5000000, # Example adjusted sizes
            'num_cases_per_type': 5
        }
        # Use the correct generator module reference
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
