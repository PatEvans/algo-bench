"""
Configuration specific to the LLM C Compression Benchmark.
"""
import os
import sys
import ctypes
# Ensure the parent directory (project root) is in the path to find the framework
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary functions/modules from the benchmark and framework
# Use relative imports within the same package
from . import benchmark as compression_benchmark # Prompt generation
from . import test_suite_generator as compression_test_suite # Test suite loading
# Framework components will be used by app.py, not directly imported here usually

# --- Core Benchmark Settings ---
BENCHMARK_NAME = "LLM C Compression"
# Use the framework's generic database by default, but specify a benchmark-specific file
DATABASE_FILE = "compress-bench/compression_benchmark_results.db"
# Test suite file specific to this benchmark
TEST_SUITE_FILENAME = compression_test_suite.DEFAULT_TEST_SUITE_FILE # e.g., "compression_test_suite.json"
# LLM Code filename (used within Docker)
LLM_CODE_FILENAME = "llm_compress.c"

# --- Docker Settings ---
# DOCKER_IMAGE setting is removed/commented out.
# The BenchmarkRunner uses the unified image built by the root Dockerfile.
# DOCKER_IMAGE = "compression-benchmark"
CONTAINER_MEM_LIMIT = "1g"
CONTAINER_CPU_SHARES = 512 # Relative CPU weight (default 1024)
EXEC_TIMEOUT_SECONDS = 600 # Timeout for the entire docker exec run

# --- Framework Script Paths ---
# Path relative to the project root where the framework wrapper script lives
FRAMEWORK_DIR = os.path.join(project_root, 'framework')
WRAPPER_SCRIPT_PATH = os.path.abspath(os.path.join(FRAMEWORK_DIR, 'docker_exec_wrapper.py'))

# --- LLM and Baseline Settings ---
C_COPY_BASELINE_LABEL = "C Copy [Baseline]"
BASELINE_C_CODE_FILENAME = "baseline_c_compress.c" # Assumed relative to compress-bench dir

# Load C baseline code
BASELINE_C_CODE_SNIPPET = None
try:
    baseline_c_path = os.path.join(os.path.dirname(__file__), BASELINE_C_CODE_FILENAME)
    with open(baseline_c_path, 'r', encoding='utf-8') as f:
        BASELINE_C_CODE_SNIPPET = f.read()
    print(f"Compression Config: Loaded C baseline code from '{baseline_c_path}'.")
except FileNotFoundError:
    print(f"Compression Config WARNING: Baseline C code file '{baseline_c_path}' not found. C baseline will be unavailable.")
except Exception as e:
    print(f"Compression Config WARNING: Failed to read baseline C code file '{baseline_c_path}': {e}. C baseline will be unavailable.")

# Available LLMs/Baselines for this benchmark
AVAILABLE_LLMS = []
if BASELINE_C_CODE_SNIPPET:
    AVAILABLE_LLMS.append(C_COPY_BASELINE_LABEL)
AVAILABLE_LLMS.extend(["Gemini 2.5 Pro Exp"]) # Add real LLMs

# Map baseline labels to their code snippets
BASELINE_CODE_SNIPPETS = {}
if BASELINE_C_CODE_SNIPPET:
    BASELINE_CODE_SNIPPETS[C_COPY_BASELINE_LABEL] = BASELINE_C_CODE_SNIPPET

# --- Function Configuration (Standardized Keys) ---
# Names of the functions the LLM should generate / baseline provides
FUNCTION_NAMES = {
    "primary": "compress",
    "secondary": "decompress",
    "free": "free_buffer"
}

# Signatures of the C functions using STRING representations of types
# These strings MUST match keys in CTYPES_MAP in docker_exec_wrapper.py
FUNCTION_SIGNATURES = {
    # Struct definition using string types
    "Buffer": {
        "is_struct": True,
        "fields": [("data", "POINTER_ubyte"), ("size", "size_t")]
    },
    # Function signatures using string types
    "compress": {
        "argtypes": ["POINTER_ubyte", "size_t"],
        "restype": "Buffer" # Use the struct name defined above
    },
    "decompress": {
        "argtypes": ["POINTER_ubyte", "size_t"],
        "restype": "Buffer" # Use the struct name defined above
    },
    "free_buffer": {
        "argtypes": ["Buffer"], # Use the struct name defined above
        "restype": "void" # Use string "void" or None
    }
}

# --- Benchmark Execution Parameters ---
# These are passed to the docker_exec_wrapper via environment variables
BENCHMARK_TYPE = "c_compression" # Identifier for the wrapper script logic (must match BENCHMARK_HELPERS key)
CALCULATE_RATIO = True
TIME_SECONDARY_FUNCTION = True

# --- Prompt Generation Function ---
# Function reference that the framework's background task will call
def prompt_generator_func():
    """Generates the prompt for the compression benchmark."""
    # Use functions from the local benchmark module
    examples = compression_benchmark.generate_prompt_examples(num_examples=2)
    return compression_benchmark.create_compression_prompt(examples=examples)

# --- Test Suite Loader Function ---
# Function reference that the framework's app base will call on init
def test_suite_loader_func():
    """Loads the test suite data for the compression benchmark."""
    # Use the loader from the local test suite generator module
    # This should return the loaded data (base64 encoded dict for compression)
    # It will raise exceptions on failure, which the framework app base handles
    test_suite_path = os.path.join(os.path.dirname(__file__), TEST_SUITE_FILENAME)
    return compression_test_suite.load_test_suite(test_suite_path)

# --- Framework Integration ---
# These will be set in app.py after importing this config
benchmark_runner = None
run_benchmark_func = None
