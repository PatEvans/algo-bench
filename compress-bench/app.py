"""
Compression Benchmark Flask Application.
Uses the shared benchmark framework.
"""
import os
import sys
import threading

# Ensure the framework directory is in the Python path
# This allows importing 'framework' modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import framework components
from framework.app_base import BenchmarkApp, run_benchmark_background_base, STATUS_LOCK, BENCHMARK_STATUS
from framework.benchmark_runner import BenchmarkRunner
from framework.database import BenchmarkDB # Import framework DB

# Import benchmark-specific configuration
from compress_bench import config as compression_config

# --- Initialize Framework Components ---

# 1. Initialize the Benchmark Runner with compression-specific config
#    The runner needs the config to know Docker image, C functions, etc.
try:
    compression_runner = BenchmarkRunner(compression_config)
    compression_config.benchmark_runner = compression_runner # Store in config for background task
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize BenchmarkRunner: {e}")
    # Optionally exit or prevent app startup
    sys.exit(1)

# 2. Define the background task function specific to this benchmark
#    This function will be called by the framework's app base in a thread.
def run_compression_benchmark_task(task_id, llm_name, config, db: BenchmarkDB, test_suite_data):
    """Background task function wrapper for compression benchmark."""
    # Calls the framework's base background task function, passing the
    # compression-specific runner instance.
    run_benchmark_background_base(
        task_id=task_id,
        llm_name=llm_name,
        config=config, # Pass the compression_config module
        db=db, # Pass the DB handler initialized by BenchmarkApp
        test_suite_data=test_suite_data, # Pass the loaded test suite
        benchmark_runner=compression_runner # Pass the compression-specific runner
    )

# 3. Store the background task function in the config object
compression_config.run_benchmark_func = run_compression_benchmark_task

# 4. Initialize the Framework Flask App
#    Pass the compression-specific configuration module.
#    The BenchmarkApp class handles Flask setup, routes, status tracking,
#    test suite loading (using config.test_suite_loader_func),
#    and triggering the background task (using config.run_benchmark_func).
try:
    # The BenchmarkApp init calls the test_suite_loader_func from the config
    benchmark_app_instance = BenchmarkApp(compression_config)
    # Get the underlying Flask app object to run
    app = benchmark_app_instance.app
except Exception as e:
     print(f"CRITICAL ERROR: Failed to initialize BenchmarkApp: {e}")
     # Optionally exit
     sys.exit(1)


# --- Main Execution ---
if __name__ == '__main__':
    # The framework's BenchmarkDB handles initialization via its __init__
    # The framework's BenchmarkApp handles test suite loading via its __init__
    # Just need to run the Flask app instance obtained from BenchmarkApp
    # Use host='0.0.0.0' to make it accessible externally if needed
    # Use a different port for each benchmark app if running simultaneously
    app_port = int(os.environ.get("COMPRESS_BENCH_PORT", 5001))
    print(f"Starting Compression Benchmark server on port {app_port}...")
    app.run(debug=True, host='0.0.0.0', port=app_port)
