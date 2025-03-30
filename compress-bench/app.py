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
from framework.app_base import BenchmarkBlueprint, run_benchmark_background_base
from framework.benchmark_runner import BenchmarkRunner
from framework.database import BenchmarkDB # Still needed for background task type hint

# Import benchmark-specific configuration
from . import config as compression_config # Use relative import

# --- Initialize Benchmark Runner (once per process) ---
# This should ideally be managed by the main app or dependency injection,
# but for simplicity, we initialize it here. Be mindful of resource usage if
# multiple benchmarks are run in the same process.
try:
    compression_runner = BenchmarkRunner(compression_config)
    # Store runner in config for background task access (might need a better way)
    compression_config.benchmark_runner = compression_runner
except Exception as e:
    print(f"CRITICAL ERROR [compress-bench]: Failed to initialize BenchmarkRunner: {e}")
    # Raise the error to prevent blueprint creation if runner fails
    raise RuntimeError(f"Failed to initialize BenchmarkRunner for compression: {e}") from e

# --- Define Background Task Function ---
# This function will be called by the framework's base logic in a thread.
def run_compression_benchmark_task(task_id, llm_name, config, db: BenchmarkDB, test_suite_data):
    """Background task function wrapper for compression benchmark."""
    # Calls the framework's base background task function, passing the
    # compression-specific runner instance (retrieved from config).
    runner = getattr(config, 'benchmark_runner', None)
    if not runner:
         # Handle error: runner not initialized or passed correctly
         print(f"ERROR [Task {task_id}]: BenchmarkRunner not found in config.")
         # Update status to reflect error (requires access to STATUS_LOCK/BENCHMARK_STATUS)
         # This highlights complexity of direct threading; task queues are better.
         return # Or raise an exception

    run_benchmark_background_base(
        task_id=task_id,
        llm_name=llm_name,
        config=config, # Pass the compression_config module
        db=db, # Pass the DB handler initialized by BenchmarkBlueprint
        test_suite_data=test_suite_data, # Pass the loaded test suite
        benchmark_runner=runner # Pass the compression-specific runner
    )

# Store the background task function in the config object so BenchmarkBlueprint can find it
compression_config.run_benchmark_func = run_compression_benchmark_task

# --- Blueprint Factory Function ---
def create_blueprint():
    """Factory function to create and configure the compression benchmark Blueprint."""
    print(f"Creating blueprint for: {compression_config.BENCHMARK_NAME}")
    # Instantiate the framework's BenchmarkBlueprint class
    # Pass blueprint name, import name (__name__), config, and URL prefix
    benchmark_bp_instance = BenchmarkBlueprint(
        blueprint_name='compress',
        import_name=__name__,
        config=compression_config,
        url_prefix='/compress' # Define the URL prefix for this benchmark
    )
    # Return the created blueprint and the config (for the main app)
    return benchmark_bp_instance.get_blueprint(), compression_config

# --- Remove Standalone Execution ---
# The app is now run via the main_app.py entry point.
# if __name__ == '__main__':
#     # ... (old code removed) ...
