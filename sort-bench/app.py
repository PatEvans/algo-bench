"""
Sort Benchmark Flask Application.

This module provides the specific logic for the Sort benchmark,
integrating with the common framework components.
It defines how to configure and create the Flask Blueprint for this benchmark.
"""
import os
import sys
import traceback # Import traceback for detailed error logging

# Ensure the framework directory is in the path
# This allows importing framework modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
framework_dir = os.path.join(project_root, 'framework')
if framework_dir not in sys.path:
    sys.path.insert(0, framework_dir)

# Framework imports - Use try-except for better error reporting if framework is missing/broken
try:
    from app_base import BenchmarkBlueprint, run_benchmark_background_base
    from benchmark_runner import BenchmarkRunner
    from database import BenchmarkDB # Use framework DB
except ImportError as e:
     print(f"ERROR: Failed to import framework components in sort-bench/app.py: {e}")
     print("Ensure the 'framework' directory exists and is accessible.")
     # Re-raise or define dummy classes/functions to prevent NameErrors later?
     # Re-raising is cleaner as the app cannot function without the framework.
     raise ImportError(f"Framework import failed: {e}") from e


# Sort benchmark specific imports
try:
    from . import config as sort_config # Import the benchmark-specific config
except ImportError as e:
     print(f"ERROR: Failed to import sort-bench/config.py: {e}")
     print("Ensure 'config.py' exists in the 'sort-bench' directory.")
     raise ImportError(f"Sort config import failed: {e}") from e

# --- Background Task Wrapper ---
# This function acts as the entry point for the background thread
# started by the framework's BenchmarkBlueprint.
def run_sort_benchmark_task(task_id, llm_name, config, db: BenchmarkDB, test_suite_data):
    """
    Wrapper function to start the sort benchmark task in the background.
    It uses the framework's base background runner function.
    """
    # The benchmark_runner instance is expected to be attached to the config
    # object by the create_blueprint function before this is called.
    if not hasattr(config, 'benchmark_runner') or not config.benchmark_runner:
        print(f"ERROR [sort-bench]: BenchmarkRunner not initialized in config for task {task_id}.")
        # Update status to reflect this critical error? (Difficult without access to STATUS_LOCK here)
        # For now, the exception in run_benchmark_background_base will handle it.
        raise RuntimeError("BenchmarkRunner not configured.")

    print(f"Sort-Bench: Kicking off background task {task_id} for LLM: {llm_name}")
    # Call the framework's base function, passing all necessary components
    run_benchmark_background_base(
        task_id=task_id,
        llm_name=llm_name,
        config=config, # Pass the sort-specific config
        db=db,         # Pass the framework DB instance configured for sort
        test_suite_data=test_suite_data,
        benchmark_runner=config.benchmark_runner # Pass the runner instance
    )

# --- Blueprint Factory ---
# This function is discovered and called by main_app.py
def create_blueprint():
    """
    Factory function to create and configure the Flask Blueprint for the Sort benchmark.
    """
    print("Sort-Bench: Creating Blueprint...")

    # Initialize the framework's BenchmarkRunner with sort-specific config
    # This might try to connect to Docker etc. based on framework logic
    try:
        # Pass the imported sort_config object
        runner = BenchmarkRunner(sort_config)
        print("Sort-Bench: BenchmarkRunner initialized.")
    except Exception as e:
        print(f"Sort-Bench: CRITICAL ERROR initializing BenchmarkRunner: {e}")
        print(traceback.format_exc()) # Print traceback for runner init errors
        # Re-raise or handle? Re-raising will stop the main app from registering this blueprint.
        raise RuntimeError(f"Failed to initialize BenchmarkRunner for Sort benchmark: {e}") from e

    # --- Assign framework hooks in the config ---
    # These tell the BenchmarkBlueprint how to run tasks and load data for *this* benchmark
    sort_config.benchmark_runner = runner # Attach the runner instance to the config
    sort_config.run_benchmark_func = run_sort_benchmark_task # Set the background task entry point
    # The test_suite_loader_func and prompt_generator_func are already defined in sort_config.py

    # Create the BenchmarkBlueprint instance using the framework class
    # It encapsulates routes, status handling, etc.
    sort_bp = BenchmarkBlueprint(
        blueprint_name='sort',         # Unique name for this blueprint's routes/url_for
        import_name=__name__,          # Standard Flask practice
        config=sort_config,            # Pass the fully configured sort_config object
        url_prefix='/sort'             # URL prefix for all routes in this blueprint
    )
    print(f"Sort-Bench: Blueprint '{sort_bp.blueprint_name}' created with prefix '{sort_bp.bp.url_prefix}'.")

    # Return the configured blueprint instance *and* the config object
    # (main_app might use the config for display purposes)
    return sort_bp.get_blueprint(), sort_config
