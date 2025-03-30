from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from . import database
from . import benchmark
from . import test_suite_generator # Import the new module
from . import llm_interface # Import the missing module
import threading
import uuid # For generating unique task IDs
from collections import deque # For storing recent progress updates
import time # For timestamping progress updates
import os
import json # For loading test suite

app = Flask(__name__)
# Required for flash messages
app.secret_key = 'super secret key' # Change this to a random secret key, maybe from env var

# --- Constants ---
# Use the default from the test_suite_generator module
TEST_SUITE_FILENAME = test_suite_generator.DEFAULT_TEST_SUITE_FILE
# Label for the baseline benchmark run inside Docker using sorted()
PYTHON_SORTED_BENCHMARK_LABEL = "Python sorted() [Docker]"
# The actual code snippet for the baseline benchmark
PYTHON_SORTED_CODE_SNIPPET = """
# Baseline implementation using Python's built-in sorted()
def sort_algorithm(data: list) -> list:
    # Return a new sorted list, as required by the benchmark structure
    return sorted(data)
"""
# Add real LLM identifiers here. The baseline is now just another option.
AVAILABLE_LLMS = [PYTHON_SORTED_BENCHMARK_LABEL, "Gemini 2.5 Pro Exp"]
# AVAILABLE_ALGORITHMS removed as specific algorithms are no longer selected

# --- Benchmark Status Tracking ---
# Store status of running/completed benchmarks. Use a deque to limit memory usage for progress lists.
# WARNING: This is in-memory and will be lost on server restart.
# For production, consider a more persistent store (e.g., Redis, DB table).
BENCHMARK_STATUS = {}
MAX_PROGRESS_UPDATES = 50 # Store the last N updates per task
STATUS_LOCK = threading.Lock() # To safely update BENCHMARK_STATUS from multiple threads

# --- Global Test Suite ---
# Loaded once at startup
GLOBAL_TEST_SUITE = None
TEST_SUITE_LOAD_ERROR = None # Store any error during loading

def initialize_test_suite():
    """Generates (if needed) and loads the global test suite."""
    global GLOBAL_TEST_SUITE, TEST_SUITE_LOAD_ERROR
    try:
        if not os.path.exists(TEST_SUITE_FILENAME):
            print(f"Test suite file '{TEST_SUITE_FILENAME}' not found. Generating...")
            # Use default generation parameters from benchmark.py's main block for consistency
            # Or define specific parameters here if needed
            gen_params = {
                'size_small': 20,
                'size_medium': 20000,
                'size_large': 2000000,
                'num_cases_per_type': 5
            }
            # Use the function from the new module
            test_suite_generator.generate_and_save_test_suite(TEST_SUITE_FILENAME, **gen_params)
            print(f"Test suite generated and saved to '{TEST_SUITE_FILENAME}'.")
        else:
            print(f"Using existing test suite file: '{TEST_SUITE_FILENAME}'")

        print("Loading test suite...")
        # Use the function from the new module
        GLOBAL_TEST_SUITE = test_suite_generator.load_test_suite(TEST_SUITE_FILENAME)
        print(f"Test suite loaded successfully ({len(GLOBAL_TEST_SUITE)} categories).")
        TEST_SUITE_LOAD_ERROR = None # Clear any previous error
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load or generate test suite '{TEST_SUITE_FILENAME}': {e}")
        TEST_SUITE_LOAD_ERROR = f"Failed to load/generate test suite: {e}"
        GLOBAL_TEST_SUITE = None # Ensure it's None if loading failed

# --- Initialize Test Suite on Startup ---
initialize_test_suite()


@app.route('/')
def index():
    """Display the benchmark results page."""
    try:
        current_results = database.get_all_results()
    except Exception as e:
        flash(f"Error fetching results from database: {e}", "error")
        current_results = []
    return render_template('index.html', results=current_results)

@app.route('/admin')
def admin():
    """Display the admin page to run benchmarks."""
    # Pass only LLMs to the template
    return render_template('admin.html', llms=AVAILABLE_LLMS)


# algorithm_name parameter removed
def run_benchmark_background(task_id, llm_name):
    """Function to run benchmark in a separate thread and update status."""
    # All benchmarks run via run_single_benchmark now use the same label
    algorithm_label = benchmark.BENCHMARKED_ALGORITHM_LABEL
    print(f"Starting background benchmark task {task_id}: {llm_name} - {algorithm_label}")

    def progress_callback(update_data):
        """Callback function passed to benchmark methods."""
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                # Add timestamp and limit the number of progress updates stored
                update_data['timestamp'] = time.time()
                BENCHMARK_STATUS[task_id]['progress'].append(update_data)
                # Update overall status fields as well
                BENCHMARK_STATUS[task_id]['status'] = update_data.get('status', BENCHMARK_STATUS[task_id]['status']) # Keep existing status if not provided
                BENCHMARK_STATUS[task_id]['current_case'] = update_data.get('current_case', BENCHMARK_STATUS[task_id]['current_case'])
                BENCHMARK_STATUS[task_id]['total_cases'] = update_data.get('total_cases', BENCHMARK_STATUS[task_id]['total_cases'])
                BENCHMARK_STATUS[task_id]['last_update'] = update_data['timestamp']
                # Store generated code if provided in the update
                if 'generated_code' in update_data:
                     BENCHMARK_STATUS[task_id]['generated_code'] = update_data['generated_code']


    # Initialize status
    with STATUS_LOCK:
        # Use the determined algorithm_label in the status
        BENCHMARK_STATUS[task_id] = {
            'task_id': task_id,
            'llm': llm_name,
            'algorithm': algorithm_label, # Use the determined label
            'status': 'Initializing',
            'start_time': time.time(),
            'end_time': None,
            'current_case': 0,
            'total_cases': None, # Will be updated by callback
            'progress': deque(maxlen=MAX_PROGRESS_UPDATES), # Store recent updates
            'final_result': None,
            'error': None,
            'generated_code': None, # Field to store generated code
            'last_update': time.time()
        }

    generated_code_for_llm = None # Variable to hold generated code if applicable
    result = None # Variable to hold the final result dict


    try:
        # Ensure DB is initialized within this thread's context before saving
        database.init_db()

        # --- Check if Test Suite is Loaded ---
        if GLOBAL_TEST_SUITE is None:
            # Use the stored error message from startup
            raise ValueError(f"Test suite not available. Load error: {TEST_SUITE_LOAD_ERROR}")

        print(f"Task {task_id}: Using pre-loaded test suite.")
        # The test suite is already loaded in GLOBAL_TEST_SUITE

        # --- Generate Code (if LLM) or Use Baseline Code ---
        if llm_name == PYTHON_SORTED_BENCHMARK_LABEL:
             # --- Use Baseline Code (sorted()) ---
            progress_callback({'status': 'Using Baseline Code', 'category': 'Setup'})
            generated_code_for_llm = PYTHON_SORTED_CODE_SNIPPET
            print(f"Task {task_id}: Using baseline code (sorted()).")

            # --- Update Status with Baseline Code BEFORE Evaluation ---
            with STATUS_LOCK:
                if task_id in BENCHMARK_STATUS:
                    BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
                    BENCHMARK_STATUS[task_id]['status'] = 'Evaluating Baseline Code...'
                    BENCHMARK_STATUS[task_id]['last_update'] = time.time()
            progress_callback({
                'status': 'Evaluating Baseline Code...',
                'category': 'Setup',
                'generated_code': generated_code_for_llm
            })

            # --- Run benchmark evaluation using the baseline code ---
            result = benchmark.run_single_benchmark(
                llm_name=llm_name, # Pass the baseline label as the identifier
                generated_code=generated_code_for_llm, # Pass the baseline code snippet
                categorized_test_cases=GLOBAL_TEST_SUITE, # Pass loaded suite
                progress_callback=progress_callback
            )
        else:
            # --- Generate Code using Actual LLM ---
            progress_callback({'status': 'Generating Code', 'category': 'Setup'})
            prompt_examples = benchmark.generate_prompt_examples(num_examples=5)
            prompt = benchmark.create_sort_prompt(examples=prompt_examples)
            print(f"Task {task_id}: Generating code using {llm_name}...")
            generated_code_for_llm = llm_interface.generate_code(llm_name, prompt)

            if not generated_code_for_llm:
                raise ValueError(f"LLM '{llm_name}' failed to generate code.")

            # --- Update Status with Generated Code BEFORE Evaluation ---
            with STATUS_LOCK:
                if task_id in BENCHMARK_STATUS:
                    BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
                    BENCHMARK_STATUS[task_id]['status'] = 'Code Generated, Evaluating...'
                    BENCHMARK_STATUS[task_id]['last_update'] = time.time()
            # Send an update specifically containing the code
            progress_callback({
                'status': 'Code Generated, Evaluating...',
                'category': 'Setup',
                'generated_code': generated_code_for_llm
            })
            print(f"Task {task_id}: Code generated. Starting evaluation.")

            # --- Run benchmark evaluation using the generated code ---
            result = benchmark.run_single_benchmark(
                llm_name=llm_name,
                generated_code=generated_code_for_llm, # Pass the generated code
                categorized_test_cases=GLOBAL_TEST_SUITE, # Pass loaded suite
                progress_callback=progress_callback
            )

        # --- Save final result to DB ---
        # Ensure the result dict includes the algorithm label
        if result:
            # generated_code is already included in the result dict from run_single_benchmark
            # Ensure algorithm label is in the result dict before saving
            result['algorithm'] = algorithm_label # Add/overwrite algorithm label

        if result: # Only save if a result was actually produced
            database.save_result(result)
        else:
             # This case might happen if code generation failed before evaluation started
             print(f"Task {task_id}: No result dictionary generated, skipping database save.")


        # --- Update final status ---
        with STATUS_LOCK:
            BENCHMARK_STATUS[task_id]['status'] = 'Completed'
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = 'Completed'
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['final_result'] = result # Store the summary
                BENCHMARK_STATUS[task_id]['error'] = result.get('error') if result else None # Store potential eval errors
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
                # Ensure generated code is in the final status if it exists
                if generated_code_for_llm:
                    BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm

        # Use algorithm_label in the finished message
        print(f"Finished background benchmark task {task_id}: {llm_name} - {algorithm_label}")

    except Exception as e:
        # Use algorithm_label in error message
        print(f"Error in background benchmark task {task_id} ({llm_name} - {algorithm_label}): {e}")
        error_message = f"Benchmark execution failed: {e}"
        with STATUS_LOCK:
            # Ensure the status dict exists before updating
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = 'Error'
                BENCHMARK_STATUS[task_id]['error'] = error_message
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
            else:
                # Should not happen ideally, but log if it does
                print(f"Error: Task ID {task_id} not found in BENCHMARK_STATUS during exception handling.")


        # Optionally save error state to DB
        error_result = {
            'llm': llm_name,
            'algorithm': algorithm_label, # Use the determined label
            'error': error_message, # Use the formatted error message
            'correctness': None,
            'avg_time_ms': None,
            'baseline_avg_time_ms': None,
            'performance_details': None,
            'generated_code': generated_code_for_llm # Include code if generated before error
        }
        # Also save this minimal error result in the status dict
        with STATUS_LOCK:
             if task_id in BENCHMARK_STATUS:
                 BENCHMARK_STATUS[task_id]['final_result'] = error_result
                 # Ensure generated code is stored even in error status if available
                 if generated_code_for_llm:
                      BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm


        # Try saving to DB (best effort)
        try:
            database.save_result(error_result)
        except Exception as db_e:
            print(f"Failed to save error result to DB: {db_e}")


@app.route('/run', methods=['POST'])
def run_benchmark():
    """Endpoint to trigger a new benchmark run."""
    # Get parameters from form
    llm_name = request.form.get('llm')
    # algorithm_name is no longer submitted by the form

    # Validate parameters
    if not llm_name or llm_name not in AVAILABLE_LLMS:
        flash(f"Invalid or missing LLM selected: {llm_name}", "error")
        return redirect(url_for('admin'))
    # Algorithm validation removed

    # Determine the algorithm label for the flash message (always the same now)
    algorithm_label = benchmark.BENCHMARKED_ALGORITHM_LABEL

    # Generate a unique ID for this benchmark task
    task_id = str(uuid.uuid4())

    # Run benchmark in a background thread - pass only task_id and llm_name
    thread = threading.Thread(target=run_benchmark_background, args=(task_id, llm_name))
    thread.daemon = True # Allow app to exit even if background threads are running
    thread.start()

    flash(f"Benchmark task {task_id} started for {llm_name} ({algorithm_label}).", "info")
    # Redirect to the progress page for this task
    return redirect(url_for('benchmark_progress', task_id=task_id))


@app.route('/benchmark_progress/<task_id>')
def benchmark_progress(task_id):
    """Display the progress page for a specific benchmark task."""
    # Pass the task ID to the template
    return render_template('progress.html', task_id=task_id)


@app.route('/benchmark_status/<task_id>')
def benchmark_status(task_id):
    """API endpoint to get the current status of a benchmark task."""
    with STATUS_LOCK:
        status = BENCHMARK_STATUS.get(task_id)

    if not status:
        return jsonify({'status': 'Not Found'}), 404

    # Convert deque to list for JSON serialization
    status_copy = status.copy()
    if 'progress' in status_copy:
        status_copy['progress'] = list(status_copy['progress'])

    return jsonify(status_copy)


# --- Optional: Add cleanup for old statuses ---
# You might want a background task or periodic check to remove very old entries
# from BENCHMARK_STATUS to prevent memory leaks if the server runs for a long time.

if __name__ == '__main__':
    # Initialize the database if it doesn't exist
    database.init_db()
    app.run(debug=True) # debug=True for development, remove for production
