from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import database
import benchmark
import threading
import uuid # For generating unique task IDs
from collections import deque # For storing recent progress updates
import time # For timestamping progress updates

app = Flask(__name__)
# Required for flash messages
app.secret_key = 'super secret key' # Change this to a random secret key, maybe from env var

# Define available LLMs (can be moved to config later)
PYTHON_SORTED_BENCHMARK = "Python sorted()"
AVAILABLE_LLMS = [PYTHON_SORTED_BENCHMARK, "dummy_llm"] # Add real LLM identifiers here
# AVAILABLE_ALGORITHMS removed as specific algorithms are no longer selected

# --- Benchmark Status Tracking ---
# Store status of running/completed benchmarks. Use a deque to limit memory usage for progress lists.
# WARNING: This is in-memory and will be lost on server restart.
# For production, consider a more persistent store (e.g., Redis, DB table).
BENCHMARK_STATUS = {}
MAX_PROGRESS_UPDATES = 50 # Store the last N updates per task
STATUS_LOCK = threading.Lock() # To safely update BENCHMARK_STATUS from multiple threads

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
    # Determine the algorithm label based on the LLM type
    algorithm_label = benchmark.GENERAL_ALGORITHM_NAME if llm_name != PYTHON_SORTED_BENCHMARK else benchmark.BASELINE_ALGORITHM_NAME
    print(f"Starting background benchmark task {task_id}: {llm_name} - {algorithm_label}")

    def progress_callback(update_data):
        """Callback function passed to benchmark methods."""
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                # Add timestamp and limit the number of progress updates stored
                update_data['timestamp'] = time.time()
                BENCHMARK_STATUS[task_id]['progress'].append(update_data)
                # Update overall status fields as well
                BENCHMARK_STATUS[task_id]['status'] = update_data.get('status', 'Running')
                BENCHMARK_STATUS[task_id]['current_case'] = update_data.get('current_case')
                BENCHMARK_STATUS[task_id]['total_cases'] = update_data.get('total_cases')
                BENCHMARK_STATUS[task_id]['last_update'] = update_data['timestamp']


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
            'last_update': time.time()
        }

    try:
        # Ensure DB is initialized within the thread context
        # database.init_db() # Already called at startup and potentially problematic here if DB file is locked

        if llm_name == PYTHON_SORTED_BENCHMARK:
            # Run benchmark using Python's built-in sorted() - no algorithm name needed
            result = benchmark.run_python_sorted_benchmark(progress_callback=progress_callback)
        else:
            # Run benchmark using LLM generation - no algorithm name needed
            result = benchmark.run_single_benchmark(llm_name, progress_callback=progress_callback)


        # Save final result to DB
        database.save_result(result)

        # Update final status
        with STATUS_LOCK:
            BENCHMARK_STATUS[task_id]['status'] = 'Completed'
            BENCHMARK_STATUS[task_id]['end_time'] = time.time()
            BENCHMARK_STATUS[task_id]['final_result'] = result # Store the summary
            BENCHMARK_STATUS[task_id]['error'] = result.get('error') # Store potential eval errors
            BENCHMARK_STATUS[task_id]['last_update'] = time.time()

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
            'error': f"Benchmark execution failed: {e}",
            'correctness': None,
            # Lines updating BENCHMARK_STATUS removed from here, handled above in STATUS_LOCK block
            'avg_time_ms': None,
            'baseline_avg_time_ms': None,
            'performance_details': None, # Add placeholder
            'generated_code': None
        }
        # Also save this minimal error result in the status dict
        with STATUS_LOCK:
             if task_id in BENCHMARK_STATUS:
                 BENCHMARK_STATUS[task_id]['final_result'] = error_result

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

    # Determine the algorithm label for the flash message
    algorithm_label = benchmark.GENERAL_ALGORITHM_NAME if llm_name != PYTHON_SORTED_BENCHMARK else benchmark.BASELINE_ALGORITHM_NAME

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
