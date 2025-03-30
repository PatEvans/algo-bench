from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import database
import benchmark
import test_suite_generator # Use the updated module
import llm_interface
import threading
import uuid # For generating unique task IDs
import traceback # For better error logging
from collections import deque # For storing recent progress updates
import time # For timestamping progress updates
import os
import json # For loading test suite

app = Flask(__name__)
# Required for flash messages
app.secret_key = 'super secret key' # Change this to a random secret key, maybe from env var

# --- Constants ---
# Use the default from the updated test_suite_generator module
TEST_SUITE_FILENAME = test_suite_generator.DEFAULT_TEST_SUITE_FILE # e.g., "compression_test_suite.json"
# Label for the baseline benchmark run using zlib
PYTHON_ZLIB_BASELINE_LABEL = "Python zlib [Baseline]"
# Label for the baseline benchmark using pre-written C code (copying)
C_COPY_BASELINE_LABEL = "C Copy [Baseline]"
# Filename for the baseline C code
BASELINE_C_CODE_FILENAME = "baseline_c_compress.c"

# --- Read Baseline Code Snippets ---
# Read Python zlib baseline (already defined as string)
PYTHON_ZLIB_CODE_SNIPPET = """
import zlib

# Baseline implementation using Python's built-in zlib
# Note: These functions match the required signature (bytes in, bytes out)

def compress(data: bytes) -> bytes:
    # Use zlib compression (level 6 is a good default balance)
    return zlib.compress(data, level=6)

def decompress(data: bytes) -> bytes:
    # Use zlib decompression
    return zlib.decompress(data)
"""

# Read C baseline code from file
BASELINE_C_CODE_SNIPPET = None
try:
    with open(BASELINE_C_CODE_FILENAME, 'r', encoding='utf-8') as f:
        BASELINE_C_CODE_SNIPPET = f.read()
    print(f"Successfully loaded C baseline code from '{BASELINE_C_CODE_FILENAME}'.")
except FileNotFoundError:
    print(f"WARNING: Baseline C code file '{BASELINE_C_CODE_FILENAME}' not found. C baseline will be unavailable.")
    BASELINE_C_CODE_SNIPPET = None # Ensure it's None if file not found
except Exception as e:
    print(f"WARNING: Failed to read baseline C code file '{BASELINE_C_CODE_FILENAME}': {e}. C baseline will be unavailable.")
    BASELINE_C_CODE_SNIPPET = None # Ensure it's None on error

# Add real LLM identifiers here, plus the baselines
AVAILABLE_LLMS = [PYTHON_ZLIB_BASELINE_LABEL]
if BASELINE_C_CODE_SNIPPET: # Only add C baseline if code was loaded successfully
    AVAILABLE_LLMS.append(C_COPY_BASELINE_LABEL)
AVAILABLE_LLMS.extend(["Gemini 2.5 Pro Exp"]) # Add more LLMs as needed

# --- Benchmark Status Tracking ---
# Store status of running/completed compression benchmarks. Use a deque to limit memory usage for progress lists.
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
            print(f"Compression test suite file '{TEST_SUITE_FILENAME}' not found. Generating...")
            # Use default generation parameters from the updated test_suite_generator
            # These might be defined in the generator module itself or passed here
            # Example: Use defaults from the generator's main block or define here
            gen_params = {
                 # sizes: Use defaults from test_suite_generator.DEFAULT_SIZES
                 # num_cases_per_type: Use default from test_suite_generator
            }
            # Use the updated function from the generator module
            test_suite_generator.generate_and_save_test_suite(TEST_SUITE_FILENAME) # Pass params if needed
            print(f"Compression test suite generated and saved to '{TEST_SUITE_FILENAME}'.")
        else:
            print(f"Using existing compression test suite file: '{TEST_SUITE_FILENAME}'")

        print("Loading compression test suite...")
        # Use the updated function from the generator module
        # This loads the base64 encoded data
        GLOBAL_TEST_SUITE = test_suite_generator.load_test_suite(TEST_SUITE_FILENAME)
        print(f"Compression test suite loaded successfully ({len(GLOBAL_TEST_SUITE)} categories).")
        TEST_SUITE_LOAD_ERROR = None # Clear any previous error
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load or generate compression test suite '{TEST_SUITE_FILENAME}': {e}\n{traceback.format_exc()}")
        TEST_SUITE_LOAD_ERROR = f"Failed to load/generate compression test suite: {e}"
        GLOBAL_TEST_SUITE = None # Ensure it's None if loading failed

# --- Initialize Test Suite on Startup ---
initialize_test_suite()


@app.route('/')
def index():
    """Display the compression benchmark results page."""
    try:
        current_results = database.get_all_results()
    except Exception as e:
        flash(f"Error fetching compression results from database: {e}", "error")
        current_results = []
    # Pass the test suite load error to the template
    return render_template('index.html', results=current_results, test_suite_error=TEST_SUITE_LOAD_ERROR)

@app.route('/admin')
def admin():
    """Display the admin page to run compression benchmarks."""
    # Pass LLMs and test suite load error to the template
    return render_template('admin.html', llms=AVAILABLE_LLMS, test_suite_error=TEST_SUITE_LOAD_ERROR)


# algorithm_name parameter removed
def run_benchmark_background(task_id, llm_name):
    """Function to run compression benchmark in a separate thread and update status."""
    # Use the updated label from benchmark.py
    algorithm_label = benchmark.BENCHMARKED_ALGORITHM_LABEL # e.g., "LLM Compression [Docker]"
    print(f"Starting background compression benchmark task {task_id}: {llm_name} - {algorithm_label}")

    def progress_callback(update_data):
        """Callback function passed to benchmark methods (receives updates from Docker wrapper)."""
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
            'algorithm': algorithm_label, # e.g., "LLM Compression [Docker]"
            'status': 'Initializing',
            'start_time': time.time(),
            'end_time': None,
            'current_case': 0,
            'total_cases': None, # Will be updated by callback from wrapper
            'progress': deque(maxlen=MAX_PROGRESS_UPDATES), # Store recent updates
            'final_result': None, # Will store the final dict from run_single_benchmark
            'error': None,
            'generated_code': None, # Field to store generated code
            'last_update': time.time()
        }

    generated_code_for_llm = None # Variable to hold generated code
    result = None # Variable to hold the final result dict from run_single_benchmark


    try:
        # Ensure DB is initialized within this thread's context before saving
        database.init_db()

        # --- Check if Test Suite is Loaded ---
        if GLOBAL_TEST_SUITE is None:
            # Use the stored error message from startup
            raise ValueError(f"Compression test suite not available. Load error: {TEST_SUITE_LOAD_ERROR}")

        print(f"Task {task_id}: Using pre-loaded compression test suite.")
        # The test suite (base64 encoded) is already loaded in GLOBAL_TEST_SUITE

        # --- Generate Code (if LLM) or Use Baseline Code ---
        if llm_name == PYTHON_ZLIB_BASELINE_LABEL:
            # --- Use Baseline Code (zlib) ---
            progress_callback({'status': 'Using Baseline Code', 'category': 'Setup'})
            generated_code_for_llm = PYTHON_ZLIB_CODE_SNIPPET
            print(f"Task {task_id}: Using baseline code (Python zlib).")

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
                categorized_test_cases_b64=GLOBAL_TEST_SUITE, # Pass loaded base64 suite
                progress_callback=progress_callback
            )
        elif llm_name == C_COPY_BASELINE_LABEL:
            # --- Use Baseline C Code (Copy) ---
            if not BASELINE_C_CODE_SNIPPET:
                 raise ValueError(f"Baseline C code ('{BASELINE_C_CODE_FILENAME}') was not loaded successfully at startup.")

            progress_callback({'status': 'Using Baseline C Code', 'category': 'Setup'})
            generated_code_for_llm = BASELINE_C_CODE_SNIPPET
            print(f"Task {task_id}: Using baseline C code (copy).")

            # --- Update Status with Baseline C Code BEFORE Evaluation ---
            with STATUS_LOCK:
                if task_id in BENCHMARK_STATUS:
                    BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
                    BENCHMARK_STATUS[task_id]['status'] = 'Evaluating Baseline C Code...'
                    BENCHMARK_STATUS[task_id]['last_update'] = time.time()
            progress_callback({
                'status': 'Evaluating Baseline C Code...',
                'category': 'Setup',
                'generated_code': generated_code_for_llm
            })

            # --- Run benchmark evaluation using the baseline C code ---
            # This uses the same C evaluation pipeline as LLM-generated C code
            result = benchmark.run_single_benchmark(
                llm_name=llm_name, # Pass the C baseline label as the identifier
                generated_code=generated_code_for_llm, # Pass the baseline C code snippet
                categorized_test_cases_b64=GLOBAL_TEST_SUITE, # Pass loaded base64 suite
                progress_callback=progress_callback
            )
        else:
            # --- Generate Code using Actual LLM ---
            progress_callback({'status': 'Generating Code', 'category': 'Setup'})
            # Generate examples for the compression prompt
            prompt_examples = benchmark.generate_prompt_examples(num_examples=2)
            # Create the compression prompt
            prompt = benchmark.create_compression_prompt(examples=prompt_examples)
            print(f"Task {task_id}: Generating compression code using {llm_name}...")
            generated_code_for_llm = llm_interface.generate_code(llm_name, prompt)

            if not generated_code_for_llm:
                raise ValueError(f"LLM '{llm_name}' failed to generate compression code.")

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
            print(f"Task {task_id}: Compression code generated. Starting evaluation.")

            # --- Run benchmark evaluation using the generated code ---
            # Pass the base64 encoded test suite
            result = benchmark.run_single_benchmark(
                llm_name=llm_name,
                generated_code=generated_code_for_llm, # Pass the generated code
                categorized_test_cases_b64=GLOBAL_TEST_SUITE, # Pass loaded base64 suite
                progress_callback=progress_callback
            )

        # --- Save final result to DB ---
        # The result dict from run_single_benchmark should now match the DB schema
        if result: # Only save if a result was actually produced
            database.save_result(result)
            print(f"Task {task_id}: Result dictionary saved to database.")
        else:
             # This might happen if run_single_benchmark itself fails critically
             print(f"Task {task_id}: No result dictionary generated by run_single_benchmark, skipping database save.")


        # --- Update final status ---
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = 'Completed' if not (result and result.get('error')) else 'Error'
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['final_result'] = result # Store the full result dict
                BENCHMARK_STATUS[task_id]['error'] = result.get('error') if result else "Evaluation failed to produce result." # Store potential eval errors
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
                # Ensure generated code is in the final status if it exists
                if generated_code_for_llm:
                    BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm

        print(f"Finished background compression benchmark task {task_id}: {llm_name} - {algorithm_label}")

    except Exception as e:
        print(f"Error in background compression benchmark task {task_id} ({llm_name} - {algorithm_label}): {e}\n{traceback.format_exc()}")
        error_message = f"Benchmark execution failed: {e}"
        with STATUS_LOCK:
            # Ensure the status dict exists before updating
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = 'Error'
                BENCHMARK_STATUS[task_id]['error'] = error_message
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
                # Store generated code if it was generated before the error
                if generated_code_for_llm:
                     BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
            else:
                print(f"Error: Task ID {task_id} not found in BENCHMARK_STATUS during exception handling.")


        # Optionally save error state to DB - structure matches compression results
        error_result = {
            'llm': llm_name,
            'algorithm': algorithm_label,
            'error': error_message,
            'correctness': 0, # Mark as incorrect due to error
            'avg_compression_time_ms': None,
            'avg_decompression_time_ms': None,
            'avg_compression_ratio': None,
            'generated_code': generated_code_for_llm # Include code if generated
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
    """Endpoint to trigger a new compression benchmark run."""
    # Get parameters from form
    llm_name = request.form.get('llm')

    # Validate parameters
    if not llm_name or llm_name not in AVAILABLE_LLMS:
        flash(f"Invalid or missing LLM selected: {llm_name}", "error")
        return redirect(url_for('admin'))

    # Check if test suite loaded correctly before starting
    if TEST_SUITE_LOAD_ERROR:
        flash(f"Cannot start benchmark: Test suite failed to load ({TEST_SUITE_LOAD_ERROR})", "error")
        return redirect(url_for('admin'))

    # Determine the algorithm label for the flash message
    algorithm_label = benchmark.BENCHMARKED_ALGORITHM_LABEL

    # Generate a unique ID for this benchmark task
    task_id = str(uuid.uuid4())

    # Run benchmark in a background thread
    thread = threading.Thread(target=run_benchmark_background, args=(task_id, llm_name))
    thread.daemon = True # Allow app to exit even if background threads are running
    thread.start()

    flash(f"Compression benchmark task {task_id} started for {llm_name} ({algorithm_label}).", "info")
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
