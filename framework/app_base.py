"""
Base Flask application structure for benchmark web UI.
Handles status tracking, background tasks, and common routes.
Specific benchmark apps will configure and extend this.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import threading
import uuid
from collections import deque
import time
import os
import json
import traceback # For logging errors
from .database import BenchmarkDB # Use the framework DB handler

# --- Constants ---
MAX_PROGRESS_UPDATES = 100 # Store more updates

# --- Benchmark Status Tracking ---
BENCHMARK_STATUS = {}
STATUS_LOCK = threading.Lock()

class BenchmarkApp:
    """Encapsulates the Flask app and benchmark logic."""

    def __init__(self, config):
        """
        Initializes the BenchmarkApp.

        Args:
            config: A configuration object/module for the specific benchmark.
                    Expected attributes: BENCHMARK_NAME, AVAILABLE_LLMS,
                                         DATABASE_FILE, TEST_SUITE_FILE,
                                         run_benchmark_func (function to start background task),
                                         benchmark_runner (instance of BenchmarkRunner),
                                         test_suite_loader_func (function to load/check test suite)
        """
        self.app = Flask(__name__, template_folder='templates') # Use framework templates
        self.app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key-change-me')
        self.config = config
        self.db = BenchmarkDB(self.config.DATABASE_FILE)

        # --- Global Test Suite Status ---
        self.test_suite_data = None
        self.test_suite_load_error = None
        self._initialize_test_suite() # Load or check test suite on startup

        # --- Register Routes ---
        self.register_routes()

    def _initialize_test_suite(self):
        """Loads the test suite using the configured function."""
        print(f"Framework: Initializing test suite using function: {getattr(self.config, 'test_suite_loader_func', None)}")
        loader_func = getattr(self.config, 'test_suite_loader_func', None)
        if loader_func and callable(loader_func):
             try:
                 # The loader function should handle generation if needed and return loaded data or raise error
                 self.test_suite_data = loader_func()
                 self.test_suite_load_error = None
                 print(f"Framework: Test suite loaded successfully for {self.config.BENCHMARK_NAME}.")
             except Exception as e:
                 self.test_suite_load_error = f"Failed to load/generate test suite '{self.config.TEST_SUITE_FILE}': {e}"
                 print(f"CRITICAL ERROR: {self.test_suite_load_error}\n{traceback.format_exc()}")
                 self.test_suite_data = None
        else:
             self.test_suite_load_error = "Configuration error: test_suite_loader_func not defined or not callable."
             print(f"ERROR: {self.test_suite_load_error}")
             self.test_suite_data = None


    def register_routes(self):
        """Registers Flask routes."""
        self.app.route('/')(self.index)
        self.app.route('/admin')(self.admin)
        self.app.route('/run', methods=['POST'])(self.run_benchmark)
        self.app.route('/benchmark_progress/<task_id>')(self.benchmark_progress)
        self.app.route('/benchmark_status/<task_id>')(self.benchmark_status)

    def index(self):
        """Display the benchmark results page."""
        try:
            current_results = self.db.get_all_results()
        except Exception as e:
            flash(f"Error fetching results from database '{self.db.db_file}': {e}", "error")
            current_results = []
        return render_template('index.html',
                               results=current_results,
                               test_suite_error=self.test_suite_load_error,
                               benchmark_name=self.config.BENCHMARK_NAME)

    def admin(self):
        """Display the admin page to run benchmarks."""
        return render_template('admin.html',
                               llms=self.config.AVAILABLE_LLMS,
                               test_suite_error=self.test_suite_load_error,
                               benchmark_name=self.config.BENCHMARK_NAME)

    def run_benchmark(self):
        """Endpoint to trigger a new benchmark run."""
        llm_name = request.form.get('llm')

        # Validate parameters
        if not llm_name or llm_name not in self.config.AVAILABLE_LLMS:
            flash(f"Invalid or missing LLM selected: {llm_name}", "error")
            return redirect(url_for('admin'))

        # Check if test suite loaded correctly before starting
        if self.test_suite_load_error:
            flash(f"Cannot start benchmark: Test suite failed to load ({self.test_suite_load_error})", "error")
            return redirect(url_for('admin'))
        if self.test_suite_data is None:
             flash(f"Cannot start benchmark: Test suite data is not available.", "error")
             return redirect(url_for('admin'))


        # Generate a unique ID for this benchmark task
        task_id = str(uuid.uuid4())

        # Get the background task function from config
        run_background_func = getattr(self.config, 'run_benchmark_func', None)
        if not run_background_func or not callable(run_background_func):
             flash(f"Configuration error: Background task function not set.", "error")
             return redirect(url_for('admin'))

        # Run benchmark in a background thread
        # Pass necessary components: task_id, llm_name, config, db_handler, test_suite_data
        thread = threading.Thread(target=run_background_func, args=(
            task_id,
            llm_name,
            self.config,
            self.db, # Pass the initialized DB handler
            self.test_suite_data # Pass the loaded test data
        ))
        thread.daemon = True
        thread.start()

        flash(f"{self.config.BENCHMARK_NAME} task {task_id} started for {llm_name}.", "info")
        return redirect(url_for('benchmark_progress', task_id=task_id))

    def benchmark_progress(self, task_id):
        """Display the progress page for a specific benchmark task."""
        # Check if task exists to provide better initial info?
        with STATUS_LOCK:
            status = BENCHMARK_STATUS.get(task_id)
        if not status:
             # Optionally flash a message or handle differently
             pass
        return render_template('progress.html', task_id=task_id, benchmark_name=self.config.BENCHMARK_NAME)

    def benchmark_status(self, task_id):
        """API endpoint to get the current status of a benchmark task."""
        with STATUS_LOCK:
            status = BENCHMARK_STATUS.get(task_id)

        if not status:
            return jsonify({'status': 'Not Found', 'message': f'Task ID {task_id} not found.'}), 404

        # Convert deque to list for JSON serialization
        status_copy = status.copy()
        if 'progress' in status_copy and isinstance(status_copy['progress'], deque):
            status_copy['progress'] = list(status_copy['progress'])

        return jsonify(status_copy)

    def run(self, **kwargs):
        """Runs the Flask development server."""
        self.app.run(**kwargs)

# --- Background Task Management (Common Logic) ---

def run_benchmark_background_base(task_id, llm_name, config, db: BenchmarkDB, test_suite_data, benchmark_runner):
    """
    Base function for running a benchmark in the background.
    Handles status updates, code generation, calling the benchmark runner, and saving results.
    """
    benchmark_name = config.BENCHMARK_NAME
    print(f"Framework: Starting background task {task_id}: {benchmark_name} - {llm_name}")

    def progress_callback(update_data):
        """Callback function passed to benchmark runner."""
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                update_data['timestamp'] = time.time()
                BENCHMARK_STATUS[task_id]['progress'].append(update_data)
                # Update overall status fields
                BENCHMARK_STATUS[task_id]['status'] = update_data.get('status', BENCHMARK_STATUS[task_id]['status'])
                BENCHMARK_STATUS[task_id]['current_case'] = update_data.get('current_case', BENCHMARK_STATUS[task_id].get('current_case'))
                BENCHMARK_STATUS[task_id]['total_cases'] = update_data.get('total_cases', BENCHMARK_STATUS[task_id].get('total_cases'))
                BENCHMARK_STATUS[task_id]['last_update'] = update_data['timestamp']
                if 'generated_code' in update_data:
                     BENCHMARK_STATUS[task_id]['generated_code'] = update_data['generated_code']

    # Initialize status
    with STATUS_LOCK:
        BENCHMARK_STATUS[task_id] = {
            'task_id': task_id,
            'llm': llm_name,
            'algorithm': benchmark_name, # Use benchmark_name from config
            'status': 'Initializing',
            'start_time': time.time(),
            'end_time': None,
            'current_case': 0,
            'total_cases': None,
            'progress': deque(maxlen=MAX_PROGRESS_UPDATES),
            'final_result': None,
            'error': None,
            'generated_code': None,
            'last_update': time.time()
        }

    generated_code_for_llm = None
    result = None
    is_baseline = llm_name in getattr(config, 'BASELINE_CODE_SNIPPETS', {})

    try:
        # --- Get Code: Generate or Use Baseline ---
        if is_baseline:
            progress_callback({'status': 'Using Baseline Code', 'category': 'Setup'})
            generated_code_for_llm = config.BASELINE_CODE_SNIPPETS[llm_name]
            print(f"Task {task_id}: Using baseline code for '{llm_name}'.")
            # Update status immediately with baseline code
            with STATUS_LOCK:
                 if task_id in BENCHMARK_STATUS:
                     BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
                     BENCHMARK_STATUS[task_id]['status'] = 'Evaluating Baseline Code...'
                     BENCHMARK_STATUS[task_id]['last_update'] = time.time()
            progress_callback({
                'status': 'Evaluating Baseline Code...', 'category': 'Setup',
                'generated_code': generated_code_for_llm
            })
        else:
            # --- Generate Code using LLM ---
            progress_callback({'status': 'Generating Code', 'category': 'Setup'})
            # Get prompt generation function from config
            prompt_func = getattr(config, 'prompt_generator_func', None)
            if not prompt_func or not callable(prompt_func):
                 raise ValueError("Configuration error: prompt_generator_func not defined or not callable.")

            prompt = prompt_func() # Call the benchmark-specific prompt generator
            print(f"Task {task_id}: Generating code using {llm_name}...")
            # Use the framework's llm_interface
            from . import llm_interface # Local import
            generated_code_for_llm = llm_interface.generate_code(llm_name, prompt)

            if not generated_code_for_llm:
                raise ValueError(f"LLM '{llm_name}' failed to generate code.")

            # Update status with generated code
            with STATUS_LOCK:
                if task_id in BENCHMARK_STATUS:
                    BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
                    BENCHMARK_STATUS[task_id]['status'] = 'Code Generated, Evaluating...'
                    BENCHMARK_STATUS[task_id]['last_update'] = time.time()
            progress_callback({
                'status': 'Code Generated, Evaluating...', 'category': 'Setup',
                'generated_code': generated_code_for_llm
            })
            print(f"Task {task_id}: Code generated. Starting evaluation.")

        # --- Run Benchmark Evaluation using the Runner ---
        # The runner needs the code, test suite, and config details
        result = benchmark_runner.run_evaluation(
            generated_code=generated_code_for_llm,
            test_suite_data=test_suite_data,
            progress_callback=progress_callback
            # Runner uses its own config internally now
        )

        # --- Prepare result for saving ---
        if result:
            result['benchmark_name'] = benchmark_name # Ensure benchmark name is in result
            result['llm'] = llm_name
            result['generated_code'] = generated_code_for_llm # Add code to result dict
            db.save_result(result)
        else:
             print(f"Task {task_id}: No result dictionary generated by runner, skipping database save.")


        # --- Update final status ---
        final_status_str = 'Completed' if not (result and result.get('error')) else 'Error'
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = final_status_str
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['final_result'] = result
                BENCHMARK_STATUS[task_id]['error'] = result.get('error') if result else "Runner failed to produce result."
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
                # Ensure generated code is stored
                if generated_code_for_llm:
                     BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm

        print(f"Framework: Finished background task {task_id}: {benchmark_name} - {llm_name} ({final_status_str})")

    except Exception as e:
        print(f"Framework: Error in background task {task_id} ({benchmark_name} - {llm_name}): {e}\n{traceback.format_exc()}")
        error_message = f"Benchmark execution failed: {e}"
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = 'Error'
                BENCHMARK_STATUS[task_id]['error'] = error_message
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
                if generated_code_for_llm: # Store code if generated before error
                     BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm

        # Save error state to DB
        error_result = {
            'benchmark_name': benchmark_name,
            'llm': llm_name,
            'error': error_message,
            'correctness': 0, # Error means incorrect
            'avg_time_ms': None,
            'avg_secondary_time_ms': None,
            'avg_ratio': None,
            'generated_code': generated_code_for_llm
        }
        db.save_result(error_result)
        # Update status dict with error result
        with STATUS_LOCK:
             if task_id in BENCHMARK_STATUS:
                 BENCHMARK_STATUS[task_id]['final_result'] = error_result
