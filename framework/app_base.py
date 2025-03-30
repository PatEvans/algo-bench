"""
Base Blueprint structure for benchmark web UI components.
Handles status tracking, background tasks, and common routes within a Blueprint.
Specific benchmark apps will configure and instantiate this.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
import threading
import uuid
from collections import deque
import time
import os
import json
import traceback # For logging errors
import pprint # For detailed logging
import math # For checking inf/nan
import sys # For stderr access
from framework.database import BenchmarkDB # Use the framework DB handler

# --- Constants ---
MAX_PROGRESS_UPDATES = 100 # Store more updates

# --- Benchmark Status Tracking ---
BENCHMARK_STATUS = {}
STATUS_LOCK = threading.Lock()

class BenchmarkBlueprint:
    """
    Encapsulates the Blueprint, routes, and logic for a specific benchmark.
    Designed to be created and registered by a main Flask application.
    """

    def __init__(self, blueprint_name: str, import_name: str, config, url_prefix: str = None):
        """
        Initializes the BenchmarkBlueprint.

        Args:
            blueprint_name: Unique name for the blueprint (e.g., 'compress').
            import_name: The import name for the Blueprint (usually __name__).
            config: A configuration object/module for the specific benchmark.
                    Expected attributes: BENCHMARK_NAME, AVAILABLE_LLMS,
                                         DATABASE_FILE, TEST_SUITE_FILE,
                                         run_benchmark_func (function to start background task),
                                         benchmark_runner (instance of BenchmarkRunner),
                                         test_suite_loader_func (function to load/check test suite)
            url_prefix: URL prefix for this benchmark's routes (e.g., '/compress').
        """
        # Use framework templates relative to the framework directory
        framework_dir = os.path.dirname(os.path.abspath(__file__))
        template_folder = os.path.join(framework_dir, 'templates')

        self.bp = Blueprint(blueprint_name, import_name,
                            template_folder=template_folder,
                            url_prefix=url_prefix)
        self.blueprint_name = blueprint_name
        self.config = config
        self.db = BenchmarkDB(self.config.DATABASE_FILE) # DB is specific to this benchmark instance

        # --- Test Suite Status (specific to this benchmark instance) ---
        self.test_suite_data = None
        self.test_suite_load_error = None
        self._initialize_test_suite() # Load or check test suite on instantiation

        # --- Register Routes on the Blueprint ---
        self._register_routes()

    def get_blueprint(self):
        """Returns the configured Flask Blueprint instance."""
        return self.bp

    def _initialize_test_suite(self):
        """Loads the test suite using the configured function."""
        # print(f"Framework [{self.blueprint_name}]: Initializing test suite using function: {getattr(self.config, 'test_suite_loader_func', None)}") # DEBUG
        loader_func = getattr(self.config, 'test_suite_loader_func', None)
        if loader_func and callable(loader_func):
             try:
                 # The loader function should handle generation if needed and return loaded data or raise error
                 self.test_suite_data = loader_func() # Store loaded data instance variable
                 self.test_suite_load_error = None
                 print(f"Framework [{self.blueprint_name}]: Test suite loaded successfully for {self.config.BENCHMARK_NAME}.")
             except Exception as e:
                 self.test_suite_load_error = f"Failed to load/generate test suite '{self.config.TEST_SUITE_FILE}': {e}"
                 print(f"CRITICAL ERROR [{self.blueprint_name}]: {self.test_suite_load_error}\n{traceback.format_exc()}")
                 self.test_suite_data = None
        else:
             self.test_suite_load_error = "Configuration error: test_suite_loader_func not defined or not callable."
             print(f"ERROR [{self.blueprint_name}]: {self.test_suite_load_error}")
             self.test_suite_data = None


    def _register_routes(self):
        """Registers routes on the blueprint."""
        # Use self.bp.route instead of self.app.route
        self.bp.route('/')(self.index)
        self.bp.route('/admin')(self.admin)
        self.bp.route('/run', methods=['POST'])(self.run_benchmark)
        # Progress/Status routes don't need the benchmark prefix if task_id is globally unique
        # Let's keep them under the blueprint for now for consistency
        self.bp.route('/progress/<task_id>')(self.benchmark_progress)
        self.bp.route('/status/<task_id>')(self.benchmark_status)

    # --- Route Handlers ---
    # These methods remain largely the same but use the blueprint context

    def index(self):
        """Display the benchmark results page (within the blueprint)."""
        try:
            current_results = self.db.get_all_results()
        except Exception as e:
            flash(f"Error fetching results from database '{self.db.db_file}': {e}", "error")
            current_results = []
        # Render the template associated with this blueprint
        return render_template('index.html',
                               results=current_results,
                               test_suite_error=self.test_suite_load_error,
                               benchmark_name=self.config.BENCHMARK_NAME,
                               blueprint_name=self.blueprint_name) # Pass blueprint name for url_for

    def admin(self):
        """Display the admin page to run benchmarks (within the blueprint)."""
        return render_template('admin.html',
                               llms=self.config.AVAILABLE_LLMS,
                               test_suite_error=self.test_suite_load_error,
                               benchmark_name=self.config.BENCHMARK_NAME,
                               blueprint_name=self.blueprint_name) # Pass blueprint name for url_for

    def run_benchmark(self):
        """Endpoint to trigger a new benchmark run (within the blueprint)."""
        llm_name = request.form.get('llm')

        # Validate parameters
        if not llm_name or llm_name not in self.config.AVAILABLE_LLMS:
            flash(f"Invalid or missing LLM selected: {llm_name}", "error")
            # Use relative url_for within blueprint: '.admin'
            return redirect(url_for(f'.admin'))

        # Check if test suite loaded correctly before starting
        if self.test_suite_load_error:
            flash(f"Cannot start benchmark: Test suite failed to load ({self.test_suite_load_error})", "error")
            return redirect(url_for('.admin'))
        if self.test_suite_data is None:
             flash(f"Cannot start benchmark: Test suite data is not available.", "error")
             return redirect(url_for('.admin'))


        # Generate a unique ID for this benchmark task
        task_id = str(uuid.uuid4())

        # Get the background task function from config
        run_background_func = getattr(self.config, 'run_benchmark_func', None)
        if not run_background_func or not callable(run_background_func):
             flash(f"Configuration error: Background task function not set.", "error")
             return redirect(url_for('.admin'))

        # Run benchmark in a background thread
        # Pass necessary components: task_id, llm_name, config, db_handler, test_suite_data
        # The background function needs access to the app context for logging etc.
        # It's generally better practice to use a task queue (Celery, RQ) for background tasks,
        # but threading is simpler for this example.
        thread = threading.Thread(target=run_background_func, args=(
            task_id,
            llm_name,
            self.config,
            self.db, # Pass the initialized DB handler
            self.test_suite_data # Pass the loaded test data
        ))
        thread.daemon = True # Allow main thread to exit even if background task is running
        thread.start()

        flash(f"{self.config.BENCHMARK_NAME} task {task_id} started for {llm_name}.", "info")
        # Redirect to the progress page within this blueprint
        return redirect(url_for('.benchmark_progress', task_id=task_id))

    def benchmark_progress(self, task_id):
        """Display the progress page for a specific benchmark task (within the blueprint)."""
        # Check if task exists to provide better initial info?
        with STATUS_LOCK:
            status = BENCHMARK_STATUS.get(task_id)
        # Render the template associated with this blueprint
        return render_template('progress.html',
                               task_id=task_id,
                               benchmark_name=self.config.BENCHMARK_NAME,
                               blueprint_name=self.blueprint_name) # Pass blueprint name for url_for

    def benchmark_status(self, task_id):
        """API endpoint to get the current status of a benchmark task (within the blueprint)."""
        with STATUS_LOCK:
            status = BENCHMARK_STATUS.get(task_id)

        if not status:
            return jsonify({'status': 'Not Found', 'message': f'Task ID {task_id} not found.'}), 404

        # Convert deque to list for JSON serialization
        status_copy = status.copy()
        if 'progress' in status_copy and isinstance(status_copy['progress'], deque):
            # Convert deque to list *before* logging/serializing
            status_copy['progress'] = list(status_copy['progress'])

        # --- Enhanced Logging --- # DEBUG - Removed
        # print(f"DEBUG: Preparing to jsonify status for task {task_id}. Content:")
        # Use pprint for potentially large/nested dicts
        # pprint.pprint(status_copy, stream=sys.stderr)
        # --- End Enhanced Logging ---

        try:
            # Attempt to jsonify the potentially complex status dictionary
            return jsonify(status_copy)
        except TypeError as e:
            # Log the error and return a simplified status indicating the serialization issue
            print(f"ERROR: Failed to jsonify status for task {task_id}: {e}\n{traceback.format_exc()}")

            # --- Granular Cleanup Attempt ---
            cleaned_status = status_copy.copy() # Work on a copy

            # 1. Check final_result
            if 'final_result' in cleaned_status:
                try:
                    json.dumps(cleaned_status['final_result']) # Test serialization
                except TypeError as res_err:
                    print(f"DEBUG: TypeError serializing final_result for task {task_id}: {res_err}")
                    # Replace with string representation only if it fails
                    cleaned_status['final_result'] = f"Error serializing result: {res_err}. Raw: {repr(cleaned_status['final_result'])}"

            # 2. Check progress list items
            if 'progress' in cleaned_status and isinstance(cleaned_status['progress'], list):
                cleaned_progress = []
                for i, item in enumerate(cleaned_status['progress']):
                    try:
                        json.dumps(item) # Test serialization of individual item
                        cleaned_progress.append(item)
                    except TypeError as prog_err:
                        print(f"DEBUG: TypeError serializing progress item {i} for task {task_id}: {prog_err}")
                        cleaned_progress.append(f"Error serializing progress item {i}: {prog_err}. Raw: {repr(item)}")
                cleaned_status['progress'] = cleaned_progress

            # 3. Check generated_code (less likely, but possible if not string)
            if 'generated_code' in cleaned_status and not isinstance(cleaned_status['generated_code'], str):
                 # print(f"DEBUG: generated_code is not a string for task {task_id} (type: {type(cleaned_status['generated_code'])}). Converting.") # DEBUG
                 cleaned_status['generated_code'] = repr(cleaned_status['generated_code'])

            # --- End Granular Cleanup Attempt ---

            # Return the modified, hopefully serializable, status
            try:
                 # print(f"DEBUG: Attempting jsonify again for task {task_id} after cleanup.") # DEBUG
                 return jsonify(cleaned_status)
            except Exception as final_json_err:
                 # If it STILL fails, return a very basic error
                 print(f"ERROR: STILL Failed to jsonify status for task {task_id} after cleanup: {final_json_err}")
                 # Log the cleaned status that still failed - Removed for less noise
                 # print("--- Cleaned Status that Failed ---")
                 # pprint.pprint(cleaned_status, stream=sys.stderr)
                 # print("---------------------------------")
                 return jsonify({
                     'task_id': task_id,
                     'status': 'Error',
                     'error': f'Failed to serialize status details after cleanup: {final_json_err}. Original error: {e}',
                     'last_update': status.get('last_update') # Use original status for last_update
                 }), 500

# --- Background Task Management (Remains mostly the same, but uses config passed in) ---
# This function is called by the specific benchmark's app.py wrapper function

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
            from framework import llm_interface # Absolute import
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
            try:
                save_success = db.save_result(result)
                if not save_success:
                     print(f"Task {task_id}: db.save_result indicated failure.")
                     # Optionally update the result error before final status update
                     result['error'] = result.get('error', '') + " [DB Save Failed]"
            except Exception as db_err:
                 print(f"Task {task_id}: CRITICAL ERROR saving result to database: {db_err}\n{traceback.format_exc()}")
                 # Update the result error before final status update
                 result['error'] = result.get('error', '') + f" [DB Save Exception: {db_err}]"
        else:
             print(f"Task {task_id}: No result dictionary generated by runner, skipping database save.")
             # Ensure result is a dict for final status update consistency
             result = {'error': 'Runner failed to produce result.', 'correctness': 0}

        # --- Clean the result dictionary before storing ---
        cleaned_result = {}
        if result:
            try:
                # Attempt to serialize/deserialize to clean non-standard types
                # Use a helper similar to the wrapper's final print logic
                def clean_value(v):
                    if isinstance(v, float):
                        if math.isinf(v): return "Infinity" if v > 0 else "-Infinity"
                        if math.isnan(v): return "NaN"
                    elif isinstance(v, bytes):
                        # Try decoding, fallback to repr for non-utf8 bytes
                        try:
                            return v.decode('utf-8')
                        except UnicodeDecodeError:
                            return repr(v)
                    # Add other type conversions if needed (e.g., numpy types)
                    return v

                # Recursively clean the dictionary
                def clean_dict_recursive(d):
                    if isinstance(d, dict):
                        return {k: clean_dict_recursive(v) for k, v in d.items()}
                    elif isinstance(d, list):
                        return [clean_dict_recursive(i) for i in d]
                    else:
                        return clean_value(d)

                cleaned_result = clean_dict_recursive(result)
                # Verify it's serializable now
                json.dumps(cleaned_result)
            except Exception as clean_err:
                 print(f"ERROR Task {task_id}: Failed to clean/serialize final result dict: {clean_err}")
                 cleaned_result = {'error': f"Failed to serialize result: {clean_err}. Original: {repr(result)}", 'correctness': 0}
                 # Ensure essential keys are present if possible
                 cleaned_result['benchmark_name'] = result.get('benchmark_name', benchmark_name)
                 cleaned_result['llm'] = result.get('llm', llm_name)
                 cleaned_result['generated_code'] = result.get('generated_code') # Keep original code if possible
        else:
             cleaned_result = {'error': 'Runner failed to produce result.', 'correctness': 0}


        # --- Update final status ---
        # Determine final status based on the *cleaned* result dict
        final_status_str = 'Completed' if cleaned_result.get('correctness') == 1 and not cleaned_result.get('error') else 'Error'
        with STATUS_LOCK:
            if task_id in BENCHMARK_STATUS:
                BENCHMARK_STATUS[task_id]['status'] = final_status_str
                BENCHMARK_STATUS[task_id]['end_time'] = time.time()
                BENCHMARK_STATUS[task_id]['final_result'] = cleaned_result # Store the cleaned result
                BENCHMARK_STATUS[task_id]['error'] = cleaned_result.get('error') # Use error from cleaned result
                BENCHMARK_STATUS[task_id]['last_update'] = time.time()
                # Ensure generated code is stored (use original if cleaning failed)
                if 'generated_code' not in cleaned_result and generated_code_for_llm:
                     BENCHMARK_STATUS[task_id]['generated_code'] = generated_code_for_llm
                elif 'generated_code' in cleaned_result:
                     BENCHMARK_STATUS[task_id]['generated_code'] = cleaned_result['generated_code']


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
