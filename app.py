from flask import Flask, render_template, request, redirect, url_for, flash
import database
import benchmark
import threading # To run benchmarks in the background

app = Flask(__name__)
# Required for flash messages
app.secret_key = 'super secret key' # Change this to a random secret key, maybe from env var

# Define available LLMs and Algorithms (can be moved to config later)
AVAILABLE_LLMS = ["dummy_llm"] # Add real LLM identifiers here
AVAILABLE_ALGORITHMS = ["Bubble Sort", "Quick Sort", "Merge Sort", "Insertion Sort"]


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
    return render_template('admin.html', llms=AVAILABLE_LLMS, algorithms=AVAILABLE_ALGORITHMS)


def run_benchmark_background(llm_name, algorithm_name):
    """Function to run benchmark in a separate thread."""
    print(f"Starting background benchmark: {llm_name} - {algorithm_name}")
    try:
        result = benchmark.run_single_benchmark(llm_name, algorithm_name)
        database.save_result(result)
        print(f"Finished background benchmark: {llm_name} - {algorithm_name}")
    except Exception as e:
        print(f"Error in background benchmark ({llm_name} - {algorithm_name}): {e}")
        # Optionally save error state to DB
        error_result = {
            'llm': llm_name,
            'algorithm': algorithm_name,
            'error': f"Benchmark execution failed: {e}",
            'correctness': None, 'avg_time_ms': None, 'baseline_avg_time_ms': None
        }
        try:
            database.save_result(error_result)
        except Exception as db_e:
            print(f"Failed to save error result to DB: {db_e}")


@app.route('/run', methods=['POST'])
def run_benchmark():
    """Endpoint to trigger a new benchmark run."""
    # Get parameters from form
    llm_name = request.form.get('llm')
    algorithm_name = request.form.get('algorithm')

    # Validate parameters
    if not llm_name or llm_name not in AVAILABLE_LLMS:
        flash(f"Invalid or missing LLM selected: {llm_name}", "error")
        return redirect(url_for('admin'))
    if not algorithm_name or algorithm_name not in AVAILABLE_ALGORITHMS:
        flash(f"Invalid or missing Algorithm selected: {algorithm_name}", "error")
        return redirect(url_for('admin'))

    # Run benchmark in a background thread to avoid blocking the web request
    thread = threading.Thread(target=run_benchmark_background, args=(llm_name, algorithm_name))
    thread.start()

    flash(f"Benchmark started for {llm_name} - {algorithm_name}. Results will appear shortly.", "info")
    return redirect(url_for('index')) # Redirect to results page

if __name__ == '__main__':
    # Initialize the database if it doesn't exist
    database.init_db()
    app.run(debug=True) # debug=True for development, remove for production
