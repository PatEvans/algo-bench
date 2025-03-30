"""
Main Flask application entry point.
Discovers and registers benchmark blueprints.
"""

import os
import sys
import importlib
import traceback # Import traceback
from flask import Flask, render_template, url_for

# --- Configuration ---
# Define benchmark directories relative to the project root
BENCHMARK_DIRS = ['compress-bench'] # Add other benchmark directories here
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- App Creation ---
app = Flask(__name__, template_folder='templates') # Use root templates dir
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key-change-me-in-prod')

# --- Blueprint Discovery and Registration ---
registered_benchmarks = []

# Loop through benchmark directories
for bench_dir in BENCHMARK_DIRS:
    try:
        module_name = f"{bench_dir}.app"
        # Dynamically import the app module from the benchmark directory
        bench_app_module = importlib.import_module(module_name)

        # Check if the module has the expected factory function
        if hasattr(bench_app_module, 'create_blueprint') and callable(bench_app_module.create_blueprint):
            # Call the factory function to get the blueprint and its config
            blueprint, bench_config = bench_app_module.create_blueprint()
            # Register the blueprint with the main app
            app.register_blueprint(blueprint)
            print(f"Registered blueprint '{blueprint.name}' from {module_name} at prefix '{blueprint.url_prefix}'")
            registered_benchmarks.append({
                'name': getattr(bench_config, 'BENCHMARK_NAME', blueprint.name),
                'endpoint': f"{blueprint.name}.index" # Store endpoint name instead of URL
            })
        else:
            print(f"Warning: Module {module_name} does not have a callable 'create_blueprint' function. Skipping.")

    except ImportError as e:
        print(f"ERROR: Could not import module {module_name}: {e}. Check for syntax errors or missing dependencies in the module and its imports.")
        print(traceback.format_exc()) # Print full traceback for import errors
    except RuntimeError as e:
        # Catch RuntimeErrors specifically, often from runner init
        print(f"ERROR: Runtime error during blueprint creation/registration from {module_name}: {e}. Is Docker running or configured correctly?")
        print(traceback.format_exc())
    except Exception as e:
            print(f"ERROR: Unexpected error registering blueprint from {module_name}: {e}.")
            print(traceback.format_exc()) # Print full traceback for other errors

# --- Main Route ---
@app.route('/')
def main_index():
    """Displays the main index page linking to benchmarks."""
    # Generate URLs within the request context
    benchmarks_with_urls = []
    for benchmark in registered_benchmarks:
        try:
            benchmarks_with_urls.append({
                'name': benchmark['name'],
                'url': url_for(benchmark['endpoint'])
            })
        except Exception as e:
            print(f"Error generating URL for endpoint {benchmark.get('endpoint', 'N/A')}: {e}")
            # Optionally skip or add a placeholder if URL generation fails
            # benchmarks_with_urls.append({'name': benchmark['name'], 'url': '#error'})

    return render_template('main_index.html', benchmarks=benchmarks_with_urls)

# --- Run ---
if __name__ == '__main__':
    # Use a different port than the individual benchmark apps might use
    app_port = int(os.environ.get("MAIN_APP_PORT", 5001))
    print(f"Starting Main Benchmark server on port {app_port}...")
    # Set debug=False for production use
    app.run(debug=True, host='0.0.0.0', port=app_port)
