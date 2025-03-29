"""
Module for database interactions using SQLite.
Stores benchmark results.
"""

import sqlite3
import json # For handling performance details

DATABASE_FILE = 'benchmark_results.db'

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            llm TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            correctness REAL, -- Percentage (0-100)
            avg_time_ms REAL, -- Overall average execution time in milliseconds for LLM code (on correct runs)
            baseline_avg_time_ms REAL, -- Overall average execution time in milliseconds for Python's sorted()
            performance_details TEXT, -- JSON blob containing per-category performance breakdown
            error TEXT, -- Store any errors encountered during generation or execution
            generated_code TEXT -- Optionally store the full generated code
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def save_result(result_data: dict):
    """Saves a single benchmark result to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Serialize performance details to JSON
    details_json = None
    if 'performance_details' in result_data and result_data['performance_details']:
        try:
            details_json = json.dumps(result_data['performance_details'])
        except TypeError as e:
            print(f"Warning: Could not serialize performance_details: {e}. Storing as NULL.")
            # Optionally store the error message instead or log it more formally
            # details_json = json.dumps({'serialization_error': str(e)})


    cursor.execute('''
        INSERT INTO results (llm, algorithm, correctness, avg_time_ms, baseline_avg_time_ms, performance_details, error, generated_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result_data.get('llm'),
        result_data.get('algorithm'),
        result_data.get('correctness'),
        result_data.get('avg_time_ms'),
        result_data.get('baseline_avg_time_ms'),
        details_json, # Use the serialized JSON string
        result_data.get('error'),
        result_data.get('generated_code')
    ))
    conn.commit()
    conn.close()
    print(f"Result saved for {result_data.get('llm')} - {result_data.get('algorithm')}")

def get_all_results() -> list[dict]:
    """Retrieves all benchmark results from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()

    # Convert Row objects to standard dictionaries and deserialize performance_details
    results_list = []
    for row in rows:
        result_dict = dict(row)
        details_json = result_dict.get('performance_details')
        if details_json:
            try:
                result_dict['performance_details'] = json.loads(details_json)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not decode performance_details JSON for result ID {result_dict.get('id')}: {e}")
                # Keep the raw string or set to an error indicator
                result_dict['performance_details'] = {'decoding_error': str(e), 'raw': details_json}
        else:
             # Ensure the key exists, even if null in DB
             result_dict['performance_details'] = None # Or {} if preferred
        results_list.append(result_dict)

    return results_list

# Example usage
if __name__ == '__main__':
    init_db()
    # Example data saving
    test_result = {
        'llm': 'test_llm',
        'algorithm': 'test_sort',
        'correctness': 90.0,
        'avg_time_ms': 15.5,
        'baseline_avg_time_ms': 1.2, # Example baseline time
        'error': None,
        'generated_code': 'def sort_algorithm(arr): return sorted(arr)'
    }
    # save_result(test_result) # Uncomment to save test data if needed after schema change
    all_data = get_all_results()
    print("\nAll results from DB:")
    for row in all_data:
        print(dict(row))
