"""
Module for database interactions using SQLite.
Stores compression benchmark results.
"""

import sqlite3
import json # For handling performance details (though less relevant now)

DATABASE_FILE = 'compression_benchmark_results.db' # Changed DB filename

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
            algorithm TEXT NOT NULL, -- e.g., "LLM Compression"
            correctness INTEGER, -- 1 for all correct, 0 if any failed
            avg_compression_time_ms REAL, -- Average time for compress() in milliseconds
            avg_decompression_time_ms REAL, -- Average time for decompress() in milliseconds
            avg_compression_ratio REAL, -- Average ratio (original_size / compressed_size)
            -- performance_details TEXT, -- JSON blob (Maybe remove or adapt if needed later)
            error TEXT, -- Store any critical errors encountered during the run
            generated_code TEXT -- Store the full generated code (compress + decompress)
        )
    ''')
    # Consider adding an index on timestamp or llm if querying becomes frequent
    # cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm ON results (llm)')
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
        INSERT INTO results (llm, algorithm, correctness, avg_compression_time_ms, avg_decompression_time_ms, avg_compression_ratio, error, generated_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result_data.get('llm'),
        result_data.get('algorithm'),
        result_data.get('correctness'), # Should be 0 or 1
        result_data.get('avg_compression_time_ms'),
        result_data.get('avg_decompression_time_ms'),
        result_data.get('avg_compression_ratio'),
        # details_json, # Removed for now
        result_data.get('error'),
        result_data.get('generated_code')
    ))
    conn.commit()
    # Ensure connection is closed even if commit fails? (Context manager preferred)
    conn.close()
    print(f"Result saved for {result_data.get('llm')} - {result_data.get('algorithm')}")

def get_all_results() -> list[dict]:
    """Retrieves all benchmark results from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()

    # Convert Row objects to standard dictionaries
    # No complex deserialization needed for now unless performance_details is re-added
    results_list = [dict(row) for row in rows]
    return results_list

# Example usage
if __name__ == '__main__':
    init_db()
    # Example data saving
    test_result = {
        'llm': 'test_llm',
        'algorithm': 'test_sort',
        'llm': 'test_compress_llm',
        'algorithm': 'LLM Compression',
        'correctness': 1, # Example: 1 (True)
        'avg_compression_time_ms': 55.2,
        'avg_decompression_time_ms': 25.8,
        'avg_compression_ratio': 2.5,
        'error': None,
        'generated_code': 'def compress(data): ...\ndef decompress(data): ...'
    }
    # init_db() # Ensure DB exists
    # save_result(test_result) # Uncomment to save test data
    all_data = get_all_results()
    print(f"\nAll results from {DATABASE_FILE}:")
    for row in all_data:
        print(dict(row))
