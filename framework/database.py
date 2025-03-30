"""
Module for generic database interactions using SQLite.
Stores benchmark results.
"""

import sqlite3
import json
import os
from typing import Optional

# Default database filename, can be overridden by config
DEFAULT_DATABASE_FILE = 'framework_results.db'

class BenchmarkDB:
    """Handles database operations for benchmark results."""

    def __init__(self, db_file: Optional[str] = None):
        """
        Initializes the database handler.

        Args:
            db_file: Path to the SQLite database file. Uses DEFAULT_DATABASE_FILE if None.
        """
        self.db_file = db_file or DEFAULT_DATABASE_FILE
        # Ensure the directory exists if db_file includes a path
        db_dir = os.path.dirname(self.db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Created directory for database: {db_dir}")
        self._init_db() # Ensure table exists on instantiation

    def _get_db_connection(self):
        """Establishes a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_file, timeout=10) # Added timeout
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        # Improve concurrency handling (optional but recommended for Flask)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # Use a generic schema based on compression-bench initially
                # We might need to make this more flexible later if schemas diverge significantly
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        benchmark_name TEXT NOT NULL, -- e.g., "LLM C Compression", "LLM C Sort"
                        llm TEXT NOT NULL, -- The specific LLM used (e.g., Gemini 2.5 Pro Exp)
                        correctness INTEGER, -- 1 for all correct, 0 if any failed/error
                        avg_time_ms REAL, -- Generic primary timing metric (e.g., avg compress time)
                        avg_secondary_time_ms REAL, -- Generic secondary timing (e.g., avg decompress time)
                        avg_ratio REAL, -- Generic ratio metric (e.g., compression ratio)
                        error TEXT, -- Store any critical errors encountered during the run
                        generated_code TEXT, -- Store the full generated code
                        performance_details TEXT -- JSON blob for extra details if needed
                    )
                ''')
                # Consider adding indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON results (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_llm ON results (benchmark_name, llm)')
                conn.commit()
            print(f"Database initialized/checked: {self.db_file}")
        except sqlite3.Error as e:
            print(f"Error initializing database {self.db_file}: {e}")
            raise # Re-raise after logging

    def save_result(self, result_data: dict):
        """Saves a single benchmark result to the database."""
        required_keys = ['benchmark_name', 'llm']
        if not all(key in result_data for key in required_keys):
             missing = [key for key in required_keys if key not in result_data]
             print(f"Error saving result: Missing required keys: {missing}. Data: {result_data}")
             # Optionally raise an error or return False
             return False

        # Serialize performance details if present
        details_json = None
        if 'performance_details' in result_data and result_data['performance_details']:
            try:
                details_json = json.dumps(result_data['performance_details'])
            except TypeError as e:
                print(f"Warning: Could not serialize performance_details: {e}. Storing as NULL.")

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO results (
                        benchmark_name, llm, correctness,
                        avg_time_ms, avg_secondary_time_ms, avg_ratio,
                        error, generated_code, performance_details
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_data.get('benchmark_name'), # Use the generic name
                    result_data.get('llm'),
                    result_data.get('correctness'), # Should be 0 or 1
                    # Map specific metrics to generic columns
                    result_data.get('avg_compression_time_ms') or result_data.get('avg_sort_time_ms') or result_data.get('avg_time_ms'), # Primary time
                    result_data.get('avg_decompression_time_ms') or result_data.get('avg_secondary_time_ms'), # Secondary time
                    result_data.get('avg_compression_ratio') or result_data.get('avg_ratio'), # Ratio
                    result_data.get('error'),
                    result_data.get('generated_code'),
                    details_json # Store the JSON blob
                ))
                conn.commit()
            print(f"Result saved to {self.db_file} for {result_data.get('benchmark_name')} - {result_data.get('llm')}")
            return True
        except sqlite3.Error as e:
            print(f"Error saving result to database {self.db_file}: {e}. Data: {result_data}")
            return False # Indicate failure

    def get_all_results(self) -> list[dict]:
        """Retrieves all benchmark results from the database."""
        results_list = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # Order by timestamp descending
                cursor.execute('SELECT * FROM results ORDER BY timestamp DESC')
                rows = cursor.fetchall()

            # Convert Row objects to standard dictionaries
            for row in rows:
                result_dict = dict(row)
                # Deserialize performance_details if needed (currently not used heavily)
                details_json = result_dict.get('performance_details')
                if details_json:
                    try:
                        result_dict['performance_details'] = json.loads(details_json)
                    except json.JSONDecodeError:
                        # Keep raw string or indicate error if deserialization fails
                        result_dict['performance_details'] = {"error": "Failed to decode JSON", "raw": details_json}
                results_list.append(result_dict)

        except sqlite3.Error as e:
            print(f"Error retrieving results from database {self.db_file}: {e}")
            # Return empty list or re-raise depending on desired behavior
        return results_list

# Example usage (for testing purposes)
if __name__ == '__main__':
    db = BenchmarkDB('test_framework_db.sqlite') # Use a test file name
    print(f"Using database: {db.db_file}")

    # Example data saving
    test_result_compress = {
        'benchmark_name': 'Test Compression',
        'llm': 'test_llm_compress',
        'correctness': 1,
        'avg_compression_time_ms': 55.2, # Specific metric
        'avg_decompression_time_ms': 25.8, # Specific metric
        'avg_compression_ratio': 2.5, # Specific metric
        'error': None,
        'generated_code': 'typedef struct { ... } Buffer; ...',
        'performance_details': {'category1': 'detail1'}
    }
    test_result_sort = {
        'benchmark_name': 'Test Sort',
        'llm': 'test_llm_sort',
        'correctness': 0,
        'avg_sort_time_ms': 101.1, # Specific metric
        # No secondary time or ratio for sort
        'error': "Sort failed on edge case",
        'generated_code': 'void sort_int_array(...) { ... }',
        'performance_details': {'category_sorted': 'detail_sort'}
    }

    print("\nAttempting to save results...")
    db.save_result(test_result_compress)
    db.save_result(test_result_sort)

    print("\nRetrieving all results...")
    all_data = db.get_all_results()
    print(f"\nAll results from {db.db_file}:")
    for i, row in enumerate(all_data):
        print(f"--- Result {i+1} ---")
        print(json.dumps(row, indent=2))

    # Clean up the test database file
    # try:
    #     os.remove(db.db_file)
    #     print(f"\nCleaned up test database: {db.db_file}")
    # except OSError as e:
    #     print(f"\nError cleaning up test database {db.db_file}: {e}")
