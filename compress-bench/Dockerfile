# Use an official Python runtime as a parent image
# Using 3.10-slim as it's generally a good balance of size and compatibility
# Explicitly specify linux/amd64 platform to match host and avoid architecture issues
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# --- Install Dependencies ---
# Install OS-level dependencies first (gcc for C compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    file \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify gcc installation
RUN gcc --version

# Copy only the requirements files first to leverage Docker cache
COPY framework/requirements.txt /app/framework/requirements.txt
COPY compress-bench/requirements.txt /app/compress-bench/requirements.txt

# Install Python dependencies for both framework and benchmark
# Using --no-cache-dir reduces image size slightly
RUN pip install --no-cache-dir -r /app/framework/requirements.txt
RUN pip install --no-cache-dir -r /app/compress-bench/requirements.txt

# --- Copy Application Code ---
# Copy framework code (needed for imports)
COPY framework/ /app/framework/
# Copy benchmark-specific code
COPY compress-bench/ /app/compress-bench/

# --- Generate Test Suite ---
# Generate the test suite during the image build process using the benchmark-specific generator
# The output file (e.g., 'compression_test_suite.json') will be created inside /app/compress-bench/
RUN python /app/compress-bench/test_suite_generator.py --generate-suite --suite-file /app/compress-bench/compression_test_suite.json

# Verify the test suite file was created (optional, useful for debugging build issues)
RUN ls -lh /app/compress-bench/compression_test_suite.json

# --- Runtime Configuration ---
# Make port 5001 available (adjust if needed)
EXPOSE 5001

# Define environment variable for Flask to run on 0.0.0.0 to be accessible externally
ENV FLASK_RUN_HOST=0.0.0.0
# Set the port for this specific benchmark app
ENV COMPRESS_BENCH_PORT=5001

# Set the working directory for running the app
WORKDIR /app/compress-bench

# Run app.py using Flask's built-in server when the container launches
# Note: For production, consider using a more robust WSGI server like Gunicorn
CMD ["flask", "run"]
