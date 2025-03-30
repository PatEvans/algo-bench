# Use an official Python runtime that includes build tools, allowing easy installation of GCC
FROM python:3.10-bullseye # Debian-based, includes common build essentials

# Set the working directory in the container
WORKDIR /app

# Install GCC and any other system dependencies needed for C benchmarks
# Use non-interactive frontend to avoid prompts during build
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy application requirements first (leverages Docker cache)
# Assuming a requirements.txt exists in the root and/or framework/sort-bench/compress-bench
# Adjust paths as necessary if requirements are structured differently
COPY requirements.txt ./
# If framework has its own requirements:
# COPY framework/requirements.txt ./framework_requirements.txt
# If benchmarks have their own:
# COPY compress-bench/requirements.txt ./compress_requirements.txt
# COPY sort-bench/requirements.txt ./sort_requirements.txt

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt
# If framework/benchmark requirements exist:
# RUN pip install --no-cache-dir -r framework_requirements.txt
# RUN pip install --no-cache-dir -r compress_requirements.txt
# RUN pip install --no-cache-dir -r sort_requirements.txt

# Copy the entire application context into the container
# This includes main_app.py, framework/, compress-bench/, sort-bench/, templates/, etc.
COPY . .

# Test suite generation should happen at runtime via the UI/config, not during build
# Remove the previous RUN python test_suite_generator.py line

# Make port 5000 available (Flask's default port)
EXPOSE 5000

# Define environment variable for Flask to run on 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0
# Set FLASK_APP to point to the main application entry point
ENV FLASK_APP=main_app.py

# Run the main Flask application when the container launches
CMD ["flask", "run"]
