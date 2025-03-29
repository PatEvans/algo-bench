# Use an official Python runtime as a parent image
# Using 3.10-slim as it's generally a good balance of size and compatibility
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces image size slightly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Ensure all necessary Python files and the templates directory are copied
COPY app.py .
COPY benchmark.py .
COPY database.py .
COPY llm_interface.py .
COPY templates/ ./templates/
# If you have other .py files or data files needed by the app, add COPY lines for them here

# Generate the test suite during the image build process
# This runs the __main__ block in benchmark.py with the --generate-suite flag
# It will use the default parameters defined in benchmark.py unless overridden here
# The output file 'test_suite.json' will be created inside the /app directory in the image
RUN python benchmark.py --generate-suite --suite-file test_suite.json

# Verify the test suite file was created (optional, useful for debugging build issues)
RUN ls -lh test_suite.json

# Make port 5000 available to the world outside this container (Flask's default port)
EXPOSE 5000

# Define environment variable for Flask to run on 0.0.0.0 to be accessible externally
ENV FLASK_RUN_HOST=0.0.0.0
# Optional: Set FLASK_APP if needed, though Flask usually detects app.py
# ENV FLASK_APP=app.py

# Run app.py using Flask's built-in server when the container launches
# Note: For production, consider using a more robust WSGI server like Gunicorn
CMD ["flask", "run"]
# Alternative using python directly (less common for Flask now):
# CMD ["python", "app.py"]
