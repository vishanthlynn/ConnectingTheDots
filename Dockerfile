# Use a lightweight Python base image compatible with AMD64
# We use python:3.9-slim-buster as an example, which is generally suitable for AMD64
FROM python:3.9-slim-buster

# Explicitly specify the platform for AMD64 compatibility if needed (optional but good practice)
# FROM --platform=linux/amd64 python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories as defined in your Python script
RUN mkdir -p /app/input /app/output /app/config

# Copy your application code into the container
COPY process_pdf.py .

# Copy your persona_config.json into the config directory
# You will need to create this file with your desired weights
COPY config/persona_config.json /app/config/persona_config.json

# Command to run your script.
# The `docker run` command provided by the hackathon will mount input/output directories:
# `docker run ... -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ...` [cite: 3]
# So, your script just needs to operate on /app/input and write to /app/output.
CMD ["python", "process_pdf.py"]