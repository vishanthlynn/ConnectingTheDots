# Use minimal base image compatible with amd64
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- CORRECTED MODEL HANDLING ---

# 1. Set an environment variable to define a local cache directory
ENV TRANSFORMERS_CACHE=/app/models

# 2. Copy the model from your host's cache to the new directory in the container
#    Replace 'user' with your actual username on your machine, or provide the full path.
COPY /home/user/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2 ${TRANSFORMERS_CACHE}/sentence-transformers_all-MiniLM-L6-v2

# --- END CORRECTION ---

# Copy the rest of the application code into the container
COPY . .

# Set entry point to the script
CMD ["python", "main.py", "--mode", "analyze"]