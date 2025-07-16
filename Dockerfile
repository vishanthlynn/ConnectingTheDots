# Use a lightweight Python base image for AMD64 architecture
FROM --platform=linux/amd64 python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# The --no-cache-dir option helps keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application code into the container
# This assumes your main Python script (e.g., process_pdf.py) is in the root of your project
COPY . .

# Create the input and output directories as required by the hackathon setup
RUN mkdir -p /app/input /app/output

# Specify the command to run when the container starts
# This command should process all PDFs in /app/input and put JSONs in /app/output

ENTRYPOINT ["python", "process_pdf.py"]