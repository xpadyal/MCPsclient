# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install nodejs and npm for npx support
RUN apt-get update && apt-get install -y nodejs npm && apt-get clean

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY main.py .
COPY servers_config.json .
COPY .env .

# Define the command to run the application with debug output
CMD ["python", "-u", "main.py"]
