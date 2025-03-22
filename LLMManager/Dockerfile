FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files
COPY requirements.txt .
COPY app.py .
COPY model_implementations.py .
COPY initialize_models.sh .

# Make the initialization script executable
RUN chmod +x initialize_models.sh

# Install Python dependencies
# Only install the needed packages based on your actual usage
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    gunicorn==21.2.0 \
    pydantic==2.4.2 \
    llama-cpp-python==0.2.56 \
    transformers==4.34.0 \
    accelerate==0.23.0 \
    bitsandbytes==0.41.1 \
    torch==2.0.1 \
    numpy==1.24.4 \
    tqdm==4.66.1

# Create directories for models and data
RUN mkdir -p models data

# Expose the Flask API port
EXPOSE 5000

# Command is specified in docker-compose.yaml
CMD ["python", "app.py"]
