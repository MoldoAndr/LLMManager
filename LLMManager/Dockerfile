FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY app.py .
COPY model_implementations.py .
COPY initialize_models.sh .

RUN chmod +x initialize_models.sh

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

RUN mkdir -p models data

EXPOSE 5000

CMD ["python", "app.py"]