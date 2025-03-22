#!/bin/bash

echo "Initializing LLM models..."

# Wait for the API to be ready
until $(curl --output /dev/null --silent --head --fail http://localhost:5000/health); do
  echo "Waiting for API to be available..."
  sleep 2
done

# Initialize Phi-2 model
echo "Initializing Phi-2 model..."
curl -X POST http://localhost:5000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {
        "id": "phi2",
        "type": "llama",
        "path": "phi-2.Q3_K_S.gguf"
      }
    ]
  }'

# Initialize TinyLlama model
echo "Initializing TinyLlama model..."
curl -X POST http://localhost:5000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {
        "id": "tinyllama",
        "type": "llama",
        "path": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
      }
    ]
  }'

echo "Initialization complete. The following models are available:"
curl -s http://localhost:5000/api/models | jq
