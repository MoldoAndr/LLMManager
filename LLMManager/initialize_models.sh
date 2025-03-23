#!/bin/bash

echo "Initializing all LLM models from the models directory..."

# Wait for the API to be ready
until $(curl --output /dev/null --silent --head --fail http://localhost:5000/health); do
  echo "Waiting for API to be available..."
  sleep 2
done

# Directory where models are stored
MODEL_DIR=${MODEL_DIR:-/app/models}

# Check if models directory exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Models directory $MODEL_DIR not found!"
  exit 1
fi

# Count of successfully initialized models
success_count=0

# Function to guess model type from filename
guess_model_type() {
  local filename=$(basename "$1" | tr '[:upper:]' '[:lower:]')
  
  # Check for known model types in filename
  if [[ "$filename" == *"phi"* ]]; then
    echo "phi2"
  elif [[ "$filename" == *"rwkv"* ]]; then
    echo "rwkv"
  else
    echo "llama"
  fi
}

# Function to generate a model ID from filename
generate_model_id() {
  local filename=$(basename "$1")
  # Remove extension and replace special chars with dashes
  local model_id=$(echo "$filename" | sed 's/\.gguf$//' | sed 's/[^a-zA-Z0-9]/-/g' | tr '[:upper:]' '[:lower:]')
  echo "$model_id"
}

# Initialize each .gguf model in the directory
for model_file in "$MODEL_DIR"/*.gguf; do
  # Skip if no matching files
  [ -e "$model_file" ] || continue
  
  # Generate a model ID from the filename
  model_id=$(generate_model_id "$model_file")
  
  # Guess the model type based on the filename
  model_type=$(guess_model_type "$model_file")
  
  # Get just the filename for the API call
  model_filename=$(basename "$model_file")
  
  echo "Initializing model: $model_id (type: $model_type) from file: $model_filename"
  
  # Call the API to initialize this model
  response=$(curl -s -X POST http://localhost:5000/api/initialize \
    -H "Content-Type: application/json" \
    -d "{
      \"models\": [
        {
          \"id\": \"$model_id\",
          \"type\": \"$model_type\",
          \"path\": \"$model_filename\"
        }
      ]
    }")
  
  # Check if initialization was successful
  if echo "$response" | grep -q "\"success\":true"; then
    echo "✓ Successfully initialized model: $model_id"
    success_count=$((success_count + 1))
  else
    echo "✗ Failed to initialize model: $model_id"
    echo "Error: $response"
  fi
done

echo "Initialization complete. Successfully initialized $success_count models."
echo "The following models are available:"
curl -s http://localhost:5000/api/models | jq
