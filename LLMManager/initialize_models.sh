

echo "Initializing all LLM models from the models directory..."


until $(curl --output /dev/null --silent --head --fail http://localhost:5000/health); do
  echo "Waiting for API to be available..."
  sleep 2
done


MODEL_DIR=${MODEL_DIR:-/app/models}


if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Models directory $MODEL_DIR not found!"
  exit 1
fi


success_count=0


guess_model_type() {
  local filename=$(basename "$1" | tr '[:upper:]' '[:lower:]')
  
  
  if [[ "$filename" == *"phi"* ]]; then
    echo "phi2"
  elif [[ "$filename" == *"rwkv"* ]]; then
    echo "rwkv"
  else
    echo "llama"
  fi
}


for model_file in "$MODEL_DIR"/*.gguf; do
  
  [ -e "$model_file" ] || continue
  
  
  model_id=$(basename "$model_file" .gguf)
  
  
  
  model_id=$(echo "$model_id" | sed 's/[^a-zA-Z0-9]/-/g' | tr '[:upper:]' '[:lower:]')
  
  
  model_type=$(guess_model_type "$model_file")
  
  
  model_filename=$(basename "$model_file")
  
  echo "Initializing model: $model_id (type: $model_type) from file: $model_filename"
  
  
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