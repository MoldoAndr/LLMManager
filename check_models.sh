#!/bin/bash

# Check local models directory
echo "=== Local models directory ==="
ls -lh ./models/
echo ""

# Check container models directory
echo "=== Container models directory ==="
docker exec llm-api ls -lh /app/models/
echo ""

# Compare the two directories
echo "=== Checking volume mapping ==="
local_models=$(ls -1 ./models/ | sort)
container_models=$(docker exec llm-api ls -1 /app/models/ | sort)

if [ "$local_models" = "$container_models" ]; then
  echo "✓ Volume mapping is working correctly - model files match"
else
  echo "✗ Volume mapping issue detected - model files don't match"
  echo ""
  echo "Files in local directory only:"
  comm -23 <(echo "$local_models") <(echo "$container_models")
  echo ""
  echo "Files in container only:"
  comm -13 <(echo "$local_models") <(echo "$container_models")
fi

# Check available models in API
echo ""
echo "=== Available models in API ==="
curl -s http://localhost:5000/api/models | jq
