import os
import logging
import requests
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify

# Import model implementations
from model_implementations import LLMInterface, TinyLlamaModel, Phi2Model, RWKVModel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM Conversation Manager class
class LLMConversationManager:
    """A lightweight manager for LLM conversations"""
    
    def __init__(self):
        self.models = {}  # Dictionary to store LLM models
        self.conversations = {}  # Dictionary to store conversations
    
    def add_model(self, model_id: str, model_instance: Any) -> None:
        """Add a model to the manager"""
        logger.info(f"Adding model: {model_id}")
        self.models[model_id] = model_instance
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the manager"""
        if model_id not in self.models:
            return False
        
        logger.info(f"Removing model: {model_id}")
        # First, close any active conversations with this model
        conv_to_remove = []
        for conv_id, conv_data in self.conversations.items():
            if conv_data["model_id"] == model_id:
                conv_to_remove.append(conv_id)
        
        for conv_id in conv_to_remove:
            del self.conversations[conv_id]
            logger.info(f"Removed conversation {conv_id} associated with model {model_id}")
        
        # Remove the model itself
        del self.models[model_id]
        return True
    
    def create_conversation(self, model_id: str, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation with a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if conversation_id is None:
            existing_count = sum(1 for conv in self.conversations.values() 
                               if conv["model_id"] == model_id)
            conversation_id = f"{model_id}_{existing_count + 1}"
        
        logger.info(f"Creating conversation: {conversation_id} with model: {model_id}")
        # Start the conversation with a system message to enforce English responses
        self.conversations[conversation_id] = {
            "model_id": model_id,
            "history": [{
                "role": "system",
                "content": "You are an English language assistant. Always respond in English only, " +
                          "regardless of the language used to ask questions. If asked in another language, " +
                          "politely request English. Maintain a helpful, concise, and informative tone."
            }]
        }
        
        return conversation_id
    
    def get_response(self, conversation_id: str, message: str) -> str:
        """Get a response from the model for the given conversation"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Add user message to conversation
        self.conversations[conversation_id]["history"].append({
            "role": "user",
            "content": message
        })
        
        # Get model and generate response
        model_id = self.conversations[conversation_id]["model_id"]
        model = self.models[model_id]
        
        logger.info(f"Generating response for conversation: {conversation_id} with model: {model_id}")
        history = self.conversations[conversation_id]["history"]
        
        try:
            response = model.generate(history)
            
            # Simple check to ensure English response
            if not self._is_english(response):
                logger.warning(f"Non-English response detected from {model_id}, enforcing English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = f"Error generating response: {str(e)}"
        
        # Add assistant response to conversation
        self.conversations[conversation_id]["history"].append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _is_english(self, text: str) -> bool:
        """Simple check to detect if text is primarily English"""
        non_english_chars = 0
        total_chars = max(1, len(text.strip()))
        
        for char in text:
            # Check if character is outside basic Latin alphabet range
            if char.isalpha() and ord(char) > 127:
                non_english_chars += 1
        
        # If more than 15% non-Latin chars, likely not English
        return (non_english_chars / total_chars) < 0.15
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get the history of a conversation"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        return self.conversations[conversation_id]["history"]
    
    def reset_conversation(self, conversation_id: str) -> None:
        """Reset a conversation while keeping the same model"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        model_id = self.conversations[conversation_id]["model_id"]
        
        # Recreate the conversation with just the system message
        self.conversations[conversation_id] = {
            "model_id": model_id,
            "history": [{
                "role": "system",
                "content": "You are an English language assistant. Always respond in English only, " +
                          "regardless of the language used to ask questions. If asked in another language, " +
                          "politely request English. Maintain a helpful, concise, and informative tone."
            }]
        }
    
    def model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        return {
            "id": model_id,
            "type": type(model).__name__,
            "size_mb": model.get_model_size()
        }

# Dynamic model loader
def load_model(model_type, **kwargs):
    """Dynamically load model implementations based on type"""
    if model_type.lower() == "llama":
        try:
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("model_path is required for llama models")
            return TinyLlamaModel(model_path=model_path)
        except ImportError:
            logger.error("llama-cpp-python not installed, cannot load llama models")
            raise
    
    if model_type.lower() == "phi2":
        try:
            model_path = kwargs.get("model_path", "microsoft/phi-2")
            return Phi2Model(model_path=model_path)
        except ImportError:
            logger.error("transformers not installed, cannot load phi2 models")
            raise
    
    if model_type.lower() == "rwkv":
        try:
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("model_path is required for RWKV models")
            return RWKVModel(model_path=model_path)
        except ImportError:
            logger.error("RWKV not installed, cannot load RWKV models")
            raise
    
    logger.error(f"Unknown model type: {model_type}")
    raise ValueError(f"Unknown model type: {model_type}")

# Function to analyze a GGUF file to check compatibility
def analyze_gguf_file(file_path):
    """Extract metadata from a GGUF file to check compatibility"""
    try:
        import struct
        import json
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": "File not found"}
            
        with open(file_path, 'rb') as f:
            # GGUF magic number
            magic = f.read(4)
            if magic != b'GGUF':
                return {"error": "Not a valid GGUF file (missing GGUF magic)"}
                
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            
            # This is a simplified check - real GGUF parsing is more complex
            metadata = {
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "gguf_version": version
            }
            
            # Try to detect architecture - this is simplified
            # Read a bit more to find architecture info
            f.seek(0)
            sample = f.read(8192).decode('utf-8', errors='ignore')
            
            # Look for known architectures in the binary data
            for arch in ["llama", "falcon", "mpt", "gpt2", "gptj", "gpt_neox", "phi", "rwkv", "exaone"]:
                if arch in sample.lower():
                    metadata["detected_architecture"] = arch
                    break
            
            return metadata
            
    except Exception as e:
        return {"error": f"Error analyzing GGUF file: {e}"}

# Function to download a GGUF file from a URL
def download_gguf_model(url, save_path):
    """Download a GGUF model from a URL to the specified path"""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download with progress reporting
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Log download information
        logger.info(f"Downloading model from {url} to {save_path}")
        if total_size:
            logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")
        
        # Write the file
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress for large files
                    if total_size > 0 and downloaded % (100 * 1024 * 1024) < 8192:  # Log every 100MB
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Downloaded: {downloaded / (1024 * 1024):.2f} MB ({progress:.2f}%)")
        
        logger.info(f"Model download complete: {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        # Clean up partial download if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

# Flask application
app = Flask(__name__)
manager = LLMConversationManager()

@app.route('/api/initialize', methods=['POST'])
def initialize_models():
    """Initialize models from configuration"""
    data = request.json
    models_config = data.get('models', [])
    
    initialized_models = []
    
    for model_config in models_config:
        model_id = model_config.get('id')
        model_type = model_config.get('type')
        
        if not model_id or not model_type:
            continue
            
        try:
            if model_type.lower() == 'llama':
                model_path = model_config.get('path')
                if not model_path:
                    continue
                
                full_path = os.path.join(os.environ.get('MODEL_DIR', './models'), model_path)
                model = load_model("llama", model_path=full_path)
                manager.add_model(model_id, model)
                initialized_models.append(model_id)
                
            elif model_type.lower() == 'phi2':
                model_path = model_config.get('path', "microsoft/phi-2")
                full_path = os.path.join(os.environ.get('MODEL_DIR', './models'), model_path)
                model = load_model("phi2", model_path=full_path)
                manager.add_model(model_id, model)
                initialized_models.append(model_id)
                
            elif model_type.lower() == 'rwkv':
                model_path = model_config.get('path')
                if not model_path:
                    continue
                
                full_path = os.path.join(os.environ.get('MODEL_DIR', './models'), model_path)
                model = load_model("rwkv", model_path=full_path)
                manager.add_model(model_id, model)
                initialized_models.append(model_id)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_id}: {e}")
    
    return jsonify({"success": True, "models": initialized_models})

@app.route('/api/add-llm', methods=['POST'])
def add_llm_model():
    """Add a new LLM model by URL"""
    data = request.json
    model_id = data.get('model_id')
    model_type = data.get('model_type', 'llama').lower()
    model_url = data.get('model_url')
    
    # Validate required parameters
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if not model_url:
        return jsonify({"error": "model_url is required"}), 400
    
    # Check if model_id already exists
    if model_id in manager.models:
        return jsonify({"error": f"Model ID '{model_id}' already exists"}), 400
    
    # Determine file name from URL or use provided name
    file_name = data.get('file_name')
    if not file_name:
        file_name = model_url.split('/')[-1]
        if not file_name or '?' in file_name:  # Handle URLs with query parameters
            file_name = f"{model_id}.gguf"
    
    # Set save path
    model_dir = os.environ.get('MODEL_DIR', './models')
    save_path = os.path.join(model_dir, file_name)
    
    # Download the model
    logger.info(f"Downloading model from {model_url} to {save_path}")
    success = download_gguf_model(model_url, save_path)
    
    if not success:
        return jsonify({"error": "Failed to download model"}), 500
    
    # Check if we should keep the model even if loading fails
    keep_file = data.get('keep_file_on_error', False)
    
    # Check if we should only download without loading
    download_only = data.get('download_only', False)
    
    # Analyze the model before loading
    analysis = analyze_gguf_file(save_path)
    
    # Return early if download_only is True
    if download_only:
        return jsonify({
            "success": True,
            "message": "Model downloaded successfully but not loaded",
            "model_id": model_id,
            "file_path": save_path,
            "analysis": analysis
        })
    
    # Check if we need to adjust the model type based on analysis
    if "detected_architecture" in analysis:
        detected_arch = analysis["detected_architecture"]
        if detected_arch != model_type and detected_arch in ["llama", "phi", "rwkv"]:
            logger.info(f"Detected architecture '{detected_arch}' differs from specified type '{model_type}'")
            
            # Auto-correct if requested
            if data.get('auto_correct_type', False):
                if detected_arch == "phi":
                    model_type = "phi2"
                else:
                    model_type = detected_arch
                logger.info(f"Auto-corrected model type to '{model_type}'")
    
    # Load the model
    try:
        model = load_model(model_type, model_path=save_path)
        manager.add_model(model_id, model)
        
        # Get model information
        model_info = manager.model_info(model_id)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "file_path": save_path,
            "model_info": model_info,
            "analysis": analysis
        })
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to load model {model_id}: {error_msg}")
        
        # Clean up the downloaded file if loading fails, unless keep_file is True
        if os.path.exists(save_path) and not keep_file:
            logger.info(f"Removing incompatible model file: {save_path}")
            os.remove(save_path)
            file_status = "Model file was removed. Set 'keep_file_on_error' to true to keep downloaded files despite errors."
        else:
            file_status = f"Model file was kept at {save_path}"
        
        # Check for common compatibility issues
        if "unknown model architecture" in error_msg:
            suggestion = "This GGUF model appears to use a custom architecture not supported by llama.cpp."
        elif "Failed to load model from file" in error_msg:
            suggestion = "The model format might be incompatible with the selected model type."
        else:
            suggestion = "Check if the model is compatible with the selected model_type."
            
        return jsonify({
            "error": f"Failed to load model: {error_msg}",
            "file_status": file_status,
            "suggestion": suggestion,
            "supported_types": ["llama (for llama.cpp compatible GGUF files)", 
                               "phi2 (for Phi-2 models)", 
                               "rwkv (for RWKV models)"]
        }), 500

@app.route('/api/delete-llm/<model_id>', methods=['POST', 'DELETE'])
def delete_llm_model(model_id):
    """Delete an LLM model"""
    if model_id not in manager.models:
        return jsonify({"error": f"Model '{model_id}' not found"}), 404
    
    # Get model info before removing
    try:
        model_info = manager.model_info(model_id)
    except:
        model_info = {"id": model_id}
    
    # Remove the model from the manager
    success = manager.remove_model(model_id)
    
    if success:
        return jsonify({
            "success": True,
            "model_id": model_id,
            "message": f"Model '{model_id}' successfully removed"
        })
    else:
        return jsonify({"error": f"Failed to remove model '{model_id}'"}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models with their information"""
    models_info = {}
    for model_id in manager.models:
        try:
            models_info[model_id] = manager.model_info(model_id)
        except Exception as e:
            models_info[model_id] = {"id": model_id, "error": str(e)}
    
    return jsonify(models_info)

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    return jsonify({
        conv_id: {
            "model_id": data["model_id"],
            "message_count": len(data["history"])
        }
        for conv_id, data in manager.conversations.items()
    })

@app.route('/api/conversation', methods=['POST'])
def create_conversation():
    data = request.json
    model_id = data.get('model_id')
    conversation_id = data.get('conversation_id')
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
        
    try:
        conv_id = manager.create_conversation(model_id, conversation_id)
        return jsonify({"conversation_id": conv_id})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    try:
        history = manager.get_conversation_history(conversation_id)
        return jsonify({"conversation_id": conversation_id, "history": history})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/conversation/<conversation_id>/reset', methods=['POST'])
def reset_conversation(conversation_id):
    """Reset a conversation's history"""
    try:
        manager.reset_conversation(conversation_id)
        return jsonify({"success": True, "conversation_id": conversation_id})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    conversation_id = data.get('conversation_id')
    message = data.get('message')
    
    if not conversation_id or not message:
        return jsonify({"error": "conversation_id and message are required"}), 400
    
    try:
        response = manager.get_response(conversation_id, message)
        return jsonify({
            "conversation_id": conversation_id,
            "response": response
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/analyze-model', methods=['POST'])
def analyze_model():
    """Analyze a model file to check compatibility before loading"""
    data = request.json
    model_path = data.get('model_path')
    
    if not model_path:
        return jsonify({"error": "model_path is required"}), 400
    
    # Resolve path if relative
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.environ.get('MODEL_DIR', './models'), model_path)
    
    # Check if file exists
    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found"}), 404
    
    # Analyze the file
    analysis = analyze_gguf_file(model_path)
    
    # Add recommendations based on analysis
    if "detected_architecture" in analysis:
        arch = analysis["detected_architecture"]
        if arch == "llama":
            analysis["recommendation"] = "Use model_type: llama"
        elif arch == "phi":
            analysis["recommendation"] = "Use model_type: phi2"
        elif arch == "rwkv":
            analysis["recommendation"] = "Use model_type: rwkv"
        else:
            analysis["recommendation"] = f"Architecture '{arch}' detected, but might not be compatible with available model types"
    
    return jsonify(analysis)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
