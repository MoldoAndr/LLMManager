import os
import logging
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify

# Import model implementations
from model_implementations import LLMInterface, TinyLlamaModel, Phi2Model

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
    
    logger.error(f"Unknown model type: {model_type}")
    raise ValueError(f"Unknown model type: {model_type}")

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
        except Exception as e:
            logger.error(f"Failed to initialize model {model_id}: {e}")
    
    return jsonify({"success": True, "models": initialized_models})

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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
