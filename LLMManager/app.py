import os
import logging
import requests
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
import torch


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMModel:
    """A general-purpose LLM model class that can handle different model types"""
    
    def __init__(self, model_path: str, model_type: str, context_window: int = 2048, 
                 n_threads: int = 4, n_gpu_layers: int = 0, temperature: float = 0.7):
        """
        Initialize the LLM model based on the provided type.
        
        Args:
            model_path: Path to the model file or Hugging Face model ID
            model_type: Type of model ('llama', 'phi2', 'rwkv')
            context_window: Size of the context window (token limit)
            n_threads: Number of CPU threads to use
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU only)
            temperature: Sampling temperature for generation
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.context_window = context_window
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        
        logger.info(f"Initializing {model_type} model from {model_path}")
        logger.info(f"Parameters: context_window={context_window}, threads={n_threads}, gpu_layers={n_gpu_layers}")
        
        try:

            is_local_file = os.path.exists(model_path) and model_path.endswith('.gguf')
            self.device = "cuda" if torch.cuda.is_available() and n_gpu_layers > 0 else "cpu"
            
            if is_local_file:

                from llama_cpp import Llama
                
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=context_window,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers
                )
                self.using_llama_cpp = True
                logger.info(f"Loaded model with llama.cpp backend")
                
            elif self.model_type == "phi2":

                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_4bit=True,
                    device_map="auto"
                )
                self.using_llama_cpp = False
                logger.info(f"Loaded Phi-2 model from Hugging Face")
                
            elif self.model_type == "rwkv" and not is_local_file:

                try:
                    import rwkv
                    from rwkv.model import RWKV
                    from rwkv.utils import PIPELINE
                    
                    self.model = RWKV(model=model_path, strategy=f'{self.device} fp16')
                    self.pipeline = PIPELINE(self.model)
                    self.using_llama_cpp = False
                    self.using_rwkv_native = True
                    logger.info(f"Loaded RWKV model with native backend")
                except ImportError:
                    logger.warning("RWKV package not available, falling back to llama.cpp")
                    raise ValueError("RWKV package not available and model is not in GGUF format")
            else:

                if not is_local_file:
                    raise ValueError(f"Non-GGUF files for model type {model_type} are not supported")
            

            model_size = self.get_model_size()
            logger.info(f"Loaded model with size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB"""
        if hasattr(self, 'using_llama_cpp') and self.using_llama_cpp:
            if os.path.exists(self.model_path):
                return os.path.getsize(self.model_path) / (1024 * 1024)  # Convert to MB
        elif hasattr(self, 'model') and not hasattr(self, 'using_rwkv_native'):
            try:
                model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
                return model_size_bytes / (1024 * 1024)  # Convert to MB
            except:
                pass
        return 0.0
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a response based on conversation history"""
        logger.info(f"Generating response with {self.model_type} model")
        
        try:

            if self.model_type == "phi2":
                prompt = self._format_phi2_prompt(conversation_history)
            elif self.model_type == "rwkv":
                prompt = self._format_rwkv_prompt(conversation_history)
            else:  # Default to llama format
                prompt = self._format_llama_prompt(conversation_history)
            
            logger.debug(f"Using prompt: {prompt[:100]}...")
            

            if hasattr(self, 'using_llama_cpp') and self.using_llama_cpp:

                output = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=self.context_window // 4,
                    temperature=self.temperature,
                    stop=self._get_stop_tokens()
                )
                
                response = ""
                if isinstance(output, dict) and "choices" in output:
                    response = output["choices"][0]["text"]
                elif isinstance(output, dict) and "text" in output:
                    response = output["text"]
                else:
                    response = str(output)
                    
            elif hasattr(self, 'using_rwkv_native') and self.using_rwkv_native:

                state = None
                

                for i in range(0, len(prompt), 64):
                    chunk = prompt[i:i+64]
                    output, state = self.pipeline.forward(chunk, state)
                

                response = ""
                for _ in range(self.context_window // 4):  # Generate up to 1/4 of context window
                    output, state = self.pipeline.forward("\n", state)
                    token = self.pipeline.sample_logits(output, temperature=self.temperature)
                    if token == '\n\n':
                        break
                    response += token
                
            else:

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_length=len(inputs["input_ids"][0]) + (self.context_window // 4),
                        do_sample=True,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
            

            if not self._is_english(response):
                logger.warning("Non-English response detected, falling back to English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
            return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def _format_llama_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for Llama models"""
        prompt = ""
        for msg in conversation_history:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def _format_phi2_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for Phi-2 model"""
        prompt = ""
        for msg in conversation_history:
            if msg["role"] == "system":
                prompt += f"<|system|>\n{msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt
    
    def _format_rwkv_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for RWKV model"""
        prompt = ""
        for msg in conversation_history:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        prompt += "Assistant:"
        return prompt
    
    def _get_stop_tokens(self) -> List[str]:
        """Get appropriate stop tokens based on model type"""
        if self.model_type == "phi2":
            return ["<|im_end|>", "<|user|>", "<|system|>"]
        elif self.model_type == "rwkv":
            return ["User:", "System:", "Assistant:"]
        else:  # llama
            return ["<|im_end|>", "<|user|>", "<|system|>"]
    
    def _is_english(self, text: str) -> bool:
        """Simple check to detect if text is primarily English"""
        try:

            from langdetect import detect
            lang = detect(text)
            return lang == 'en'
        except ImportError:

            non_english_chars = 0
            total_chars = max(1, len(text.strip()))
            
            for char in text:

                if char.isalpha() and ord(char) > 127:
                    non_english_chars += 1
            

            return (non_english_chars / total_chars) < 0.15


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

        conv_to_remove = []
        for conv_id, conv_data in self.conversations.items():
            if conv_data["model_id"] == model_id:
                conv_to_remove.append(conv_id)
        
        for conv_id in conv_to_remove:
            del self.conversations[conv_id]
            logger.info(f"Removed conversation {conv_id} associated with model {model_id}")
        

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
        

        self.conversations[conversation_id]["history"].append({
            "role": "user",
            "content": message
        })
        

        model_id = self.conversations[conversation_id]["model_id"]
        model = self.models[model_id]
        
        logger.info(f"Generating response for conversation: {conversation_id} with model: {model_id}")
        history = self.conversations[conversation_id]["history"]
        
        try:
            response = model.generate(history)
            

            if not model._is_english(response):
                logger.warning(f"Non-English response detected from {model_id}, enforcing English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = f"Error generating response: {str(e)}"
        

        self.conversations[conversation_id]["history"].append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
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
            "type": model.model_type,
            "size_mb": model.get_model_size(),
            "context_window": model.context_window,
            "n_threads": model.n_threads,
            "n_gpu_layers": model.n_gpu_layers
        }


def analyze_gguf_file(file_path):
    """Extract metadata from a GGUF file to check compatibility"""
    try:
        import struct
        import json
        

        if not os.path.exists(file_path):
            return {"error": "File not found"}
            
        with open(file_path, 'rb') as f:

            magic = f.read(4)
            if magic != b'GGUF':
                return {"error": "Not a valid GGUF file (missing GGUF magic)"}
                

            version = struct.unpack('<I', f.read(4))[0]
            

            metadata = {
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "gguf_version": version
            }
            


            f.seek(0)
            sample = f.read(8192).decode('utf-8', errors='ignore')
            

            for arch in ["llama", "falcon", "mpt", "gpt2", "gptj", "gpt_neox", "phi", "rwkv", "exaone"]:
                if arch in sample.lower():
                    metadata["detected_architecture"] = arch
                    break
            
            return metadata
            
    except Exception as e:
        return {"error": f"Error analyzing GGUF file: {e}"}


def download_gguf_model(url, save_path):
    """Download a GGUF model from a URL to the specified path"""
    try:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        

        response = requests.get(url, stream=True)
        response.raise_for_status()
        

        total_size = int(response.headers.get('content-length', 0))
        

        logger.info(f"Downloading model from {url} to {save_path}")
        if total_size:
            logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")
        

        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    

                    if total_size > 0 and downloaded % (100 * 1024 * 1024) < 8192:  # Log every 100MB
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Downloaded: {downloaded / (1024 * 1024):.2f} MB ({progress:.2f}%)")
        
        logger.info(f"Model download complete: {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")

        if os.path.exists(save_path):
            os.remove(save_path)
        return False


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
        model_type = model_config.get('type', 'llama')
        
        if not model_id:
            continue
            
        try:

            model_path = model_config.get('path')
            if not model_path:
                continue
                
            full_path = os.path.join(os.environ.get('MODEL_DIR', './models'), model_path)
            

            context_window = model_config.get('context_window', 2048)
            n_threads = model_config.get('n_threads', 4)
            n_gpu_layers = model_config.get('n_gpu_layers', 0)
            temperature = model_config.get('temperature', 0.7)
            

            model = LLMModel(
                model_path=full_path,
                model_type=model_type,
                context_window=context_window,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                temperature=temperature
            )
            
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
    

    context_window = data.get('context_window', 2048)
    n_threads = data.get('n_threads', 4)
    n_gpu_layers = data.get('n_gpu_layers', 0)
    temperature = data.get('temperature', 0.7)
    

    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if not model_url:
        return jsonify({"error": "model_url is required"}), 400
    

    if model_id in manager.models:
        return jsonify({"error": f"Model ID '{model_id}' already exists"}), 400
    

    file_name = data.get('file_name')
    if not file_name:
        file_name = model_url.split('/')[-1]
        if not file_name or '?' in file_name:  # Handle URLs with query parameters
            file_name = f"{model_id}.gguf"
    

    model_dir = os.environ.get('MODEL_DIR', './models')
    save_path = os.path.join(model_dir, file_name)
    

    logger.info(f"Downloading model from {model_url} to {save_path}")
    success = download_gguf_model(model_url, save_path)
    
    if not success:
        return jsonify({"error": "Failed to download model"}), 500
    

    keep_file = data.get('keep_file_on_error', False)
    

    download_only = data.get('download_only', False)
    

    analysis = analyze_gguf_file(save_path)
    

    if download_only:
        return jsonify({
            "success": True,
            "message": "Model downloaded successfully but not loaded",
            "model_id": model_id,
            "file_path": save_path,
            "analysis": analysis
        })
    

    if "detected_architecture" in analysis:
        detected_arch = analysis["detected_architecture"]
        if detected_arch != model_type and detected_arch in ["llama", "phi", "rwkv"]:
            logger.info(f"Detected architecture '{detected_arch}' differs from specified type '{model_type}'")
            

            if data.get('auto_correct_type', False):
                if detected_arch == "phi":
                    model_type = "phi2"
                else:
                    model_type = detected_arch
                logger.info(f"Auto-corrected model type to '{model_type}'")
    

    try:
        model = LLMModel(
            model_path=save_path,
            model_type=model_type,
            context_window=context_window,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature
        )
        
        manager.add_model(model_id, model)
        

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
        

        if os.path.exists(save_path) and not keep_file:
            logger.info(f"Removing incompatible model file: {save_path}")
            os.remove(save_path)
            file_status = "Model file was removed. Set 'keep_file_on_error' to true to keep downloaded files despite errors."
        else:
            file_status = f"Model file was kept at {save_path}"
        

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
    

    try:
        model_info = manager.model_info(model_id)
    except:
        model_info = {"id": model_id}
    

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
    

    if not os.path.isabs(model_path):
        model_path = os.path.join(os.environ.get('MODEL_DIR', './models'), model_path)
    

    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found"}), 404
    

    analysis = analyze_gguf_file(model_path)
    

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

    app.run(host='0.0.0.0', port=5000)