import os
import logging
import torch
from typing import List, Dict

logger = logging.getLogger(__name__)

class LLMInterface:
    """Base interface for LLM models"""
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a response based on conversation history"""
        raise NotImplementedError("Subclasses must implement generate")
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB"""
        return 0.0  # Default implementation

class Phi2Model(LLMInterface):
    """Microsoft's Phi-2 model implementation using Transformers"""
    
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Initializing Phi2Model with path: {model_path}")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load in 4-bit quantization for reduced memory footprint
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load in 4-bit to stay under 400MB
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                device_map="auto"
            )
            
            # Check model size
            model_size = self.get_model_size()
            logger.info(f"Loaded Phi2 model with size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phi2Model: {e}")
            raise
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB"""
        if hasattr(self, 'model'):
            model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            return model_size_bytes / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        # Format conversation history for Phi-2
        prompt = self._format_phi2_prompt(conversation_history)
        
        logger.info("Generating response with Phi2Model")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=len(inputs["input_ids"][0]) + 256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Ensure response is in English
            if not self._is_english(response):
                logger.warning("Non-English response detected, falling back to English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error processing your request."
    
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
    
    def _is_english(self, text: str) -> bool:
        """Simple check to detect if text is primarily English"""
        # This is a basic implementation - could be enhanced with language detection libraries
        non_english_chars = 0
        total_chars = max(1, len(text.strip()))
        
        for char in text:
            # Check if character is outside basic Latin alphabet range
            if char.isalpha() and ord(char) > 127:
                non_english_chars += 1
        
        # If more than 15% non-Latin chars, likely not English
        return (non_english_chars / total_chars) < 0.15


class TinyLlamaModel(LLMInterface):
    """TinyLlama 1.1B implementation using llama.cpp"""
    
    def __init__(self, model_path: str):
        from llama_cpp import Llama
        logger.info(f"Initializing TinyLlamaModel with path: {model_path}")
        try:
            # Use 4-bit quantization for reduced memory footprint
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,       # Context window size
                n_threads=4,      # Number of threads to use
                n_gpu_layers=0    # CPU only by default, adjust as needed
            )
            
            # Get model size from file
            self.model_path = model_path
            model_size = self.get_model_size()
            logger.info(f"Loaded TinyLlama model with size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize TinyLlamaModel: {e}")
            raise
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB from file size"""
        if hasattr(self, 'model_path') and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        # Format conversation history for TinyLlama
        prompt = self._format_tinyllama_prompt(conversation_history)
        
        logger.info("Generating response with TinyLlamaModel")
        try:
            output = self.model.create_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.7,
                stop=["<|im_end|>", "<|user|>"]
            )
            
            response = ""
            if isinstance(output, dict) and "choices" in output:
                response = output["choices"][0]["text"]
            elif isinstance(output, dict) and "text" in output:
                response = output["text"]
            else:
                response = str(output)
            
            # Ensure response is in English
            if not self._is_english(response):
                logger.warning("Non-English response detected, falling back to English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
            return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def _format_tinyllama_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for TinyLlama model"""
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
    
    def _is_english(self, text: str) -> bool:
        """Simple check to detect if text is primarily English"""
        # This is a basic implementation - could be enhanced with language detection libraries
        non_english_chars = 0
        total_chars = max(1, len(text.strip()))
        
        for char in text:
            # Check if character is outside basic Latin alphabet range
            if char.isalpha() and ord(char) > 127:
                non_english_chars += 1
        
        # If more than 15% non-Latin chars, likely not English
        return (non_english_chars / total_chars) < 0.15


class RWKVModel(LLMInterface):
    """RWKV model implementation - RNN with transformer-level performance"""
    
    def __init__(self, model_path: str):
        logger.info(f"Initializing RWKVModel with path: {model_path}")
        try:
            # Import needed for RWKV
            import rwkv
            from rwkv.model import RWKV
            from rwkv.utils import PIPELINE
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize RWKV model
            self.model = RWKV(model=model_path, strategy=f'{self.device} fp16')
            self.pipeline = PIPELINE(self.model)
            
            # Get model size from file
            self.model_path = model_path
            model_size = self.get_model_size()
            logger.info(f"Loaded RWKV model with size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize RWKVModel: {e}")
            raise
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB from file size"""
        if hasattr(self, 'model_path') and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        # Format conversation history for RWKV
        prompt = self._format_rwkv_prompt(conversation_history)
        
        logger.info("Generating response with RWKVModel")
        try:
            # Initialize state
            state = None
            
            # Process prompt to get state
            for i in range(0, len(prompt), 64):
                chunk = prompt[i:i+64]
                output, state = self.pipeline.forward(chunk, state)
            
            # Generate response
            response = ""
            for _ in range(256):  # Generate up to 256 tokens
                output, state = self.pipeline.forward("\n", state)
                token = self.pipeline.sample_logits(output, temperature=0.7)
                if token == '\n\n':
                    break
                response += token
            
            # Ensure response is in English
            if not self._is_english(response):
                logger.warning("Non-English response detected, falling back to English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error processing your request."
    
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
    
    def _is_english(self, text: str) -> bool:
        """Simple check to detect if text is primarily English"""
        # This is a basic implementation - could be enhanced with language detection libraries
        non_english_chars = 0
        total_chars = max(1, len(text.strip()))
        
        for char in text:
            # Check if character is outside basic Latin alphabet range
            if char.isalpha() and ord(char) > 127:
                non_english_chars += 1
        
        # If more than 15% non-Latin chars, likely not English
        return (non_english_chars / total_chars) < 0.15
