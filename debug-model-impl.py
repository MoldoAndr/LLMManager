import os
import logging
import torch
from typing import List, Dict

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMInterface:
    """Base interface for LLM models"""
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a response based on conversation history"""
        raise NotImplementedError("Subclasses must implement generate")
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB"""
        return 0.0

class Phi2Model(LLMInterface):
    """Microsoft's Phi-2 model implementation using Transformers"""
    
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Initializing Phi2Model with path: {model_path}")
        try:
            is_local_file = os.path.exists(model_path) and model_path.endswith('.gguf')
            logger.debug(f"Is local file: {is_local_file}")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            if is_local_file:
                logger.debug("Loading from local GGUF file")
                from llama_cpp import Llama
                
                try:
                    logger.debug(f"Attempting to load with llama_cpp from {model_path}")
                    self.model = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=4,
                        n_gpu_layers=0
                    )
                    logger.info("Successfully loaded GGUF file with llama_cpp")
                    self.model_path = model_path
                    self.is_llama_cpp = True
                except Exception as llama_err:
                    logger.error(f"Failed to load with llama_cpp: {llama_err}")
                    raise
            else:
                logger.debug("Loading from Hugging Face model ID")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        load_in_4bit=True,
                        device_map="auto"
                    )
                    logger.info("Successfully loaded model from Hugging Face")
                    self.is_llama_cpp = False
                except Exception as hf_err:
                    logger.error(f"Failed to load from Hugging Face: {hf_err}")
                    raise
            
            # Check model size
            model_size = self.get_model_size()
            logger.info(f"Loaded Phi2 model with size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phi2Model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB"""
        if hasattr(self, 'is_llama_cpp') and self.is_llama_cpp:
            if hasattr(self, 'model_path') and os.path.exists(self.model_path):
                return os.path.getsize(self.model_path) / (1024 * 1024)
        elif hasattr(self, 'model'):
            model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            return model_size_bytes / (1024 * 1024)
        return 0.0
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
        logger.info("Generating response with Phi2Model")
        
        try:
            if hasattr(self, 'is_llama_cpp') and self.is_llama_cpp:
                prompt = self._format_phi2_prompt(conversation_history)
                logger.debug(f"Using prompt for llama_cpp: {prompt[:100]}...")
                
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
                
            else:
                prompt = self._format_phi2_prompt(conversation_history)
                logger.debug(f"Using prompt for Hugging Face: {prompt[:100]}...")
                
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
            
            if not self._is_english(response):
                logger.warning("Non-English response detected, falling back to English")
                response = "I apologize, but I can only respond in English. " + \
                           "Please ask your question in English."
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        non_english_chars = 0
        total_chars = max(1, len(text.strip()))
        
        for char in text:
            if char.isalpha() and ord(char) > 127:
                non_english_chars += 1
        
        return (non_english_chars / total_chars) < 0.15


class TinyLlamaModel(LLMInterface):
    """TinyLlama 1.1B implementation using llama.cpp"""
    
    def __init__(self, model_path: str):
        from llama_cpp import Llama
        logger.info(f"Initializing TinyLlamaModel with path: {model_path}")
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4, 
                n_gpu_layers=0
            )
            
            # Get model size from file
            self.model_path = model_path
            model_size = self.get_model_size()
            logger.info(f"Loaded TinyLlama model with size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize TinyLlamaModel: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_model_size(self) -> float:
        """Get the size of the model in MB from file size"""
        if hasattr(self, 'model_path') and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def generate(self, conversation_history: List[Dict[str, str]]) -> str:
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
        non_english_chars = 0
        total_chars = max(1, len(text.strip()))
        for char in text:
            if char.isalpha() and ord(char) > 127:
                non_english_chars += 1
        return (non_english_chars / total_chars) < 0.15
