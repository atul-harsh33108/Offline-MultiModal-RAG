"""
Ollama LLM integration for text generation
"""
import logging
from typing import Optional, List, Dict, Any
import ollama
from config import OLLAMA_BASE_URL, OLLAMA_MODELS, DEFAULT_LLM

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Wrapper for Ollama LLM inference"""
    
    def __init__(self, model_name: str = DEFAULT_LLM):
        self.model_name = model_name
        self.ollama_model = OLLAMA_MODELS.get(model_name, OLLAMA_MODELS[DEFAULT_LLM])
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Generate text response using Ollama
        
        Args:
            prompt: User query or prompt
            context: Retrieved context for RAG
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Construct full prompt with context
            full_prompt = prompt
            if context:
                full_prompt = f"""Context:
{context}

Question: {prompt}

Answer based on the context provided above. If the context doesn't contain relevant information, say so."""
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": full_prompt
            })
            
            # Generate response
            response = self.client.chat(
                model=self.ollama_model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise
    
    def stream_generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Stream text response using Ollama (generator)
        
        Args:
            prompt: User query or prompt
            context: Retrieved context for RAG
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Generated text chunks
        """
        try:
            # Construct full prompt with context
            full_prompt = prompt
            if context:
                full_prompt = f"""Context:
{context}

Question: {prompt}

Answer based on the context provided above. If the context doesn't contain relevant information, say so."""
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": full_prompt
            })
            
            # Stream response
            stream = self.client.chat(
                model=self.ollama_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error streaming response with Ollama: {e}")
            raise
    
    def set_model(self, model_name: str):
        """Change the model being used"""
        if model_name in OLLAMA_MODELS:
            self.model_name = model_name
            self.ollama_model = OLLAMA_MODELS[model_name]
            logger.info(f"Switched to model: {model_name}")
        else:
            logger.warning(f"Model {model_name} not found, keeping current model")

