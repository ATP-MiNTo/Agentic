# LLM interface for Llama via Ollama
import requests
from typing import Optional, Generator
import config
from src.utils.logger import get_logger

logger = get_logger()


class LLMInterface:
    """Interface for Llama LLM via Ollama."""
    
    def __init__(self, model_name: str = None, endpoint: str = None):
        """Initialize LLM interface.
        
        Args:
            model_name: Name of model. If None, uses config.LLM_MODEL
            endpoint: Ollama endpoint. If None, uses config.LLM_ENDPOINT
        """
        self.model_name = model_name or config.LLM_MODEL
        self.endpoint = endpoint or config.LLM_ENDPOINT
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is running and model is available."""
        try:
            logger.info(f"Checking Ollama connection at {self.endpoint}")
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            if self.model_name in model_names or any(self.model_name in name for name in model_names):
                logger.info(f"✓ Model '{self.model_name}' is available")
            else:
                logger.warning(f"Model '{self.model_name}' not found in Ollama")
                logger.info(f"Available models: {model_names}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.endpoint}")
            logger.info("Make sure Ollama is running: 'ollama serve'")
            raise
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {str(e)}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        stream: bool = False
    ) -> str:
        """Generate response from LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate. If None, uses config
            temperature: Sampling temperature. If None, uses config
            top_p: Nucleus sampling parameter. If None, uses config
            stream: Whether to stream response
        
        Returns:
            Generated text
        """
        if max_tokens is None:
            max_tokens = config.MAX_TOKENS
        if temperature is None:
            temperature = config.TEMPERATURE
        if top_p is None:
            top_p = config.TOP_P
        
        logger.debug(f"Generating response with {self.model_name}")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": stream,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                timeout=config.LLM_TIMEOUT
            )
            response.raise_for_status()
            
            if stream:
                # For streaming, handle line by line
                generated_text = ""
                for line in response.iter_lines():
                    if line:
                        data = response.json()
                        if "response" in data:
                            generated_text += data["response"]
                            if data.get("done", False):
                                break
                return generated_text
            else:
                # For non-streaming, combine all response chunks
                generated_text = ""
                for line in response.iter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                generated_text += data["response"]
                        except json.JSONDecodeError:
                            continue
                
                return generated_text.strip()
                
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timed out (>{config.LLM_TIMEOUT}s)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None
    ) -> Generator[str, None, None]:
        """Generate response from LLM with streaming.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Yields:
            Generated text chunks
        """
        if max_tokens is None:
            max_tokens = config.MAX_TOKENS
        if temperature is None:
            temperature = config.TEMPERATURE
        if top_p is None:
            top_p = config.TOP_P
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                stream=True,
                timeout=config.LLM_TIMEOUT
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough estimate: 1 token ≈ 4 chars).
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        # Rough heuristic: ~4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> dict:
        """Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        try:
            response = requests.get(
                f"{self.endpoint}/api/show",
                json={"name": self.model_name},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
