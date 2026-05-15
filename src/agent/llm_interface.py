# Ollama-only LLM interface for standalone local inference
import json
import time
from typing import Generator, Optional, List

import requests
import config
from src.utils.logger import get_logger

logger = get_logger()


class LLMInterface:
    """Interface for local Ollama chat generation."""

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize Ollama model names for comparisons."""
        return (model_name or "").strip().lower().split("@")[0].split(":")[0]
    
    def __init__(self, model_name: str = None, endpoint: str = None):
        """Initialize LLM interface.
        
        Args:
            model_name: Name of model. If None, uses config.LLM_MODEL
            endpoint: Ollama endpoint. If None, uses config.LLM_ENDPOINT
        """
        self.model_name = model_name or config.LLM_MODEL
        self.endpoint = (endpoint or config.LLM_ENDPOINT).rstrip("/")

        # API path may vary across Ollama versions or deployments. Allow
        # an explicit override via config.LLM_API_PATH; otherwise autodetect.
        self.api_path = getattr(config, "LLM_API_PATH", None)

        # Initial lightweight connection check (models list) then determine API path
        self._check_connection()
        if not self.api_path:
            self.api_path = self._determine_api_path()
        logger.info(f"Using API path: {self.api_path}")
    
    def _check_connection(self):
        """Check if Ollama is running and model is available."""
        try:
            logger.info(f"Checking Ollama connection at {self.endpoint}")
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            # Find a server model name matching the requested normalized model.
            model_names = [m.get("name", "") for m in models]
            normalized_models = [self._normalize_model_name(n) for n in model_names]
            requested_model = self._normalize_model_name(self.model_name)

            resolved = None
            for orig, norm in zip(model_names, normalized_models):
                if requested_model == norm or requested_model in norm:
                    resolved = orig
                    break

            if resolved:
                # If the server exposes a more specific name (e.g., 'llama3.1:8b'),
                # use that exact identifier for generation calls.
                if resolved != self.model_name:
                    logger.info(f"Mapping requested model '{self.model_name}' to server model '{resolved}'")
                self.model_name = resolved
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
            response = self._post_with_retry(
                payload={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": stream,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                stream=stream,
            )
            return self._extract_chat_response(response, stream=stream)

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
            response = self._post_with_retry(
                payload={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                stream=True,
            )
            
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                if "content" in message:
                    yield message["content"]
                        
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            raise

    def _post_with_retry(self, payload: dict, stream: bool = False) -> requests.Response:
        """POST to Ollama with one retry for transient 5xx failures."""
        url = f"{self.endpoint}{self.api_path}"
        last_error = None

        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    stream=stream,
                    timeout=config.LLM_TIMEOUT,
                )

                if response.status_code >= 500:
                    logger.error(
                        f"LLM server error on attempt {attempt + 1} for {url}: "
                        f"{response.status_code} {response.text[:500]}"
                    )
                    response.raise_for_status()

                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = getattr(exc.response, "status_code", None)
                if status_code and 500 <= status_code < 600 and attempt == 0:
                    time.sleep(1)
                    continue
                raise

            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("Failed to post to LLM endpoint")

    @staticmethod
    def _extract_chat_response(response: requests.Response, stream: bool) -> str:
        """Extract assistant text from Ollama chat responses."""
        if stream:
            generated_text = ""
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                if "content" in message:
                    generated_text += message["content"]

            return generated_text.strip()

        data = response.json()
        message = data.get("message", {})
        return message.get("content", "").strip()
    
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
            response = requests.get(f"{self.endpoint}/api/show", json={"name": self.model_name}, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}

    def _determine_api_path(self) -> str:
        """Probe the Ollama endpoint to find a working chat/generation API path.

        Tries a small list of known candidate paths and returns the first path
        that does not return HTTP 404. If none match, returns '/api/generate'
        as a sensible default.
        """
        candidates: List[str] = [
            "/api/chat",
            "/api/generate",
            "/api/completions",
            "/v1/chat/completions",
            "/api/complete",
        ]

        for path in candidates:
            try:
                probe_url = f"{self.endpoint}{path}"
                logger.debug(f"Probing Ollama path: {probe_url}")
                resp = requests.options(probe_url, timeout=3)
                # Accept 200 OK, 204 No Content, 405 Method Not Allowed (endpoint exists),
                # or 401/403 (protected) as indicators the path exists.
                if resp.status_code in (200, 204, 405, 401, 403):
                    return path
            except requests.exceptions.RequestException:
                continue

        logger.warning("Could not detect Ollama API path; defaulting to /api/generate."
                       " If you continue to see 404s, set `LLM_API_PATH` in config.py to the correct path.")
        return "/api/generate"
