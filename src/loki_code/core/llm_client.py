"""
LLM Client for Loki Code - Basic communication with local LLMs.

This module provides a simple interface for communicating with local LLMs,
particularly Ollama. It handles both streaming and non-streaming responses,
includes error handling and retry logic, and integrates with the logging system.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
from enum import Enum
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

from ..utils.logging import get_logger, log_performance
from ..config.models import LokiCodeConfig


class ResponseFormat(Enum):
    """Response format options."""
    STREAMING = "streaming"
    COMPLETE = "complete"


@dataclass
class LLMResponse:
    """LLM response container."""
    content: str
    model: str
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    response_time_ms: Optional[float] = None
    finished: bool = True
    error: Optional[str] = None


@dataclass
class LLMRequest:
    """LLM request container."""
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    system_prompt: Optional[str] = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMClientError):
    """Connection-related errors."""
    pass


class LLMModelError(LLMClientError):
    """Model-related errors."""
    pass


class LLMTimeoutError(LLMClientError):
    """Timeout-related errors."""
    pass


class OllamaClient:
    """
    Simple client for communicating with Ollama LLMs.
    
    Provides methods for sending prompts and receiving responses,
    with support for both streaming and non-streaming modes.
    """
    
    def __init__(self, config: LokiCodeConfig):
        """Initialize the Ollama client with configuration.
        
        Args:
            config: LokiCodeConfig instance containing LLM settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.llm.base_url.rstrip('/')
        self.default_model = config.llm.model
        self.timeout = config.llm.timeout
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.timeout = self.timeout
        
        # Track conversation for logging
        self.conversation_count = 0
        
    def send_prompt(
        self, 
        request: Union[str, LLMRequest],
        stream: bool = False,
        retry_count: int = 3
    ) -> Union[LLMResponse, Iterator[str]]:
        """Send a prompt to the LLM and get response.
        
        Args:
            request: Prompt string or LLMRequest object
            stream: Whether to stream the response
            retry_count: Number of retries on failure
            
        Returns:
            LLMResponse for complete responses, Iterator[str] for streaming
            
        Raises:
            LLMConnectionError: If connection fails
            LLMModelError: If model returns error
            LLMTimeoutError: If request times out
        """
        # Convert string to LLMRequest if needed
        if isinstance(request, str):
            request = LLMRequest(prompt=request, stream=stream)
        else:
            request.stream = stream
            
        self.conversation_count += 1
        start_time = time.perf_counter()
        
        # Log the request (truncated for privacy)
        prompt_preview = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
        self.logger.debug(f"Sending prompt #{self.conversation_count}: {prompt_preview}")
        
        for attempt in range(retry_count + 1):
            try:
                if stream:
                    return self._send_streaming_request(request, attempt)
                else:
                    return self._send_complete_request(request, attempt, start_time)
                    
            except (ConnectionError, LLMConnectionError) as e:
                if attempt < retry_count:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Connection failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise LLMConnectionError(f"Failed to connect after {retry_count + 1} attempts: {e}")
                    
            except LLMTimeoutError as e:
                if attempt < retry_count:
                    self.logger.warning(f"Request timed out (attempt {attempt + 1}), retrying: {e}")
                    continue
                else:
                    raise
                    
            except LLMModelError as e:
                # Don't retry model errors
                raise
                
        # Should never reach here
        raise LLMClientError("Unexpected error in retry loop")
    
    def _send_complete_request(self, request: LLMRequest, attempt: int, start_time: float) -> LLMResponse:
        """Send a complete (non-streaming) request."""
        payload = self._build_payload(request)
        
        try:
            with log_performance(f"LLM request (attempt {attempt + 1})", level=logging.DEBUG):
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=max(self.timeout, 30)  # Allow extra time for generation
                )
            
            if response.status_code != 200:
                error_msg = f"LLM request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                raise LLMModelError(error_msg)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise LLMModelError(f"Invalid JSON response: {e}")
            
            # Extract response content
            content = data.get('response', '').strip()
            if not content:
                self.logger.warning("LLM returned empty response")
            
            # Calculate timing
            response_time = (time.perf_counter() - start_time) * 1000
            
            # Log successful response (truncated)
            content_preview = content[:100] + "..." if len(content) > 100 else content
            self.logger.debug(f"Received response #{self.conversation_count} ({response_time:.1f}ms): {content_preview}")
            
            return LLMResponse(
                content=content,
                model=data.get('model', request.model or self.default_model),
                response_time_ms=response_time,
                finished=data.get('done', True),
                total_tokens=self._extract_token_count(data),
                prompt_tokens=self._extract_prompt_tokens(data),
                completion_tokens=self._extract_completion_tokens(data)
            )
            
        except Timeout as e:
            raise LLMTimeoutError(f"Request timed out after {self.timeout}s: {e}")
        except ConnectionError as e:
            raise LLMConnectionError(f"Connection failed: {e}")
        except RequestException as e:
            raise LLMConnectionError(f"Request failed: {e}")
    
    def _send_streaming_request(self, request: LLMRequest, attempt: int) -> Iterator[str]:
        """Send a streaming request and yield tokens as they arrive."""
        payload = self._build_payload(request)
        payload['stream'] = True
        
        try:
            with log_performance(f"LLM streaming request (attempt {attempt + 1})", level=logging.DEBUG):
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=max(self.timeout, 30),
                    stream=True
                )
            
            if response.status_code != 200:
                error_msg = f"Streaming request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                raise LLMModelError(error_msg)
            
            # Process streaming response
            full_content = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    try:
                        data = json.loads(line)
                        token = data.get('response', '')
                        if token:
                            full_content += token
                            yield token
                        
                        # Check if done
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse streaming response line: {e}")
                        continue
            
            # Log completed streaming response
            content_preview = full_content[:100] + "..." if len(full_content) > 100 else full_content
            self.logger.debug(f"Completed streaming response #{self.conversation_count}: {content_preview}")
            
        except Timeout as e:
            raise LLMTimeoutError(f"Streaming request timed out: {e}")
        except ConnectionError as e:
            raise LLMConnectionError(f"Streaming connection failed: {e}")
        except RequestException as e:
            raise LLMConnectionError(f"Streaming request failed: {e}")
    
    def _build_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Build the request payload for Ollama API."""
        payload = {
            "model": request.model or self.default_model,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {}
        }
        
        # Add temperature if specified
        temperature = request.temperature if request.temperature is not None else self.temperature
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        
        # Add max tokens if specified
        max_tokens = request.max_tokens if request.max_tokens is not None else self.max_tokens
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        # Add system prompt if specified
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        return payload
    
    def _extract_token_count(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract total token count from response data."""
        # Ollama provides various timing/count fields
        eval_count = data.get('eval_count')
        prompt_eval_count = data.get('prompt_eval_count')
        
        if eval_count is not None and prompt_eval_count is not None:
            return eval_count + prompt_eval_count
        return None
    
    def _extract_prompt_tokens(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract prompt token count from response data."""
        return data.get('prompt_eval_count')
    
    def _extract_completion_tokens(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract completion token count from response data."""
        return data.get('eval_count')
    
    def test_connection(self) -> bool:
        """Test if the LLM is available and responding.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.send_prompt("Hello! Please respond with 'OK' to confirm connection.")
            return isinstance(response, LLMResponse) and response.content.strip().lower() in ['ok', 'hello']
        except Exception as e:
            self.logger.debug(f"Connection test failed: {e}")
            return False
    
    def __del__(self):
        """Clean up the requests session."""
        if hasattr(self, 'session'):
            self.session.close()


def create_llm_client(config: LokiCodeConfig) -> OllamaClient:
    """Factory function to create an LLM client.
    
    Args:
        config: LokiCodeConfig instance
        
    Returns:
        Configured LLM client instance
    """
    # For now, we only support Ollama, but this can be extended
    # to support other providers based on config.llm.provider
    return OllamaClient(config)