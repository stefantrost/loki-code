"""
Ollama provider implementation for Loki Code.

This module implements the Ollama LLM provider, wrapping the existing
llm_client functionality and adapting it to the provider abstraction interface.
"""

import asyncio
import time
from typing import List, Optional, AsyncIterator, Dict, Any
import json

from ...utils.error_handling import handle_provider_operation, AsyncProviderMixin

from .base import (
    BaseLLMProvider, 
    ProviderCapabilities, 
    ProviderType,
    ModelInfo, 
    ModelType,
    GenerationRequest, 
    GenerationResponse,
    ProviderConnectionError,
    ProviderModelError,
    ProviderTimeoutError
)
from ..llm_client import LLMClient, LLMClientError, LLMConnectionError, LLMServerError, LLMTimeoutError
from ...config.models import LokiCodeConfig


class OllamaProvider(BaseLLMProvider, AsyncProviderMixin):
    """
    Ollama provider implementation.
    
    This provider wraps the existing LLMClient functionality and adapts it
    to the provider abstraction interface, maintaining backward compatibility
    while enabling the provider abstraction system.
    """
    
    def __init__(self, config: LokiCodeConfig):
        """Initialize the Ollama provider.
        
        Args:
            config: LokiCodeConfig instance containing LLM settings
        """
        super().__init__(config)
        self.client = LLMClient(
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        self._cached_models: Optional[List[ModelInfo]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = 300  # Cache models for 5 minutes
        
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a prompt using Ollama.
        
        Args:
            request: Generation request with prompt and parameters
            
        Returns:
            GenerationResponse with generated text and metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare inference config
            config = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": request.stop_sequences
            }
            
            # Use new client interface
            response = await self.client.inference(
                message=request.prompt,
                config=config,
                model_name=request.model_name
            )
            
            # Convert client response to provider response
            return self._convert_to_generation_response(response, request, start_time)
            
        except LLMConnectionError as e:
            raise ProviderConnectionError(str(e), "ollama") from e
        except LLMServerError as e:
            raise ProviderModelError(str(e), "ollama") from e
        except LLMTimeoutError as e:
            raise ProviderTimeoutError(str(e), "ollama") from e
        except LLMClientError as e:
            raise ProviderConnectionError(str(e), "ollama") from e
    
    @handle_provider_operation("stream_generate", timeout=180.0)
    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate text with streaming response using Ollama.
        
        Args:
            request: Generation request with prompt and parameters
            
        Yields:
            Individual tokens or text chunks as they're generated
        """
        # Prepare inference config
        config = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop_sequences
        }
        
        # Use streaming inference
        async for chunk in self.client.inference_stream(
            message=request.prompt,
            config=config,
            model_name=request.model_name
        ):
            yield chunk
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy and responding.
        
        Returns:
            True if Ollama is healthy, False otherwise
        """
        try:
            # Use existing client's test connection method
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.client.test_connection)
        except Exception as e:
            self.logger.debug(f"Ollama health check failed: {e}")
            return False
    
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Ollama provider capabilities.
        
        Returns:
            ProviderCapabilities describing Ollama's features
        """
        if self._capabilities is None:
            # Get available models for capabilities
            try:
                models = await self.list_models()
                max_context = max((model.context_length for model in models), default=4096)
                default_model = self.config.llm.model
            except Exception:
                models = []
                max_context = 4096
                default_model = None
            
            self._capabilities = ProviderCapabilities(
                provider_type=ProviderType.OLLAMA,
                provider_name="Ollama",
                supports_streaming=True,
                supports_function_calling=False,  # Ollama doesn't have native function calling
                supports_multimodal=True,  # Some Ollama models support vision
                supports_model_switching=True,
                supports_async=True,
                max_context_length=max_context,
                available_models=models,
                default_model=default_model,
                connection_pooling=True,
                rate_limiting=None  # Ollama doesn't have built-in rate limiting
            )
        
        return self._capabilities
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from Ollama.
        
        Returns:
            List of ModelInfo objects describing available Ollama models
        """
        # Check cache first
        current_time = time.time()
        if (self._cached_models is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._cached_models
        
        try:
            # Use existing client session to get model list
            loop = asyncio.get_event_loop()
            
            def _get_models():
                response = self.client.session.get(
                    f"{self.client.base_url}/api/tags",
                    timeout=self.client.timeout
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    raise ProviderConnectionError(
                        f"Failed to list models: HTTP {response.status_code}",
                        "ollama"
                    )
            
            data = await loop.run_in_executor(None, _get_models)
            models_data = data.get('models', [])
            
            # Convert to ModelInfo objects
            models = []
            for model_data in models_data:
                model_name = model_data.get('name', 'unknown')
                model_info = self._parse_ollama_model(model_name, model_data)
                models.append(model_info)
            
            # Cache the results
            self._cached_models = models
            self._cache_timestamp = current_time
            
            return models
            
        except Exception as e:
            if isinstance(e, ProviderConnectionError):
                raise
            raise ProviderConnectionError(f"Failed to list Ollama models: {e}", "ollama") from e
    
    
    def _convert_to_generation_response(
        self, 
        response, 
        original_request: GenerationRequest,
        start_time: float
    ) -> GenerationResponse:
        """Convert client response to provider response format.
        
        Args:
            response: AgentResponse from the LLM client
            original_request: Original generation request
            start_time: Request start time for timing calculation
            
        Returns:
            GenerationResponse in provider format
        """
        response_time = (time.perf_counter() - start_time) * 1000
        
        return GenerationResponse(
            content=response.content,
            model=original_request.model_name or "unknown",
            provider="ollama",
            finish_reason="stop",
            total_tokens=response.metadata.get("total_tokens", 0) if response.metadata else 0,
            prompt_tokens=response.metadata.get("prompt_tokens", 0) if response.metadata else 0,
            completion_tokens=response.metadata.get("completion_tokens", 0) if response.metadata else 0,
            response_time_ms=response_time,
            provider_metadata={
                "response_time": response_time,
                "model_loaded": True,
                "confidence": response.confidence
            }
        )
    
    def _parse_ollama_model(self, model_name: str, model_data: Dict[str, Any]) -> ModelInfo:
        """Parse Ollama model data into ModelInfo format.
        
        Args:
            model_name: Name of the model
            model_data: Raw model data from Ollama API
            
        Returns:
            ModelInfo object with parsed information
        """
        # Determine model type based on name patterns
        model_type = ModelType.GENERAL_CHAT
        if any(keyword in model_name.lower() for keyword in ['code', 'coding', 'codellama', 'starcoder']):
            model_type = ModelType.CODE_GENERATION
        elif 'embed' in model_name.lower():
            model_type = ModelType.EMBEDDING
        elif any(keyword in model_name.lower() for keyword in ['vision', 'multimodal', 'llava']):
            model_type = ModelType.MULTIMODAL
        
        # Extract parameter size if available
        parameters = None
        for size in ['7b', '13b', '30b', '65b', '70b']:
            if size in model_name.lower():
                parameters = size.upper()
                break
        
        # Estimate context length based on model
        context_length = 4096  # Default
        if 'llama' in model_name.lower():
            context_length = 8192
        elif any(keyword in model_name.lower() for keyword in ['mistral', 'mixtral']):
            context_length = 32768
        elif 'claude' in model_name.lower():
            context_length = 200000
        
        # Check for multimodal support
        supports_multimodal = any(keyword in model_name.lower() 
                                for keyword in ['vision', 'multimodal', 'llava', 'clip'])
        
        return ModelInfo(
            name=model_name,
            display_name=model_name.replace(':', ' ').title(),
            description=f"Ollama model: {model_name}",
            model_type=model_type,
            context_length=context_length,
            parameters=parameters,
            supports_streaming=True,
            supports_function_calling=False,
            supports_multimodal=supports_multimodal
        )
    
    async def cleanup(self):
        """Clean up Ollama client resources."""
        if hasattr(self.client, 'session'):
            self.client.session.close()
    
    # Additional Ollama-specific methods
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model in Ollama.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _pull_model():
                response = self.client.session.post(
                    f"{self.client.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=300  # Pulling can take a while
                )
                return response.status_code == 200
            
            success = await loop.run_in_executor(None, _pull_model)
            if success:
                # Clear model cache since we have a new model
                self._cached_models = None
                self._cache_timestamp = None
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _delete_model():
                response = self.client.session.delete(
                    f"{self.client.base_url}/api/delete",
                    json={"name": model_name},
                    timeout=self.client.timeout
                )
                return response.status_code == 200
            
            success = await loop.run_in_executor(None, _delete_model)
            if success:
                # Clear model cache since we removed a model
                self._cached_models = None
                self._cache_timestamp = None
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    async def get_model_details(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with detailed model information, or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _get_model_details():
                response = self.client.session.post(
                    f"{self.client.base_url}/api/show",
                    json={"name": model_name},
                    timeout=self.client.timeout
                )
                if response.status_code == 200:
                    return response.json()
                return None
            
            return await loop.run_in_executor(None, _get_model_details)
            
        except Exception as e:
            self.logger.error(f"Failed to get model details for {model_name}: {e}")
            return None