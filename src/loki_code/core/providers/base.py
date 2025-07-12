"""
Abstract base classes for LLM providers in Loki Code.

This module defines the interfaces and base classes that all LLM providers
must implement, ensuring consistent behavior across different providers
while allowing provider-specific optimizations and capabilities.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncIterator, Any, Union
from enum import Enum
import logging

from ...utils.logging import get_logger


class ProviderType(Enum):
    """Supported provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelType(Enum):
    """Model specialization types for task-aware selection."""
    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"
    MULTIMODAL = "multimodal"


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    display_name: str
    description: str
    model_type: ModelType
    context_length: int
    cost_per_input_token: Optional[float] = None
    cost_per_output_token: Optional[float] = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_multimodal: bool = False
    parameters: Optional[str] = None  # e.g., "7B", "13B"
    

@dataclass
class ProviderCapabilities:
    """Capabilities and metadata for an LLM provider."""
    provider_type: ProviderType
    provider_name: str
    supports_streaming: bool
    supports_function_calling: bool
    supports_multimodal: bool
    supports_model_switching: bool
    supports_async: bool
    max_context_length: int
    available_models: List[ModelInfo] = field(default_factory=list)
    default_model: Optional[str] = None
    connection_pooling: bool = False
    rate_limiting: Optional[Dict[str, Any]] = None
    

@dataclass
class GenerationRequest:
    """Standard request format for text generation."""
    prompt: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    # Provider-specific options
    provider_options: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Standard response format for text generation."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    response_time_ms: Optional[float] = None
    # Provider-specific metadata
    provider_metadata: Optional[Dict[str, Any]] = None


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class ProviderConnectionError(LLMProviderError):
    """Connection-related provider errors."""
    pass


class ProviderModelError(LLMProviderError):
    """Model-related provider errors."""
    pass


class ProviderTimeoutError(LLMProviderError):
    """Timeout-related provider errors."""
    pass


class ProviderRateLimitError(LLMProviderError):
    """Rate limiting errors."""
    pass


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM providers must implement,
    ensuring consistent behavior across different providers while allowing
    for provider-specific optimizations and capabilities.
    """
    
    def __init__(self, config: Any):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration object
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._capabilities: Optional[ProviderCapabilities] = None
        
    # Core generation methods (must be implemented by all providers)
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a prompt.
        
        Args:
            request: Generation request with prompt and parameters
            
        Returns:
            GenerationResponse with generated text and metadata
            
        Raises:
            ProviderConnectionError: If connection fails
            ProviderModelError: If model returns error
            ProviderTimeoutError: If request times out
        """
        pass
    
    @abstractmethod
    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate text with streaming response.
        
        Args:
            request: Generation request with prompt and parameters
            
        Yields:
            Individual tokens or text chunks as they're generated
            
        Raises:
            ProviderConnectionError: If connection fails
            ProviderModelError: If model returns error
            ProviderTimeoutError: If request times out
        """
        pass
    
    # Health and capability methods
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and responding.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities and metadata.
        
        Returns:
            ProviderCapabilities object describing what this provider supports
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models from this provider.
        
        Returns:
            List of ModelInfo objects describing available models
        """
        pass
    
    # Optional methods with default implementations
    
    async def validate_model(self, model_name: str) -> bool:
        """Validate that a model is available and accessible.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is valid and accessible, False otherwise
        """
        try:
            models = await self.list_models()
            return any(model.name == model_name for model in models)
        except Exception as e:
            self.logger.error(f"Failed to validate model {model_name}: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object if found, None otherwise
        """
        try:
            models = await self.list_models()
            return next((model for model in models if model.name == model_name), None)
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    async def estimate_cost(self, request: GenerationRequest) -> Optional[float]:
        """Estimate the cost for a generation request.
        
        Args:
            request: Generation request to estimate cost for
            
        Returns:
            Estimated cost in USD, or None if cost estimation not available
        """
        # Default implementation - providers can override for accurate estimates
        return None
    
    async def supports_feature(self, feature: str) -> bool:
        """Check if provider supports a specific feature.
        
        Args:
            feature: Feature name (e.g., "streaming", "function_calling")
            
        Returns:
            True if feature is supported, False otherwise
        """
        try:
            capabilities = await self.get_capabilities()
            return getattr(capabilities, f"supports_{feature}", False)
        except Exception:
            return False
    
    # Synchronous wrapper methods for backward compatibility
    
    def generate_sync(self, request: GenerationRequest) -> GenerationResponse:
        """Synchronous wrapper for generate method.
        
        Args:
            request: Generation request
            
        Returns:
            GenerationResponse
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(request))
    
    def health_check_sync(self) -> bool:
        """Synchronous wrapper for health_check method.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.health_check())
    
    def list_models_sync(self) -> List[ModelInfo]:
        """Synchronous wrapper for list_models method.
        
        Returns:
            List of ModelInfo objects
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.list_models())
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources (override in subclasses if needed)."""
        pass
    
    # Provider identification
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(provider_name='{self.provider_name}', config={self.config})"


class ProviderRegistry:
    """Registry for managing available LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, type] = {}
        self.logger = get_logger(__name__)
    
    def register(self, provider_name: str, provider_class: type):
        """Register a provider class.
        
        Args:
            provider_name: Name to register the provider under
            provider_class: Provider class (must inherit from BaseLLMProvider)
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"Provider class must inherit from BaseLLMProvider")
        
        self._providers[provider_name.lower()] = provider_class
        self.logger.debug(f"Registered provider: {provider_name}")
    
    def get_provider_class(self, provider_name: str) -> Optional[type]:
        """Get a provider class by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider class if found, None otherwise
        """
        return self._providers.get(provider_name.lower())
    
    def list_providers(self) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List of registered provider names
        """
        return list(self._providers.keys())
    
    def is_registered(self, provider_name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider is registered, False otherwise
        """
        return provider_name.lower() in self._providers


# Global provider registry instance
provider_registry = ProviderRegistry()