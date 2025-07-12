"""
LLM Provider abstraction system for Loki Code.

This module provides a flexible, extensible system for integrating multiple
LLM providers while maintaining a consistent interface. It supports provider
switching, fallback mechanisms, and task-aware model selection.

Usage:
    from loki_code.core.providers import create_llm_provider
    
    provider = create_llm_provider(config)
    response = await provider.generate(GenerationRequest(prompt="Hello!"))
"""

from .base import (
    # Core abstractions
    BaseLLMProvider,
    ProviderRegistry,
    provider_registry,
    
    # Data classes
    ProviderCapabilities,
    ModelInfo,
    GenerationRequest,
    GenerationResponse,
    
    # Enums
    ProviderType,
    ModelType,
    
    # Exceptions
    LLMProviderError,
    ProviderConnectionError,
    ProviderModelError,
    ProviderTimeoutError,
    ProviderRateLimitError,
)

from .factory import (
    # Factory classes
    LLMProviderFactory,
    ProviderConfig,
    llm_factory,
    
    # Convenience functions
    create_llm_provider,
    create_llm_provider_with_fallback,
    list_available_providers,
    get_recommended_provider,
)

from .ollama import OllamaProvider

# Version info
__version__ = "0.1.0"

# Provider registry - automatically populated by imports
__all__ = [
    # Core abstractions
    "BaseLLMProvider",
    "ProviderRegistry",
    "provider_registry",
    
    # Data classes
    "ProviderCapabilities", 
    "ModelInfo",
    "GenerationRequest",
    "GenerationResponse",
    
    # Enums
    "ProviderType",
    "ModelType",
    
    # Exceptions
    "LLMProviderError",
    "ProviderConnectionError", 
    "ProviderModelError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    
    # Factory
    "LLMProviderFactory",
    "ProviderConfig",
    "llm_factory",
    "create_llm_provider",
    "create_llm_provider_with_fallback", 
    "list_available_providers",
    "get_recommended_provider",
    
    # Providers
    "OllamaProvider",
]

# Provider information for discovery
PROVIDERS = {
    "ollama": {
        "class": "OllamaProvider",
        "module": "loki_code.core.providers.ollama",
        "description": "Local Ollama LLM provider",
        "capabilities": ["streaming", "model_switching", "local_execution"],
        "supported": True
    },
    # Future providers can be added here
    "openai": {
        "class": "OpenAIProvider",
        "module": "loki_code.core.providers.openai", 
        "description": "OpenAI API provider",
        "capabilities": ["streaming", "function_calling", "multimodal"],
        "supported": False  # Not implemented yet
    },
    "anthropic": {
        "class": "AnthropicProvider",
        "module": "loki_code.core.providers.anthropic",
        "description": "Anthropic Claude API provider", 
        "capabilities": ["streaming", "large_context"],
        "supported": False  # Not implemented yet
    }
}


def get_provider_info(provider_name: str = None):
    """Get information about providers.
    
    Args:
        provider_name: Specific provider name, or None for all providers
        
    Returns:
        Provider information dictionary or list of dictionaries
    """
    if provider_name:
        return PROVIDERS.get(provider_name.lower())
    return PROVIDERS


def is_provider_supported(provider_name: str) -> bool:
    """Check if a provider is supported.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        True if provider is supported and available
    """
    info = get_provider_info(provider_name)
    return info is not None and info.get("supported", False)


def get_supported_providers():
    """Get list of supported provider names.
    
    Returns:
        List of supported provider names
    """
    return [name for name, info in PROVIDERS.items() if info.get("supported", False)]