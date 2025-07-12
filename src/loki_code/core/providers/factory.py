"""
Provider factory for creating LLM providers in Loki Code.

This module implements the factory pattern for creating LLM providers,
allowing dynamic provider selection based on configuration while maintaining
a clean separation of concerns and easy extensibility.
"""

from typing import Optional, Dict, Any, List
import importlib
from dataclasses import dataclass

from .base import BaseLLMProvider, ProviderType, provider_registry, ProviderConnectionError
from ...config.models import LokiCodeConfig
from ...utils.logging import get_logger


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    provider_type: str
    enabled: bool = True
    priority: int = 0  # Higher priority providers are preferred
    config: Optional[Dict[str, Any]] = None
    fallback: bool = False  # Whether this provider can be used as fallback


class LLMProviderFactory:
    """
    Factory for creating and managing LLM providers.
    
    This factory handles provider creation, registration, and selection
    based on configuration. It supports dynamic loading of providers
    and provides fallback mechanisms for robustness.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._provider_configs: Dict[str, ProviderConfig] = {}
        self._initialize_builtin_providers()
    
    def _initialize_builtin_providers(self):
        """Initialize built-in providers."""
        try:
            # Import and register Ollama provider
            from .ollama import OllamaProvider
            provider_registry.register("ollama", OllamaProvider)
            self.logger.debug("Registered Ollama provider")
        except ImportError as e:
            self.logger.warning(f"Could not register Ollama provider: {e}")
    
    def create_provider(
        self, 
        config: LokiCodeConfig, 
        provider_name: Optional[str] = None
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.
        
        Args:
            config: Configuration object containing LLM settings
            provider_name: Specific provider to create (uses config.llm.provider if None)
            
        Returns:
            BaseLLMProvider instance
            
        Raises:
            ValueError: If provider is not supported
            ProviderConnectionError: If provider creation fails
        """
        # Use specified provider or fall back to config
        target_provider = provider_name or config.llm.provider.lower()
        
        self.logger.debug(f"Creating provider: {target_provider}")
        
        # Get provider class from registry
        provider_class = provider_registry.get_provider_class(target_provider)
        if provider_class is None:
            # Try to load provider dynamically
            provider_class = self._load_provider_dynamically(target_provider)
            
        if provider_class is None:
            available = provider_registry.list_providers()
            raise ValueError(
                f"Unsupported LLM provider: {target_provider}. "
                f"Available providers: {', '.join(available)}"
            )
        
        try:
            # Create provider instance
            provider = provider_class(config)
            self.logger.info(f"Created {target_provider} provider successfully")
            return provider
            
        except Exception as e:
            self.logger.error(f"Failed to create {target_provider} provider: {e}")
            raise ProviderConnectionError(
                f"Failed to initialize {target_provider} provider: {e}",
                target_provider
            ) from e
    
    def create_provider_with_fallback(
        self,
        config: LokiCodeConfig,
        primary_provider: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None
    ) -> BaseLLMProvider:
        """Create a provider with fallback options.
        
        Args:
            config: Configuration object
            primary_provider: Primary provider to try first
            fallback_providers: List of fallback providers to try
            
        Returns:
            BaseLLMProvider instance (primary or fallback)
            
        Raises:
            ProviderConnectionError: If all providers fail
        """
        providers_to_try = []
        
        # Add primary provider
        if primary_provider:
            providers_to_try.append(primary_provider)
        else:
            providers_to_try.append(config.llm.provider)
        
        # Add fallback providers
        if fallback_providers:
            providers_to_try.extend(fallback_providers)
        else:
            # Default fallbacks based on primary provider
            if providers_to_try[0].lower() != "ollama":
                providers_to_try.append("ollama")  # Ollama as fallback
        
        errors = []
        for provider_name in providers_to_try:
            try:
                provider = self.create_provider(config, provider_name)
                # Test if provider is actually working
                if provider.health_check_sync():
                    self.logger.info(f"Successfully created and tested {provider_name} provider")
                    return provider
                else:
                    self.logger.warning(f"Provider {provider_name} created but health check failed")
                    errors.append(f"{provider_name}: health check failed")
            except Exception as e:
                self.logger.warning(f"Failed to create {provider_name} provider: {e}")
                errors.append(f"{provider_name}: {str(e)}")
                continue
        
        # All providers failed
        error_details = "; ".join(errors)
        raise ProviderConnectionError(
            f"All providers failed. Tried: {', '.join(providers_to_try)}. Errors: {error_details}",
            "factory"
        )
    
    def _load_provider_dynamically(self, provider_name: str) -> Optional[type]:
        """Attempt to load a provider dynamically.
        
        Args:
            provider_name: Name of the provider to load
            
        Returns:
            Provider class if successfully loaded, None otherwise
        """
        try:
            # Try to import from providers module
            module_name = f"loki_code.core.providers.{provider_name}"
            provider_module = importlib.import_module(module_name)
            
            # Look for provider class (e.g., OpenAIProvider, AnthropicProvider)
            class_name = f"{provider_name.title()}Provider"
            if hasattr(provider_module, class_name):
                provider_class = getattr(provider_module, class_name)
                provider_registry.register(provider_name, provider_class)
                self.logger.info(f"Dynamically loaded {provider_name} provider")
                return provider_class
                
        except ImportError as e:
            self.logger.debug(f"Could not dynamically load {provider_name} provider: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading {provider_name} provider: {e}")
        
        return None
    
    def list_available_providers(self) -> List[str]:
        """List all available (registered) providers.
        
        Returns:
            List of provider names
        """
        return provider_registry.list_providers()
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider is available, False otherwise
        """
        return provider_registry.is_registered(provider_name.lower())
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dictionary with provider information, or None if not found
        """
        provider_class = provider_registry.get_provider_class(provider_name)
        if provider_class is None:
            return None
        
        return {
            "name": provider_name,
            "class": provider_class.__name__,
            "module": provider_class.__module__,
            "description": provider_class.__doc__ or "No description available"
        }
    
    def register_provider(self, provider_name: str, provider_class: type):
        """Register a custom provider.
        
        Args:
            provider_name: Name to register the provider under
            provider_class: Provider class (must inherit from BaseLLMProvider)
        """
        provider_registry.register(provider_name, provider_class)
        self.logger.info(f"Registered custom provider: {provider_name}")
    
    def configure_provider(self, provider_name: str, config: ProviderConfig):
        """Configure a provider.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
        """
        self._provider_configs[provider_name.lower()] = config
        self.logger.debug(f"Configured provider {provider_name}: {config}")
    
    def get_recommended_provider(self, config: LokiCodeConfig) -> str:
        """Get the recommended provider based on configuration and availability.
        
        Args:
            config: Configuration object
            
        Returns:
            Name of the recommended provider
        """
        # Start with configured provider
        configured_provider = config.llm.provider.lower()
        
        # Check if it's available
        if self.is_provider_available(configured_provider):
            return configured_provider
        
        # Fall back to available providers in order of preference
        preferred_order = ["ollama", "openai", "anthropic", "huggingface"]
        
        for provider in preferred_order:
            if self.is_provider_available(provider):
                self.logger.info(f"Configured provider {configured_provider} not available, using {provider}")
                return provider
        
        # If nothing is available, return the configured one anyway
        # (will fail later with a clear error message)
        return configured_provider


# Global factory instance
llm_factory = LLMProviderFactory()


def create_llm_provider(
    config: LokiCodeConfig, 
    provider_name: Optional[str] = None
) -> BaseLLMProvider:
    """Convenience function to create an LLM provider.
    
    Args:
        config: Configuration object containing LLM settings
        provider_name: Specific provider to create (uses config.llm.provider if None)
        
    Returns:
        BaseLLMProvider instance
    """
    return llm_factory.create_provider(config, provider_name)


def create_llm_provider_with_fallback(
    config: LokiCodeConfig,
    primary_provider: Optional[str] = None,
    fallback_providers: Optional[List[str]] = None
) -> BaseLLMProvider:
    """Convenience function to create an LLM provider with fallback.
    
    Args:
        config: Configuration object
        primary_provider: Primary provider to try first
        fallback_providers: List of fallback providers to try
        
    Returns:
        BaseLLMProvider instance (primary or fallback)
    """
    return llm_factory.create_provider_with_fallback(config, primary_provider, fallback_providers)


def list_available_providers() -> List[str]:
    """Convenience function to list available providers.
    
    Returns:
        List of provider names
    """
    return llm_factory.list_available_providers()


def get_recommended_provider(config: LokiCodeConfig) -> str:
    """Convenience function to get recommended provider.
    
    Args:
        config: Configuration object
        
    Returns:
        Name of the recommended provider
    """
    return llm_factory.get_recommended_provider(config)