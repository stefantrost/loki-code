"""
Loki Code Configuration System

This module provides easy access to configuration loading and management.
Import the main functions you need:

    from loki_code.config import get_config, load_config

Basic usage:
    config = get_config()
    print(config.app.name)          # "Loki Code"
    print(config.llm.model)         # "codellama:7b"
    print(config.tools.enabled)     # ["file_reader", "file_writer", ...]
"""

from .loader import (
    load_config,
    get_config,
    reload_config,
    validate_config_file,
    ConfigurationError,
)

from .models import (
    LokiCodeConfig,
    AppConfig,
    LLMConfig,
    ToolsConfig,
    UIConfig,
    SafetyConfig,
    DevelopmentConfig,
    LoggingConfig,
    PluginsConfig,
    CacheConfig,
    IntegrationsConfig,
)

__all__ = [
    # Main functions
    "get_config",
    "load_config", 
    "reload_config",
    "validate_config_file",
    
    # Exception
    "ConfigurationError",
    
    # Configuration models
    "LokiCodeConfig",
    "AppConfig",
    "LLMConfig", 
    "ToolsConfig",
    "UIConfig",
    "SafetyConfig",
    "DevelopmentConfig",
    "LoggingConfig",
    "PluginsConfig",
    "CacheConfig",
    "IntegrationsConfig",
]