"""
Loki Code Utilities

This module provides utility functions and classes used throughout Loki Code.
"""

from .logging import (
    setup_logging,
    get_logger,
    log_performance,
    performance_timer,
    is_logging_initialized,
    log_startup,
    log_config_info,
    log_shutdown,
)

from .error_handling import (
    LokiCodeError,
    ToolExecutionError,
    ConfigurationError,
    ProviderError,
    ValidationError,
    handle_tool_execution,
    handle_provider_operation,
    handle_configuration_operation,
    AsyncProviderMixin,
    validate_input,
)

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",
    "log_performance",
    "performance_timer", 
    "is_logging_initialized",
    "log_startup",
    "log_config_info",
    "log_shutdown",
    
    # Error handling utilities
    "LokiCodeError",
    "ToolExecutionError",
    "ConfigurationError", 
    "ProviderError",
    "ValidationError",
    "handle_tool_execution",
    "handle_provider_operation",
    "handle_configuration_operation",
    "AsyncProviderMixin",
    "validate_input",
]