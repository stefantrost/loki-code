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

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",
    "log_performance",
    "performance_timer", 
    "is_logging_initialized",
    
    # Convenience functions
    "log_startup",
    "log_config_info",
    "log_shutdown",
]