"""
Tool registry for Loki Code.

This module provides backward compatibility while using the new modular tool system.
"""

# Import from the new modular system
from .tool_system import (
    ToolRegistry, ToolDiscovery, ToolSearch,
    ToolRegistration, ToolFilter, ToolExecutionRecord
)

# Re-export the global registry function for compatibility
from .tool_system.tool_registry_core import get_global_registry

__all__ = [
    "ToolRegistry",
    "ToolDiscovery", 
    "ToolSearch",
    "ToolRegistration",
    "ToolFilter",
    "ToolExecutionRecord",
    "get_global_registry"
]