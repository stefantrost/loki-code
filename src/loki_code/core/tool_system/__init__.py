"""
Tool system for Loki Code.

Simplified and modular tool system with clear separation of concerns.
"""

from .tool_registry_core import ToolRegistry
from .tool_discovery import ToolDiscovery  
from .tool_search import ToolSearch
from .tool_types import ToolRegistration, ToolFilter, ToolExecutionRecord

# Backward compatibility
try:
    from ..tool_registry import ToolRegistry as ToolRegistryOriginal
except ImportError:
    ToolRegistryOriginal = None

__all__ = [
    "ToolRegistry",
    "ToolDiscovery", 
    "ToolSearch",
    "ToolRegistration",
    "ToolFilter", 
    "ToolExecutionRecord",
    "ToolRegistryOriginal"
]