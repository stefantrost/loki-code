"""
Tools package for Loki Code.

This package contains the tool system foundation and implementations,
designed for MCP compatibility from the start while providing a rich
framework for local tool development.

Usage:
    from loki_code.tools import BaseTool, ToolRegistry, tool_registry
    
    # Register a tool
    tool_registry.register_tool(MyTool)
    
    # Execute a tool
    result = await tool_registry.execute_tool("my_tool", input_data, context)
"""

from .base import BaseTool, SimpleFileTool, MCPResourceAdapter, create_simple_tool_schema
from .registry import ToolRegistry, ToolRegistration, tool_registry
from .types import (
    # Core types
    ToolSchema,
    ToolContext, 
    ToolResult,
    ToolCall,
    ToolExecution,
    
    # Enums
    SecurityLevel,
    ToolCapability,
    ConfirmationLevel,
    ToolStatus,
    
    # Configuration
    SafetySettings,
    
    # Schema helpers
    InputValidationSchema,
    OutputValidationSchema,
    COMMON_SCHEMAS
)
from .exceptions import (
    # Base exception
    ToolException,
    
    # Specific exceptions
    ToolValidationError,
    ToolExecutionError,
    ToolPermissionError,
    ToolSecurityError,
    ToolTimeoutError,
    ToolResourceError,
    ToolConfigurationError,
    ToolDependencyError,
    ToolNotFoundError,
    ToolRegistrationError,
    MCPToolError,
    
    # Utility functions
    handle_tool_exception,
    is_recoverable_error,
    get_error_severity
)

# Version info
__version__ = "0.1.0"

# Export all public APIs
__all__ = [
    # Base classes
    "BaseTool",
    "SimpleFileTool", 
    "MCPResourceAdapter",
    "create_simple_tool_schema",
    
    # Registry
    "ToolRegistry",
    "ToolRegistration",
    "tool_registry",
    
    # Core types
    "ToolSchema",
    "ToolContext",
    "ToolResult", 
    "ToolCall",
    "ToolExecution",
    
    # Enums
    "SecurityLevel",
    "ToolCapability",
    "ConfirmationLevel",
    "ToolStatus",
    
    # Configuration
    "SafetySettings",
    
    # Schema helpers
    "InputValidationSchema",
    "OutputValidationSchema",
    "COMMON_SCHEMAS",
    
    # Exceptions
    "ToolException",
    "ToolValidationError",
    "ToolExecutionError", 
    "ToolPermissionError",
    "ToolSecurityError",
    "ToolTimeoutError",
    "ToolResourceError",
    "ToolConfigurationError",
    "ToolDependencyError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "MCPToolError",
    "handle_tool_exception",
    "is_recoverable_error",
    "get_error_severity"
]

# Tool system information for discovery
TOOL_SYSTEM_INFO = {
    "version": __version__,
    "mcp_compatible": True,
    "supported_security_levels": [level.value for level in SecurityLevel],
    "supported_capabilities": [cap.value for cap in ToolCapability],
    "confirmation_levels": [level.value for level in ConfirmationLevel]
}


def get_tool_system_info() -> dict:
    """Get information about the tool system.
    
    Returns:
        Dictionary with tool system information
    """
    return TOOL_SYSTEM_INFO.copy()


def list_available_tools() -> list:
    """Get list of available tools.
    
    Returns:
        List of available tool names
    """
    return tool_registry.list_tool_names()


def get_tool_by_name(name: str) -> BaseTool:
    """Get a tool instance by name.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool instance
        
    Raises:
        ToolNotFoundError: If tool is not found
    """
    tool = tool_registry.get_tool(name)
    if tool is None:
        available_tools = list_available_tools()
        raise ToolNotFoundError(
            name,
            available_tools=available_tools,
            suggested_alternatives=tool_registry._suggest_similar_tools(name)
        )
    return tool


def register_tool_class(tool_class: type, config: dict = None) -> None:
    """Register a tool class.
    
    Args:
        tool_class: Tool class to register
        config: Optional configuration for the tool
    """
    tool_registry.register_tool(tool_class, config)


def register_tool_instance(tool_instance: BaseTool) -> None:
    """Register a tool instance.
    
    Args:
        tool_instance: Tool instance to register
    """
    tool_registry.register_tool_instance(tool_instance)


# Register built-in tools
def _register_builtin_tools():
    """Register all built-in tools with the global registry."""
    try:
        # Use the new comprehensive registry system
        from ..core.tool_registry import get_global_registry
        from .file_reader import FileReaderTool
        
        registry = get_global_registry()
        registry.register_tool_class(FileReaderTool, source="builtin")
    except Exception as e:
        # Import time error handling - log but don't fail the import
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to register built-in tool FileReaderTool: {e}")

# Auto-register built-in tools on import (disabled due to circular import)
# _register_builtin_tools()