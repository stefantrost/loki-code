"""
Core tool registry implementation.

Simplified and focused registry that coordinates with specialized components.
"""

import time
from typing import Dict, List, Optional, Any, Type

from ...tools.base import BaseTool
from ...tools.types import ToolSchema, ToolCapability
from .tool_types import ToolRegistration, ToolFilter, ToolExecutionRecord
from .tool_search import ToolSearch
from .tool_discovery import ToolDiscovery
from ...utils.logging import get_logger


# Global registry instance
_global_registry: Optional['ToolRegistry'] = None


def get_global_registry() -> 'ToolRegistry':
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


class ToolRegistry:
    """
    Simplified core tool registry.
    
    Focuses on registration and retrieval, delegates search and discovery
    to specialized components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tool registry."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Core registry data
        self._tools: Dict[str, ToolRegistration] = {}
        self._execution_history: List[ToolExecutionRecord] = []
        
        # Specialized components
        self.search = ToolSearch()
        self.discovery = ToolDiscovery(self)
        
        # Auto-discover built-in tools if enabled
        if self.config.get("auto_discover", True):
            self._auto_discover_builtin_tools()
    
    def _auto_discover_builtin_tools(self) -> None:
        """Auto-discover and register built-in tools."""
        try:
            self.discovery.auto_discover_and_register()
        except Exception as e:
            self.logger.warning(f"Auto-discovery failed: {e}")
    
    def register_tool_class(
        self, 
        tool_class: Type[BaseTool], 
        config: Optional[Dict[str, Any]] = None,
        source: str = "manual"
    ) -> None:
        """Register a tool class with the registry."""
        # Validate tool class
        issues = self.discovery.validate_tool_class(tool_class)
        if issues:
            raise ValueError(f"Tool validation failed: {', '.join(issues)}")
        
        # Create instance to get schema
        instance = tool_class(config)
        schema = instance.get_schema()
        
        if schema.name in self._tools:
            existing = self._tools[schema.name]
            if existing.source == "builtin" and source == "manual":
                self.logger.info(f"Overriding built-in tool {schema.name}")
            else:
                raise ValueError(f"Tool {schema.name} already registered")
        
        # Create registration
        registration = ToolRegistration(
            tool_class=tool_class,
            tool_instance=instance,
            schema=schema,
            source=source,
            config=config or {}
        )
        
        self._tools[schema.name] = registration
        self.logger.info(f"Registered tool: {schema.name} (source: {source})")
    
    def register_tool_instance(
        self, 
        tool_instance: BaseTool,
        source: str = "manual"
    ) -> None:
        """Register a tool instance directly."""
        schema = tool_instance.get_schema()
        
        if schema.name in self._tools:
            raise ValueError(f"Tool {schema.name} already registered")
        
        registration = ToolRegistration(
            tool_class=tool_instance.__class__,
            tool_instance=tool_instance,
            schema=schema,
            source=source
        )
        
        self._tools[schema.name] = registration
        self.logger.info(f"Registered tool instance: {schema.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        registration = self._tools.get(name)
        if registration and registration.enabled:
            # Update usage statistics
            registration.last_used = time.time()
            registration.usage_count += 1
            return registration.tool_instance
        return None
    
    def list_tools(
        self, 
        filter_by: Optional[ToolFilter] = None,
        include_disabled: bool = False
    ) -> List[ToolSchema]:
        """List available tools with optional filtering."""
        # Get base list of tools
        tools = []
        for registration in self._tools.values():
            if not include_disabled and not registration.enabled:
                continue
            if registration.schema:
                tools.append(registration.schema)
        
        # Apply filtering using search component
        if filter_by:
            tools = self.search.filter_tools(tools, filter_by)
        
        return tools
    
    def list_tool_names(self, include_disabled: bool = False) -> List[str]:
        """Get list of tool names."""
        return [
            name for name, reg in self._tools.items()
            if include_disabled or reg.enabled
        ]
    
    def get_tool_schema(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        registration = self._tools.get(name)
        return registration.schema if registration else None
    
    def get_tool_registration(self, name: str) -> Optional[ToolRegistration]:
        """Get tool registration by name."""
        return self._tools.get(name)
    
    def get_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        """Get tool names that have a specific capability."""
        tools = []
        for name, registration in self._tools.items():
            if (registration.enabled and 
                registration.schema and 
                capability in registration.schema.capabilities):
                tools.append(name)
        return tools
    
    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        registration = self._tools.get(name)
        if registration:
            registration.enabled = True
            self.logger.info(f"Enabled tool: {name}")
            return True
        return False
    
    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        registration = self._tools.get(name)
        if registration:
            registration.enabled = False
            self.logger.info(f"Disabled tool: {name}")
            return True
        return False
    
    def search_tools(self, query: str) -> List[ToolSchema]:
        """Search tools by description."""
        all_tools = self.list_tools()
        return self.search.search_by_description(all_tools, query)
    
    def get_tool_statistics(self, name: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a tool."""
        registration = self._tools.get(name)
        if not registration:
            return None
        
        return {
            "usage_count": registration.usage_count,
            "error_count": registration.error_count,
            "last_used": registration.last_used,
            "registered_at": registration.registered_at,
            "source": registration.source,
            "enabled": registration.enabled
        }
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        total_tools = len(self._tools)
        enabled_tools = sum(1 for reg in self._tools.values() if reg.enabled)
        
        sources = {}
        for reg in self._tools.values():
            sources[reg.source] = sources.get(reg.source, 0) + 1
        
        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": total_tools - enabled_tools,
            "sources": sources,
            "execution_history_size": len(self._execution_history)
        }
    
    async def execute_tool(
        self,
        name: str,
        input_data: Any,
        context: Any,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute a tool by name.
        
        Args:
            name: Name of the tool to execute
            input_data: Input data for the tool
            context: Execution context
            timeout: Optional timeout in seconds
            
        Returns:
            ToolResult from the execution
        """
        # Get the tool instance
        tool = self.get_tool(name)
        if not tool:
            from ...tools.exceptions import ToolNotFoundError
            raise ToolNotFoundError(
                name,
                available_tools=self.list_tool_names(),
                suggested_alternatives=[]
            )
        
        # Start execution record
        record = ToolExecutionRecord.start(name, input_data, context)
        
        # Execute the tool directly
        try:
            result = await tool.safe_execute(input_data, context, timeout)
            
            # Record successful execution
            record.complete(result)
            self.record_execution(record)
            
            return result
            
        except Exception as e:
            # Record failed execution
            record.fail(str(e))
            self.record_execution(record)
            raise

    def record_execution(self, record: ToolExecutionRecord) -> None:
        """Record a tool execution."""
        self._execution_history.append(record)
        
        # Update error count if execution failed
        if record.error and record.tool_name in self._tools:
            self._tools[record.tool_name].error_count += 1
        
        # Limit history size
        max_history = self.config.get("max_execution_history", 1000)
        if len(self._execution_history) > max_history:
            self._execution_history = self._execution_history[-max_history:]
    
    def get_execution_history(self, limit: int = 100) -> List[ToolExecutionRecord]:
        """Get recent execution history."""
        return self._execution_history[-limit:]
    
    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
        self.logger.info("Execution history cleared")
    
    def reset(self) -> None:
        """Reset the registry to initial state."""
        self._tools.clear()
        self._execution_history.clear()
        
        # Re-discover built-in tools
        if self.config.get("auto_discover", True):
            self._auto_discover_builtin_tools()
        
        self.logger.info("Registry reset")