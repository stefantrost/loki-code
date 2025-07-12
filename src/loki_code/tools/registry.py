"""
Tool registry for managing and discovering tools in Loki Code.

This module provides a centralized registry for all tools, enabling
discovery, validation, and execution coordination.
"""

import asyncio
from typing import Dict, List, Optional, Any, Type, Set
from dataclasses import dataclass, field
import time
import logging

from .base import BaseTool, MCPResourceAdapter
from .types import (
    ToolSchema, ToolContext, ToolResult, ToolCall, ToolExecution,
    SecurityLevel, ToolCapability, ConfirmationLevel
)
from .exceptions import (
    ToolException, ToolNotFoundError, ToolRegistrationError,
    ToolValidationError, handle_tool_exception
)
from ..utils.logging import get_logger


@dataclass
class ToolRegistration:
    """Information about a registered tool."""
    tool_class: Type[BaseTool]
    tool_instance: Optional[BaseTool] = None
    schema: Optional[ToolSchema] = None
    registered_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    usage_count: int = 0
    error_count: int = 0


class ToolRegistry:
    """
    Registry for managing tools in Loki Code.
    
    Provides centralized registration, discovery, and execution
    coordination for all tools in the system.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.logger = get_logger(__name__)
        self._tools: Dict[str, ToolRegistration] = {}
        self._tool_schemas: Dict[str, ToolSchema] = {}
        self._capabilities_index: Dict[ToolCapability, Set[str]] = {}
        self._security_index: Dict[SecurityLevel, Set[str]] = {}
        self._execution_history: List[ToolExecution] = []
        self._mcp_adapter = MCPResourceAdapter()
        
        # Initialize capability index
        for capability in ToolCapability:
            self._capabilities_index[capability] = set()
        
        # Initialize security index
        for level in SecurityLevel:
            self._security_index[level] = set()
    
    def register_tool(self, tool_class: Type[BaseTool], config: Optional[Dict[str, Any]] = None) -> None:
        """Register a tool class with the registry.
        
        Args:
            tool_class: Tool class to register
            config: Optional configuration for the tool
            
        Raises:
            ToolRegistrationError: If registration fails
        """
        try:
            # Create temporary instance to get schema
            temp_instance = tool_class(config)
            schema = temp_instance.get_schema()
            
            # Validate schema
            self._validate_tool_schema(schema)
            
            # Check for name conflicts
            if schema.name in self._tools:
                raise ToolRegistrationError(
                    f"Tool with name '{schema.name}' is already registered",
                    schema.name,
                    "name_conflict"
                )
            
            # Register the tool
            registration = ToolRegistration(
                tool_class=tool_class,
                schema=schema
            )
            
            self._tools[schema.name] = registration
            self._tool_schemas[schema.name] = schema
            
            # Update indexes
            for capability in schema.capabilities:
                self._capabilities_index[capability].add(schema.name)
            
            self._security_index[schema.security_level].add(schema.name)
            
            self.logger.info(f"Registered tool: {schema.name}")
            
        except Exception as e:
            if isinstance(e, ToolRegistrationError):
                raise
            raise ToolRegistrationError(
                f"Failed to register tool {tool_class.__name__}: {str(e)}",
                getattr(tool_class, '__name__', 'unknown'),
                str(e)
            ) from e
    
    def register_tool_instance(self, tool_instance: BaseTool) -> None:
        """Register a tool instance directly.
        
        Args:
            tool_instance: Tool instance to register
            
        Raises:
            ToolRegistrationError: If registration fails
        """
        try:
            schema = tool_instance.get_schema()
            
            # Validate schema
            self._validate_tool_schema(schema)
            
            # Check for name conflicts
            if schema.name in self._tools:
                raise ToolRegistrationError(
                    f"Tool with name '{schema.name}' is already registered",
                    schema.name,
                    "name_conflict"
                )
            
            # Register the tool
            registration = ToolRegistration(
                tool_class=tool_instance.__class__,
                tool_instance=tool_instance,
                schema=schema
            )
            
            self._tools[schema.name] = registration
            self._tool_schemas[schema.name] = schema
            
            # Update indexes
            for capability in schema.capabilities:
                self._capabilities_index[capability].add(schema.name)
            
            self._security_index[schema.security_level].add(schema.name)
            
            self.logger.info(f"Registered tool instance: {schema.name}")
            
        except Exception as e:
            if isinstance(e, ToolRegistrationError):
                raise
            raise ToolRegistrationError(
                f"Failed to register tool instance {tool_instance.__class__.__name__}: {str(e)}",
                tool_instance.get_name(),
                str(e)
            ) from e
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name not in self._tools:
            return False
        
        registration = self._tools[tool_name]
        schema = registration.schema
        
        # Remove from indexes
        if schema:
            for capability in schema.capabilities:
                self._capabilities_index[capability].discard(tool_name)
            
            self._security_index[schema.security_level].discard(tool_name)
        
        # Remove from registry
        del self._tools[tool_name]
        del self._tool_schemas[tool_name]
        
        self.logger.info(f"Unregistered tool: {tool_name}")
        return True
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool instance if found, None otherwise
        """
        if name not in self._tools:
            return None
        
        registration = self._tools[name]
        
        # Create instance if needed
        if registration.tool_instance is None:
            try:
                registration.tool_instance = registration.tool_class()
            except Exception as e:
                self.logger.error(f"Failed to create tool instance for {name}: {e}")
                return None
        
        return registration.tool_instance
    
    def list_tools(
        self, 
        capability: Optional[ToolCapability] = None,
        security_level: Optional[SecurityLevel] = None,
        include_disabled: bool = False
    ) -> List[ToolSchema]:
        """List available tools with optional filtering.
        
        Args:
            capability: Filter by capability
            security_level: Filter by security level
            include_disabled: Include disabled tools
            
        Returns:
            List of tool schemas matching the criteria
        """
        tool_names = set(self._tools.keys())
        
        # Filter by capability
        if capability:
            tool_names &= self._capabilities_index.get(capability, set())
        
        # Filter by security level
        if security_level:
            tool_names &= self._security_index.get(security_level, set())
        
        return [self._tool_schemas[name] for name in tool_names if name in self._tool_schemas]
    
    def list_tool_names(
        self, 
        capability: Optional[ToolCapability] = None,
        security_level: Optional[SecurityLevel] = None
    ) -> List[str]:
        """List tool names with optional filtering.
        
        Args:
            capability: Filter by capability
            security_level: Filter by security level
            
        Returns:
            List of tool names matching the criteria
        """
        schemas = self.list_tools(capability, security_level)
        return [schema.name for schema in schemas]
    
    def get_tool_schema(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool schema if found, None otherwise
        """
        return self._tool_schemas.get(name)
    
    def validate_tool_call(self, name: str, input_data: Any) -> bool:
        """Validate a tool call before execution.
        
        Args:
            name: Name of the tool
            input_data: Input data for the tool
            
        Returns:
            True if the call is valid
            
        Raises:
            ToolNotFoundError: If tool is not found
            ToolValidationError: If validation fails
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                name,
                available_tools=list(self._tools.keys()),
                suggested_alternatives=self._suggest_similar_tools(name)
            )
        
        tool = self.get_tool(name)
        if tool is None:
            raise ToolValidationError(
                f"Failed to get tool instance for {name}",
                name
            )
        
        # This will be async in the actual validation
        # For now, we'll assume it's valid if we can get the tool
        return True
    
    async def execute_tool(
        self, 
        name: str, 
        input_data: Any, 
        context: ToolContext,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            name: Name of the tool to execute
            input_data: Input data for the tool
            context: Execution context
            timeout: Optional timeout in seconds
            
        Returns:
            ToolResult from the execution
            
        Raises:
            ToolNotFoundError: If tool is not found
            ToolException: If execution fails
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                name,
                available_tools=list(self._tools.keys()),
                suggested_alternatives=self._suggest_similar_tools(name)
            )
        
        tool = self.get_tool(name)
        if tool is None:
            raise ToolValidationError(
                f"Failed to get tool instance for {name}",
                name
            )
        
        registration = self._tools[name]
        call = tool.create_tool_call(input_data, context)
        
        try:
            # Execute the tool
            start_time = time.perf_counter()
            result = await tool.safe_execute(input_data, context, timeout)
            end_time = time.perf_counter()
            
            # Record execution
            execution = ToolExecution(
                call=call,
                result=result,
                started_at=start_time,
                completed_at=end_time
            )
            
            self._record_execution(registration, execution)
            
            return result
            
        except Exception as e:
            # Record failed execution
            end_time = time.perf_counter()
            failed_result = ToolResult.failure_result(f"Tool execution failed: {str(e)}")
            
            execution = ToolExecution(
                call=call,
                result=failed_result,
                started_at=start_time,
                completed_at=end_time
            )
            
            registration.error_count += 1
            self._record_execution(registration, execution)
            
            # Re-raise the exception
            raise
    
    def get_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        """Get tools that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of tool names with the capability
        """
        return list(self._capabilities_index.get(capability, set()))
    
    def get_tools_by_security_level(self, security_level: SecurityLevel) -> List[str]:
        """Get tools at a specific security level.
        
        Args:
            security_level: Security level to search for
            
        Returns:
            List of tool names at the security level
        """
        return list(self._security_index.get(security_level, set()))
    
    def get_tool_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            Dictionary with tool statistics, None if tool not found
        """
        if name not in self._tools:
            return None
        
        registration = self._tools[name]
        
        return {
            "name": name,
            "registered_at": registration.registered_at,
            "last_used": registration.last_used,
            "usage_count": registration.usage_count,
            "error_count": registration.error_count,
            "success_rate": (
                (registration.usage_count - registration.error_count) / registration.usage_count
                if registration.usage_count > 0 else 0.0
            )
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total_tools = len(self._tools)
        total_executions = len(self._execution_history)
        
        capability_counts = {
            cap.value: len(tools) 
            for cap, tools in self._capabilities_index.items() 
            if tools
        }
        
        security_counts = {
            level.value: len(tools)
            for level, tools in self._security_index.items()
            if tools
        }
        
        return {
            "total_tools": total_tools,
            "total_executions": total_executions,
            "capability_distribution": capability_counts,
            "security_distribution": security_counts,
            "execution_history_size": len(self._execution_history)
        }
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()
        self.logger.info("Cleared tool execution history")
    
    def _validate_tool_schema(self, schema: ToolSchema) -> None:
        """Validate a tool schema.
        
        Args:
            schema: Schema to validate
            
        Raises:
            ToolRegistrationError: If schema is invalid
        """
        if not schema.name:
            raise ToolRegistrationError(
                "Tool schema must have a name",
                "unknown",
                "missing_name"
            )
        
        if not schema.description:
            raise ToolRegistrationError(
                "Tool schema must have a description",
                schema.name,
                "missing_description"
            )
        
        if not schema.capabilities:
            raise ToolRegistrationError(
                "Tool schema must declare at least one capability",
                schema.name,
                "missing_capabilities"
            )
    
    def _suggest_similar_tools(self, name: str) -> List[str]:
        """Suggest similar tool names for a missing tool.
        
        Args:
            name: Name that was not found
            
        Returns:
            List of suggested similar tool names
        """
        suggestions = []
        available_names = list(self._tools.keys())
        
        # Simple similarity based on common substrings
        for available_name in available_names:
            if name.lower() in available_name.lower() or available_name.lower() in name.lower():
                suggestions.append(available_name)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _record_execution(self, registration: ToolRegistration, execution: ToolExecution) -> None:
        """Record a tool execution.
        
        Args:
            registration: Tool registration
            execution: Execution record
        """
        # Update registration stats
        registration.last_used = execution.completed_at
        registration.usage_count += 1
        
        # Add to history
        self._execution_history.append(execution)
        
        # Keep history size manageable
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]


# Global tool registry instance
tool_registry = ToolRegistry()