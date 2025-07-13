"""
Comprehensive Tool Registry system for Loki Code.

This module provides the core tool registry that manages tool discovery,
registration, and execution coordination. Designed for both local tools
and future MCP integration.

Features:
- Dynamic tool discovery and registration
- MCP-compatible tool management  
- Tool metadata and capability querying
- Execution routing and error handling
- Security-aware tool access
- Plugin system support
"""

import asyncio
import inspect
import importlib
import pkgutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Set, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import time
import logging

from ..tools.base import BaseTool
from ..tools.types import (
    ToolSchema, ToolContext, ToolResult, ToolCall, ToolExecution,
    SecurityLevel, ToolCapability, ConfirmationLevel, ToolStatus,
    SafetySettings
)
from ..tools.exceptions import (
    ToolException, ToolNotFoundError, ToolRegistrationError,
    ToolValidationError, ToolExecutionError, ToolSecurityError,
    handle_tool_exception
)
from ..utils.logging import get_logger


@dataclass
class ToolFilter:
    """Filter criteria for tool queries."""
    capabilities: Optional[List[ToolCapability]] = None
    security_levels: Optional[List[SecurityLevel]] = None
    mcp_compatible: Optional[bool] = None
    keywords: Optional[List[str]] = None
    confirmation_required: Optional[bool] = None
    enabled_only: bool = True


@dataclass
class ToolRegistration:
    """Complete tool registration information."""
    tool_class: Type[BaseTool]
    tool_instance: Optional[BaseTool] = None
    schema: Optional[ToolSchema] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    error_count: int = 0
    enabled: bool = True
    source: str = "unknown"  # builtin, plugin, mcp
    config: Optional[Dict[str, Any]] = None


@dataclass
class MCPServer:
    """MCP server information (future use)."""
    url: str
    name: str
    version: str
    capabilities: List[str]
    tools: List[Dict[str, Any]] = field(default_factory=list)
    trusted: bool = False
    last_seen: Optional[datetime] = None
    status: str = "unknown"  # connected, disconnected, error


@dataclass
class ToolExecutionRecord:
    """Record of tool execution for tracking and analysis."""
    execution_id: str
    tool_name: str
    input_data: Any
    context_summary: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    security_level: Optional[SecurityLevel] = None
    
    @classmethod
    def start(cls, tool_name: str, input_data: Any, context: ToolContext) -> 'ToolExecutionRecord':
        """Start a new execution record."""
        execution_id = f"{tool_name}_{int(time.time() * 1000)}"
        context_summary = {
            "project_path": context.project_path,
            "session_id": context.session_id,
            "dry_run": context.dry_run
        }
        
        return cls(
            execution_id=execution_id,
            tool_name=tool_name,
            input_data=input_data,
            context_summary=context_summary,
            start_time=datetime.now()
        )
    
    def complete(self, result: ToolResult) -> None:
        """Complete the execution record."""
        self.end_time = datetime.now()
        self.result = result
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def fail(self, error: str) -> None:
        """Mark execution as failed."""
        self.end_time = datetime.now()
        self.error = error
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class ToolSearch:
    """Tool search and filtering utilities."""
    
    @staticmethod
    def filter_tools(tools: List[ToolSchema], filter_obj: ToolFilter) -> List[ToolSchema]:
        """Filter tools based on criteria."""
        filtered = tools
        
        if filter_obj.capabilities:
            filtered = [
                tool for tool in filtered
                if any(cap in tool.capabilities for cap in filter_obj.capabilities)
            ]
        
        if filter_obj.security_levels:
            filtered = [
                tool for tool in filtered
                if tool.security_level in filter_obj.security_levels
            ]
        
        if filter_obj.mcp_compatible is not None:
            filtered = [
                tool for tool in filtered
                if tool.mcp_compatible == filter_obj.mcp_compatible
            ]
        
        if filter_obj.keywords:
            filtered = [
                tool for tool in filtered
                if any(
                    keyword.lower() in tool.description.lower() or
                    keyword.lower() in tool.name.lower() or
                    any(keyword.lower() in tag.lower() for tag in tool.tags)
                    for keyword in filter_obj.keywords
                )
            ]
        
        if filter_obj.confirmation_required is not None:
            filtered = [
                tool for tool in filtered
                if (tool.confirmation_level != ConfirmationLevel.NONE) == filter_obj.confirmation_required
            ]
        
        return filtered
    
    @staticmethod
    def search_by_description(tools: List[ToolSchema], query: str) -> List[ToolSchema]:
        """Search tools by description content."""
        query_lower = query.lower()
        matching = []
        
        for tool in tools:
            score = 0
            
            # Exact name match gets highest score
            if query_lower == tool.name.lower():
                score = 100
            elif query_lower in tool.name.lower():
                score = 80
            
            # Description matches
            if query_lower in tool.description.lower():
                score += 50
            
            # Tag matches
            for tag in tool.tags:
                if query_lower in tag.lower():
                    score += 30
            
            # Capability matches
            for cap in tool.capabilities:
                if query_lower in cap.value.lower():
                    score += 20
            
            if score > 0:
                matching.append((tool, score))
        
        # Sort by score descending
        matching.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in matching]
    
    @staticmethod
    def rank_tools_by_relevance(tools: List[ToolSchema], task_description: str) -> List[ToolSchema]:
        """Rank tools by relevance to a task description."""
        task_lower = task_description.lower()
        scored_tools = []
        
        for tool in tools:
            score = 0
            
            # Check for relevant keywords in task
            if "read" in task_lower and ToolCapability.READ_FILE in tool.capabilities:
                score += 50
            if "write" in task_lower and ToolCapability.WRITE_FILE in tool.capabilities:
                score += 50
            if "execute" in task_lower and ToolCapability.EXECUTE_COMMAND in tool.capabilities:
                score += 40
            if "analyze" in task_lower and ToolCapability.CODE_ANALYSIS in tool.capabilities:
                score += 40
            
            # Description relevance
            tool_words = tool.description.lower().split()
            task_words = task_lower.split()
            common_words = set(tool_words) & set(task_words)
            score += len(common_words) * 10
            
            scored_tools.append((tool, score))
        
        # Sort by score descending
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scored_tools]


class ToolDiscovery:
    """Tool discovery system for finding and loading tools."""
    
    def __init__(self, registry: 'ToolRegistry'):
        self.registry = registry
        self.logger = get_logger(__name__)
    
    def discover_builtin_tools(self) -> List[Type[BaseTool]]:
        """Discover built-in tools in the tools package."""
        builtin_tools = []
        
        try:
            # Manually import known tool modules to avoid circular imports
            from ..tools import file_reader
            
            # Scan known modules for BaseTool subclasses
            modules_to_scan = [file_reader]
            
            for module in modules_to_scan:
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    
                    if (inspect.isclass(item) and 
                        issubclass(item, BaseTool) and 
                        item != BaseTool and
                        not inspect.isabstract(item)):  # Skip abstract classes
                        builtin_tools.append(item)
                        self.logger.debug(f"Discovered tool: {item.__name__}")
                        
        except ImportError as e:
            self.logger.error(f"Failed to import tool modules: {e}")
        except Exception as e:
            self.logger.error(f"Error during tool discovery: {e}")
        
        return builtin_tools
    
    def load_plugin_tools(self, plugin_dirs: List[str]) -> List[Type[BaseTool]]:
        """Load tools from plugin directories (future feature)."""
        plugin_tools = []
        
        for plugin_dir in plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                self.logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                continue
            
            self.logger.info(f"Scanning plugin directory: {plugin_dir}")
            
            # Future: Implement plugin loading system
            # For now, just log the intent
            
        return plugin_tools
    
    async def discover_mcp_tools(self, server_url: str) -> List[Dict[str, Any]]:
        """Discover tools from MCP servers (future feature)."""
        # Future: Implement MCP server discovery
        self.logger.info(f"MCP tool discovery from {server_url} - not yet implemented")
        return []


class ToolRegistry:
    """
    Comprehensive tool registry for managing tool discovery, registration, and execution.
    
    Supports both local tools and future MCP integration with advanced
    filtering, search, and execution tracking capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tool registry."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Tool storage
        self._local_tools: Dict[str, ToolRegistration] = {}
        self._mcp_tools: Dict[str, Dict[str, Any]] = {}  # Future MCP tools
        self._tool_schemas: Dict[str, ToolSchema] = {}
        
        # MCP servers (future)
        self._mcp_servers: Dict[str, MCPServer] = {}
        
        # Indexes for fast lookup
        self._capability_index: Dict[ToolCapability, Set[str]] = {}
        self._security_index: Dict[SecurityLevel, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        
        # Execution tracking
        self._execution_history: List[ToolExecutionRecord] = []
        self._max_history_size = self.config.get('max_execution_history', 1000)
        
        # Initialize capability and security indexes
        for capability in ToolCapability:
            self._capability_index[capability] = set()
        for level in SecurityLevel:
            self._security_index[level] = set()
        
        # Discovery system
        self.discovery = ToolDiscovery(self)
        self.search = ToolSearch()
        
        # Auto-discover tools if configured
        if self.config.get('discovery', {}).get('auto_discover_builtin', True):
            self._auto_discover_builtin_tools()
    
    def _auto_discover_builtin_tools(self) -> None:
        """Automatically discover and register builtin tools."""
        try:
            tool_classes = self.discovery.discover_builtin_tools()
            for tool_class in tool_classes:
                try:
                    self.register_tool_class(tool_class, source="builtin")
                except Exception as e:
                    self.logger.warning(f"Failed to auto-register {tool_class.__name__}: {e}")
        except Exception as e:
            self.logger.error(f"Auto-discovery failed: {e}")
    
    def register_tool_class(
        self, 
        tool_class: Type[BaseTool], 
        config: Optional[Dict[str, Any]] = None,
        source: str = "manual"
    ) -> None:
        """Register a tool class with the registry."""
        try:
            # Create temporary instance to get schema
            temp_instance = tool_class(config)
            schema = temp_instance.get_schema()
            
            # Validate schema
            self._validate_tool_schema(schema)
            
            # Check for name conflicts
            if schema.name in self._local_tools:
                existing = self._local_tools[schema.name]
                if existing.source == "builtin" and source == "manual":
                    # Allow manual registration to override builtin
                    self.logger.info(f"Overriding builtin tool {schema.name} with manual registration")
                else:
                    raise ToolRegistrationError(
                        f"Tool with name '{schema.name}' is already registered",
                        schema.name,
                        "name_conflict"
                    )
            
            # Create registration
            registration = ToolRegistration(
                tool_class=tool_class,
                schema=schema,
                source=source,
                config=config
            )
            
            # Register the tool
            self._local_tools[schema.name] = registration
            self._tool_schemas[schema.name] = schema
            
            # Update indexes
            self._update_indexes_for_tool(schema, add=True)
            
            self.logger.info(f"Registered tool: {schema.name} (source: {source})")
            
        except Exception as e:
            if isinstance(e, ToolRegistrationError):
                raise
            raise ToolRegistrationError(
                f"Failed to register tool {tool_class.__name__}: {str(e)}",
                getattr(tool_class, '__name__', 'unknown'),
                str(e)
            ) from e
    
    def register_tool_instance(
        self, 
        tool_instance: BaseTool,
        source: str = "manual"
    ) -> None:
        """Register a tool instance directly."""
        try:
            schema = tool_instance.get_schema()
            
            # Validate schema
            self._validate_tool_schema(schema)
            
            # Check for name conflicts
            if schema.name in self._local_tools:
                raise ToolRegistrationError(
                    f"Tool with name '{schema.name}' is already registered",
                    schema.name,
                    "name_conflict"
                )
            
            # Create registration
            registration = ToolRegistration(
                tool_class=tool_instance.__class__,
                tool_instance=tool_instance,
                schema=schema,
                source=source
            )
            
            # Register the tool
            self._local_tools[schema.name] = registration
            self._tool_schemas[schema.name] = schema
            
            # Update indexes
            self._update_indexes_for_tool(schema, add=True)
            
            self.logger.info(f"Registered tool instance: {schema.name} (source: {source})")
            
        except Exception as e:
            if isinstance(e, ToolRegistrationError):
                raise
            raise ToolRegistrationError(
                f"Failed to register tool instance {tool_instance.__class__.__name__}: {str(e)}",
                tool_instance.get_name(),
                str(e)
            ) from e
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry."""
        if tool_name not in self._local_tools:
            return False
        
        registration = self._local_tools[tool_name]
        schema = registration.schema
        
        # Remove from indexes
        if schema:
            self._update_indexes_for_tool(schema, add=False)
        
        # Remove from registry
        del self._local_tools[tool_name]
        if tool_name in self._tool_schemas:
            del self._tool_schemas[tool_name]
        
        self.logger.info(f"Unregistered tool: {tool_name}")
        return True
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        if name not in self._local_tools:
            return None
        
        registration = self._local_tools[name]
        
        # Create instance if needed
        if registration.tool_instance is None:
            try:
                registration.tool_instance = registration.tool_class(registration.config)
            except Exception as e:
                self.logger.error(f"Failed to create tool instance for {name}: {e}")
                return None
        
        return registration.tool_instance
    
    def list_tools(
        self, 
        filter_by: Optional[ToolFilter] = None,
        include_disabled: bool = False
    ) -> List[ToolSchema]:
        """List available tools with optional filtering."""
        # Get base list of tools
        tools = []
        for name, registration in self._local_tools.items():
            if not include_disabled and not registration.enabled:
                continue
            if registration.schema:
                tools.append(registration.schema)
        
        # Apply filtering
        if filter_by:
            tools = self.search.filter_tools(tools, filter_by)
        
        return tools
    
    def list_tool_names(
        self, 
        filter_by: Optional[ToolFilter] = None,
        include_disabled: bool = False
    ) -> List[str]:
        """List tool names with optional filtering."""
        schemas = self.list_tools(filter_by, include_disabled)
        return [schema.name for schema in schemas]
    
    def get_tool_schema(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        return self._tool_schemas.get(name)
    
    def get_tool_registration(self, name: str) -> Optional[ToolRegistration]:
        """Get full tool registration information."""
        return self._local_tools.get(name)
    
    def get_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        """Get tools that have a specific capability."""
        return list(self._capability_index.get(capability, set()))
    
    def get_tools_by_security_level(self, security_level: SecurityLevel) -> List[str]:
        """Get tools at a specific security level."""
        return list(self._security_index.get(security_level, set()))
    
    def get_tools_by_tag(self, tag: str) -> List[str]:
        """Get tools that have a specific tag."""
        return list(self._tag_index.get(tag.lower(), set()))
    
    def search_tools(self, query: str) -> List[ToolSchema]:
        """Search tools by query string."""
        all_tools = self.list_tools()
        return self.search.search_by_description(all_tools, query)
    
    def rank_tools_for_task(self, task_description: str) -> List[ToolSchema]:
        """Rank tools by relevance to a task."""
        all_tools = self.list_tools()
        return self.search.rank_tools_by_relevance(all_tools, task_description)
    
    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        if name in self._local_tools:
            self._local_tools[name].enabled = True
            self.logger.info(f"Enabled tool: {name}")
            return True
        return False
    
    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        if name in self._local_tools:
            self._local_tools[name].enabled = False
            self.logger.info(f"Disabled tool: {name}")
            return True
        return False
    
    def get_tool_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a tool."""
        if name not in self._local_tools:
            return None
        
        registration = self._local_tools[name]
        
        return {
            "name": name,
            "source": registration.source,
            "enabled": registration.enabled,
            "registered_at": registration.registered_at.isoformat(),
            "last_used": registration.last_used.isoformat() if registration.last_used else None,
            "usage_count": registration.usage_count,
            "error_count": registration.error_count,
            "success_rate": (
                (registration.usage_count - registration.error_count) / registration.usage_count
                if registration.usage_count > 0 else 0.0
            )
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        total_tools = len(self._local_tools)
        enabled_tools = sum(1 for reg in self._local_tools.values() if reg.enabled)
        total_executions = len(self._execution_history)
        
        # Source distribution
        source_counts = {}
        for registration in self._local_tools.values():
            source = registration.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Capability distribution
        capability_counts = {
            cap.value: len(tools) 
            for cap, tools in self._capability_index.items() 
            if tools
        }
        
        # Security level distribution
        security_counts = {
            level.value: len(tools)
            for level, tools in self._security_index.items()
            if tools
        }
        
        # Recent execution stats
        recent_executions = [
            exec_record for exec_record in self._execution_history
            if exec_record.start_time > datetime.now() - timedelta(hours=24)
        ]
        
        successful_recent = sum(
            1 for exec_record in recent_executions
            if exec_record.result and exec_record.result.success
        )
        
        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": total_tools - enabled_tools,
            "total_executions": total_executions,
            "recent_executions_24h": len(recent_executions),
            "recent_success_rate": (
                successful_recent / len(recent_executions)
                if recent_executions else 0.0
            ),
            "source_distribution": source_counts,
            "capability_distribution": capability_counts,
            "security_distribution": security_counts,
            "execution_history_size": len(self._execution_history)
        }
    
    def add_execution_record(self, record: ToolExecutionRecord) -> None:
        """Add an execution record to the history."""
        self._execution_history.append(record)
        
        # Update tool stats
        if record.tool_name in self._local_tools:
            registration = self._local_tools[record.tool_name]
            registration.last_used = record.start_time
            registration.usage_count += 1
            
            if record.error:
                registration.error_count += 1
        
        # Trim history if too large
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]
    
    def get_execution_history(
        self, 
        tool_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ToolExecutionRecord]:
        """Get execution history with optional filtering."""
        history = self._execution_history
        
        if tool_name:
            history = [record for record in history if record.tool_name == tool_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()
        self.logger.info("Cleared tool execution history")
    
    # MCP Integration Methods (Future)
    
    async def discover_mcp_servers(self) -> List[MCPServer]:
        """Discover available MCP servers (future feature)."""
        # Future: Implement MCP server discovery
        return list(self._mcp_servers.values())
    
    async def register_mcp_server(self, server_url: str) -> None:
        """Register an MCP server (future feature)."""
        # Future: Implement MCP server registration
        self.logger.info(f"MCP server registration for {server_url} - not yet implemented")
    
    def _validate_tool_schema(self, schema: ToolSchema) -> None:
        """Validate a tool schema."""
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
    
    def _update_indexes_for_tool(self, schema: ToolSchema, add: bool = True) -> None:
        """Update indexes when adding or removing a tool."""
        tool_name = schema.name
        
        # Update capability index
        for capability in schema.capabilities:
            if add:
                self._capability_index[capability].add(tool_name)
            else:
                self._capability_index[capability].discard(tool_name)
        
        # Update security index
        if add:
            self._security_index[schema.security_level].add(tool_name)
        else:
            self._security_index[schema.security_level].discard(tool_name)
        
        # Update tag index
        for tag in schema.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            
            if add:
                self._tag_index[tag_lower].add(tool_name)
            else:
                self._tag_index[tag_lower].discard(tool_name)
    
    def _suggest_similar_tools(self, name: str) -> List[str]:
        """Suggest similar tool names for a missing tool."""
        suggestions = []
        available_names = list(self._local_tools.keys())
        
        # Simple similarity based on common substrings
        for available_name in available_names:
            if name.lower() in available_name.lower() or available_name.lower() in name.lower():
                suggestions.append(available_name)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._local_tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._local_tools
    
    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._local_tools.keys())


# Global registry instance (can be replaced with dependency injection)
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def set_global_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry instance."""
    global _global_registry
    _global_registry = registry