"""
Type definitions for the tool system.

Extracted from tool_registry.py for better organization.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type

from ...tools.base import BaseTool
from ...tools.types import ToolSchema, ToolContext, ToolResult, ToolCapability, SecurityLevel


@dataclass
class ToolFilter:
    """Filter criteria for tool queries."""
    capabilities: Optional[List[ToolCapability]] = None
    security_levels: Optional[List[SecurityLevel]] = None
    mcp_compatible: Optional[bool] = None
    keywords: Optional[List[str]] = None
    enabled_only: bool = True


@dataclass
class ToolRegistration:
    """Information about a registered tool."""
    tool_class: Type[BaseTool]
    tool_instance: Optional[BaseTool] = None
    schema: Optional[ToolSchema] = None
    enabled: bool = True
    source: str = "manual"  # manual, builtin, plugin, mcp
    registered_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    usage_count: int = 0
    error_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServer:
    """Information about an MCP server."""
    url: str
    name: str
    description: str = ""
    available_tools: List[str] = field(default_factory=list)
    last_ping: Optional[float] = None
    enabled: bool = True


@dataclass
class ToolExecutionRecord:
    """Record of a tool execution for monitoring and debugging."""
    tool_name: str
    input_data: Any
    context: ToolContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    execution_id: str = field(default_factory=lambda: f"exec_{int(time.time() * 1000)}")
    
    @classmethod
    def start(cls, tool_name: str, input_data: Any, context: ToolContext) -> 'ToolExecutionRecord':
        """Start a new execution record."""
        return cls(
            tool_name=tool_name,
            input_data=input_data,
            context=context
        )
    
    def complete(self, result: ToolResult) -> None:
        """Mark execution as completed with result."""
        self.end_time = time.time()
        self.result = result
    
    def fail(self, error: str) -> None:
        """Mark execution as failed with error."""
        self.end_time = time.time()
        self.error = error
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.end_time is not None