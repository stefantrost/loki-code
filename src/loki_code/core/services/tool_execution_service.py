"""
Tool Execution Service - Simplified integration layer for tool usage.

This service eliminates complex wrapper patterns by providing a clean,
standardized interface for tool execution across all system components.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..tool_system.tool_registry_core import get_global_registry
from ...tools.types import ToolResult, ToolContext
from ...tools.exceptions import ToolException, ToolNotFoundError
from ...utils.logging import get_logger


@dataclass
class ToolExecutionRequest:
    """Standardized request for tool execution."""
    tool_name: str
    input_data: Any
    context: Optional[ToolContext] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResponse:
    """Standardized response from tool execution."""
    success: bool
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolExecutionService:
    """
    Simplified tool execution service that eliminates wrapper complexity.
    
    Provides a clean interface for:
    1. Direct tool execution with standardized context
    2. Batch tool execution
    3. Tool capability discovery
    4. Execution monitoring and statistics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tool execution service."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.registry = get_global_registry()
        
        # Default context settings
        self.default_context = self._create_default_context()
        
        # Ensure LangChain tools are registered
        self._ensure_tools_registered()
        
        self.logger.info("ToolExecutionService initialized")
    
    def _ensure_tools_registered(self):
        """Ensure LangChain tools are registered via bridge service."""
        try:
            # Import here to avoid circular imports
            from .tool_bridge import get_global_bridge_service
            bridge = get_global_bridge_service()
            
            # Check if tools are already registered
            if not self.registry.list_tool_names():
                # Register tools if none exist
                count = bridge.register_all_langchain_tools()
                self.logger.info(f"Auto-registered {count} LangChain tools")
            
        except Exception as e:
            self.logger.warning(f"Failed to auto-register tools: {e}")
    
    def _create_default_context(self) -> ToolContext:
        """Create default tool context."""
        return ToolContext(
            project_path=self.config.get("project_path", "."),
            user_id=self.config.get("user_id", "system"),
            session_id=self.config.get("session_id", "default"),
            environment=self.config.get("environment", {}),
        )
    
    async def execute_tool(
        self,
        tool_name: str,
        input_data: Any,
        context: Optional[ToolContext] = None,
        timeout: Optional[float] = None
    ) -> ToolExecutionResponse:
        """
        Execute a single tool with simplified interface.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for the tool
            context: Optional execution context (uses default if not provided)
            timeout: Optional timeout in seconds
            
        Returns:
            ToolExecutionResponse with standardized result
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Use default context if none provided
            if context is None:
                context = self.default_context
            
            # Execute tool through registry
            result = await self.registry.execute_tool(
                name=tool_name,
                input_data=input_data,
                context=context,
                timeout=timeout
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolExecutionResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                metadata={
                    "tool_name": tool_name,
                    "context_used": "default" if context == self.default_context else "custom"
                }
            )
            
        except ToolNotFoundError as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Tool not found: {tool_name}")
            
            return ToolExecutionResponse(
                success=False,
                error=f"Tool '{tool_name}' not found. Available: {', '.join(e.available_tools)}",
                execution_time_ms=execution_time,
                metadata={"available_tools": e.available_tools}
            )
            
        except ToolException as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Tool execution failed: {e}")
            
            return ToolExecutionResponse(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                metadata={"tool_name": tool_name, "error_type": type(e).__name__}
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Unexpected error executing tool {tool_name}: {e}", exc_info=True)
            
            return ToolExecutionResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"tool_name": tool_name, "error_type": type(e).__name__}
            )
    
    async def execute_tool_simple(
        self,
        tool_name: str,
        **kwargs
    ) -> Union[str, Any]:
        """
        Simple tool execution that returns just the output data.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Input parameters for the tool
            
        Returns:
            Tool output data or error message
        """
        response = await self.execute_tool(tool_name, kwargs)
        
        if response.success and response.result:
            return response.result.output
        else:
            return response.error or "Tool execution failed"
    
    async def execute_batch(
        self,
        requests: List[ToolExecutionRequest]
    ) -> List[ToolExecutionResponse]:
        """
        Execute multiple tools in batch (parallel execution).
        
        Args:
            requests: List of tool execution requests
            
        Returns:
            List of execution responses in same order as requests
        """
        self.logger.info(f"Executing batch of {len(requests)} tools")
        
        # Create tasks for parallel execution
        tasks = []
        for request in requests:
            task = self.execute_tool(
                tool_name=request.tool_name,
                input_data=request.input_data,
                context=request.context,
                timeout=request.timeout
            )
            tasks.append(task)
        
        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert any exceptions to error responses
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                final_responses.append(ToolExecutionResponse(
                    success=False,
                    error=str(response),
                    metadata={"request_index": i, "tool_name": requests[i].tool_name}
                ))
            else:
                final_responses.append(response)
        
        return final_responses
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.registry.list_tool_names()
    
    def get_tools_by_capability(self, capability_name: str) -> List[str]:
        """Get tools that have a specific capability."""
        from ...tools.types import ToolCapability
        
        # Convert string to enum if needed
        if isinstance(capability_name, str):
            try:
                capability = ToolCapability(capability_name.upper())
            except ValueError:
                self.logger.warning(f"Unknown capability: {capability_name}")
                return []
        else:
            capability = capability_name
        
        return self.registry.get_tools_by_capability(capability)
    
    def search_tools(self, query: str) -> List[str]:
        """Search for tools by description."""
        schemas = self.registry.search_tools(query)
        return [schema.name for schema in schemas]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        schema = self.registry.get_tool_schema(tool_name)
        if not schema:
            return None
        
        stats = self.registry.get_tool_statistics(tool_name)
        
        return {
            "name": schema.name,
            "description": schema.description,
            "capabilities": [cap.value for cap in schema.capabilities],
            "security_level": schema.security_level.value,
            "confirmation_level": schema.confirmation_level.value,
            "input_schema": schema.input_schema,
            "statistics": stats
        }
    
    def create_context(
        self,
        project_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ToolContext:
        """Create a custom tool context."""
        return ToolContext(
            project_path=project_path or self.default_context.project_path,
            user_id=user_id or self.default_context.user_id,
            session_id=session_id or self.default_context.session_id,
            environment={**self.default_context.environment, **kwargs}
        )
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics from the registry."""
        return self.registry.get_registry_statistics()


# Global service instance
_global_service: Optional[ToolExecutionService] = None


def get_global_tool_service() -> ToolExecutionService:
    """Get the global tool execution service instance."""
    global _global_service
    if _global_service is None:
        _global_service = ToolExecutionService()
    return _global_service


def reset_global_tool_service():
    """Reset the global tool service (useful for testing)."""
    global _global_service
    _global_service = None


# Convenience functions for simple tool execution
async def execute_tool(tool_name: str, **kwargs) -> Union[str, Any]:
    """Global convenience function for simple tool execution."""
    service = get_global_tool_service()
    return await service.execute_tool_simple(tool_name, **kwargs)


async def read_file(file_path: str, **kwargs) -> str:
    """Convenience function to read a file."""
    return await execute_tool("file_reader", file_path=file_path, **kwargs)


async def write_file(file_path: str, content: str, **kwargs) -> str:
    """Convenience function to write a file."""
    return await execute_tool("file_writer", file_path=file_path, content=content, **kwargs)


def list_tools() -> List[str]:
    """Convenience function to list available tools."""
    service = get_global_tool_service()
    return service.get_available_tools()


def search_tools(query: str) -> List[str]:
    """Convenience function to search tools."""
    service = get_global_tool_service()
    return service.search_tools(query)