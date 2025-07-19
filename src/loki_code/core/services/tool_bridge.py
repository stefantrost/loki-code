"""
Tool Bridge Service - Bridges LangChain tools with our internal tool system.

This service creates adapter classes that allow LangChain tools to work
seamlessly with our internal tool registry and execution services.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from langchain_core.tools import BaseTool as LangChainBaseTool
from ..tool_system.tool_registry_core import get_global_registry
from ...tools.base import BaseTool as InternalBaseTool
from ...tools.types import ToolSchema, ToolResult, ToolContext, ToolCapability, SecurityLevel, ConfirmationLevel
from ...tools.langchain_tools import create_langchain_tools
from ...utils.logging import get_logger


class LangChainToolAdapter(InternalBaseTool):
    """
    Adapter that wraps a LangChain tool to work with our internal tool system.
    """
    
    def __init__(self, langchain_tool: LangChainBaseTool, config: Optional[Dict[str, Any]] = None):
        """Initialize the adapter with a LangChain tool."""
        super().__init__(config)
        self.langchain_tool = langchain_tool
        self.logger = get_logger(__name__)
        
        # Cache schema
        self._cached_schema: Optional[ToolSchema] = None
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema by converting from LangChain tool."""
        if self._cached_schema is None:
            self._cached_schema = self._convert_langchain_schema()
        return self._cached_schema
    
    def _convert_langchain_schema(self) -> ToolSchema:
        """Convert LangChain tool schema to our internal format."""
        # Get basic info from LangChain tool
        name = self.langchain_tool.name
        description = self.langchain_tool.description
        
        # Create input schema from LangChain args_schema
        input_schema = {}
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            # Convert Pydantic schema to JSON schema
            try:
                pydantic_schema = self.langchain_tool.args_schema.schema()
                input_schema = {
                    "type": "object",
                    "properties": pydantic_schema.get("properties", {}),
                    "required": pydantic_schema.get("required", []),
                    "description": f"Input schema for {name}"
                }
            except Exception as e:
                self.logger.warning(f"Failed to convert schema for {name}: {e}")
                input_schema = {"type": "object", "properties": {}}
        
        # Create output schema
        output_schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "Tool execution result"},
                "success": {"type": "boolean", "description": "Whether execution succeeded"}
            },
            "required": ["result", "success"]
        }
        
        # Determine capabilities based on tool name and description
        capabilities = []
        
        # File operations
        if "read" in name.lower() and "file" in name.lower():
            capabilities.append(ToolCapability.READ_FILE)
        if "write" in name.lower() and "file" in name.lower():
            capabilities.append(ToolCapability.WRITE_FILE)
        
        # Code generation
        if "code" in name.lower() or "generate" in name.lower():
            capabilities.append(ToolCapability.CODE_GENERATION)
        
        # Default to code analysis if no specific capabilities
        if not capabilities:
            capabilities.append(ToolCapability.CODE_ANALYSIS)
        
        # Security level - conservative default
        security_level = SecurityLevel.SAFE
        if "write" in name.lower() or "create" in name.lower():
            security_level = SecurityLevel.CAUTION
        
        return ToolSchema(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            capabilities=capabilities,
            security_level=security_level,
            confirmation_level=ConfirmationLevel.NONE
        )
    
    async def execute(self, input_data: Any, context: ToolContext) -> ToolResult:
        """Execute the LangChain tool."""
        try:
            # Convert input_data to the format expected by LangChain tool
            if isinstance(input_data, dict):
                # Use the dict directly
                tool_input = input_data
            else:
                # Try to convert to dict if it's a simple type
                tool_input = {"input": input_data}
            
            # Execute the LangChain tool
            try:
                # Check if the tool has async support
                if hasattr(self.langchain_tool, '_arun'):
                    result = await self.langchain_tool._arun(**tool_input)
                else:
                    # Run sync method in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.langchain_tool._run, **tool_input)
            except Exception as e:
                # Try alternative execution method
                try:
                    result = await self.langchain_tool.ainvoke(tool_input)
                except:
                    # Last resort - try sync invoke
                    result = self.langchain_tool.invoke(tool_input)
            
            # Convert result to our format
            return ToolResult.success_result(
                output=result,
                metadata={
                    "langchain_tool": self.langchain_tool.name,
                    "input_data": tool_input
                }
            )
            
        except Exception as e:
            self.logger.error(f"LangChain tool execution failed: {e}", exc_info=True)
            return ToolResult.failure_result(
                message=f"Tool execution failed: {str(e)}",
                metadata={
                    "langchain_tool": self.langchain_tool.name,
                    "error": str(e),
                    "error_code": "langchain_execution_error"
                }
            )


class ToolBridgeService:
    """
    Service that bridges LangChain tools with our internal tool system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tool bridge service."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.registry = get_global_registry()
        
        # Track registered adapters
        self._adapters: Dict[str, LangChainToolAdapter] = {}
        
        self.logger.info("ToolBridgeService initialized")
    
    def register_langchain_tool(self, langchain_tool: LangChainBaseTool) -> bool:
        """
        Register a LangChain tool with our internal registry.
        
        Args:
            langchain_tool: LangChain tool to register
            
        Returns:
            True if registration succeeded
        """
        try:
            # Create adapter
            adapter = LangChainToolAdapter(langchain_tool)
            
            # Register with internal registry
            self.registry.register_tool_instance(adapter, source="langchain_bridge")
            
            # Track adapter
            self._adapters[langchain_tool.name] = adapter
            
            self.logger.info(f"Registered LangChain tool: {langchain_tool.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register LangChain tool {langchain_tool.name}: {e}")
            return False
    
    def register_all_langchain_tools(self) -> int:
        """
        Register all available LangChain tools.
        
        Returns:
            Number of tools registered
        """
        registered_count = 0
        
        try:
            # Get all LangChain tools
            langchain_tools = create_langchain_tools()
            
            # Register each tool
            for tool in langchain_tools:
                if self.register_langchain_tool(tool):
                    registered_count += 1
            
            self.logger.info(f"Registered {registered_count} LangChain tools")
            
        except Exception as e:
            self.logger.error(f"Failed to register LangChain tools: {e}")
        
        return registered_count
    
    def unregister_langchain_tool(self, tool_name: str) -> bool:
        """
        Unregister a LangChain tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistration succeeded
        """
        try:
            # Remove from registry
            success = self.registry.unregister_tool(tool_name)
            
            # Remove from adapters
            if tool_name in self._adapters:
                del self._adapters[tool_name]
            
            if success:
                self.logger.info(f"Unregistered LangChain tool: {tool_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to unregister LangChain tool {tool_name}: {e}")
            return False
    
    def get_adapter(self, tool_name: str) -> Optional[LangChainToolAdapter]:
        """Get adapter for a tool."""
        return self._adapters.get(tool_name)
    
    def list_bridged_tools(self) -> List[str]:
        """Get list of bridged tool names."""
        return list(self._adapters.keys())
    
    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get bridge service statistics."""
        return {
            "bridged_tools": len(self._adapters),
            "tool_names": list(self._adapters.keys()),
            "registry_tools": len(self.registry.list_tool_names()),
            "total_tools_in_registry": len(self.registry.list_tool_names())
        }
    
    def auto_register_on_startup(self) -> int:
        """Auto-register LangChain tools on startup."""
        if self.config.get("auto_register", True):
            return self.register_all_langchain_tools()
        return 0


# Global bridge service instance
_global_bridge_service: Optional[ToolBridgeService] = None


def get_global_bridge_service() -> ToolBridgeService:
    """Get the global tool bridge service."""
    global _global_bridge_service
    if _global_bridge_service is None:
        _global_bridge_service = ToolBridgeService()
        # Auto-register tools on first access
        _global_bridge_service.auto_register_on_startup()
    return _global_bridge_service


def ensure_langchain_tools_registered() -> int:
    """Ensure LangChain tools are registered. Returns number of tools registered."""
    bridge = get_global_bridge_service()
    return bridge.register_all_langchain_tools()


def get_bridged_tools() -> List[str]:
    """Get list of bridged tool names."""
    bridge = get_global_bridge_service()
    return bridge.list_bridged_tools()