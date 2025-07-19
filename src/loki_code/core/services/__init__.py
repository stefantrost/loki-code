"""
Core services for Loki Code.

This module provides high-level service interfaces that simplify
integration and eliminate complex wrapper patterns.
"""

from .tool_execution_service import (
    ToolExecutionService,
    ToolExecutionRequest,
    ToolExecutionResponse,
    get_global_tool_service,
    execute_tool,
    read_file,
    write_file,
    list_tools,
    search_tools
)

# Unified command processor removed - replaced by AgentService

from .dynamic_tool_registry import (
    DynamicToolRegistry,
    get_global_dynamic_registry
)

from .unified_context_manager import (
    UnifiedContextManager,
    UnifiedContext,
    get_global_context_manager,
    get_context,
    create_context,
    get_or_create_context
)

from .unified_memory_manager import (
    UnifiedMemoryManager,
    MemoryType,
    MemoryEntry,
    get_global_memory_manager,
    add_conversation_memory,
    add_operation_memory,
    get_conversation_context
)

from .tool_bridge import (
    ToolBridgeService,
    LangChainToolAdapter,
    get_global_bridge_service,
    ensure_langchain_tools_registered,
    get_bridged_tools
)

from .agent_service import (
    AgentService,
    get_agent_service,
    reset_agent_service
)

from .http_agent_service import HttpAgentService

__all__ = [
    # Tool Execution Service
    "ToolExecutionService",
    "ToolExecutionRequest", 
    "ToolExecutionResponse",
    "get_global_tool_service",
    "execute_tool",
    "read_file",
    "write_file",
    "list_tools",
    "search_tools",
    
# Unified Command Processor removed - replaced by AgentService
    
    # Dynamic Tool Registry
    "DynamicToolRegistry",
    "get_global_dynamic_registry",
    
    # Unified Context Manager
    "UnifiedContextManager",
    "UnifiedContext", 
    "get_global_context_manager",
    "get_context",
    "create_context",
    "get_or_create_context",
    
    # Unified Memory Manager
    "UnifiedMemoryManager",
    "MemoryType",
    "MemoryEntry",
    "get_global_memory_manager",
    "add_conversation_memory",
    "add_operation_memory",
    "get_conversation_context",
    
    # Tool Bridge Service
    "ToolBridgeService",
    "LangChainToolAdapter",
    "get_global_bridge_service",
    "ensure_langchain_tools_registered",
    "get_bridged_tools",
    
    # Agent Service
    "AgentService",
    "HttpAgentService",
    "get_agent_service",
    "reset_agent_service"
]