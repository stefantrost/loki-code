"""
Agent system for Loki Code.

LangChain-based agent architecture with ReAct pattern.

The agent system now consists of:
- LangChain ReAct agent (langchain_agent.py)
- Type definitions (types.py)
- Permission, safety, and conversation managers
- LangChain tool adapters

This implements proper LangChain integration as specified in Phase 4.
"""

# Main agent classes (LangGraph-based)
from .langgraph_agent import LokiLangGraphAgent, LokiLangGraphAgentFactory

# Default to LangGraph agent (modern approach)
LokiCodeAgent = LokiLangGraphAgent
LokiCodeAgentFactory = LokiLangGraphAgentFactory

# Type definitions
from .types import (
    AgentConfig, AgentResponse, RequestContext,
    RequestUnderstanding, ExecutionPlan, AgentState
)

# Backward compatibility
LokiAgent = LokiCodeAgent  # Alias for backward compatibility

# Existing managers
from .permission_manager import (
    PermissionManager, PermissionLevel, PermissionResult,
    ToolAction, PermissionConfig
)
from .safety_manager import (
    SafetyManager, SafetyResult, RecoveryPlan, RecoveryStrategy,
    SafetyConfig
)
from .conversation_manager import (
    ConversationManager, InteractionType, UserPreferences,
    ConversationConfig
)

__all__ = [
    # Main agents
    "LokiCodeAgent",  # Default (LangGraph)
    "LokiCodeAgentFactory",  # Default (LangGraph)
    "LokiLangGraphAgent",  # Modern
    "LokiLangGraphAgentFactory",  # Modern
    
    # Types
    "AgentConfig", 
    "AgentResponse",
    "RequestContext",
    "RequestUnderstanding",
    "ExecutionPlan", 
    "AgentState",
    
    # Backward compatibility
    "LokiAgent",
    
    # Existing managers
    "PermissionManager",
    "PermissionLevel",
    "PermissionResult", 
    "ToolAction",
    "PermissionConfig",
    "SafetyManager",
    "SafetyResult",
    "RecoveryPlan",
    "RecoveryStrategy",
    "SafetyConfig",
    "ConversationManager",
    "InteractionType",
    "UserPreferences",
    "ConversationConfig"
]