"""
LangChain-based intelligent agent system for Loki Code.

This package provides a comprehensive agent system that implements:
- Permission-based autonomy with user interaction
- Intelligent reasoning with clarification capabilities  
- Safety-first error recovery and validation
- Progressive user interaction patterns
- LangChain integration with tool orchestration

The agent system builds on the existing prompt template system to create
a true coding assistant that can reason, ask for clarification, and 
operate safely within defined boundaries.
"""

from .loki_agent import LokiCodeAgent, AgentConfig, AgentResponse, RequestContext
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
    # Main agent
    "LokiCodeAgent",
    "AgentConfig",
    "AgentResponse",
    "RequestContext",
    
    # Permission system
    "PermissionManager",
    "PermissionLevel",
    "PermissionResult", 
    "ToolAction",
    "PermissionConfig",
    
    # Safety system
    "SafetyManager",
    "SafetyResult",
    "RecoveryPlan",
    "RecoveryStrategy",
    "SafetyConfig",
    
    # Conversation system
    "ConversationManager",
    "InteractionType",
    "UserPreferences",
    "ConversationConfig",
]