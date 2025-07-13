"""
Agent system for Loki Code.

Simplified and modular agent architecture with clear separation of concerns.

The agent system now consists of:
- Core agent orchestrator (agent_core.py)  
- Specialized components (request_analyzer, execution_planner)
- Type definitions (types.py)
- Permission, safety, and conversation managers

This replaces the original monolithic 807-line loki_agent.py with 
a cleaner, more maintainable structure.
"""

# Main agent class (simplified)
from .agent_core import LokiCodeAgent

# Type definitions
from .types import (
    AgentConfig, AgentResponse, RequestContext,
    RequestUnderstanding, ExecutionPlan, AgentState
)

# Specialized components
from .request_analyzer import RequestAnalyzer
from .execution_planner import ExecutionPlanner

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

# Backward compatibility - original monolithic class still available
try:
    from .loki_agent import LokiCodeAgent as LokiCodeAgentOriginal
except ImportError:
    LokiCodeAgentOriginal = None

__all__ = [
    # Main agent (simplified)
    "LokiCodeAgent",
    
    # Types
    "AgentConfig", 
    "AgentResponse",
    "RequestContext",
    "RequestUnderstanding",
    "ExecutionPlan", 
    "AgentState",
    
    # Specialized components
    "RequestAnalyzer",
    "ExecutionPlanner",
    
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
    "ConversationConfig",
    
    # Backward compatibility
    "LokiCodeAgentOriginal"
]