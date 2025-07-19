"""
Agent type definitions and data structures.

Extracted from loki_agent.py for better organization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .conversation_manager import UserPreferences


class AgentState(Enum):
    """Current state of the agent."""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_PERMISSION = "waiting_for_permission"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"


@dataclass
class RequestUnderstanding:
    """Agent's understanding of a user request."""
    user_intent: str
    confidence: float
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    ambiguous_aspects: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    risk_assessment: str = "low"
    suggested_approach: str = ""


@dataclass
class ExecutionPlan:
    """Plan for executing a user request."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration: float = 0.0
    required_permissions: List[str] = field(default_factory=list)
    safety_considerations: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from the agent."""
    content: str
    state: AgentState = AgentState.COMPLETED
    actions_taken: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    permissions_requested: int = 0
    safety_checks_passed: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Context for processing a request."""
    project_path: Optional[str] = None
    current_file: Optional[str] = None
    target_files: List[str] = field(default_factory=list)
    user_preferences: Optional[UserPreferences] = None
    session_id: str = ""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Unified context integration
    recent_operations: List[Dict[str, Any]] = field(default_factory=list)
    file_contexts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    language_consistency: Optional[str] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for the Loki Code agent."""
    # Core settings
    max_steps: int = 10
    timeout_seconds: int = 300
    auto_approve_safe_actions: bool = True
    
    # LangChain settings
    use_langchain: bool = True
    model_name: str = "qwen3:32b"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Tool settings
    max_tool_retries: int = 3
    tool_timeout_seconds: int = 30
    
    # Safety settings
    require_permission_for_writes: bool = True
    require_permission_for_commands: bool = True
    enable_safety_validation: bool = True
    
    # Conversation settings
    max_conversation_history: int = 100
    include_conversation_context: bool = True
    
    # Performance settings
    enable_parallel_tool_execution: bool = False
    max_concurrent_tools: int = 3
    
    # Development settings
    debug_mode: bool = False
    log_langchain_calls: bool = True  # Always show reasoning steps
    simulate_tools: bool = False
    show_agent_reasoning: bool = True  # Always display agent thinking process