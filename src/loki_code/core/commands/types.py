"""
Shared types for command processing.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class IntentType(Enum):
    """Types of user intents."""
    
    FILE_ANALYSIS = "file_analysis"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    SEARCH = "search"
    SYSTEM_COMMAND = "system_command"
    HELP = "help"
    CONVERSATION = "conversation"
    FOLLOW_UP = "follow_up"


class RouteType(Enum):
    """Types of routing decisions."""
    
    DIRECT_TOOL = "direct_tool"
    AGENT_CONVERSATION = "agent_conversation"
    SYSTEM_COMMAND = "system_command"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class Intent:
    """User intent with confidence score."""
    
    type: IntentType
    confidence: float
    reasoning: str = ""


@dataclass
class ParsedInput:
    """Parsed user input with extracted information."""
    
    original_text: str
    cleaned_text: str
    intent: Intent
    entities: Dict[str, List[str]]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision about how to route a command."""
    
    route_type: RouteType
    confidence: float
    target_tool: Optional[str] = None
    system_command: Optional[str] = None
    clarification_questions: List[str] = None
    reasoning: str = ""


@dataclass
class ProcessedCommand:
    """Result of processing a user command."""
    
    success: bool
    message: str
    execution_type: str  # "direct_tool", "agent_conversation", "system_command", "shortcut"
    tool_results: List[Any] = field(default_factory=list)
    agent_response: Optional[Any] = None
    direct_tool_call: Optional[tuple[str, Dict[str, Any]]] = None  # (tool_name, tool_args)
    system_command: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for a conversation session."""
    
    session_id: str
    project_path: Optional[str] = None
    current_file: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class OperationType(Enum):
    """Types of operations that can be tracked."""
    FILE_CREATION = "file_creation"
    FILE_READING = "file_reading"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    SEARCH = "search"


@dataclass
class SessionOperation:
    """Represents a recent operation in the session."""
    
    operation_type: OperationType
    tool_name: str
    tool_input: Dict[str, Any]
    result: 'ProcessedCommand'  # Forward reference
    timestamp: float
    user_input: str
    
    def is_recent(self, max_age_seconds: int = 300) -> bool:
        """Check if operation is recent enough to be referenced."""
        return time.time() - self.timestamp < max_age_seconds