"""
Conversation management and context tracking for Loki Code.

This module manages conversation flow, context windows, and conversation
state for multi-turn interactions with the intelligent agent.
"""

import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from pathlib import Path
import logging

from .parser import ParsedResponse
from ...tools.types import ToolResult
from ...utils.logging import get_logger


class ConversationIntent(Enum):
    """Types of conversation intents."""
    CODE_ANALYSIS = "code_analysis"
    CODE_MODIFICATION = "code_modification"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    EXPLORATION = "exploration"
    PLANNING = "planning"
    QUESTION_ANSWERING = "question_answering"
    GENERAL_CHAT = "general_chat"
    TOOL_USAGE = "tool_usage"
    PROJECT_MANAGEMENT = "project_management"


class ConversationPhase(Enum):
    """Phases of a conversation."""
    INITIATION = "initiation"
    EXPLORATION = "exploration"
    ACTION = "action"
    CLARIFICATION = "clarification"
    COMPLETION = "completion"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: str
    timestamp: float
    user_message: str
    agent_response: Optional[ParsedResponse] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    intent: Optional[ConversationIntent] = None
    phase: ConversationPhase = ConversationPhase.INITIATION
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get turn duration if completed."""
        if self.agent_response and hasattr(self.agent_response, 'metadata'):
            return self.agent_response.metadata.get('processing_time_ms', 0) / 1000
        return 0.0
    
    @property
    def has_errors(self) -> bool:
        """Check if turn has any errors."""
        if self.agent_response and self.agent_response.has_errors:
            return True
        return any(not result.success for result in self.tool_results)
    
    @property
    def tool_count(self) -> int:
        """Get number of tools used in this turn."""
        return len(self.tool_results)


@dataclass
class ContextWindow:
    """A window of conversation context."""
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 10
    max_tokens: int = 4000
    current_tokens: int = 0
    
    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the context window."""
        self.turns.append(turn)
        self._estimate_tokens()
        self._trim_if_needed()
    
    def get_context_text(self) -> str:
        """Get conversation context as text."""
        context_parts = []
        
        for turn in self.turns:
            context_parts.append(f"User: {turn.user_message}")
            if turn.agent_response:
                context_parts.append(f"Assistant: {turn.agent_response.text_content}")
        
        return "\n".join(context_parts)
    
    def get_recent_turns(self, count: int) -> List[ConversationTurn]:
        """Get the most recent N turns."""
        return self.turns[-count:] if count > 0 else []
    
    def _estimate_tokens(self):
        """Estimate token count for current context."""
        # Rough estimation: ~4 characters per token
        text = self.get_context_text()
        self.current_tokens = len(text) // 4
    
    def _trim_if_needed(self):
        """Trim context if it exceeds limits."""
        # Remove oldest turns if we exceed limits
        while (len(self.turns) > self.max_turns or 
               self.current_tokens > self.max_tokens) and self.turns:
            self.turns.pop(0)
            self._estimate_tokens()


@dataclass
class ConversationContext:
    """Rich conversation context with analysis."""
    project_path: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    primary_intent: Optional[ConversationIntent] = None
    active_files: List[str] = field(default_factory=list)
    active_tools: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    conversation_summary: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def add_file(self, file_path: str):
        """Add a file to active context."""
        if file_path not in self.active_files:
            self.active_files.append(file_path)
            # Keep only recent files
            if len(self.active_files) > 10:
                self.active_files = self.active_files[-10:]
    
    def add_tool(self, tool_name: str):
        """Add a tool to active context."""
        if tool_name not in self.active_tools:
            self.active_tools.append(tool_name)
    
    def add_entity(self, entity: str):
        """Add a key entity to context."""
        if entity not in self.key_entities:
            self.key_entities.append(entity)
            # Keep only recent entities
            if len(self.key_entities) > 20:
                self.key_entities = self.key_entities[-20:]


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""
    max_turns: int = 50
    context_window: int = 10
    max_context_tokens: int = 4000
    enable_context_compression: bool = True
    enable_intent_tracking: bool = True
    enable_entity_extraction: bool = True
    auto_summarize_threshold: int = 20
    session_timeout_minutes: int = 60
    persist_conversations: bool = True
    conversation_storage_path: Optional[str] = None


@dataclass
class Conversation:
    """A complete conversation session."""
    conversation_id: str
    session_id: str
    start_time: float
    last_activity: float
    turns: List[ConversationTurn] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    context_window: Optional[ContextWindow] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get conversation duration in seconds."""
        return self.last_activity - self.start_time
    
    @property
    def turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)
    
    @property
    def is_active(self) -> bool:
        """Check if conversation is still active."""
        # Consider active if last activity was within session timeout
        timeout_seconds = 60 * 60  # 1 hour default
        return (time.time() - self.last_activity) < timeout_seconds
    
    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the conversation."""
        self.turns.append(turn)
        self.last_activity = time.time()
        
        # Update context window
        if self.context_window:
            self.context_window.add_turn(turn)
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.turns:
            return "Empty conversation"
        
        total_tools = sum(turn.tool_count for turn in self.turns)
        error_count = sum(1 for turn in self.turns if turn.has_errors)
        
        return (f"Conversation with {self.turn_count} turns, "
                f"{total_tools} tool executions, {error_count} errors")


class ConversationManager:
    """
    Manager for conversation flow and context tracking.
    
    Handles multi-turn conversations, context windows, intent tracking,
    and conversation persistence for the intelligent agent system.
    """
    
    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        self.logger = get_logger(__name__)
        
        # Active conversations
        self.conversations: Dict[str, Conversation] = {}
        self.session_conversations: Dict[str, str] = {}  # session_id -> conversation_id
        
        # Intent classification patterns
        self._intent_patterns = self._compile_intent_patterns()
        
        # Entity extraction patterns
        self._entity_patterns = self._compile_entity_patterns()
    
    def start_conversation(self, session_id: str, 
                          context: Optional[ConversationContext] = None) -> str:
        """Start a new conversation.
        
        Args:
            session_id: Session identifier
            context: Initial conversation context
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        current_time = time.time()
        
        conversation = Conversation(
            conversation_id=conversation_id,
            session_id=session_id,
            start_time=current_time,
            last_activity=current_time,
            context=context or ConversationContext(session_id=session_id),
            context_window=ContextWindow(
                max_turns=self.config.context_window,
                max_tokens=self.config.max_context_tokens
            )
        )
        
        self.conversations[conversation_id] = conversation
        self.session_conversations[session_id] = conversation_id
        
        self.logger.info(f"Started conversation {conversation_id} for session {session_id}")
        return conversation_id
    
    def add_turn(self, session_id: str, user_message: str, 
                 agent_response: Optional[ParsedResponse] = None,
                 tool_results: Optional[List[ToolResult]] = None) -> str:
        """Add a turn to the conversation.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            agent_response: Agent's parsed response
            tool_results: Results from tool executions
            
        Returns:
            Turn ID
        """
        conversation_id = self.session_conversations.get(session_id)
        if not conversation_id:
            conversation_id = self.start_conversation(session_id)
        
        conversation = self.conversations[conversation_id]
        
        # Create turn
        turn_id = str(uuid.uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=time.time(),
            user_message=user_message,
            agent_response=agent_response,
            tool_results=tool_results or []
        )
        
        # Analyze and enhance turn
        self._analyze_turn(turn, conversation.context)
        
        # Add to conversation
        conversation.add_turn(turn)
        
        # Update conversation context
        self._update_conversation_context(conversation, turn)
        
        # Auto-summarize if needed
        if (self.config.enable_context_compression and 
            len(conversation.turns) >= self.config.auto_summarize_threshold):
            self._summarize_conversation(conversation)
        
        self.logger.debug(f"Added turn {turn_id} to conversation {conversation_id}")
        return turn_id
    
    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get conversation for a session."""
        conversation_id = self.session_conversations.get(session_id)
        if conversation_id:
            return self.conversations.get(conversation_id)
        return None
    
    def get_context_for_request(self, session_id: str) -> Tuple[str, ConversationContext]:
        """Get conversation context for a new request.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Tuple of (context_text, conversation_context)
        """
        conversation = self.get_conversation(session_id)
        if not conversation:
            return "", ConversationContext(session_id=session_id)
        
        # Get context from context window
        context_text = ""
        if conversation.context_window:
            context_text = conversation.context_window.get_context_text()
        
        return context_text, conversation.context
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get a summary of the conversation."""
        conversation = self.get_conversation(session_id)
        if conversation:
            return conversation.get_summary()
        return "No active conversation"
    
    def end_conversation(self, session_id: str):
        """End a conversation session."""
        conversation_id = self.session_conversations.get(session_id)
        if conversation_id:
            conversation = self.conversations.get(conversation_id)
            if conversation and self.config.persist_conversations:
                self._persist_conversation(conversation)
            
            # Clean up
            del self.conversations[conversation_id]
            del self.session_conversations[session_id]
            
            self.logger.info(f"Ended conversation {conversation_id}")
    
    def cleanup_inactive_conversations(self):
        """Clean up inactive conversations."""
        current_time = time.time()
        timeout_seconds = self.config.session_timeout_minutes * 60
        
        inactive_conversations = []
        for conv_id, conversation in self.conversations.items():
            if (current_time - conversation.last_activity) > timeout_seconds:
                inactive_conversations.append(conv_id)
        
        for conv_id in inactive_conversations:
            session_id = None
            for sid, cid in self.session_conversations.items():
                if cid == conv_id:
                    session_id = sid
                    break
            
            if session_id:
                self.end_conversation(session_id)
        
        if inactive_conversations:
            self.logger.info(f"Cleaned up {len(inactive_conversations)} inactive conversations")
    
    def _analyze_turn(self, turn: ConversationTurn, context: ConversationContext):
        """Analyze a turn to extract intent, entities, etc."""
        # Classify intent
        if self.config.enable_intent_tracking:
            turn.intent = self._classify_intent(turn.user_message)
        
        # Extract entities
        if self.config.enable_entity_extraction:
            entities = self._extract_entities(turn.user_message)
            for entity in entities:
                context.add_entity(entity)
        
        # Determine conversation phase
        turn.phase = self._determine_phase(turn, context)
        
        # Calculate confidence
        turn.confidence = self._calculate_turn_confidence(turn)
    
    def _classify_intent(self, message: str) -> ConversationIntent:
        """Classify the intent of a user message."""
        message_lower = message.lower()
        
        # Check patterns for each intent
        for intent, patterns in self._intent_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return intent
        
        return ConversationIntent.GENERAL_CHAT
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract key entities from a message."""
        entities = []
        
        # File paths
        import re
        file_patterns = [
            r'[\w/]+\.py\b',
            r'[\w/]+\.js\b',
            r'[\w/]+\.ts\b',
            r'[\w/]+\.json\b',
            r'[\w/]+\.yaml\b',
            r'[\w/]+\.md\b'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, message)
            entities.extend(matches)
        
        # Function/class names (simple heuristic)
        code_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[a-z_][a-z0-9_]*\(\)', message)
        entities.extend(code_entities)
        
        return list(set(entities))  # Remove duplicates
    
    def _determine_phase(self, turn: ConversationTurn, 
                        context: ConversationContext) -> ConversationPhase:
        """Determine the conversation phase for a turn."""
        if turn.intent in [ConversationIntent.CODE_ANALYSIS, ConversationIntent.EXPLORATION]:
            return ConversationPhase.EXPLORATION
        elif turn.intent in [ConversationIntent.CODE_MODIFICATION, ConversationIntent.TOOL_USAGE]:
            return ConversationPhase.ACTION
        elif turn.agent_response and turn.agent_response.needs_clarification:
            return ConversationPhase.CLARIFICATION
        elif turn.has_errors:
            return ConversationPhase.ERROR_RECOVERY
        else:
            return ConversationPhase.INITIATION
    
    def _calculate_turn_confidence(self, turn: ConversationTurn) -> float:
        """Calculate confidence score for a turn."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on agent response confidence
        if turn.agent_response:
            confidence = turn.agent_response.confidence_score
        
        # Adjust based on tool results
        if turn.tool_results:
            successful_tools = sum(1 for result in turn.tool_results if result.success)
            tool_success_rate = successful_tools / len(turn.tool_results)
            confidence *= tool_success_rate
        
        return min(1.0, max(0.1, confidence))
    
    def _update_conversation_context(self, conversation: Conversation, turn: ConversationTurn):
        """Update conversation context based on the turn."""
        # Update primary intent if confident
        if turn.confidence > 0.8 and turn.intent:
            conversation.context.primary_intent = turn.intent
        
        # Add files and tools from turn
        for result in turn.tool_results:
            conversation.context.add_tool(result.tool_name)
            
            # Extract file paths from tool results
            if hasattr(result, 'metadata') and 'file_path' in result.metadata:
                conversation.context.add_file(result.metadata['file_path'])
    
    def _summarize_conversation(self, conversation: Conversation):
        """Create a summary of the conversation for context compression."""
        # Simple summarization - could be enhanced with LLM
        key_points = []
        
        # Extract key intents
        intents = [turn.intent for turn in conversation.turns if turn.intent]
        if intents:
            most_common_intent = max(set(intents), key=intents.count)
            key_points.append(f"Primary activity: {most_common_intent.value}")
        
        # Extract key files
        if conversation.context.active_files:
            key_points.append(f"Working with files: {', '.join(conversation.context.active_files[-5:])}")
        
        # Extract key tools
        if conversation.context.active_tools:
            key_points.append(f"Used tools: {', '.join(set(conversation.context.active_tools))}")
        
        conversation.context.conversation_summary = "; ".join(key_points)
        
        self.logger.debug(f"Summarized conversation {conversation.conversation_id}")
    
    def _persist_conversation(self, conversation: Conversation):
        """Persist conversation to storage."""
        if not self.config.conversation_storage_path:
            return
        
        try:
            storage_path = Path(self.config.conversation_storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            file_path = storage_path / f"{conversation.conversation_id}.json"
            
            # Convert to serializable format
            data = {
                "conversation_id": conversation.conversation_id,
                "session_id": conversation.session_id,
                "start_time": conversation.start_time,
                "last_activity": conversation.last_activity,
                "duration": conversation.duration,
                "turn_count": conversation.turn_count,
                "summary": conversation.get_summary(),
                "context": {
                    "project_path": conversation.context.project_path,
                    "primary_intent": conversation.context.primary_intent.value if conversation.context.primary_intent else None,
                    "active_files": conversation.context.active_files,
                    "active_tools": conversation.context.active_tools,
                    "key_entities": conversation.context.key_entities
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Persisted conversation to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist conversation: {e}")
    
    def _compile_intent_patterns(self) -> Dict[ConversationIntent, List[str]]:
        """Compile intent classification patterns."""
        return {
            ConversationIntent.CODE_ANALYSIS: [
                "analyze", "examine", "look at", "understand", "explain", "what does", "how does"
            ],
            ConversationIntent.CODE_MODIFICATION: [
                "change", "modify", "update", "fix", "refactor", "improve", "add", "remove"
            ],
            ConversationIntent.DEBUGGING: [
                "debug", "error", "bug", "problem", "issue", "not working", "broken"
            ],
            ConversationIntent.EXPLANATION: [
                "explain", "help me understand", "what is", "how to", "why"
            ],
            ConversationIntent.EXPLORATION: [
                "show me", "list", "find", "search", "explore", "browse"
            ],
            ConversationIntent.PLANNING: [
                "plan", "strategy", "approach", "how should", "next steps"
            ],
            ConversationIntent.TOOL_USAGE: [
                "use", "run", "execute", "tool", "command"
            ]
        }
    
    def _compile_entity_patterns(self) -> Dict[str, str]:
        """Compile entity extraction patterns."""
        return {
            "file_path": r"[\w/.-]+\.[a-zA-Z]+",
            "function_name": r"\b[a-z_][a-z0-9_]*\(\)",
            "class_name": r"\b[A-Z][a-zA-Z0-9]*\b",
            "variable_name": r"\b[a-z_][a-z0-9_]*\b"
        }