"""
Unified Context Manager for comprehensive context management across all layers.

This module consolidates session context, conversation context, project context,
and task context into a single, coherent system that can be shared across
the router, agent, and tools.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path

from .types import ConversationContext, ProcessedCommand
from .types import SessionOperation, OperationType
from ..agent.types import RequestContext
from ...utils.logging import get_logger


class ContextScope(Enum):
    """Different scopes of context information."""
    SESSION = "session"          # Current session operations
    CONVERSATION = "conversation"  # Multi-turn conversation
    PROJECT = "project"          # Project-wide knowledge
    TASK = "task"               # Current task/goal
    USER = "user"               # User preferences and patterns
    FILE = "file"               # File-specific context


@dataclass
class FileContext:
    """Rich context information about a file."""
    file_path: str
    language: str
    file_extension: str
    last_modified: float
    size_bytes: int
    
    # Content analysis
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Operations
    recent_operations: List[SessionOperation] = field(default_factory=list)
    modification_count: int = 0
    
    # Metadata
    created_by_session: Optional[str] = None
    primary_purpose: Optional[str] = None  # "main", "config", "test", "util"
    complexity_score: float = 0.0


@dataclass
class ProjectContext:
    """Project-wide context information."""
    project_path: str
    project_name: str
    project_type: Optional[str] = None  # "python", "javascript", "go", "mixed"
    
    # File structure
    files: Dict[str, FileContext] = field(default_factory=dict)
    directories: List[str] = field(default_factory=list)
    
    # Dependencies and configuration
    dependencies: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    build_files: List[str] = field(default_factory=list)
    
    # Patterns and conventions
    naming_conventions: Dict[str, str] = field(default_factory=dict)
    coding_patterns: List[str] = field(default_factory=list)
    common_imports: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    last_analyzed: float = 0.0
    analysis_version: str = "1.0"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    user_input: str
    system_response: str
    timestamp: float
    intent_type: str
    entities: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    files_affected: List[str] = field(default_factory=list)
    success: bool = True


@dataclass
class TaskContext:
    """Current task/goal context."""
    task_id: str
    task_description: str
    current_step: int
    total_steps: int
    
    # Progress tracking
    steps_completed: List[str] = field(default_factory=list)
    current_files: List[str] = field(default_factory=list)
    target_outcome: Optional[str] = None
    
    # Context
    related_conversations: List[ConversationTurn] = field(default_factory=list)
    relevant_files: List[str] = field(default_factory=list)
    
    # Metadata
    started_at: float = 0.0
    estimated_completion: Optional[float] = None
    priority: str = "medium"  # "low", "medium", "high"


@dataclass
class UserContext:
    """User preferences and behavioral patterns."""
    user_id: str
    
    # Preferences
    preferred_languages: List[str] = field(default_factory=list)
    coding_style: Dict[str, str] = field(default_factory=dict)
    explanation_level: str = "standard"  # "minimal", "standard", "detailed"
    
    # Patterns
    common_requests: List[str] = field(default_factory=list)
    typical_workflows: List[str] = field(default_factory=list)
    frequent_files: List[str] = field(default_factory=list)
    
    # Learning
    successful_patterns: List[str] = field(default_factory=list)
    problem_areas: List[str] = field(default_factory=list)
    
    # Metadata
    total_interactions: int = 0
    last_active: float = 0.0


class UnifiedContextManager:
    """
    Unified context manager that consolidates all context types and provides
    a single interface for context operations across all system layers.
    """
    
    def __init__(self, max_sessions: int = 100, max_conversation_turns: int = 50):
        self.logger = get_logger(__name__)
        self.max_sessions = max_sessions
        self.max_conversation_turns = max_conversation_turns
        
        # Core context storage
        self.sessions: Dict[str, List[SessionOperation]] = {}
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.projects: Dict[str, ProjectContext] = {}
        self.tasks: Dict[str, TaskContext] = {}
        self.users: Dict[str, UserContext] = {}
        
        # Cross-references
        self.session_to_project: Dict[str, str] = {}
        self.session_to_user: Dict[str, str] = {}
        self.file_to_project: Dict[str, str] = {}
    
    def get_unified_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive context for a session including all relevant
        information from different scopes.
        """
        context = {
            "session_id": session_id,
            "timestamp": time.time(),
            "scopes": {}
        }
        
        # Session context
        context["scopes"]["session"] = self._get_session_context(session_id)
        
        # Conversation context
        context["scopes"]["conversation"] = self._get_conversation_context(session_id)
        
        # Project context
        project_path = self.session_to_project.get(session_id)
        if project_path:
            context["scopes"]["project"] = self._get_project_context(project_path)
        
        # Task context
        context["scopes"]["task"] = self._get_active_task_context(session_id)
        
        # User context
        user_id = self.session_to_user.get(session_id)
        if user_id:
            context["scopes"]["user"] = self._get_user_context(user_id)
        
        return context
    
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session-specific context."""
        operations = self.sessions.get(session_id, [])
        
        # Get recent operations
        recent_ops = [op for op in operations if op.is_recent()]
        
        # Analyze patterns
        operation_types = {}
        files_touched = set()
        tools_used = set()
        
        for op in recent_ops:
            op_type = op.operation_type.value
            operation_types[op_type] = operation_types.get(op_type, 0) + 1
            
            if op.tool_input.get("file_path"):
                files_touched.add(op.tool_input["file_path"])
            
            # Extract tool name from operation
            if hasattr(op, 'tool_name'):
                tools_used.add(op.tool_name)
        
        return {
            "total_operations": len(operations),
            "recent_operations": recent_ops,  # Return actual operation objects, not count
            "recent_operations_count": len(recent_ops),  # Keep count as separate field for backward compatibility
            "operation_types": operation_types,
            "files_touched": list(files_touched),
            "tools_used": list(tools_used),
            "last_operation": recent_ops[-1] if recent_ops else None,
            "last_file_context": self._get_last_file_context(session_id)
        }
    
    def _get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation-specific context."""
        turns = self.conversations.get(session_id, [])
        recent_turns = turns[-5:]  # Last 5 turns
        
        if not recent_turns:
            return {"turns": [], "entities": {}, "topics": []}
        
        # Extract entities and topics from recent turns
        all_entities = {}
        topics = set()
        
        for turn in recent_turns:
            for key, value in turn.entities.items():
                if key not in all_entities:
                    all_entities[key] = []
                all_entities[key].extend(value if isinstance(value, list) else [value])
            
            # Extract topics from intent types
            topics.add(turn.intent_type)
        
        return {
            "turns": len(turns),
            "recent_turns": recent_turns,
            "entities": all_entities,
            "topics": list(topics),
            "last_turn": recent_turns[-1] if recent_turns else None
        }
    
    def _get_project_context(self, project_path: str) -> Optional[ProjectContext]:
        """Get project-specific context."""
        return self.projects.get(project_path)
    
    def _get_active_task_context(self, session_id: str) -> Optional[TaskContext]:
        """Get active task context for a session."""
        # Find active task for this session
        for task_id, task in self.tasks.items():
            if session_id in task_id:  # Simple association for now
                return task
        return None
    
    def _get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Get user-specific context."""
        return self.users.get(user_id)
    
    def _get_last_file_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent file operation context."""
        operations = self.sessions.get(session_id, [])
        
        # Look for the most recent file operation
        for operation in reversed(operations):
            if operation.is_recent() and operation.operation_type in [
                OperationType.FILE_CREATION, 
                OperationType.FILE_READING
            ]:
                file_path = operation.tool_input.get("file_path")
                if file_path:
                    # Get enhanced file context
                    file_context = self._get_file_context(file_path)
                    
                    return {
                        "file_path": file_path,
                        "file_extension": file_context.file_extension if file_context else "",
                        "language": file_context.language if file_context else "",
                        "operation_type": operation.operation_type.value,
                        "timestamp": operation.timestamp,
                        "file_context": file_context
                    }
        
        return None
    
    def _get_file_context(self, file_path: str) -> Optional[FileContext]:
        """Get enhanced context for a specific file."""
        # Check if file exists in any project
        for project_path, project in self.projects.items():
            if file_path in project.files:
                return project.files[file_path]
        
        # Create basic file context if not found
        try:
            path = Path(file_path)
            if path.exists():
                return FileContext(
                    file_path=file_path,
                    language=self._get_language_from_extension(path.suffix[1:]),
                    file_extension=path.suffix[1:],
                    last_modified=path.stat().st_mtime,
                    size_bytes=path.stat().st_size
                )
        except Exception as e:
            self.logger.warning(f"Failed to create file context for {file_path}: {e}")
        
        return None
    
    def _get_language_from_extension(self, extension: str) -> str:
        """Map file extension to programming language."""
        ext_to_lang = {
            "py": "python",
            "js": "javascript", 
            "ts": "typescript",
            "java": "java",
            "go": "go",
            "rs": "rust",
            "c": "c",
            "cpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "rb": "ruby",
            "php": "php",
            "sh": "bash",
            "bash": "bash",
            "html": "html",
            "css": "css",
            "json": "json",
            "xml": "xml",
            "yaml": "yaml",
            "yml": "yaml",
            "md": "markdown",
            "txt": "text",
        }
        
        return ext_to_lang.get(extension.lower(), "text")
    
    def record_operation(
        self, 
        session_id: str, 
        operation_type: OperationType,
        tool_name: str, 
        tool_input: Dict[str, Any],
        result: ProcessedCommand,
        user_input: str,
        project_path: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """Record a session operation with enhanced context."""
        
        # Create operation
        operation = SessionOperation(
            operation_type=operation_type,
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
            timestamp=time.time(),
            user_input=user_input
        )
        
        # Store in session
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append(operation)
        
        # Maintain session limits
        if len(self.sessions[session_id]) > 50:  # Keep last 50 operations
            self.sessions[session_id] = self.sessions[session_id][-50:]
        
        # Associate with project and user
        if project_path:
            self.session_to_project[session_id] = project_path
        if user_id:
            self.session_to_user[session_id] = user_id
        
        # Update file context if applicable
        if tool_input.get("file_path"):
            self._update_file_context(tool_input["file_path"], operation)
        
        self.logger.debug(f"Recorded operation: {operation_type.value} for session {session_id}")
    
    def record_conversation_turn(
        self,
        session_id: str,
        user_input: str,
        system_response: str,
        intent_type: str,
        entities: Dict[str, Any],
        tools_used: List[str],
        files_affected: List[str],
        success: bool = True
    ) -> None:
        """Record a conversation turn."""
        
        turn = ConversationTurn(
            user_input=user_input,
            system_response=system_response,
            timestamp=time.time(),
            intent_type=intent_type,
            entities=entities,
            tools_used=tools_used,
            files_affected=files_affected,
            success=success
        )
        
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append(turn)
        
        # Maintain conversation limits
        if len(self.conversations[session_id]) > self.max_conversation_turns:
            self.conversations[session_id] = self.conversations[session_id][-self.max_conversation_turns:]
        
        self.logger.debug(f"Recorded conversation turn for session {session_id}")
    
    def _update_file_context(self, file_path: str, operation: SessionOperation) -> None:
        """Update file context based on operation."""
        # Find the project for this file
        project_path = self.file_to_project.get(file_path)
        if not project_path:
            # Try to infer project from file path
            path = Path(file_path)
            project_path = str(path.parent)
        
        # Get or create project context
        if project_path not in self.projects:
            self.projects[project_path] = ProjectContext(
                project_path=project_path,
                project_name=Path(project_path).name,
                last_analyzed=time.time()
            )
        
        project = self.projects[project_path]
        
        # Get or create file context
        if file_path not in project.files:
            project.files[file_path] = self._get_file_context(file_path) or FileContext(
                file_path=file_path,
                language=self._get_language_from_extension(Path(file_path).suffix[1:]),
                file_extension=Path(file_path).suffix[1:],
                last_modified=time.time(),
                size_bytes=0
            )
        
        file_context = project.files[file_path]
        
        # Update file context
        file_context.recent_operations.append(operation)
        file_context.modification_count += 1
        
        # Keep only recent operations
        if len(file_context.recent_operations) > 10:
            file_context.recent_operations = file_context.recent_operations[-10:]
        
        # Update project-file mapping
        self.file_to_project[file_path] = project_path
    
    def get_contextual_routing_hints(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        Provide context-aware hints for routing decisions.
        """
        context = self.get_unified_context(session_id)
        hints = {
            "confidence_boost": 0.0,
            "suggested_tool": None,
            "context_factors": [],
            "file_context": None
        }
        
        # Check for file context
        last_file = context["scopes"]["session"].get("last_file_context")
        if last_file:
            hints["file_context"] = last_file
            
            # Boost confidence for contextual requests
            if any(phrase in user_input.lower() for phrase in ["in that file", "in the file", "to that file"]):
                hints["confidence_boost"] = 0.15
                hints["suggested_tool"] = "file_writer"
                hints["context_factors"].append("contextual_file_reference")
        
        # Check conversation patterns
        conversation = context["scopes"]["conversation"]
        if conversation.get("topics"):
            recent_topics = conversation["topics"]
            if "code_generation" in recent_topics:
                hints["context_factors"].append("recent_code_generation")
        
        return hints
    
    def cleanup_expired_context(self, max_age_seconds: int = 3600) -> int:
        """Clean up expired context across all scopes."""
        current_time = time.time()
        removed_count = 0
        
        # Clean up sessions
        for session_id in list(self.sessions.keys()):
            original_count = len(self.sessions[session_id])
            
            # Keep only recent operations
            self.sessions[session_id] = [
                op for op in self.sessions[session_id]
                if current_time - op.timestamp < max_age_seconds
            ]
            
            # Remove empty sessions
            if not self.sessions[session_id]:
                del self.sessions[session_id]
                # Clean up associations
                self.session_to_project.pop(session_id, None)
                self.session_to_user.pop(session_id, None)
            
            removed_count += original_count - len(self.sessions.get(session_id, []))
        
        # Clean up conversations
        for session_id in list(self.conversations.keys()):
            original_count = len(self.conversations[session_id])
            
            # Keep only recent turns
            self.conversations[session_id] = [
                turn for turn in self.conversations[session_id]
                if current_time - turn.timestamp < max_age_seconds
            ]
            
            # Remove empty conversations
            if not self.conversations[session_id]:
                del self.conversations[session_id]
            
            removed_count += original_count - len(self.conversations.get(session_id, []))
        
        if removed_count > 0:
            self.logger.debug(f"Cleaned up {removed_count} expired context entries")
        
        return removed_count
    
    # Methods migrated from SessionContextManager for consolidation
    
    def get_last_operation(self, session_id: str, operation_type: Optional[OperationType] = None) -> Optional[SessionOperation]:
        """Get the most recent operation of a specific type or any type."""
        
        if session_id not in self.sessions:
            return None
        
        operations = self.sessions[session_id]
        
        # Filter by operation type if specified
        if operation_type:
            operations = [op for op in operations if op.operation_type == operation_type]
        
        # Get the most recent operation that's still valid
        for operation in reversed(operations):
            if operation.is_recent():
                return operation
        
        return None
    
    def get_recent_operations(self, session_id: str, max_age_seconds: int = 300) -> List[SessionOperation]:
        """Get all recent operations for a session."""
        
        if session_id not in self.sessions:
            return []
        
        return [op for op in self.sessions[session_id] if op.is_recent(max_age_seconds)]
    
    def get_operation_summary(self, operation: SessionOperation) -> str:
        """Get a human-readable summary of an operation."""
        
        if operation.operation_type == OperationType.FILE_CREATION:
            file_path = operation.tool_input.get("file_path", "unknown file")
            return f"created file '{file_path}'"
        
        elif operation.operation_type == OperationType.FILE_READING:
            file_path = operation.tool_input.get("file_path", "unknown file")
            return f"read file '{file_path}'"
        
        elif operation.operation_type == OperationType.CODE_GENERATION:
            return "generated code"
        
        elif operation.operation_type == OperationType.ANALYSIS:
            return "analyzed code"
        
        elif operation.operation_type == OperationType.SEARCH:
            query = operation.tool_input.get("query", "")
            return f"searched for '{query}'"
        
        else:
            return f"performed {operation.operation_type.value}"
    
    def get_last_file_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent file operation context for contextual requests."""
        
        if session_id not in self.sessions:
            return None
        
        operations = self.sessions[session_id]
        
        # Look for the most recent file operation
        for operation in reversed(operations):
            if operation.is_recent() and operation.operation_type in [
                OperationType.FILE_CREATION, 
                OperationType.FILE_READING
            ]:
                file_path = operation.tool_input.get("file_path")
                if file_path:
                    # Extract file extension and language
                    from pathlib import Path
                    path = Path(file_path)
                    file_extension = path.suffix.lower()
                    
                    # Simple language detection based on extension
                    language_map = {
                        '.py': 'python',
                        '.js': 'javascript', 
                        '.ts': 'typescript',
                        '.java': 'java',
                        '.cpp': 'cpp',
                        '.c': 'c',
                        '.rs': 'rust',
                        '.go': 'go',
                        '.rb': 'ruby',
                        '.php': 'php',
                        '.json': 'json',
                        '.yaml': 'yaml',
                        '.yml': 'yaml',
                        '.md': 'markdown',
                        '.txt': 'text'
                    }
                    
                    language = language_map.get(file_extension, 'unknown')
                    
                    return {
                        "file_path": file_path,
                        "file_extension": file_extension,
                        "language": language,
                        "operation_type": operation.operation_type.value,
                        "timestamp": operation.timestamp,
                        "tool_input": operation.tool_input
                    }
        
        return None
    
    def clear_session(self, session_id: str) -> None:
        """Clear all operations for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.session_to_project:
            del self.session_to_project[session_id]
        if session_id in self.session_to_user:
            del self.session_to_user[session_id]
        self.logger.debug(f"Cleared unified context for session {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the context system."""
        return {
            "sessions": len(self.sessions),
            "conversations": len(self.conversations),
            "projects": len(self.projects),
            "tasks": len(self.tasks),
            "users": len(self.users),
            "total_operations": sum(len(ops) for ops in self.sessions.values()),
            "total_conversation_turns": sum(len(turns) for turns in self.conversations.values()),
            "total_files_tracked": sum(len(project.files) for project in self.projects.values()),
            "active_sessions": len([
                session_id for session_id, ops in self.sessions.items()
                if ops and ops[-1].is_recent()
            ])
        }