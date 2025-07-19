"""
Unified Context Manager - Consolidates all context management patterns.

This manager eliminates the fragmented context handling found across the
codebase by providing a single, comprehensive context system that works
with all components (agents, tools, commands, sessions).
"""

import time
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..agent.types import RequestContext as AgentRequestContext
from ..commands.types import ConversationContext as CommandConversationContext
from ...tools.types import ToolContext
from ...utils.logging import get_logger


@dataclass
class UnifiedContext:
    """
    Unified context that works across all system components.
    
    This context can be converted to any specific context type needed
    by different components while maintaining consistency.
    """
    # Core identification
    session_id: str
    user_id: str = "default"
    request_id: Optional[str] = None
    
    # Project and file context
    project_path: str = "."
    working_directory: str = "."
    current_file: Optional[str] = None
    recent_files: List[str] = field(default_factory=list)
    
    # Conversation context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    recent_operations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tool and execution context
    available_tools: List[str] = field(default_factory=list)
    tool_usage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Agent context
    agent_state: Dict[str, Any] = field(default_factory=dict)
    memory_summary: Optional[str] = None
    
    # Security and safety context
    permissions: Dict[str, bool] = field(default_factory=dict)
    safety_settings: Dict[str, Any] = field(default_factory=dict)
    
    # User preferences and settings
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime metadata
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def update_timestamp(self):
        """Update the last_updated timestamp."""
        self.last_updated = time.time()
    
    def add_conversation_entry(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add an entry to conversation history."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(entry)
        self.update_timestamp()
    
    def add_operation(self, operation_type: str, details: Dict[str, Any]):
        """Add an operation to recent operations."""
        operation = {
            "type": operation_type,
            "timestamp": time.time(),
            "details": details
        }
        self.recent_operations.append(operation)
        
        # Keep only recent operations (last 50)
        if len(self.recent_operations) > 50:
            self.recent_operations = self.recent_operations[-50:]
        
        self.update_timestamp()
    
    def add_tool_usage(self, tool_name: str, input_data: Any, result: Any, success: bool):
        """Add tool usage to history."""
        usage = {
            "tool_name": tool_name,
            "timestamp": time.time(),
            "input_data": str(input_data)[:200] + "..." if len(str(input_data)) > 200 else str(input_data),
            "success": success,
            "result_preview": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
        }
        self.tool_usage_history.append(usage)
        
        # Keep only recent usage (last 20)
        if len(self.tool_usage_history) > 20:
            self.tool_usage_history = self.tool_usage_history[-20:]
        
        self.update_timestamp()
    
    def set_current_file(self, file_path: str):
        """Set current file and update recent files."""
        self.current_file = file_path
        
        # Add to recent files if not already there
        if file_path not in self.recent_files:
            self.recent_files.insert(0, file_path)
            # Keep only 10 recent files
            self.recent_files = self.recent_files[:10]
        elif file_path in self.recent_files:
            # Move to front
            self.recent_files.remove(file_path)
            self.recent_files.insert(0, file_path)
        
        self.update_timestamp()
    
    def get_conversation_summary(self, max_entries: int = 10) -> str:
        """Get a summary of recent conversation."""
        recent_history = self.conversation_history[-max_entries:]
        
        if not recent_history:
            return "No conversation history"
        
        summary_parts = []
        for entry in recent_history:
            role = entry["role"].title()
            content = entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry["content"]
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def to_agent_context(self) -> AgentRequestContext:
        """Convert to agent-specific context."""
        return AgentRequestContext(
            project_path=self.project_path,
            current_file=self.current_file,
            session_id=self.session_id,
            conversation_history=self.conversation_history,
            recent_operations=self.recent_operations,
            file_contexts={},  # Could be populated from recent_files
            session_metadata={
                "user_id": self.user_id,
                "preferences": self.preferences,
                "agent_state": self.agent_state,
                "memory_summary": self.memory_summary
            }
        )
    
    def to_command_context(self) -> CommandConversationContext:
        """Convert to command-specific context."""
        return CommandConversationContext(
            session_id=self.session_id,
            project_path=self.project_path,
            current_file=self.current_file,
            conversation_history=self.conversation_history,
            user_preferences=self.preferences
        )
    
    def to_tool_context(self) -> ToolContext:
        """Convert to tool-specific context."""
        return ToolContext(
            project_path=self.project_path,
            user_id=self.user_id,
            session_id=self.session_id,
            environment=self.environment,
            working_directory=self.working_directory,
            safety_settings=self._create_safety_settings()
        )
    
    def _create_safety_settings(self):
        """Create safety settings for tool context."""
        from ...tools.types import SafetySettings
        
        return SafetySettings(
            allowed_paths=[self.project_path, self.working_directory],
            max_file_size_mb=self.safety_settings.get("max_file_size_mb", 10),
            timeout_seconds=self.safety_settings.get("timeout_seconds", 30),
            require_confirmation_for=self.safety_settings.get("require_confirmation_for", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedContext':
        """Create from dictionary."""
        return cls(**data)
    
    def merge_with(self, other: 'UnifiedContext') -> 'UnifiedContext':
        """Merge with another context, keeping the most recent data."""
        # Use the more recent context as base
        if other.last_updated > self.last_updated:
            merged = UnifiedContext(**asdict(other))
            base = self
        else:
            merged = UnifiedContext(**asdict(self))
            base = other
        
        # Merge conversation histories
        merged.conversation_history = self._merge_lists(
            merged.conversation_history,
            base.conversation_history,
            key=lambda x: x.get("timestamp", 0)
        )
        
        # Merge recent operations
        merged.recent_operations = self._merge_lists(
            merged.recent_operations,
            base.recent_operations,
            key=lambda x: x.get("timestamp", 0)
        )
        
        # Merge tool usage history
        merged.tool_usage_history = self._merge_lists(
            merged.tool_usage_history,
            base.tool_usage_history,
            key=lambda x: x.get("timestamp", 0)
        )
        
        # Merge recent files (keep unique, ordered by recency)
        all_files = merged.recent_files + [f for f in base.recent_files if f not in merged.recent_files]
        merged.recent_files = all_files[:10]
        
        # Merge preferences (merged takes precedence)
        base_prefs = base.preferences.copy()
        base_prefs.update(merged.preferences)
        merged.preferences = base_prefs
        
        # Update timestamp
        merged.update_timestamp()
        
        return merged
    
    def _merge_lists(self, list1: List[Dict], list2: List[Dict], key) -> List[Dict]:
        """Merge two lists, removing duplicates and sorting by key."""
        combined = list1 + list2
        # Sort by key (timestamp) and keep unique entries
        seen = set()
        unique_combined = []
        
        for item in sorted(combined, key=key):
            item_id = (item.get("timestamp"), str(item))  # Simple deduplication
            if item_id not in seen:
                seen.add(item_id)
                unique_combined.append(item)
        
        return unique_combined


class UnifiedContextManager:
    """
    Manager for unified contexts across the system.
    
    Provides:
    - Context creation and lifecycle management
    - Context conversion between different component types
    - Session management
    - Context persistence and retrieval
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the context manager."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Active contexts by session_id
        self._contexts: Dict[str, UnifiedContext] = {}
        
        # Context templates for different scenarios
        self._templates: Dict[str, Dict[str, Any]] = {
            "default": {
                "permissions": {"read_files": True, "write_files": True},
                "safety_settings": {"max_file_size_mb": 10, "timeout_seconds": 30},
                "preferences": {"theme": "dark", "verbose": False}
            },
            "restricted": {
                "permissions": {"read_files": True, "write_files": False},
                "safety_settings": {"max_file_size_mb": 5, "timeout_seconds": 15},
                "preferences": {"verbose": True}
            }
        }
        
        self.logger.info("UnifiedContextManager initialized")
    
    def create_context(
        self,
        session_id: str,
        user_id: str = "default",
        project_path: str = ".",
        template: str = "default",
        **kwargs
    ) -> UnifiedContext:
        """
        Create a new unified context.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            project_path: Project root path
            template: Template to use for default values
            **kwargs: Additional context properties
            
        Returns:
            New UnifiedContext instance
        """
        # Get template defaults
        template_data = self._templates.get(template, self._templates["default"]).copy()
        
        # Create base context
        context_data = {
            "session_id": session_id,
            "user_id": user_id,
            "project_path": str(Path(project_path).resolve()),
            "working_directory": str(Path(project_path).resolve()),
            **template_data,
            **kwargs
        }
        
        context = UnifiedContext(**context_data)
        
        # Store in active contexts
        self._contexts[session_id] = context
        
        self.logger.info(f"Created new context for session: {session_id}")
        return context
    
    def get_context(self, session_id: str) -> Optional[UnifiedContext]:
        """Get context by session ID."""
        return self._contexts.get(session_id)
    
    def get_or_create_context(
        self,
        session_id: str,
        **kwargs
    ) -> UnifiedContext:
        """Get existing context or create new one."""
        context = self.get_context(session_id)
        if context is None:
            context = self.create_context(session_id, **kwargs)
        return context
    
    def update_context(self, session_id: str, **updates) -> bool:
        """Update context properties."""
        context = self.get_context(session_id)
        if context is None:
            return False
        
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        context.update_timestamp()
        return True
    
    def merge_contexts(self, session_id1: str, session_id2: str, target_session: str) -> bool:
        """Merge two contexts into a target session."""
        context1 = self.get_context(session_id1)
        context2 = self.get_context(session_id2)
        
        if not context1 or not context2:
            return False
        
        merged = context1.merge_with(context2)
        merged.session_id = target_session
        
        self._contexts[target_session] = merged
        return True
    
    def delete_context(self, session_id: str) -> bool:
        """Delete a context."""
        if session_id in self._contexts:
            del self._contexts[session_id]
            self.logger.info(f"Deleted context for session: {session_id}")
            return True
        return False
    
    def list_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._contexts.keys())
    
    def get_context_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of context state."""
        context = self.get_context(session_id)
        if not context:
            return None
        
        return {
            "session_id": context.session_id,
            "user_id": context.user_id,
            "project_path": context.project_path,
            "current_file": context.current_file,
            "conversation_entries": len(context.conversation_history),
            "recent_operations": len(context.recent_operations),
            "tools_used": len(context.tool_usage_history),
            "recent_files": len(context.recent_files),
            "created_at": context.created_at,
            "last_updated": context.last_updated,
            "age_seconds": time.time() - context.created_at
        }
    
    def cleanup_old_contexts(self, max_age_seconds: int = 3600) -> int:
        """Clean up contexts older than max_age_seconds."""
        current_time = time.time()
        old_sessions = []
        
        for session_id, context in self._contexts.items():
            if current_time - context.last_updated > max_age_seconds:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            self.delete_context(session_id)
        
        if old_sessions:
            self.logger.info(f"Cleaned up {len(old_sessions)} old contexts")
        
        return len(old_sessions)
    
    def add_context_template(self, name: str, template: Dict[str, Any]):
        """Add a new context template."""
        self._templates[name] = template
        self.logger.info(f"Added context template: {name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        total_contexts = len(self._contexts)
        total_conversations = sum(len(ctx.conversation_history) for ctx in self._contexts.values())
        total_operations = sum(len(ctx.recent_operations) for ctx in self._contexts.values())
        
        return {
            "active_contexts": total_contexts,
            "total_conversation_entries": total_conversations,
            "total_operations": total_operations,
            "available_templates": list(self._templates.keys())
        }


# Global context manager instance
_global_context_manager: Optional[UnifiedContextManager] = None


def get_global_context_manager() -> UnifiedContextManager:
    """Get the global unified context manager."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = UnifiedContextManager()
    return _global_context_manager


# Convenience functions
def get_context(session_id: str) -> Optional[UnifiedContext]:
    """Get context by session ID."""
    manager = get_global_context_manager()
    return manager.get_context(session_id)


def create_context(session_id: str, **kwargs) -> UnifiedContext:
    """Create a new context."""
    manager = get_global_context_manager()
    return manager.create_context(session_id, **kwargs)


def get_or_create_context(session_id: str, **kwargs) -> UnifiedContext:
    """Get existing context or create new one."""
    manager = get_global_context_manager()
    return manager.get_or_create_context(session_id, **kwargs)