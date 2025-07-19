"""
Unified Memory Manager - Simplified memory management integrated with unified context.

This manager eliminates the complexity of multiple memory systems by providing
a single, efficient memory interface that works seamlessly with the unified
context system and supports all memory patterns found in the codebase.
"""

import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .unified_context_manager import UnifiedContext, get_global_context_manager
from ...utils.logging import get_logger


class MemoryType(Enum):
    """Types of memory storage."""
    CONVERSATION = "conversation"
    OPERATION = "operation" 
    TOOL_USAGE = "tool_usage"
    FILE_CONTEXT = "file_context"
    KNOWLEDGE = "knowledge"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    memory_type: MemoryType
    content: Any
    timestamp: float
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 1.0  # 0-1 scale for memory retention
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_type": self.memory_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "importance_score": self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        data["memory_type"] = MemoryType(data["memory_type"])
        return cls(**data)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_entries: int = 0
    conversation_entries: int = 0
    operation_entries: int = 0
    tool_usage_entries: int = 0
    file_context_entries: int = 0
    knowledge_entries: int = 0
    memory_size_kb: float = 0.0
    oldest_entry_age_seconds: float = 0.0
    average_importance: float = 0.0


class UnifiedMemoryManager:
    """
    Unified memory manager that simplifies all memory patterns.
    
    Features:
    - Automatic integration with unified context
    - Intelligent memory retention based on importance
    - Multiple memory types (conversation, operations, tools, etc.)
    - Automatic cleanup and optimization
    - Simple persistence interface
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified memory manager."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.context_manager = get_global_context_manager()
        
        # Memory configuration
        self.max_entries_per_type = self.config.get("max_entries_per_type", 1000)
        self.max_age_seconds = self.config.get("max_age_seconds", 86400)  # 24 hours
        self.importance_threshold = self.config.get("importance_threshold", 0.1)
        
        # Memory storage
        self._memories: Dict[str, List[MemoryEntry]] = {
            session_id: [] for session_id in []  # Will be populated as needed
        }
        
        # Auto-cleanup timer
        self._last_cleanup = time.time()
        self._cleanup_interval = self.config.get("cleanup_interval", 3600)  # 1 hour
        
        self.logger.info("UnifiedMemoryManager initialized")
    
    def add_memory(
        self,
        session_id: str,
        memory_type: MemoryType,
        content: Any,
        importance_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a memory entry.
        
        Args:
            session_id: Session identifier
            memory_type: Type of memory
            content: Memory content
            importance_score: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            True if memory was added successfully
        """
        try:
            # Ensure session exists in memory storage
            if session_id not in self._memories:
                self._memories[session_id] = []
            
            # Create memory entry
            entry = MemoryEntry(
                memory_type=memory_type,
                content=content,
                timestamp=time.time(),
                session_id=session_id,
                metadata=metadata or {},
                importance_score=max(0.0, min(1.0, importance_score))
            )
            
            # Add to memory storage
            self._memories[session_id].append(entry)
            
            # Update unified context if it exists
            context = self.context_manager.get_context(session_id)
            if context:
                self._sync_memory_to_context(entry, context)
            
            # Trigger cleanup if needed
            self._maybe_cleanup()
            
            self.logger.debug(f"Added {memory_type.value} memory for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            return False
    
    def add_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        importance_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a conversation memory."""
        conversation_content = {
            "role": role,
            "content": content
        }
        
        return self.add_memory(
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content=conversation_content,
            importance_score=importance_score,
            metadata=metadata
        )
    
    def add_operation(
        self,
        session_id: str,
        operation_type: str,
        details: Dict[str, Any],
        success: bool = True,
        importance_score: float = None
    ) -> bool:
        """Add an operation memory."""
        # Auto-calculate importance based on success and operation type
        if importance_score is None:
            base_score = 0.8 if success else 0.6
            # Important operations get higher scores
            if operation_type in ["file_write", "code_generation", "error_fix"]:
                base_score += 0.2
            importance_score = min(1.0, base_score)
        
        operation_content = {
            "operation_type": operation_type,
            "success": success,
            "details": details
        }
        
        return self.add_memory(
            session_id=session_id,
            memory_type=MemoryType.OPERATION,
            content=operation_content,
            importance_score=importance_score
        )
    
    def add_tool_usage(
        self,
        session_id: str,
        tool_name: str,
        input_data: Any,
        result: Any,
        success: bool,
        execution_time_ms: float = 0.0
    ) -> bool:
        """Add tool usage memory."""
        # Calculate importance based on success and tool type
        importance_score = 0.7 if success else 0.3
        if tool_name in ["file_writer", "code_generator"]:
            importance_score += 0.2
        
        tool_content = {
            "tool_name": tool_name,
            "input_data": str(input_data)[:500],  # Truncate for storage
            "result": str(result)[:500],
            "success": success,
            "execution_time_ms": execution_time_ms
        }
        
        return self.add_memory(
            session_id=session_id,
            memory_type=MemoryType.TOOL_USAGE,
            content=tool_content,
            importance_score=min(1.0, importance_score)
        )
    
    def get_memories(
        self,
        session_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Get memories for a session.
        
        Args:
            session_id: Session identifier
            memory_type: Filter by memory type (optional)
            limit: Maximum number of memories to return
            min_importance: Minimum importance score
            
        Returns:
            List of memory entries
        """
        if session_id not in self._memories:
            return []
        
        memories = self._memories[session_id]
        
        # Filter by type if specified
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Filter by importance
        memories = [m for m in memories if m.importance_score >= min_importance]
        
        # Sort by timestamp (newest first)
        memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            memories = memories[:limit]
        
        return memories
    
    def get_conversation_context(
        self,
        session_id: str,
        max_entries: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent conversation context for LangChain agents."""
        conversations = self.get_memories(
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            limit=max_entries
        )
        
        # Convert to LangChain-compatible format
        context = []
        for memory in reversed(conversations):  # Reverse to get chronological order
            content = memory.content
            context.append({
                "role": content["role"],
                "content": content["content"],
                "timestamp": memory.timestamp
            })
        
        return context
    
    def get_operation_summary(self, session_id: str, max_entries: int = 20) -> str:
        """Get a summary of recent operations."""
        operations = self.get_memories(
            session_id=session_id,
            memory_type=MemoryType.OPERATION,
            limit=max_entries,
            min_importance=0.5
        )
        
        if not operations:
            return "No recent operations"
        
        summary_parts = []
        success_count = 0
        
        for op in operations:
            content = op.content
            if content["success"]:
                success_count += 1
            
            op_type = content["operation_type"]
            status = "✓" if content["success"] else "✗"
            summary_parts.append(f"{status} {op_type}")
        
        summary = f"Recent operations ({success_count}/{len(operations)} successful):\n"
        summary += "\n".join(summary_parts[:10])  # Show first 10
        
        return summary
    
    def get_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive memory summary for a session."""
        if session_id not in self._memories:
            return {"error": "Session not found"}
        
        memories = self._memories[session_id]
        
        # Calculate stats
        stats = MemoryStats()
        stats.total_entries = len(memories)
        
        current_time = time.time()
        importance_scores = []
        
        for memory in memories:
            # Count by type
            if memory.memory_type == MemoryType.CONVERSATION:
                stats.conversation_entries += 1
            elif memory.memory_type == MemoryType.OPERATION:
                stats.operation_entries += 1
            elif memory.memory_type == MemoryType.TOOL_USAGE:
                stats.tool_usage_entries += 1
            elif memory.memory_type == MemoryType.FILE_CONTEXT:
                stats.file_context_entries += 1
            elif memory.memory_type == MemoryType.KNOWLEDGE:
                stats.knowledge_entries += 1
            
            # Track importance
            importance_scores.append(memory.importance_score)
            
            # Calculate age
            age = current_time - memory.timestamp
            if stats.oldest_entry_age_seconds < age:
                stats.oldest_entry_age_seconds = age
        
        # Calculate averages
        if importance_scores:
            stats.average_importance = sum(importance_scores) / len(importance_scores)
        
        # Estimate memory size
        memory_json = json.dumps([m.to_dict() for m in memories])
        stats.memory_size_kb = len(memory_json.encode('utf-8')) / 1024
        
        return {
            "session_id": session_id,
            "stats": stats,
            "recent_conversation": self.get_conversation_context(session_id, 5),
            "recent_operations": self.get_operation_summary(session_id, 10)
        }
    
    def cleanup_session(self, session_id: str) -> int:
        """Clean up old/low-importance memories for a session."""
        if session_id not in self._memories:
            return 0
        
        memories = self._memories[session_id]
        current_time = time.time()
        
        # Filter memories to keep
        kept_memories = []
        removed_count = 0
        
        for memory in memories:
            # Remove if too old
            age = current_time - memory.timestamp
            if age > self.max_age_seconds:
                removed_count += 1
                continue
            
            # Remove if importance too low
            if memory.importance_score < self.importance_threshold:
                removed_count += 1
                continue
            
            kept_memories.append(memory)
        
        # Limit by count per type
        type_counts = {}
        final_memories = []
        
        # Sort by importance and timestamp
        kept_memories.sort(key=lambda m: (m.importance_score, m.timestamp), reverse=True)
        
        for memory in kept_memories:
            mem_type = memory.memory_type
            current_count = type_counts.get(mem_type, 0)
            
            if current_count < self.max_entries_per_type:
                final_memories.append(memory)
                type_counts[mem_type] = current_count + 1
            else:
                removed_count += 1
        
        self._memories[session_id] = final_memories
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} memories for session {session_id}")
        
        return removed_count
    
    def cleanup_all_sessions(self) -> int:
        """Clean up all sessions."""
        total_removed = 0
        for session_id in list(self._memories.keys()):
            total_removed += self.cleanup_session(session_id)
        
        # Remove empty sessions
        empty_sessions = [sid for sid, memories in self._memories.items() if not memories]
        for session_id in empty_sessions:
            del self._memories[session_id]
        
        self._last_cleanup = time.time()
        return total_removed
    
    def delete_session_memories(self, session_id: str) -> bool:
        """Delete all memories for a session."""
        if session_id in self._memories:
            count = len(self._memories[session_id])
            del self._memories[session_id]
            self.logger.info(f"Deleted {count} memories for session {session_id}")
            return True
        return False
    
    def _sync_memory_to_context(self, memory: MemoryEntry, context: UnifiedContext):
        """Sync memory entry to unified context."""
        if memory.memory_type == MemoryType.CONVERSATION:
            content = memory.content
            context.add_conversation_entry(
                role=content["role"],
                content=content["content"],
                metadata=memory.metadata
            )
        
        elif memory.memory_type == MemoryType.OPERATION:
            content = memory.content
            context.add_operation(
                operation_type=content["operation_type"],
                details=content["details"]
            )
        
        elif memory.memory_type == MemoryType.TOOL_USAGE:
            content = memory.content
            context.add_tool_usage(
                tool_name=content["tool_name"],
                input_data=content["input_data"],
                result=content["result"],
                success=content["success"]
            )
    
    def _maybe_cleanup(self):
        """Trigger cleanup if enough time has passed."""
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self.cleanup_all_sessions()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        total_sessions = len(self._memories)
        total_memories = sum(len(memories) for memories in self._memories.values())
        
        # Memory size estimation
        total_size_kb = 0
        for memories in self._memories.values():
            memory_json = json.dumps([m.to_dict() for m in memories])
            total_size_kb += len(memory_json.encode('utf-8')) / 1024
        
        return {
            "active_sessions": total_sessions,
            "total_memories": total_memories,
            "memory_size_kb": total_size_kb,
            "last_cleanup": self._last_cleanup,
            "config": {
                "max_entries_per_type": self.max_entries_per_type,
                "max_age_seconds": self.max_age_seconds,
                "importance_threshold": self.importance_threshold
            }
        }


# Global memory manager instance
_global_memory_manager: Optional[UnifiedMemoryManager] = None


def get_global_memory_manager() -> UnifiedMemoryManager:
    """Get the global unified memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = UnifiedMemoryManager()
    return _global_memory_manager


# Convenience functions
def add_conversation_memory(session_id: str, role: str, content: str, **kwargs) -> bool:
    """Add conversation memory."""
    manager = get_global_memory_manager()
    return manager.add_conversation(session_id, role, content, **kwargs)


def add_operation_memory(session_id: str, operation_type: str, details: Dict[str, Any], **kwargs) -> bool:
    """Add operation memory."""
    manager = get_global_memory_manager()
    return manager.add_operation(session_id, operation_type, details, **kwargs)


def get_conversation_context(session_id: str, max_entries: int = 10) -> List[Dict[str, Any]]:
    """Get conversation context."""
    manager = get_global_memory_manager()
    return manager.get_conversation_context(session_id, max_entries)