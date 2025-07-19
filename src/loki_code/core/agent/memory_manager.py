"""
LangChain memory management for Loki Code.

This module provides intelligent conversation memory using LangChain memory components,
replacing basic conversation history with sophisticated context retention and management.
"""

import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

try:
    # Try newer LangChain imports first
    from langchain_community.memory import (
        ConversationBufferMemory,
        ConversationSummaryMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryBufferMemory
    )
except ImportError:
    # Fallback to older imports
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationSummaryMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryBufferMemory
    )
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .conversation_manager import ConversationManager, UserPreferences, ConversationConfig
from ...utils.logging import get_logger


class MemoryStrategy(Enum):
    """Memory management strategies."""
    BUFFER = "buffer"                           # Keep all messages in memory
    WINDOW = "window"                          # Keep last N messages
    SUMMARY = "summary"                        # Summarize old conversations
    SUMMARY_BUFFER = "summary_buffer"          # Hybrid approach


class LangChainMemoryManager:
    """
    Advanced memory management using LangChain memory components.
    
    Provides intelligent conversation context retention with multiple strategies
    for different use cases and resource constraints.
    """
    
    def __init__(self, 
                 llm: Optional[BaseLanguageModel] = None,
                 strategy: MemoryStrategy = MemoryStrategy.SUMMARY_BUFFER,
                 config: Optional[ConversationConfig] = None):
        """
        Initialize the memory manager.
        
        Args:
            llm: Language model for summarization (required for summary strategies)
            strategy: Memory management strategy
            config: Conversation configuration
        """
        self.llm = llm
        self.strategy = strategy
        self.config = config or ConversationConfig()
        self.logger = get_logger(__name__)
        
        # Initialize LangChain memory based on strategy
        self.memory = self._create_memory()
        
        # Session management
        self.session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        
        self.logger.info(f"LangChainMemoryManager initialized with {strategy.value} strategy")
    
    def _create_memory(self) -> Union[ConversationBufferMemory, ConversationSummaryMemory, 
                                   ConversationBufferWindowMemory, ConversationSummaryBufferMemory]:
        """Create appropriate LangChain memory based on strategy."""
        
        memory_kwargs = {
            "memory_key": "chat_history",
            "return_messages": True,
            "human_prefix": "Human",
            "ai_prefix": "Assistant"
        }
        
        if self.strategy == MemoryStrategy.BUFFER:
            return ConversationBufferMemory(**memory_kwargs)
            
        elif self.strategy == MemoryStrategy.WINDOW:
            return ConversationBufferWindowMemory(
                k=getattr(self.config, 'context_window_size', 10),
                **memory_kwargs
            )
            
        elif self.strategy == MemoryStrategy.SUMMARY:
            if not self.llm:
                self.logger.warning("No LLM provided for summary strategy, falling back to buffer")
                return ConversationBufferMemory(**memory_kwargs)
            
            # Try to create summary memory, fall back to buffer if tokenizer issues
            try:
                return ConversationSummaryMemory(
                    llm=self.llm,
                    **memory_kwargs
                )
            except Exception as e:
                self.logger.warning(f"Failed to create summary memory (likely tokenizer issue): {e}")
                self.logger.info("Falling back to buffer memory strategy")
                return ConversationBufferMemory(**memory_kwargs)
            
        elif self.strategy == MemoryStrategy.SUMMARY_BUFFER:
            if not self.llm:
                self.logger.warning("No LLM provided for summary_buffer strategy, falling back to window")
                return ConversationBufferWindowMemory(
                    k=getattr(self.config, 'context_window_size', 10),
                    **memory_kwargs
                )
            
            # Try to create summary buffer memory, fall back to window if tokenizer issues
            try:
                return ConversationSummaryBufferMemory(
                    llm=self.llm,
                    max_token_limit=getattr(self.config, 'max_context_tokens', 2000),
                    **memory_kwargs
                )
            except Exception as e:
                self.logger.warning(f"Failed to create summary buffer memory (likely tokenizer issue): {e}")
                self.logger.info("Falling back to window memory strategy")
                return ConversationBufferWindowMemory(
                    k=getattr(self.config, 'context_window_size', 10),
                    **memory_kwargs
                )
        
        else:
            self.logger.error(f"Unknown memory strategy: {self.strategy}")
            return ConversationBufferMemory(**memory_kwargs)
    
    def start_session(self, session_id: str) -> None:
        """Start a new conversation session."""
        self.session_id = session_id
        self.session_start_time = datetime.now()
        
        # Load existing session if available
        self._load_session_memory(session_id)
        
        self.logger.info(f"Started memory session: {session_id}")
    
    def add_message(self, human_message: str, ai_response: str) -> None:
        """Add a conversation exchange to memory."""
        try:
            # Add to LangChain memory
            self.memory.save_context(
                {"input": human_message},
                {"output": ai_response}
            )
            
            # Persist if session is active
            if self.session_id:
                self._save_session_memory()
                
        except Exception as e:
            self.logger.error(f"Error adding message to memory: {e}", exc_info=True)
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for prompts."""
        try:
            # Get memory variables (contains chat_history)
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            if not chat_history:
                return ""
            
            # Format messages for context
            formatted_messages = []
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_messages.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    formatted_messages.append(f"Assistant: {message.content}")
            
            return "\n".join(formatted_messages)
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}", exc_info=True)
            return ""
    
    def get_memory_summary(self) -> Optional[str]:
        """Get conversation summary if available."""
        try:
            if hasattr(self.memory, 'moving_summary_buffer') and self.memory.moving_summary_buffer:
                return self.memory.moving_summary_buffer
            elif hasattr(self.memory, 'buffer') and self.memory.buffer:
                return self.memory.buffer
            return None
        except Exception as e:
            self.logger.error(f"Error getting memory summary: {e}", exc_info=True)
            return None
    
    def clear_memory(self) -> None:
        """Clear all conversation memory."""
        try:
            self.memory.clear()
            
            # Clear session file if exists
            if self.session_id:
                session_file = self._get_session_file_path(self.session_id)
                if session_file.exists():
                    session_file.unlink()
            
            self.logger.info("Memory cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing memory: {e}", exc_info=True)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            stats = {
                "strategy": self.strategy.value,
                "message_count": len(chat_history),
                "session_id": self.session_id,
                "session_duration": None,
                "has_summary": self.get_memory_summary() is not None,
                "memory_type": type(self.memory).__name__
            }
            
            if self.session_start_time:
                duration = datetime.now() - self.session_start_time
                stats["session_duration"] = duration.total_seconds()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """Get file path for session persistence."""
        # Create sessions directory if it doesn't exist
        sessions_dir = Path.home() / ".loki_code" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        return sessions_dir / f"{session_id}_memory.json"
    
    def _save_session_memory(self) -> None:
        """Save current memory state to file."""
        if not self.session_id:
            return
            
        try:
            session_file = self._get_session_file_path(self.session_id)
            
            # Extract memory data for serialization
            memory_data = {
                "strategy": self.strategy.value,
                "session_id": self.session_id,
                "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
                "chat_history": [],
                "summary": self.get_memory_summary()
            }
            
            # Get chat history in serializable format
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    memory_data["chat_history"].append({
                        "type": "human",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    memory_data["chat_history"].append({
                        "type": "ai", 
                        "content": message.content
                    })
            
            # Save to file
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving session memory: {e}", exc_info=True)
    
    def _load_session_memory(self, session_id: str) -> None:
        """Load memory state from file."""
        try:
            session_file = self._get_session_file_path(session_id)
            
            if not session_file.exists():
                self.logger.debug(f"No existing session file for {session_id}")
                return
            
            with open(session_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Restore session metadata
            if memory_data.get("session_start_time"):
                self.session_start_time = datetime.fromisoformat(memory_data["session_start_time"])
            
            # Restore chat history
            chat_history = memory_data.get("chat_history", [])
            for message_data in chat_history:
                if message_data["type"] == "human":
                    # Add as human message by creating a mock interaction
                    # We'll add the next AI message too to maintain pairs
                    continue
                elif message_data["type"] == "ai":
                    # Find the corresponding human message
                    human_idx = chat_history.index(message_data) - 1
                    if human_idx >= 0 and chat_history[human_idx]["type"] == "human":
                        human_content = chat_history[human_idx]["content"]
                        ai_content = message_data["content"]
                        
                        # Add the exchange to memory
                        self.memory.save_context(
                            {"input": human_content},
                            {"output": ai_content}
                        )
            
            self.logger.info(f"Loaded session memory for {session_id} with {len(chat_history)} messages")
            
        except Exception as e:
            self.logger.error(f"Error loading session memory: {e}", exc_info=True)