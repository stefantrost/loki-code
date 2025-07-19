"""
UI Interface Layer with Streaming Support

This module defines the abstract interface for all UI implementations,
with first-class streaming support for real-time user interactions.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator, Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio


class UIResponseType(Enum):
    """Types of UI responses for proper handling by different UI implementations."""
    TEXT = "text"                    # Regular text content
    THINKING = "thinking"            # Thinking/processing indicators
    TOOL_EXECUTION = "tool_execution" # Tool execution updates
    ERROR = "error"                  # Error messages
    SYSTEM = "system"                # System messages
    PROGRESS = "progress"            # Progress indicators
    COMPLETION = "completion"        # Task completion markers


@dataclass
class UIMessage:
    """Container for user input messages."""
    content: str
    message_type: str = "user_input"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


@dataclass
class UIResponse:
    """Container for UI response chunks with streaming support."""
    content: str
    response_type: UIResponseType
    metadata: Optional[Dict[str, Any]] = None
    is_complete: bool = False
    confidence: Optional[float] = None
    tools_used: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UIEvent:
    """Container for system events that UIs need to handle."""
    event_type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class UIInterface(ABC):
    """
    Abstract base class for all UI implementations.
    
    This interface defines the contract for UI implementations with
    full streaming support for real-time interactions.
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the UI interface.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def process_message_stream(self, message: UIMessage) -> AsyncIterator[UIResponse]:
        """
        Process a user message with streaming responses.
        
        Args:
            message: The user message to process
            
        Yields:
            UIResponse: Streaming response chunks
        """
        pass
    
    @abstractmethod
    async def handle_thinking_stream(self, thinking_text: str) -> None:
        """
        Handle thinking indicators during processing.
        
        Args:
            thinking_text: The thinking/processing text to display
        """
        pass
    
    @abstractmethod
    async def handle_tool_stream(self, tool_name: str, status: str, progress: Optional[float] = None) -> None:
        """
        Handle tool execution updates.
        
        Args:
            tool_name: Name of the tool being executed
            status: Current status of tool execution
            progress: Optional progress percentage (0.0-1.0)
        """
        pass
    
    @abstractmethod
    async def handle_error_stream(self, error_message: str, error_type: str) -> None:
        """
        Handle error messages with appropriate formatting.
        
        Args:
            error_message: The error message to display
            error_type: Type of error (connection, server, client, etc.)
        """
        pass
    
    @abstractmethod
    async def display_response(self, response: UIResponse) -> None:
        """
        Display a response chunk in the UI.
        
        Args:
            response: The response chunk to display
        """
        pass
    
    @abstractmethod
    async def handle_system_event(self, event: UIEvent) -> None:
        """
        Handle system events (loading, ready, shutdown, etc.).
        
        Args:
            event: The system event to handle
        """
        pass
    
    @abstractmethod
    async def get_user_input(self) -> UIMessage:
        """
        Get input from the user.
        
        Returns:
            UIMessage: The user's input message
        """
        pass
    
    @abstractmethod
    async def run(self) -> None:
        """
        Run the UI interface main loop.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the UI interface gracefully.
        """
        pass


class StreamingUIService:
    """
    Service for managing streaming UI interactions.
    
    This service provides utilities for converting agent responses
    to streaming UI responses and managing UI state.
    """
    
    def __init__(self, ui_interface: UIInterface):
        self.ui_interface = ui_interface
        self.active_streams: Dict[str, asyncio.Task] = {}
    
    async def process_agent_response_stream(self, agent_response_stream) -> AsyncIterator[UIResponse]:
        """
        Convert agent service response stream to UI response stream.
        
        Args:
            agent_response_stream: Stream from agent service
            
        Yields:
            UIResponse: UI-formatted response chunks
        """
        async for chunk in agent_response_stream:
            if isinstance(chunk, str):
                # Simple text chunk
                yield UIResponse(
                    content=chunk,
                    response_type=UIResponseType.TEXT
                )
            elif isinstance(chunk, dict):
                # Structured response chunk
                response_type = UIResponseType.TEXT
                if chunk.get("type") == "thinking":
                    response_type = UIResponseType.THINKING
                elif chunk.get("type") == "tool_execution":
                    response_type = UIResponseType.TOOL_EXECUTION
                elif chunk.get("type") == "error":
                    response_type = UIResponseType.ERROR
                elif chunk.get("type") == "system":
                    response_type = UIResponseType.SYSTEM
                
                yield UIResponse(
                    content=chunk.get("content", ""),
                    response_type=response_type,
                    metadata=chunk.get("metadata", {}),
                    is_complete=chunk.get("is_complete", False),
                    confidence=chunk.get("confidence"),
                    tools_used=chunk.get("tools_used")
                )
    
    async def start_message_stream(self, message: UIMessage, stream_id: str) -> None:
        """
        Start processing a message with streaming responses.
        
        Args:
            message: The message to process
            stream_id: Unique identifier for this stream
        """
        if stream_id in self.active_streams:
            # Cancel existing stream
            self.active_streams[stream_id].cancel()
        
        # Start new stream
        task = asyncio.create_task(self._process_message_stream(message))
        self.active_streams[stream_id] = task
    
    async def _process_message_stream(self, message: UIMessage) -> None:
        """
        Internal method to process message stream.
        
        Args:
            message: The message to process
        """
        try:
            async for response in self.ui_interface.process_message_stream(message):
                await self.ui_interface.display_response(response)
        except Exception as e:
            await self.ui_interface.handle_error_stream(
                f"Error processing message: {str(e)}", 
                "processing_error"
            )
    
    async def stop_stream(self, stream_id: str) -> None:
        """
        Stop a specific message stream.
        
        Args:
            stream_id: The stream to stop
        """
        if stream_id in self.active_streams:
            self.active_streams[stream_id].cancel()
            del self.active_streams[stream_id]
    
    async def stop_all_streams(self) -> None:
        """Stop all active message streams."""
        for task in self.active_streams.values():
            task.cancel()
        self.active_streams.clear()


class UIResponseFormatter:
    """
    Utility class for formatting responses for different UI types.
    """
    
    @staticmethod
    def format_thinking_response(thinking_text: str) -> UIResponse:
        """Format thinking text as UI response."""
        return UIResponse(
            content=thinking_text,
            response_type=UIResponseType.THINKING,
            metadata={"animated": True}
        )
    
    @staticmethod
    def format_tool_response(tool_name: str, status: str, progress: Optional[float] = None) -> UIResponse:
        """Format tool execution as UI response."""
        return UIResponse(
            content=f"🔧 {tool_name}: {status}",
            response_type=UIResponseType.TOOL_EXECUTION,
            metadata={"tool_name": tool_name, "status": status, "progress": progress}
        )
    
    @staticmethod
    def format_error_response(error_message: str, error_type: str) -> UIResponse:
        """Format error as UI response."""
        return UIResponse(
            content=f"❌ {error_message}",
            response_type=UIResponseType.ERROR,
            metadata={"error_type": error_type}
        )
    
    @staticmethod
    def format_system_response(message: str) -> UIResponse:
        """Format system message as UI response."""
        return UIResponse(
            content=f"🤖 {message}",
            response_type=UIResponseType.SYSTEM,
            metadata={"system": True}
        )
    
    @staticmethod
    def format_completion_response(summary: str, tools_used: List[str]) -> UIResponse:
        """Format task completion as UI response."""
        content = summary
        if tools_used:
            content += f"\n\n🔧 Tools used: {', '.join(tools_used)}"
        
        return UIResponse(
            content=content,
            response_type=UIResponseType.COMPLETION,
            metadata={"completion": True},
            is_complete=True,
            tools_used=tools_used
        )