"""
UI Message Bus for Decoupled Communication

This module provides a message bus system for decoupled communication
between the core services and UI implementations.
"""

import asyncio
from typing import Dict, Set, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import weakref
import time
from collections import defaultdict

from .interface import UIMessage, UIResponse, UIEvent, UIResponseType
from ..utils.logging import get_logger


class MessageType(Enum):
    """Types of messages that can be sent through the message bus."""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_EVENT = "system_event"
    UI_EVENT = "ui_event"
    ERROR = "error"


@dataclass
class BusMessage:
    """Container for messages sent through the message bus."""
    message_type: MessageType
    source: str
    target: Optional[str] = None
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None


class UIMessageBus:
    """
    Message bus for UI communication.
    
    This bus provides decoupled communication between core services
    and UI implementations using publish-subscribe pattern.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.subscribers: Dict[MessageType, Set[Callable]] = defaultdict(set)
        self.wildcard_subscribers: Set[Callable] = set()
        self.message_history: list[BusMessage] = []
        self.max_history: int = 1000
        self.running = False
        self.message_queue = asyncio.Queue()
        self.worker_task = None
        
    async def start(self):
        """Start the message bus."""
        if self.running:
            return
            
        self.running = True
        self.worker_task = asyncio.create_task(self._message_worker())
        self.logger.info("📡 UI Message Bus started")
    
    async def stop(self):
        """Stop the message bus."""
        if not self.running:
            return
            
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("📡 UI Message Bus stopped")
    
    async def _message_worker(self):
        """Worker task to process messages from the queue."""
        while self.running:
            try:
                message = await self.message_queue.get()
                await self._deliver_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _deliver_message(self, message: BusMessage):
        """Deliver a message to all subscribers."""
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Deliver to specific subscribers
        subscribers = self.subscribers.get(message.message_type, set())
        
        # Deliver to wildcard subscribers
        all_subscribers = subscribers.union(self.wildcard_subscribers)
        
        # Call subscribers
        for subscriber in all_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(message)
                else:
                    subscriber(message)
            except Exception as e:
                self.logger.error(f"Error calling subscriber {subscriber}: {e}")
    
    async def publish(self, message: BusMessage):
        """
        Publish a message to the bus.
        
        Args:
            message: The message to publish
        """
        if not self.running:
            await self.start()
        
        await self.message_queue.put(message)
    
    async def publish_user_input(self, content: str, source: str = "ui", session_id: str = None):
        """
        Publish a user input message.
        
        Args:
            content: The user input content
            source: Source of the input
            session_id: Session identifier
        """
        message = BusMessage(
            message_type=MessageType.USER_INPUT,
            source=source,
            data=UIMessage(content=content, message_type="user_input"),
            session_id=session_id
        )
        await self.publish(message)
    
    async def publish_agent_response(self, response: UIResponse, source: str = "agent", session_id: str = None):
        """
        Publish an agent response message.
        
        Args:
            response: The agent response
            source: Source of the response
            session_id: Session identifier
        """
        message = BusMessage(
            message_type=MessageType.AGENT_RESPONSE,
            source=source,
            data=response,
            session_id=session_id
        )
        await self.publish(message)
    
    async def publish_system_event(self, event: UIEvent, source: str = "system", session_id: str = None):
        """
        Publish a system event message.
        
        Args:
            event: The system event
            source: Source of the event
            session_id: Session identifier
        """
        message = BusMessage(
            message_type=MessageType.SYSTEM_EVENT,
            source=source,
            data=event,
            session_id=session_id
        )
        await self.publish(message)
    
    async def publish_error(self, error_message: str, error_type: str = "general", source: str = "system", session_id: str = None):
        """
        Publish an error message.
        
        Args:
            error_message: The error message
            error_type: Type of error
            source: Source of the error
            session_id: Session identifier
        """
        message = BusMessage(
            message_type=MessageType.ERROR,
            source=source,
            data={"message": error_message, "error_type": error_type},
            session_id=session_id
        )
        await self.publish(message)
    
    def subscribe(self, message_type: MessageType, callback: Callable):
        """
        Subscribe to messages of a specific type.
        
        Args:
            message_type: Type of messages to subscribe to
            callback: Callback function to call when message is received
        """
        self.subscribers[message_type].add(callback)
        self.logger.debug(f"Subscribed to {message_type} messages")
    
    def subscribe_all(self, callback: Callable):
        """
        Subscribe to all messages (wildcard subscription).
        
        Args:
            callback: Callback function to call for any message
        """
        self.wildcard_subscribers.add(callback)
        self.logger.debug("Subscribed to all messages")
    
    def unsubscribe(self, message_type: MessageType, callback: Callable):
        """
        Unsubscribe from messages of a specific type.
        
        Args:
            message_type: Type of messages to unsubscribe from
            callback: Callback function to remove
        """
        self.subscribers[message_type].discard(callback)
        self.logger.debug(f"Unsubscribed from {message_type} messages")
    
    def unsubscribe_all(self, callback: Callable):
        """
        Unsubscribe from all messages.
        
        Args:
            callback: Callback function to remove
        """
        self.wildcard_subscribers.discard(callback)
        self.logger.debug("Unsubscribed from all messages")
    
    def get_message_history(self, 
                           message_type: Optional[MessageType] = None,
                           session_id: Optional[str] = None,
                           limit: int = 100) -> list[BusMessage]:
        """
        Get message history with optional filtering.
        
        Args:
            message_type: Filter by message type
            session_id: Filter by session ID
            limit: Maximum number of messages to return
            
        Returns:
            List of filtered messages
        """
        messages = self.message_history
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        if session_id:
            messages = [m for m in messages if m.session_id == session_id]
        
        return messages[-limit:]
    
    def clear_history(self):
        """Clear message history."""
        self.message_history.clear()
        self.logger.info("Message history cleared")


# Global message bus instance
_global_message_bus: Optional[UIMessageBus] = None


def get_global_message_bus() -> UIMessageBus:
    """
    Get the global message bus instance.
    
    Returns:
        UIMessageBus: The global message bus
    """
    global _global_message_bus
    if _global_message_bus is None:
        _global_message_bus = UIMessageBus()
    return _global_message_bus


async def publish_user_input(content: str, source: str = "ui", session_id: str = None):
    """Convenience function to publish user input."""
    bus = get_global_message_bus()
    await bus.publish_user_input(content, source, session_id)


async def publish_agent_response(response: UIResponse, source: str = "agent", session_id: str = None):
    """Convenience function to publish agent response."""
    bus = get_global_message_bus()
    await bus.publish_agent_response(response, source, session_id)


async def publish_system_event(event: UIEvent, source: str = "system", session_id: str = None):
    """Convenience function to publish system event."""
    bus = get_global_message_bus()
    await bus.publish_system_event(event, source, session_id)


def subscribe_to_user_input(callback: Callable):
    """Convenience function to subscribe to user input."""
    bus = get_global_message_bus()
    bus.subscribe(MessageType.USER_INPUT, callback)


def subscribe_to_agent_responses(callback: Callable):
    """Convenience function to subscribe to agent responses."""
    bus = get_global_message_bus()
    bus.subscribe(MessageType.AGENT_RESPONSE, callback)