"""
TUI Adapter for Textual-based UI

This adapter implements the UIInterface for the existing Textual TUI,
providing proper decoupling while maintaining all existing functionality.
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any
from pathlib import Path

from .interface import UIInterface, UIMessage, UIResponse, UIEvent, UIResponseType, UIResponseFormatter
from .message_bus import get_global_message_bus, MessageType, BusMessage
from ..core.services import get_agent_service
from ..utils.logging import get_logger

try:
    from textual.app import App
    from textual.widgets import Static
    from textual import work
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = None
    Static = None


class TUIAdapter(UIInterface):
    """
    TUI Adapter that implements UIInterface for Textual-based TUI.
    
    This adapter bridges the new UI interface with the existing TUI implementation,
    providing proper decoupling while maintaining backward compatibility.
    """
    
    def __init__(self, config, textual_app=None):
        self.config = config
        self.textual_app = textual_app
        self.logger = get_logger(__name__)
        self.agent_service = None
        self.is_initialized = False
        self.current_thinking_widget = None
        self.message_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.message_bus = get_global_message_bus()
        self.session_id = "tui_session"
        
    async def initialize(self) -> bool:
        """Initialize the TUI adapter and agent service."""
        try:
            self.logger.info("🚀 Initializing TUI adapter...")
            
            # Start message bus
            await self.message_bus.start()
            
            # Subscribe to agent responses
            self.message_bus.subscribe(MessageType.AGENT_RESPONSE, self._handle_agent_response)
            self.message_bus.subscribe(MessageType.SYSTEM_EVENT, self._handle_system_event)
            
            # Initialize agent service
            self.agent_service = await get_agent_service(self.config, self.session_id)
            
            if not self.agent_service:
                self.logger.error("❌ Failed to initialize agent service")
                return False
            
            self.is_initialized = True
            self.logger.info("✅ TUI adapter initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TUI adapter initialization failed: {e}")
            return False
    
    async def process_message_stream(self, message: UIMessage) -> AsyncIterator[UIResponse]:
        """
        Process a user message with streaming responses.
        
        Args:
            message: The user message to process
            
        Yields:
            UIResponse: Streaming response chunks
        """
        if not self.is_initialized or not self.agent_service:
            yield UIResponseFormatter.format_error_response(
                "TUI adapter not initialized", "initialization_error"
            )
            return
        
        try:
            # Show thinking indicator
            yield UIResponseFormatter.format_thinking_response(
                self._get_thinking_message(message.content)
            )
            
            # Process message through agent service
            if hasattr(self.agent_service, 'process_message_stream'):
                # Use streaming if available
                async for response_chunk in self.agent_service.process_message_stream(message.content):
                    # Convert string chunks to UI responses
                    if isinstance(response_chunk, str):
                        yield UIResponse(
                            content=response_chunk,
                            response_type=UIResponseType.TEXT,
                            is_complete=False
                        )
                    else:
                        yield self._convert_agent_response_to_ui(response_chunk)
                
                # Send completion marker
                yield UIResponse(
                    content="",
                    response_type=UIResponseType.COMPLETION,
                    is_complete=True
                )
            else:
                # Fallback to non-streaming
                response = await self.agent_service.process_message(message.content)
                yield self._convert_agent_response_to_ui(response)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            yield UIResponseFormatter.format_error_response(
                f"Error processing message: {str(e)}", "processing_error"
            )
    
    def _convert_agent_response_to_ui(self, agent_response) -> UIResponse:
        """Convert agent service response to UI response format."""
        if isinstance(agent_response, str):
            return UIResponse(
                content=agent_response,
                response_type=UIResponseType.TEXT,
                is_complete=True
            )
        
        # Handle structured agent response
        content = getattr(agent_response, 'content', str(agent_response))
        tools_used = getattr(agent_response, 'tools_used', None)
        confidence = getattr(agent_response, 'confidence', None)
        metadata = getattr(agent_response, 'metadata', {})
        
        # Add reasoning information if available
        if metadata and 'reasoning_steps' in metadata:
            reasoning_steps = metadata['reasoning_steps']
            if reasoning_steps:
                content += "\n\n💭 Reasoning steps:\n" + "\n".join(f"  - {step}" for step in reasoning_steps)
        
        return UIResponse(
            content=content,
            response_type=UIResponseType.TEXT,
            metadata=metadata,
            is_complete=True,
            confidence=confidence,
            tools_used=tools_used
        )
    
    def _clean_conversation_format(self, content: str) -> str:
        """Clean conversation format labels from LLM response content."""
        import re
        
        # Remove common conversation format patterns
        patterns = [
            r'^User:\s*',           # "User: " at start
            r'^Assistant:\s*',      # "Assistant: " at start  
            r'\nUser:\s*',          # "\nUser: " anywhere
            r'\nAssistant:\s*',     # "\nAssistant: " anywhere
            r'^Human:\s*',          # "Human: " at start
            r'^AI:\s*',             # "AI: " at start
            r'\nHuman:\s*',         # "\nHuman: " anywhere
            r'\nAI:\s*',            # "\nAI: " anywhere
        ]
        
        # Apply all patterns
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE)
        
        # Handle the specific format we're seeing: clean up extra whitespace and newlines
        content = content.strip()
        
        # If the content is just whitespace or conversation labels, return empty
        if content.lower().strip() in ['user:', 'assistant:', 'human:', 'ai:', '']:
            return ""
        
        return content
    
    def _get_thinking_message(self, user_input: str) -> str:
        """Get appropriate thinking message based on input (from original TUI)."""
        input_lower = user_input.lower().strip()
        
        # System commands
        if input_lower in ['help', 'h', '?', 'status', 'clear']:
            return "📋 Processing system command..."
        
        # File operations
        if any(word in input_lower for word in ['read', 'analyze', 'file', '.py', '.js', '.ts']):
            return "📖 Analyzing file..."
        
        # Code generation
        if any(word in input_lower for word in ['create', 'generate', 'write', 'implement']):
            return "⚡ Generating code..."
        
        # Explanation requests
        if any(word in input_lower for word in ['explain', 'what', 'how', 'why']):
            return "🧠 Understanding and explaining..."
        
        # Debugging
        if any(word in input_lower for word in ['debug', 'fix', 'error', 'bug']):
            return "🐛 Debugging and analyzing..."
        
        # Shortcuts
        if len(input_lower) <= 3 and any(input_lower.startswith(s) for s in ['r ', 'a ', 'h', '?']):
            return "⚡ Processing shortcut..."
        
        # Default
        return "🤔 Thinking..."
    
    async def handle_thinking_stream(self, thinking_text: str) -> None:
        """Handle thinking indicators during processing."""
        if self.textual_app:
            # Update thinking indicator in the TUI
            try:
                # Check if this is the new REPL app
                if hasattr(self.textual_app, 'start_thinking'):
                    # New REPL interface
                    self.textual_app.start_thinking(thinking_text)
                else:
                    # Old chat interface
                    from .textual_app import ChatMessage
                    thinking_msg = ChatMessage(thinking_text, is_user=False)
                    
                    # Store reference to remove later
                    self.current_thinking_widget = thinking_msg
                    
                    # Add to chat view
                    if hasattr(self.textual_app, 'query_one'):
                        chat_view = self.textual_app.query_one("#chat-view")
                        await chat_view.mount(thinking_msg)
                        chat_view.scroll_end()
                        
            except Exception as e:
                self.logger.error(f"Error updating thinking indicator: {e}")
    
    async def handle_tool_stream(self, tool_name: str, status: str, progress: Optional[float] = None) -> None:
        """Handle tool execution updates."""
        if self.textual_app:
            try:
                from .textual_app import ChatMessage
                
                progress_text = f" ({progress:.1%})" if progress is not None else ""
                tool_msg = ChatMessage(f"🔧 {tool_name}: {status}{progress_text}", is_user=False)
                
                if hasattr(self.textual_app, 'query_one'):
                    chat_view = self.textual_app.query_one("#chat-view")
                    await chat_view.mount(tool_msg)
                    chat_view.scroll_end()
                    
            except Exception as e:
                self.logger.error(f"Error updating tool indicator: {e}")
    
    async def handle_error_stream(self, error_message: str, error_type: str) -> None:
        """Handle error messages with appropriate formatting."""
        if self.textual_app:
            try:
                from .textual_app import ChatMessage
                error_msg = ChatMessage(f"❌ {error_message}", is_user=False)
                
                if hasattr(self.textual_app, 'query_one'):
                    chat_view = self.textual_app.query_one("#chat-view")
                    await chat_view.mount(error_msg)
                    chat_view.scroll_end()
                    
            except Exception as e:
                self.logger.error(f"Error displaying error message: {e}")
    
    async def display_response(self, response: UIResponse) -> None:
        """Display a response chunk in the TUI."""
        if not self.textual_app:
            return
        
        try:
            # Check if this is the new REPL app
            if hasattr(self.textual_app, 'add_output'):
                # New REPL interface
                await self._display_response_repl(response)
            else:
                # Old chat interface
                await self._display_response_chat(response)
                    
        except Exception as e:
            self.logger.error(f"Error displaying response: {e}")
    
    async def _display_response_repl(self, response: UIResponse) -> None:
        """Display response in the new REPL interface."""
        if response.response_type == UIResponseType.THINKING:
            self.textual_app.start_thinking(response.content)
        elif response.response_type == UIResponseType.TOOL_EXECUTION:
            tool_name = response.metadata.get('tool_name', 'Unknown')
            status = response.metadata.get('status', 'Running')
            self.textual_app.update_thinking(f"Running {tool_name}: {status}")
        elif response.response_type == UIResponseType.ERROR:
            self.textual_app.stop_thinking()
            self.textual_app.add_line(response.content, "error")
        elif response.response_type == UIResponseType.COMPLETION:
            # Handle completion - this is the end of streaming
            self.textual_app.stop_thinking()
        else:
            # Regular text response - could be streaming or complete
            if response.content:  # Only display if there's content
                self.textual_app.stop_thinking()
                self.textual_app.display_response(response.content)
    
    async def _display_response_chat(self, response: UIResponse) -> None:
        """Display response in the chat interface."""
        from .textual_app import ChatMessage
        
        # Handle different response types
        if response.response_type == UIResponseType.THINKING:
            await self.handle_thinking_stream(response.content)
        elif response.response_type == UIResponseType.TOOL_EXECUTION:
            tool_name = response.metadata.get('tool_name', 'Unknown')
            status = response.metadata.get('status', 'Running')
            progress = response.metadata.get('progress')
            await self.handle_tool_stream(tool_name, status, progress)
        elif response.response_type == UIResponseType.ERROR:
            await self.handle_error_stream(response.content, response.metadata.get('error_type', 'unknown'))
        elif response.response_type == UIResponseType.COMPLETION:
            # Handle completion - this is the end of streaming
            if self.current_thinking_widget:
                self.current_thinking_widget.remove()
                self.current_thinking_widget = None
            
            # Clean up streaming state
            if hasattr(self, '_streaming_message_widget'):
                # Final scroll to bottom
                if hasattr(self.textual_app, 'query_one'):
                    chat_view = self.textual_app.query_one("#chat-view")
                    chat_view.scroll_end()
                delattr(self, '_streaming_message_widget')
            if hasattr(self, '_streaming_content'):
                delattr(self, '_streaming_content')
            if hasattr(self, '_word_buffer'):
                delattr(self, '_word_buffer')
            if hasattr(self, '_last_update_time'):
                delattr(self, '_last_update_time')
        else:
            # Regular text response - handle streaming properly
            if response.content and response.content.strip():  # Only display if there's meaningful content
                # Remove thinking indicator if this is the first real response
                if self.current_thinking_widget:
                    self.current_thinking_widget.remove()
                    self.current_thinking_widget = None
                
                # For streaming, we need to accumulate chunks
                if not hasattr(self, '_streaming_message_widget'):
                    # Create new message widget for streaming
                    self._streaming_message_widget = ChatMessage("", is_user=False)
                    if hasattr(self.textual_app, 'query_one'):
                        chat_view = self.textual_app.query_one("#chat-view")
                        await chat_view.mount(self._streaming_message_widget)
                        chat_view.scroll_end()
                
                # Clean up streaming data format (remove "data: " prefix)
                cleaned_content = response.content
                if cleaned_content.startswith("data: "):
                    cleaned_content = cleaned_content[6:]  # Remove "data: " prefix
                
                # Skip empty chunks or control messages
                if cleaned_content.strip() in ["", "[DONE]", "\\[DONE\\]"]:
                    return
                
                # Remove conversation format labels from LLM response
                cleaned_content = self._clean_conversation_format(cleaned_content)
                
                # Skip if nothing left after cleaning
                if not cleaned_content.strip():
                    return
                
                # Debug: log what we're receiving (reduced logging)
                self.logger.debug(f"Streaming chunk: '{cleaned_content}'")
                
                # Update the streaming message with accumulated content
                # Add proper spacing between words/tokens
                if hasattr(self, '_streaming_content'):
                    # Add space before new content if the previous content doesn't end with space
                    # and the new content doesn't start with punctuation
                    if (self._streaming_content and 
                        not self._streaming_content.endswith(' ') and 
                        not cleaned_content.startswith((' ', '.', '!', '?', ',', ';', ':', '\n'))):
                        self._streaming_content += ' ' + cleaned_content
                    else:
                        self._streaming_content += cleaned_content
                else:
                    self._streaming_content = cleaned_content
                
                # Initialize word buffer if not exists
                if not hasattr(self, '_word_buffer'):
                    self._word_buffer = []
                    self._last_update_time = time.time()
                
                # Add words to buffer
                words = cleaned_content.split()
                self._word_buffer.extend(words)
                
                # Update UI for each word/token as it comes in (typewriter effect)
                if hasattr(self, '_streaming_message_widget'):
                    self._streaming_message_widget.update(self._streaming_content)
                    
                    # Scroll to bottom to show new content
                    if hasattr(self.textual_app, 'query_one'):
                        chat_view = self.textual_app.query_one("#chat-view")
                        chat_view.scroll_end()
                    
                    # Small delay to make streaming more visible (optional)
                    await asyncio.sleep(0.01)  # 10ms delay for typewriter effect
                
                # If this is marked as complete, finalize the message
                if response.is_complete:
                    # Final scroll to bottom
                    if hasattr(self.textual_app, 'query_one'):
                        chat_view = self.textual_app.query_one("#chat-view")
                        chat_view.scroll_end()
                    
                    # Clean up streaming state
                    if hasattr(self, '_streaming_message_widget'):
                        delattr(self, '_streaming_message_widget')
                    if hasattr(self, '_streaming_content'):
                        delattr(self, '_streaming_content')
                    if hasattr(self, '_word_buffer'):
                        delattr(self, '_word_buffer')
                    if hasattr(self, '_last_update_time'):
                        delattr(self, '_last_update_time')
    
    async def handle_system_event(self, event: UIEvent) -> None:
        """Handle system events (loading, ready, shutdown, etc.)."""
        if event.event_type == "loading":
            # Handle loading events
            if self.textual_app and hasattr(self.textual_app, 'loading_widget'):
                loading_widget = self.textual_app.loading_widget
                if loading_widget:
                    message = event.data.get('message', 'Loading...')
                    progress = event.data.get('progress', 0.0)
                    loading_widget.update_status(message, progress)
                    
        elif event.event_type == "ready":
            # System ready event
            self.logger.info("🎉 TUI system ready")
            
        elif event.event_type == "shutdown":
            # Handle shutdown
            await self.shutdown()
    
    async def get_user_input(self) -> UIMessage:
        """Get input from the user (handled by Textual event system)."""
        # This is handled by the Textual app's input handlers
        # Return queued message if available
        if not self.message_queue.empty():
            return await self.message_queue.get()
        
        # For now, return a placeholder - actual input comes via Textual events
        return UIMessage(content="", message_type="placeholder")
    
    async def run(self) -> None:
        """Run the TUI interface main loop."""
        if not TEXTUAL_AVAILABLE:
            raise RuntimeError("Textual not available. Install with: pip install textual")
        
        if not self.textual_app:
            # Create TUI app if not provided - use new REPL app by default
            from .repl_app import create_repl_app
            self.textual_app = create_repl_app(self.config, use_adapter=False)
            
            # Set adapter reference in TUI app
            self.textual_app.ui_adapter = self
        
        # Initialize adapter
        await self.initialize()
        
        # Run the Textual app - this will block until the app exits
        def run_textual():
            self.textual_app.run()
        
        # Run in a separate thread since Textual is synchronous
        import threading
        textual_thread = threading.Thread(target=run_textual)
        textual_thread.start()
        
        # Wait for the thread to complete
        textual_thread.join()
    
    async def shutdown(self) -> None:
        """Shutdown the TUI interface gracefully."""
        self.logger.info("🔄 Shutting down TUI adapter...")
        
        # Stop any running tasks
        if self.agent_service and hasattr(self.agent_service, 'shutdown'):
            await self.agent_service.shutdown()
        
        self.is_initialized = False
        self.logger.info("✅ TUI adapter shutdown complete")
    
    # Methods to support integration with existing TUI app
    
    async def queue_user_message(self, content: str) -> None:
        """Queue a user message for processing."""
        message = UIMessage(
            content=content,
            message_type="user_input",
            timestamp=time.time()
        )
        await self.message_queue.put(message)
        
        # Also publish to message bus for decoupled communication
        await self.message_bus.publish_user_input(content, "tui", self.session_id)
    
    async def process_queued_message(self) -> None:
        """Process the next queued message."""
        if not self.message_queue.empty():
            message = await self.message_queue.get()
            
            # Don't add user message here - it should already be displayed
            # This prevents duplicate user input display
            
            # Process with streaming
            async for response in self.process_message_stream(message):
                await self.display_response(response)
                
                # Don't publish to message bus - we're already displaying directly
                # This prevents duplicate display
    
    async def _handle_agent_response(self, bus_message: BusMessage) -> None:
        """Handle agent response messages from the message bus."""
        if bus_message.session_id != self.session_id:
            return  # Not for this session
            
        response = bus_message.data
        if isinstance(response, UIResponse):
            await self.display_response(response)
    
    async def _handle_system_event(self, bus_message: BusMessage) -> None:
        """Handle system event messages from the message bus."""
        if bus_message.session_id != self.session_id:
            return  # Not for this session
            
        event = bus_message.data
        if isinstance(event, UIEvent):
            await self.handle_system_event(event)


def create_tui_adapter(config) -> TUIAdapter:
    """
    Factory function to create a TUI adapter.
    
    Args:
        config: Application configuration
        
    Returns:
        TUIAdapter: Configured TUI adapter instance
    """
    return TUIAdapter(config)