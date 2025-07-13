"""
Simplified Textual-based TUI application for Loki Code.

A clean, modern TUI using the Textual framework for chat-based interaction.
"""

import sys
import logging
from io import StringIO
from typing import Optional, Any

try:
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll, Vertical, Horizontal
    from textual.widgets import Input, Static, Header, Footer
    from textual.logging import TextualHandler
    from rich.text import Text
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    # Define App as Any for type hints when Textual not available
    App = Any

from ..utils.logging import get_logger


if TEXTUAL_AVAILABLE:
    
    class ChatMessage(Static):
        """A single chat message widget."""
        
        def __init__(self, content: str, is_user: bool = False):
            prefix = "👤 You: " if is_user else "🤖 Assistant: "
            super().__init__(f"{prefix}{content}")
            self.add_class("user-message" if is_user else "assistant-message")
    

    class LokiApp(App):
        """Simplified Loki Code TUI using Textual."""
        
        CSS = """
        Screen {
            layout: vertical;
        }
        
        Header {
            dock: top;
            height: 3;
        }
        
        Footer {
            dock: bottom; 
            height: 3;
        }
        
        #main-container {
            height: 1fr;
            layout: vertical;
        }
        
        #chat-view {
            height: 1fr;
            border: solid $primary;
            margin: 1;
            padding: 1;
            overflow-y: auto;
        }
        
        #input-container {
            height: 5;
            padding: 1;
        }
        
        #user-input {
            height: 3;
            width: 100%;
            border: solid $accent;
            background: $surface;
            color: $text;
        }
        
        #user-input:focus {
            border: solid $success;
            background: $surface-lighten-1;
        }
        
        .user-message {
            background: $primary 30%;
            margin: 0 0 1 0;
            padding: 1;
        }
        
        .assistant-message {
            background: $success 30%;
            margin: 0 0 1 0;  
            padding: 1;
        }
        """
        
        TITLE = "Loki Code"
        SUB_TITLE = "AI Coding Assistant"
        
        BINDINGS = [
            ("ctrl+c", "quit", "Quit"),
            ("ctrl+d", "quit", "Quit"),
            ("escape", "focus_input", "Focus Input"),
        ]
        
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.logger = get_logger(__name__)
            
            # Configure logging to use Textual handler and suppress stdout/stderr
            self._setup_textual_logging()
            
        def _setup_textual_logging(self):
            """Set up Textual logging to use developer console."""
            # Configure Python logging to use Textual's handler for the dev console
            logging.basicConfig(
                level="INFO",
                handlers=[TextualHandler()],
                force=True  # Override existing handlers
            )
            
            # Store original streams but don't redirect them
            # The TextualHandler will send logs to the dev console
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            
        def _restore_stdio(self):
            """Restore original stdout/stderr."""
            # Nothing to restore since we're using TextualHandler
            pass
            
        def compose(self) -> ComposeResult:
            """Create the TUI layout."""
            yield Header()
            
            # Main container with proper layout
            with Vertical(id="main-container"):
                with VerticalScroll(id="chat-view"):
                    yield ChatMessage("Welcome to Loki Code! How can I help you today?", is_user=False)
                
                with Vertical(id="input-container"):
                    yield Input(
                        placeholder="Type your message and press Enter...", 
                        id="user-input",
                        value=""
                    )
            
            yield Footer()
            
        def on_mount(self) -> None:
            """Initialize the app."""
            self.query_one("#user-input").focus()
        
        def action_focus_input(self) -> None:
            """Focus the input field."""
            self.query_one("#user-input").focus()
            
        def on_unmount(self) -> None:
            """Restore stdio when app unmounts."""
            self._restore_stdio()
            
        async def action_quit(self) -> None:
            """Custom quit action to ensure cleanup."""
            self._restore_stdio()
            await super().action_quit()
        
        def _get_thinking_message(self, user_input: str) -> str:
            """Get appropriate thinking message based on input."""
            
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
            
        async def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle user input submission."""
            user_input = event.value.strip()
            
            if not user_input:
                return
                
            # Clear the input immediately
            event.input.value = ""
            
            # Add user message immediately and scroll to show it
            chat_view = self.query_one("#chat-view")
            await chat_view.mount(ChatMessage(user_input, is_user=True))
            chat_view.scroll_end()
            
            # Add enhanced thinking indicator based on command type
            thinking_text = self._get_thinking_message(user_input)
            thinking_msg = ChatMessage(thinking_text, is_user=False)
            await chat_view.mount(thinking_msg)
            chat_view.scroll_end()
            
            try:
                # Get AI response (simplified for now)
                response = await self._get_ai_response(user_input)
                
                # Remove thinking indicator and add real response
                thinking_msg.remove()
                await chat_view.mount(ChatMessage(response, is_user=False))
                
            except Exception as e:
                thinking_msg.remove()
                await chat_view.mount(ChatMessage(f"❌ Error: {e}", is_user=False))
            
            # Final scroll to bottom
            chat_view.scroll_end()
            
        async def _get_ai_response(self, user_input: str) -> str:
            """Get AI response using command processor."""
            try:
                # Use the new command processor for intelligent routing
                from .commands import CommandProcessor, ConversationContext
                from ..core.agent.loki_agent import LokiCodeAgent
                from ..core.agent.types import AgentConfig
                from ..core.tool_registry import get_global_registry
                
                # Create simple context
                context = ConversationContext(
                    session_id="tui_session",
                    project_path=getattr(self.config, 'project_path', None)
                )
                
                # Create agent and processor
                agent_config = AgentConfig()
                agent = LokiCodeAgent(agent_config)
                tool_registry = get_global_registry()
                
                processor = CommandProcessor(agent, tool_registry)
                
                # Process the input with logging
                self.log(f"Processing user input: {user_input[:50]}...")
                result = await processor.process_input(user_input, context)
                
                if result.success:
                    self.log(f"Command processed successfully: {result.execution_type}")
                    
                    # For system commands, return the message directly
                    if result.execution_type == "system_command":
                        return result.message
                    
                    # For shortcuts and direct tools, show what was executed
                    elif result.execution_type in ["shortcut", "direct_tool"]:
                        if result.direct_tool_call:
                            tool_name, tool_args = result.direct_tool_call
                            return f"✅ Executed {tool_name} successfully!\n\nResult: {result.message}"
                        else:
                            return result.message
                    
                    # For agent conversations, try to use the LLM
                    elif result.execution_type == "agent_conversation":
                        try:
                            # Try to get actual LLM response
                            from ..core.providers import create_llm_provider, GenerationRequest
                            provider = create_llm_provider(self.config)
                            request = GenerationRequest(prompt=user_input)
                            response = await provider.generate(request)
                            return response.content
                        except:
                            # Fallback to processor message
                            return result.message
                    
                    else:
                        return result.message
                        
                else:
                    suggestions = "\n".join(f"• {s}" for s in result.suggestions[:3])
                    return f"{result.message}\n\nSuggestions:\n{suggestions}"
                    
            except Exception as e:
                return f"Sorry, I encountered an error: {e}"


def create_loki_app(config) -> Optional[App]:
    """Factory function to create the Loki TUI app."""
    if not TEXTUAL_AVAILABLE:
        return None
    
    app = LokiApp(config)
    
    # Ensure cleanup on any exit
    import atexit
    atexit.register(app._restore_stdio)
    
    return app