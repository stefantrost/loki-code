"""
Simplified Textual-based TUI application for Loki Code.

A clean, modern TUI using the Textual framework for chat-based interaction.
"""

import sys
import logging
import os
import time
from io import StringIO
from typing import Optional, Any
import asyncio

try:
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll, Vertical, Horizontal
    from textual.widgets import Input, Static, Header, Footer, ProgressBar
    from textual.logging import TextualHandler
    from textual.reactive import reactive
    from textual import work
    from rich.text import Text
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    # Define App as Any for type hints when Textual not available
    App = Any

from ..utils.logging import get_logger


def safe_format(value) -> str:
    """
    Safely format any value for display in the TUI, escaping markup characters.

    Args:
        value: Any value to format

    Returns:
        str: Safe string representation
    """
    if isinstance(value, str):
        # Escape Textual markup characters
        text = value.replace('[', '\\[').replace(']', '\\]')
        text = text.replace('=', '\\=')
        return text
    elif value is None:
        return "No content"
    else:
        # Convert to string and escape markup characters
        str_value = str(value)
        str_value = str_value.replace('[', '\\[').replace(']', '\\]')
        str_value = str_value.replace('=', '\\=')
        return str_value


class TUILogHandler(logging.Handler):
    """Custom log handler that displays agent reasoning logs in the TUI chat interface."""

    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.setLevel(logging.INFO)

        # Define which log messages should appear in chat
        self.reasoning_patterns = [
            "🤔 Thinking...",
            "💭 Thought:",
            "🔧 Action #",
            "📝 Action Input:",
            "⚡ Executing tool:",
            "👀 Observation:",
            "✅ Final Answer:",
            "❌ Tool Error:",
        ]

    def emit(self, record):
        """Handle log records and display reasoning steps in chat."""
        try:
            message = self.format(record)

            # Only show agent reasoning messages in chat
            if any(pattern in message for pattern in self.reasoning_patterns):
                # Store message for later display to avoid deadlocks
                if not hasattr(self.app, '_pending_reasoning_messages'):
                    self.app._pending_reasoning_messages = []
                self.app._pending_reasoning_messages.append(message)

        except Exception:
            # Don't let log handling break the app
            pass

    def _add_reasoning_message_sync(self, message: str):
        """Add a reasoning message to the chat interface (synchronous for log handler)."""
        try:
            if hasattr(self.app, 'query_one'):
                chat_view = self.app.query_one("#chat-view")
                # Create a reasoning message (not user message)
                reasoning_msg = ReasoningMessage(message)
                chat_view.mount(reasoning_msg)
                chat_view.scroll_end()
        except Exception:
            # Don't let UI updates break the app
            pass

    async def _add_reasoning_message(self, message: str):
        """Add a reasoning message to the chat interface."""
        try:
            if hasattr(self.app, 'query_one'):
                chat_view = self.app.query_one("#chat-view")
                # Create a reasoning message (not user message)
                reasoning_msg = ReasoningMessage(message)
                await chat_view.mount(reasoning_msg)
                chat_view.scroll_end()
        except Exception:
            # Don't let UI updates break the app
            pass


class ReasoningMessage(Static):
    """A reasoning step message widget (distinct from regular chat messages)."""

    def __init__(self, content: str):
        # Escape markup and add reasoning styling
        content = self._escape_markup(content)
        super().__init__(content)
        self.add_class("reasoning-message")

    @staticmethod
    def _escape_markup(text: str) -> str:
        """Escape characters that could be interpreted as Textual markup."""
        text = text.replace('[', '\\[').replace(']', '\\]')
        text = text.replace('=', '\\=')
        return text


class ChatMessage(Static):
    """A single chat message widget."""

    def __init__(self, content: Any, is_user: bool = False):
        # Ensure content is always a safe string to prevent MarkupError
        content = self._safe_string_conversion(content)

        # REPL-like styling: minimal prefixes
        prefix = "> " if is_user else ""
        super().__init__(f"{prefix}{content}")
        self.add_class("user-message" if is_user else "assistant-message")

    @staticmethod
    def _safe_string_conversion(value) -> str:
        """Convert any value to a safe string for Textual markup."""
        if isinstance(value, str):
            # Even if it's already a string, escape markup characters
            return ChatMessage._escape_markup(value)
        elif value is None:
            return "No content"
        else:
            # Convert to string and escape any markup characters
            str_value = str(value)
            return ChatMessage._escape_markup(str_value)

    @staticmethod
    def _escape_markup(text: str) -> str:
        """Escape characters that could be interpreted as Textual markup."""
        # Escape square brackets and other potential markup characters
        text = text.replace('[', '\\[').replace(']', '\\]')
        # Also escape any sequences that might look like markup errors
        text = text.replace('=', '\\=')  # In case '=' at start causes issues
        return text

    def update(self, content: Any = "") -> None:
        """Override update to ensure content is always a safe string."""
        content = self._safe_string_conversion(content)
        super().update(content)

    def __setattr__(self, name, value):
        """Override setattr to intercept _content assignments."""
        if name == '_content':
            value = self._safe_string_conversion(value)
        super().__setattr__(name, value)


class LoadingWidget(Vertical):
    """Widget to show loading progress during model initialization with real-time logs."""

    def __init__(self):
        super().__init__(id="loading-container")
        self.add_class("loading-widget")
        self.logs = []  # Store recent log messages

    def compose(self) -> ComposeResult:
        yield Static("🤖 Initializing Loki Code Agent", id="loading-status")
        yield ProgressBar(id="loading-progress")
        yield Static("💡 Tip: Loading usually takes 2-3 minutes for first startup", id="loading-tips")
        yield VerticalScroll(id="loading-logs")

    def update_status(self, message: str, progress: Optional[float] = None):
        """Update loading status message and progress."""
        try:
            status_widget = self.query_one("#loading-status")
            status_widget.update(message)

            if progress is not None:
                progress_widget = self.query_one("#loading-progress")
                progress_widget.progress = progress
        except Exception:
            # Don't let UI updates break the loading process
            pass

    def update_tip(self, tip: str):
        """Update loading tip message."""
        try:
            tip_widget = self.query_one("#loading-tips")
            tip_widget.update(f"💡 {tip}")
        except Exception:
            pass

    def update_log(self, log_message: str):
        """Add a new log message to the loading log display."""
        try:
            # Add to logs list (keep last 10 messages)
            self.logs.append(log_message)
            if len(self.logs) > 10:
                self.logs.pop(0)

            # Update the log display
            log_container = self.query_one("#loading-logs")
            log_container.remove_children()

            for log in self.logs:
                log_widget = Static(self._escape_markup(log))
                log_widget.add_class("loading-log-item")
                log_container.mount(log_widget)

            # Scroll to bottom
            log_container.scroll_end()

        except Exception:
            # Don't let log updates break the loading process
            pass

    @staticmethod
    def _escape_markup(text: str) -> str:
        """Escape characters that could be interpreted as Textual markup."""
        text = text.replace('[', '\\[').replace(']', '\\]')
        text = text.replace('=', '\\=')
        return text


class LokiApp(App):
    """Simplified Loki Code TUI using Textual."""

    # Reactive loading state that triggers recompose when changed
    is_loading = reactive(False, recompose=True)
    
    # UI Adapter integration
    ui_adapter = None

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
        border: none;
        margin: 1;
        padding: 1;
        overflow-y: auto;
        background: $surface;
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
        background: transparent;
        margin: 0;
        padding: 0 1;
        color: $text;
    }
    
    .assistant-message {
        background: transparent;
        margin: 0;
        padding: 0 1;
        color: $text;
    }
    
    .reasoning-message {
        background: transparent;
        margin: 0;
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
    }
    
    #loading-container {
        height: 100%;
        display: block;
        align: center middle;
        padding: 2;
    }
    
    #loading-status {
        text-align: center;
        margin: 1 0;
        color: $accent;
    }
    
    #loading-progress {
        width: 60%;
        margin: 1 0;
    }
    
    #loading-tips {
        text-align: center;
        margin: 1 0;
        color: $text-muted;
    }
    
    #loading-logs {
        height: 1fr;
        margin: 1 0;
        padding: 1;
        border: solid $primary;
        background: $surface;
        overflow-y: auto;
    }
    
    .loading-log-item {
        color: $text-muted;
        margin: 0;
        padding: 0 1;
        background: transparent;
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

        # Model loading state management
        self.agent_service = None
        self.loading_widget = None
        self.model_loading = False
        self.model_loaded = False
        self.model_error = None

        # Configure logging to use Textual handler and suppress stdout/stderr
        self._setup_textual_logging()

    def _setup_textual_logging(self):
        """Set up comprehensive logging for TUI debugging."""
        # Create file handler for debugging TUI loading issues
        import tempfile
        import os
        
        # Set up environment to prevent multiprocessing issues in TUI
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        # Fix multiprocessing context for TUI environment
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # start method already set
            pass
        
        # Create log file in temp directory
        self.log_file_path = os.path.join(tempfile.gettempdir(), "loki_tui_loading.log")
        
        # Set up file handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file_path, mode='w')  # Overwrite each time
        file_handler.setLevel(logging.DEBUG)  # Capture everything
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Set up Textual handler for dev console
        textual_handler = TextualHandler()
        textual_handler.setLevel(logging.INFO)
        
        # Configure root logger with both handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Clear existing handlers and add our handlers
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(textual_handler)
        # Temporarily disable reasoning handler to avoid deadlocks
        # reasoning_handler = TUILogHandler(self)
        # reasoning_handler.setLevel(logging.INFO)
        # root_logger.addHandler(reasoning_handler)
        
        # Store original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Log the setup
        self.logger.info(f"TUI logging configured - file: {self.log_file_path}")
        self.logger.debug("TUI logging setup complete with DEBUG level file logging")

    def _restore_stdio(self):
        """Restore original stdout/stderr."""
        # Nothing to restore since we're using TextualHandler
        pass

    def compose(self) -> ComposeResult:
        """Create the TUI layout."""
        yield Header()

        # Show loading screen initially
        if self.is_loading:
            self.loading_widget = LoadingWidget()
            yield self.loading_widget
        else:
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
        """Initialize the app - UI-first design."""
        # Focus input immediately - no blocking operations
        self.query_one("#user-input").focus()
        # AI service will be loaded lazily on first input

    def action_focus_input(self) -> None:
        """Focus the input field."""
        if not self.is_loading:
            self.query_one("#user-input").focus()

    async def _load_model_if_needed(self) -> None:
        """Load model lazily on first use with proper state management."""
        if self.model_loaded or self.model_loading:
            return
        
        self.model_loading = True
        self.model_error = None
        
        try:
            self.logger.info("🔄 Loading model lazily on first use...")
            
            # Show loading status to user
            chat_view = self.query_one("#chat-view")
            loading_msg = ChatMessage("🔄 Loading AI model (this may take a moment)...", is_user=False)
            await chat_view.mount(loading_msg)
            chat_view.scroll_end()
            
            # Load the agent service
            from ..core.services import get_agent_service
            self.agent_service = await get_agent_service(self.config)
            
            if self.agent_service:
                self.model_loaded = True
                self.logger.info("✅ Model loaded successfully")
                
                # Update user with success message
                loading_msg.update("✅ AI model loaded successfully!")
                
            else:
                raise RuntimeError("Agent service returned None")
                
        except Exception as e:
            self.model_error = str(e)
            self.logger.error(f"❌ Model loading failed: {e}")
            
            # Update user with error message
            if 'loading_msg' in locals():
                loading_msg.update(f"❌ Model loading failed: {e}")
            
        finally:
            self.model_loading = False

    async def _start_loading_process(self) -> None:
        """Start the background loading process with proper async handling."""
        try:
            self.logger.info("🚀 TUI: Starting loading process")

            # Step 1: Configuration
            if self.loading_widget:
                self.loading_widget.update_status(
                    "🔧 Initializing configuration...", 0.1)
                self.loading_widget.update_log("📋 Loading configuration...")
                if hasattr(self, 'log_file_path'):
                    self.loading_widget.update_log(f"📄 Debug log: {self.log_file_path}")

            await asyncio.sleep(0.1)  # Allow UI to update

            # Step 2: Model loading preparation
            if self.loading_widget:
                self.loading_widget.update_status(
                    "📦 Loading AI model (this may take 2-3 minutes)...", 0.2)
                self.loading_widget.update_tip(
                    "The model is being loaded from disk - this is normal for first startup and takes 2-3 minutes")
                self.loading_widget.update_log(
                    "🤖 Preparing to load AI model (this will take 2-3 minutes)...")

            await asyncio.sleep(0.1)  # Allow UI to update

            # Step 3: Agent service initialization (with proper async handling)
            if self.ui_adapter:
                # Use adapter for initialization
                self.logger.info("🔧 TUI: Initializing via UI adapter")
                
                if self.loading_widget:
                    self.loading_widget.update_log(
                        "⚡ Initializing UI adapter (this is the slow part)...")
                    self.loading_widget.update_status(
                        "🚀 Starting model loading process...", 0.3)

                # Initialize adapter with proper timeout
                try:
                    success = await asyncio.wait_for(
                        self.ui_adapter.initialize(),
                        timeout=300.0  # 5 minutes timeout for model loading
                    )
                    if not success:
                        raise RuntimeError("UI adapter initialization failed")
                        
                    self.agent_service = self.ui_adapter.agent_service
                    
                except asyncio.TimeoutError:
                    if self.loading_widget:
                        self.loading_widget.update_log(
                            "⚠️ Timeout after 5 minutes - model loading failed")
                        self.loading_widget.update_status(
                            "❌ Model loading timed out after 5 minutes", 0.0)
                        self.loading_widget.update_tip(
                            "Try restarting or check if your model is accessible")
                    raise RuntimeError(
                        "Agent service initialization timed out after 5 minutes")
                        
            else:
                # Fallback to direct initialization
                from ..core.services import get_agent_service
                self.logger.info("🔧 TUI: About to initialize agent service directly")

                if self.loading_widget:
                    self.loading_widget.update_log(
                        "⚡ Initializing agent service (this is the slow part)...")
                    self.loading_widget.update_log(
                        "🔧 Progress bars disabled to prevent TUI issues...")
                    self.loading_widget.update_status(
                        "🚀 Starting model loading process...", 0.3)

                # Initialize agent service with proper timeout
                try:
                    self.agent_service = await asyncio.wait_for(
                        get_agent_service(self.config, "tui_session"),
                        timeout=300.0  # 5 minutes timeout for model loading
                    )
                except asyncio.TimeoutError:
                    if self.loading_widget:
                        self.loading_widget.update_log(
                            "⚠️ Timeout after 5 minutes - model loading failed")
                        self.loading_widget.update_status(
                            "❌ Model loading timed out after 5 minutes", 0.0)
                        self.loading_widget.update_tip(
                            "Try restarting or check if your model is accessible")
                    raise RuntimeError(
                        "Agent service initialization timed out after 5 minutes")

            self.logger.info("✅ TUI: Agent service initialization completed")

            if self.loading_widget:
                self.loading_widget.update_log(
                    "✅ Agent service loaded successfully!")
                self.loading_widget.update_status(
                    "✅ Model loaded successfully!", 0.9)

            # Step 4: Validate agent service
            if not self.agent_service:
                raise RuntimeError("Agent service is None")

            agent_info = self.agent_service.get_agent_info()
            self.logger.info(f"📊 TUI: Agent info: {agent_info}")

            # Step 5: Final setup
            if self.loading_widget:
                self.loading_widget.update_status(
                    "🛠️ Setting up tools and agent...", 0.8)
                self.loading_widget.update_log(
                    f"✅ Agent loaded with {len(agent_info.get('available_tools', []))} tools")

            await asyncio.sleep(0.2)  # Allow UI to update

            # Step 6: Ready to start
            if self.loading_widget:
                self.loading_widget.update_status(
                    "✅ Ready! Starting chat interface...", 1.0)
                self.loading_widget.update_log(
                    "🎉 Initialization complete! Starting chat...")

            await asyncio.sleep(0.3)  # Allow UI to update

            # Switch to chat interface
            await self._switch_to_chat_interface()

        except Exception as e:
            self.logger.error(f"Loading failed: {e}", exc_info=True)
            if self.loading_widget:
                # Provide specific error information and troubleshooting
                error_msg, tip_msg = self._analyze_loading_error(e)
                self.loading_widget.update_status(f"❌ {error_msg}", 0.0)
                self.loading_widget.update_tip(tip_msg)
                self.loading_widget.update_log(f"💥 ERROR: {str(e)[:100]}")
                self.loading_widget.update_log(f"💡 TIP: {tip_msg}")
                if hasattr(self, 'log_file_path'):
                    self.loading_widget.update_log(f"🗺️ Check full logs: {self.log_file_path}")

    async def _switch_to_chat_interface(self) -> None:
        """Switch from loading screen to chat interface using reactive recompose pattern."""
        try:
            self.logger.info("🔄 TUI: Switching to chat interface")
            
            # Set loading state to false - this triggers automatic recompose
            self.is_loading = False
            
            # Focus the input after a brief delay to allow recomposition
            self.set_timer(0.1, self._focus_input)
            
            self.logger.info("✅ TUI: Chat interface ready")

        except Exception as e:
            self.logger.error(
                f"Failed to switch to chat interface: {e}", exc_info=True)
    
    def _focus_input(self) -> None:
        """Focus the input field after UI recomposition."""
        try:
            user_input = self.query_one("#user-input")
            user_input.focus()
        except Exception:
            # Input not ready yet, ignore
            pass

    async def _show_loading_progress(self) -> None:
        """Show real-time progress updates by tailing the log file and time-based messages."""
        progress_messages = [
            (30, "Still loading model... (30 seconds elapsed)"),
            (60, "Model loading in progress... (1 minute elapsed)"),
            (90, "Loading continues... (1.5 minutes elapsed)"),
            (120, "Almost there... (2 minutes elapsed)"),
            (150, "Final loading steps... (2.5 minutes elapsed)"),
            (180, "Loading should complete soon... (3 minutes elapsed)"),
            (210, "This is taking longer than usual... (3.5 minutes elapsed)"),
            (240, "Still working on it... (4 minutes elapsed)"),
            (270, "Almost done... (4.5 minutes elapsed)")
        ]

        start_time = time.time()
        last_log_position = 0
        
        try:
            while True:
                # Check for new log entries
                try:
                    if hasattr(self, 'log_file_path') and os.path.exists(self.log_file_path):
                        with open(self.log_file_path, 'r') as f:
                            f.seek(last_log_position)
                            new_lines = f.readlines()
                            last_log_position = f.tell()
                            
                            # Show recent log entries in TUI
                            for line in new_lines[-3:]:  # Show last 3 log lines
                                if line.strip() and self.loading_widget:
                                    # Extract just the message part (after the last |)
                                    log_msg = line.strip().split(' | ')[-1] if ' | ' in line else line.strip()
                                    self.loading_widget.update_log(f"📜 {log_msg}")
                except Exception:
                    pass  # Don't let file reading break the progress display
                
                # Check for time-based progress messages
                elapsed = time.time() - start_time
                
                for delay, message in progress_messages:
                    if elapsed >= delay and elapsed < delay + 5:  # Show within 5 second window
                        if self.loading_widget:
                            self.loading_widget.update_log(f"🕰️ {message}")
                            # Update progress bar based on time elapsed
                            progress = min(0.3 + (delay / 300.0) * 0.5, 0.8)
                            self.loading_widget.update_status(
                                "🚀 Loading model...", progress)
                        break
                
                # Update every 2 seconds
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            # Loading completed or failed - stop progress updates
            pass

    def _analyze_loading_error(self, error: Exception) -> tuple[str, str]:
        """Analyze loading error and provide helpful troubleshooting information."""
        error_str = str(error).lower()

        # Log the error for debugging
        self.logger.error(f"TUI Loading Error: {error}", exc_info=True)

        # Model not found errors
        if "model path not found" in error_str or "no such file" in error_str:
            return (
                "Model files not found",
                "Check if the model path exists. You may need to download the model first."
            )

        # Memory/resource errors
        elif "out of memory" in error_str or "cuda out of memory" in error_str:
            return (
                "Not enough memory to load model",
                "Try closing other applications or use a smaller model."
            )

        # Network/download errors
        elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return (
                "Network error downloading model",
                "Check your internet connection and try again."
            )

        # Permission errors
        elif "permission" in error_str or "access" in error_str:
            return (
                "Permission denied accessing model files",
                "Check file permissions on the model directory."
            )

        # Tool validation errors
        elif "tool" in error_str and "validation" in error_str:
            return (
                "Tool validation failed",
                "There may be a configuration issue. Try restarting the application."
            )

        # Agent initialization errors
        elif "agent" in error_str and "initialize" in error_str:
            return (
                "Agent initialization failed",
                "Check logs for detailed error information. Try restarting."
            )

        # Async/threading errors
        elif "executor" in error_str or "thread" in error_str or "async" in error_str:
            return (
                "Async execution error",
                "Internal threading issue. Try restarting the application."
            )

        # Generic errors
        else:
            return (
                f"Loading failed: {str(error)[:100]}",
                "Check logs for detailed error information. Try restarting the application."
            )

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

        # Clear the input immediately and keep it focusable
        event.input.value = ""
        event.input.focus()

        # Add user message to chat view
        chat_view = self.query_one("#chat-view")
        await chat_view.mount(ChatMessage(user_input, is_user=True))
        chat_view.scroll_end()

        # Use new adapter if available, otherwise fall back to old method
        if self.ui_adapter:
            await self.ui_adapter.queue_user_message(user_input)
            await self.ui_adapter.process_queued_message()
        else:
            # Fallback to original method
            thinking_text = self._get_thinking_message(user_input)
            thinking_msg = ChatMessage(thinking_text, is_user=False)
            await chat_view.mount(thinking_msg)
            chat_view.scroll_end()

            # Start background processing (non-blocking) - don't await this
            self._get_ai_response(user_input, thinking_msg)

    @work()
    async def _get_ai_response(self, user_input: str, thinking_msg: 'ChatMessage') -> None:
        """Get AI response using proper Textual async patterns - no threads."""
        try:
            # Remove thinking indicator
            thinking_msg.remove()

            # Create streaming response widget
            streaming_widget = ChatMessage("", is_user=False)
            chat_view = self.query_one("#chat-view")
            await chat_view.mount(streaming_widget)
            chat_view.scroll_end()

            # Process message using proper async patterns
            await self._process_message_async(user_input, streaming_widget)

        except Exception as e:
            self.logger.error(f"Agent service error: {e}")
            # Handle errors in async context
            thinking_msg.remove()
            chat_view = self.query_one("#chat-view")
            error_response = safe_format(f"Sorry, I encountered an error: {e}")
            await chat_view.mount(ChatMessage(f"❌ {error_response}", is_user=False))
            chat_view.scroll_end()
    
    async def _process_message_async(self, user_input: str, streaming_widget: 'ChatMessage') -> None:
        """Process agent request using proper async patterns - no threading issues."""
        try:
            chat_view = self.query_one("#chat-view")
            
            # Check if we have an agent service available
            if not self.agent_service and not self.model_loading:
                # Try to load the model lazily on first use
                await self._load_model_if_needed()
            
            if self.agent_service and self.model_loaded:
                # Model is loaded - use it for real responses with streaming
                try:
                    # Use streaming to show reasoning steps in real-time
                    final_response = None
                    
                    async for update in self.agent_service.process_message_stream(user_input):
                        update_type = update.get("type", "unknown")
                        
                        if update_type == "reasoning":
                            # Display reasoning step in real-time
                            step = update.get("step", "")
                            if step:
                                reasoning_msg = ReasoningMessage(step)
                                await chat_view.mount(reasoning_msg)
                                chat_view.scroll_end()
                        
                        elif update_type == "final_response":
                            # Process final response
                            if "response" in update:
                                # Traditional AgentResponse format
                                final_response = update["response"]
                            else:
                                # Extract final response from messages
                                messages = update.get("messages", [])
                                if messages:
                                    # Find the last AI message
                                    for message in reversed(messages):
                                        if hasattr(message, '__class__') and "AIMessage" in message.__class__.__name__:
                                            if hasattr(message, 'content') and message.content:
                                                final_response = type('AgentResponse', (), {
                                                    'content': message.content,
                                                    'tools_used': [],
                                                    'actions_taken': []
                                                })()
                                                break
                        
                        elif update_type == "error":
                            # Handle streaming errors
                            error_content = update.get("content", "Unknown error")
                            formatted_response = safe_format(error_content)
                            streaming_widget.update(f"❌ {formatted_response}")
                            chat_view.scroll_end()
                            return
                    
                    # Display final response
                    if final_response:
                        formatted_response = safe_format(final_response.content)
                        
                        # Add additional info from the response if available
                        if hasattr(final_response, 'tools_used') and final_response.tools_used:
                            formatted_response += f"\n\n🔧 Tools used: {', '.join(final_response.tools_used)}"
                        
                        if hasattr(final_response, 'actions_taken') and final_response.actions_taken:
                            formatted_response += f"\n\n⚡ Actions taken: {', '.join(final_response.actions_taken)}"
                        
                        streaming_widget.update(formatted_response)
                        chat_view.scroll_end()
                    else:
                        # No final response received
                        streaming_widget.update("⚠️ No response received from agent")
                        chat_view.scroll_end()
                    
                except Exception as e:
                    # Fall back to echo mode if model fails
                    self.logger.error(f"Model processing failed, falling back to echo: {e}")
                    response_content = f"Echo (model failed): {user_input}"
                    formatted_response = safe_format(response_content)
                    streaming_widget.update(formatted_response)
                    chat_view.scroll_end()
                    
            elif self.model_loading:
                # Model is currently loading
                response_content = f"Model loading... Echo: {user_input}"
                formatted_response = safe_format(response_content)
                streaming_widget.update(formatted_response)
                chat_view.scroll_end()
                
            else:
                # No model available - use echo mode
                response_content = f"Echo (no model): {user_input}"
                formatted_response = safe_format(response_content)
                streaming_widget.update(formatted_response)
                chat_view.scroll_end()

        except Exception as e:
            self.logger.error(f"Agent service error in async processing: {e}")
            chat_view = self.query_one("#chat-view")
            error_response = safe_format(f"Sorry, I encountered an error: {e}")
            streaming_widget.update(f"❌ {error_response}")
            chat_view.scroll_end()

    # Legacy method - now reasoning is displayed in real-time during streaming
    async def _display_reasoning_steps(self, response: 'AgentResponse', chat_view) -> None:
        """Display agent reasoning steps in the chat interface."""
        try:
            # Always show this debug message to confirm function is called
            self.logger.info("=== _display_reasoning_steps called ===")
            
            reasoning_steps = []
            
            # Extract metadata from response
            metadata = getattr(response, 'metadata', {})
            
            # Debug: Log what we have in metadata
            self.logger.info(f"Response metadata keys: {list(metadata.keys())}")
            self.logger.info(f"Response attributes: {dir(response)}")
            
            # Check for LangGraph result in metadata
            if 'langgraph_result' in metadata:
                langgraph_result = metadata['langgraph_result']
                self.logger.info(f"Found LangGraph result with keys: {list(langgraph_result.keys())}")
                
                # Extract messages from LangGraph result
                messages = langgraph_result.get('messages', [])
                self.logger.info(f"Found {len(messages)} messages in LangGraph result")
                
                # Process each message to extract reasoning steps
                for i, message in enumerate(messages):
                    message_type = message.__class__.__name__ if hasattr(message, '__class__') else str(type(message))
                    self.logger.info(f"Message {i}: {message_type}")
                    
                    # Handle AI messages (model reasoning)
                    if "AIMessage" in message_type:
                        # Check for tool calls
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_name = tool_call.get('name', 'unknown_tool')
                                tool_args = tool_call.get('args', {})
                                reasoning_steps.append(f"🔧 Planning to use tool: {tool_name}")
                                if tool_args:
                                    args_str = str(tool_args)[:100] + "..." if len(str(tool_args)) > 100 else str(tool_args)
                                    reasoning_steps.append(f"📝 With arguments: {args_str}")
                        
                        # Show AI message content if it's reasoning
                        if hasattr(message, 'content') and message.content and not message.content.strip().startswith('I need to'):
                            content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                            reasoning_steps.append(f"💭 AI reasoning: {content}")
                    
                    # Handle Tool messages (tool results)
                    elif "ToolMessage" in message_type:
                        tool_name = getattr(message, 'name', 'unknown_tool')
                        tool_content = getattr(message, 'content', '')
                        reasoning_steps.append(f"🔧 Tool {tool_name} executed")
                        if tool_content:
                            # Show truncated tool result
                            content = tool_content[:150] + "..." if len(tool_content) > 150 else tool_content
                            reasoning_steps.append(f"👀 Result: {content}")
                    
                    # Handle Human messages (user input)
                    elif "HumanMessage" in message_type and i > 0:  # Skip the first user message
                        content = getattr(message, 'content', '')
                        if content:
                            reasoning_steps.append(f"👤 User input: {content}")
            
            # Check for reasoning_steps in metadata (if available)
            if 'reasoning_steps' in metadata:
                for step in metadata['reasoning_steps']:
                    reasoning_steps.append(f"🧠 {step}")
            
            # Show tool usage information
            if hasattr(response, 'tools_used') and response.tools_used:
                reasoning_steps.append(f"🛠️ Tools used: {', '.join(response.tools_used)}")
            
            # Show actions taken
            if hasattr(response, 'actions_taken') and response.actions_taken:
                reasoning_steps.append(f"⚡ Actions taken: {', '.join(response.actions_taken)}")
            
            # Also check if there are any attributes on response itself
            if hasattr(response, 'tools_used'):
                self.logger.info(f"Response.tools_used: {response.tools_used}")
            if hasattr(response, 'actions_taken'):
                self.logger.info(f"Response.actions_taken: {response.actions_taken}")
            
            # Display reasoning steps if any were found
            if reasoning_steps:
                # Add a separator before reasoning steps
                reasoning_msg = ReasoningMessage("🧠 Agent Reasoning:")
                await chat_view.mount(reasoning_msg)
                chat_view.scroll_end()
                
                for step in reasoning_steps:
                    reasoning_msg = ReasoningMessage(step)
                    await chat_view.mount(reasoning_msg)
                    chat_view.scroll_end()
            else:
                # If no reasoning steps found, show a debug message
                reasoning_msg = ReasoningMessage("🔍 No reasoning steps found in response")
                await chat_view.mount(reasoning_msg)
                chat_view.scroll_end()
            
            # Show intermediate steps from LangGraph/LangChain (legacy support)
            if 'intermediate_steps' in metadata:
                for i, step in enumerate(metadata['intermediate_steps']):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action_info, observation = step[0], step[1]
                        
                        # Handle AgentAction objects
                        if hasattr(action_info, 'tool') and hasattr(action_info, 'tool_input'):
                            tool_name = action_info.tool
                            tool_input = action_info.tool_input
                            reasoning_steps.append(f"🔧 Action {i+1}: {tool_name}")
                            if tool_input:
                                reasoning_steps.append(f"📝 Input: {str(tool_input)[:100]}...")
                        
                        # Handle thought logs
                        if hasattr(action_info, 'log') and action_info.log:
                            reasoning_steps.append(f"💭 Thought: {action_info.log}")
                        
                        # Show observation/result
                        if observation:
                            obs_text = str(observation)[:150]
                            reasoning_steps.append(f"👀 Result: {obs_text}...")
            
            # Show thinking process if available
            if 'thinking' in metadata:
                thinking = metadata['thinking']
                if isinstance(thinking, str):
                    reasoning_steps.append(f"🤔 Thinking: {thinking}")
                elif isinstance(thinking, list):
                    for thought in thinking:
                        reasoning_steps.append(f"🤔 Thinking: {thought}")
            
            # Show tool usage information
            if hasattr(response, 'tools_used') and response.tools_used:
                reasoning_steps.append(f"🛠️ Tools used: {', '.join(response.tools_used)}")
            
            # Show actions taken
            if hasattr(response, 'actions_taken') and response.actions_taken:
                reasoning_steps.append(f"⚡ Actions taken: {', '.join(response.actions_taken)}")
            
            # Display reasoning steps if any were found
            if reasoning_steps:
                # Add a separator before reasoning steps
                reasoning_msg = ReasoningMessage("🧠 Agent Reasoning:")
                await chat_view.mount(reasoning_msg)
                chat_view.scroll_end()
                
                for step in reasoning_steps:
                    reasoning_msg = ReasoningMessage(step)
                    await chat_view.mount(reasoning_msg)
                    chat_view.scroll_end()
                    
        except Exception as e:
            self.logger.error(f"Error displaying reasoning steps: {e}")
            # Don't break the main response if reasoning display fails

    def _format_file_reader_result(self, tool_args: dict, tool_result: dict) -> str:
        """Format file reader results for display in chat."""
        file_path = tool_args.get('file_path', 'unknown')

        if not tool_result.get('success', False):
            return f"❌ Failed to read {file_path}: {tool_result.get('message', 'Unknown error')}"

        # Extract file content and info
        content = tool_result.get('content', '')
        file_info = tool_result.get('file_info', {})
        analysis_summary = tool_result.get('analysis_summary', '')

        # Build formatted response
        response_parts = [f"📖 **File: {file_path}**"]

        # Add file info if available
        if file_info:
            lines = file_info.get('lines', 0)
            size_bytes = file_info.get('size_bytes', 0)
            language = file_info.get('language', 'text')

            size_kb = size_bytes / 1024 if size_bytes > 0 else 0
            info_line = f"📊 {lines} lines, {size_kb:.1f}KB"
            if language and language != 'text':
                info_line += f", {language}"
            response_parts.append(info_line)

        # Add analysis summary if available
        if analysis_summary:
            response_parts.append(f"🔍 **Analysis:** {analysis_summary}")

        # Add content preview (truncated)
        if content and not content.startswith('<Binary file:'):
            lines = content.split('\n')
            if len(lines) > 30:
                preview = '\n'.join(lines[:30])
                response_parts.append(
                    f"**Content Preview:**\n```\n{preview}\n... ({len(lines) - 30} more lines)\n```")
            else:
                response_parts.append(f"**Content:**\n```\n{content}\n```")
        elif content.startswith('<Binary file:'):
            response_parts.append("📄 Binary file - content not displayed")

        return '\n\n'.join(response_parts)

    def _format_directory_result(self, tool_args: dict, tool_result: dict) -> str:
        """Format directory listing results for display in chat."""
        if not tool_result.get('success', False):
            return f"❌ Directory listing failed: {tool_result.get('message', 'Unknown error')}"

        files = tool_result.get('files', [])
        return f"📁 **Directory Listing:**\n\n" + '\n'.join(f"• {file}" for file in files)


def create_loki_app(config, use_adapter: bool = False) -> Optional[App]:
    """Factory function to create the Loki TUI app."""
    if not TEXTUAL_AVAILABLE:
        return None

    app = LokiApp(config)
    
    # Optionally integrate with new UI adapter (disabled by default for now)
    if use_adapter:
        try:
            from .tui_adapter import create_tui_adapter
            adapter = create_tui_adapter(config)
            app.ui_adapter = adapter
            adapter.textual_app = app
        except ImportError:
            # Fallback to old method if adapter not available
            pass

    # Ensure cleanup on any exit
    import atexit
    atexit.register(app._restore_stdio)

    return app
