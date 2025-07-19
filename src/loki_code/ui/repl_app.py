"""
REPL-style TUI for Loki Code

A minimalist REPL-like interface that feels more like a Python interpreter
than a chat application, with minimal visual differences between user and LLM text.
"""

import asyncio
import time
from typing import Optional, Any, List
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll, Vertical, Horizontal
    from textual.widgets import Static, Input, Header, Footer
    from textual.reactive import reactive
    from textual.binding import Binding
    from textual import work
    from rich.console import Console
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.padding import Padding
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = Any

from ..utils.logging import get_logger


class REPLLine(Static):
    """A single line in the REPL output with minimal styling."""
    
    def __init__(self, content: str, line_type: str = "output", prompt: str = ""):
        self.line_type = line_type
        self.prompt = prompt
        
        # Format content based on type
        if line_type == "input":
            # User input with minimal prompt
            display_text = f"{prompt}{content}"
        elif line_type == "output":
            # LLM output with no prefix
            display_text = content
        elif line_type == "thinking":
            # Thinking indicator (subtle)
            display_text = f"# {content}"
        elif line_type == "error":
            # Error messages
            display_text = f"Error: {content}"
        elif line_type == "system":
            # System messages
            display_text = f"# {content}"
        else:
            display_text = content
        
        super().__init__(display_text)
        self.add_class(f"repl-{line_type}")


class REPLCodeBlock(Static):
    """A code block with syntax highlighting."""
    
    def __init__(self, code: str, language: str = "python"):
        self.code = code
        self.language = language
        
        # Create syntax highlighted content
        try:
            syntax = Syntax(code, language, theme="monokai", line_numbers=False, padding=0)
            super().__init__(syntax)
        except Exception:
            # Fallback to plain text
            super().__init__(code)
        
        self.add_class("repl-code")


class REPLHistory:
    """Manages command history similar to Python REPL."""
    
    def __init__(self, max_size: int = 1000):
        self.history: List[str] = []
        self.max_size = max_size
        self.position = 0
        
    def add(self, command: str):
        """Add a command to history."""
        if command and (not self.history or self.history[-1] != command):
            self.history.append(command)
            if len(self.history) > self.max_size:
                self.history.pop(0)
        self.position = len(self.history)
    
    def get_previous(self) -> Optional[str]:
        """Get previous command in history."""
        if self.position > 0:
            self.position -= 1
            return self.history[self.position]
        return None
    
    def get_next(self) -> Optional[str]:
        """Get next command in history."""
        if self.position < len(self.history) - 1:
            self.position += 1
            return self.history[self.position]
        elif self.position == len(self.history) - 1:
            self.position = len(self.history)
            return ""
        return None


class REPLInput(Input):
    """Enhanced input widget with REPL-style features."""
    
    def __init__(self, history: REPLHistory, **kwargs):
        super().__init__(**kwargs)
        self.history = history
        self.multiline_buffer = []
        self.in_multiline = False
        
    def on_key(self, event) -> None:
        """Handle special key combinations."""
        if event.key == "up":
            # Navigate history backwards
            prev_cmd = self.history.get_previous()
            if prev_cmd is not None:
                self.value = prev_cmd
                self.cursor_position = len(prev_cmd)
                event.prevent_default()
        elif event.key == "down":
            # Navigate history forwards
            next_cmd = self.history.get_next()
            if next_cmd is not None:
                self.value = next_cmd
                self.cursor_position = len(next_cmd)
                event.prevent_default()
        elif event.key == "ctrl+c":
            # Clear current input
            self.value = ""
            self.multiline_buffer = []
            self.in_multiline = False
            event.prevent_default()
        elif event.key == "ctrl+d":
            # Exit application
            self.app.exit()
            event.prevent_default()
        else:
            super().on_key(event)


class LokiREPL(App):
    """
    REPL-style TUI for Loki Code.
    
    Provides a minimalist interface similar to Python REPL with:
    - Minimal visual differences between user and LLM text
    - Continuous text flow instead of message bubbles
    - Command history and navigation
    - Syntax highlighting for code blocks
    - Multi-line input support
    """
    
    CSS = """
    Screen {
        background: #1e1e1e;
        color: #d4d4d4;
    }
    
    Header {
        dock: top;
        height: 1;
        background: #2d2d2d;
        color: #cccccc;
        padding: 0 1;
    }
    
    Footer {
        dock: bottom;
        height: 1;
        background: #2d2d2d;
        color: #cccccc;
        padding: 0 1;
    }
    
    #main-container {
        height: 1fr;
        padding: 0;
    }
    
    #repl-output {
        height: 1fr;
        padding: 1;
        background: #1e1e1e;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    
    #input-container {
        height: 3;
        padding: 0 1;
        background: #1e1e1e;
    }
    
    #repl-input {
        height: 1;
        background: #1e1e1e;
        color: #d4d4d4;
        border: none;
        padding: 0;
        margin: 0;
    }
    
    #repl-input:focus {
        background: #1e1e1e;
        border: none;
    }
    
    .repl-input {
        color: #d4d4d4;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    .repl-output {
        color: #d4d4d4;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    .repl-thinking {
        color: #6a9955;
        background: transparent;
        padding: 0;
        margin: 0;
        font-style: italic;
    }
    
    .repl-error {
        color: #f44747;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    .repl-system {
        color: #6a9955;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    .repl-code {
        background: #252526;
        padding: 1;
        margin: 1 0;
        border-left: thick #007acc;
    }
    
    /* Remove default widget styling */
    Static {
        background: transparent;
        color: inherit;
        padding: 0;
        margin: 0;
    }
    """
    
    TITLE = "Loki REPL"
    SUB_TITLE = "Interactive AI Coding Assistant"
    
    BINDINGS = [
        Binding("ctrl+c", "clear_input", "Clear", show=False),
        Binding("ctrl+d", "quit", "Exit", show=False),
        Binding("up", "history_prev", "Previous", show=False),
        Binding("down", "history_next", "Next", show=False),
        Binding("escape", "focus_input", "Focus", show=False),
    ]
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        self.history = REPLHistory()
        self.ui_adapter = None
        self.session_counter = 0
        self.current_thinking_line = None
        
        # REPL prompts
        self.input_prompt = ">>> "
        self.continuation_prompt = "... "
        self.output_prompt = ""
        
    def compose(self) -> ComposeResult:
        """Create the REPL interface."""
        yield Header()
        
        with Vertical(id="main-container"):
            with VerticalScroll(id="repl-output"):
                # Welcome message
                yield REPLLine("Loki Code REPL v0.1.0", "system")
                yield REPLLine("Type 'help' for help, 'exit' to quit", "system")
                yield REPLLine("", "output")  # Empty line
            
            with Vertical(id="input-container"):
                yield REPLInput(
                    self.history,
                    placeholder="Enter your command...",
                    id="repl-input"
                )
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the REPL."""
        self.query_one("#repl-input").focus()
        
        # Show initial prompt
        self.add_prompt_line()
    
    def add_prompt_line(self) -> None:
        """Add a new prompt line - but keep input in the bottom input field."""
        # Don't add a prompt line, just reset the session counter
        # Input stays in the bottom input field like before
        pass
    
    def add_line(self, content: str, line_type: str = "output", prompt: str = "") -> REPLLine:
        """Add a line to the REPL output."""
        if content or line_type == "input":  # Always show input lines
            line = REPLLine(content, line_type, prompt)
            output_view = self.query_one("#repl-output")
            output_view.mount(line)
            output_view.scroll_end()
            return line
        return None
    
    def add_code_block(self, code: str, language: str = "python") -> None:
        """Add a syntax-highlighted code block."""
        code_block = REPLCodeBlock(code, language)
        output_view = self.query_one("#repl-output")
        output_view.mount(code_block)
        output_view.scroll_end()
    
    def update_line(self, line: REPLLine, content: str) -> None:
        """Update an existing line's content."""
        if line:
            line.update(content)
    
    def start_thinking(self, message: str) -> None:
        """Start showing thinking indicator."""
        self.current_thinking_line = self.add_line(message, "thinking")
    
    def update_thinking(self, message: str) -> None:
        """Update thinking indicator."""
        if self.current_thinking_line:
            self.update_line(self.current_thinking_line, message)
    
    def stop_thinking(self) -> None:
        """Stop showing thinking indicator."""
        if self.current_thinking_line:
            self.current_thinking_line.remove()
            self.current_thinking_line = None
    
    def add_output(self, content: str) -> None:
        """Add output text (from LLM)."""
        # Split content into lines for better display
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if i == 0:
                # First line continues from current position
                self.add_line(line, "output")
            else:
                # Subsequent lines start fresh
                self.add_line(line, "output")
    
    def detect_code_blocks(self, text: str) -> List[tuple]:
        """Detect code blocks in text and return [(text, is_code, language)]."""
        blocks = []
        lines = text.split('\n')
        current_block = []
        in_code_block = False
        current_language = "python"
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if current_block:
                        blocks.append(('\n'.join(current_block), True, current_language))
                        current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    if current_block:
                        blocks.append(('\n'.join(current_block), False, ""))
                        current_block = []
                    in_code_block = True
                    # Extract language if specified
                    lang = line.strip()[3:].strip()
                    if lang:
                        current_language = lang
            else:
                current_block.append(line)
        
        # Add remaining block
        if current_block:
            blocks.append(('\n'.join(current_block), in_code_block, current_language))
        
        return blocks
    
    def display_response(self, content: str) -> None:
        """Display response content with code block detection."""
        blocks = self.detect_code_blocks(content)
        
        for block_content, is_code, language in blocks:
            block_content = block_content.strip()
            if not block_content:
                continue
                
            if is_code:
                self.add_code_block(block_content, language)
            else:
                # Split into lines for regular text
                lines = block_content.split('\n')
                for line in lines:
                    self.add_line(line, "output")
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        
        if not user_input:
            return
        
        # Add to history
        self.history.add(user_input)
        
        # Show the user input in the output with minimal styling
        self.add_line(f">>> {user_input}", "input")
        
        # Clear the input
        event.input.value = ""
        
        # Handle special commands
        if user_input.lower() in ['exit', 'quit']:
            self.app.exit()
            return
        elif user_input.lower() == 'help':
            self.show_help()
            return
        elif user_input.lower() == 'clear':
            self.clear_output()
            return
        
        # Process with UI adapter if available
        if self.ui_adapter:
            await self.ui_adapter.queue_user_message(user_input)
            await self.ui_adapter.process_queued_message()
        else:
            # Fallback - simulate processing with thinking indicator
            self.start_thinking("thinking...")
            await asyncio.sleep(1)
            self.stop_thinking()
            self.add_output("UI adapter not available. Please initialize the REPL properly.")
    
    def show_help(self) -> None:
        """Show help information."""
        help_text = """
Available commands:
  help     - Show this help message
  clear    - Clear the output
  exit     - Exit the REPL
  
Navigation:
  Up/Down  - Navigate command history
  Ctrl+C   - Clear current input
  Ctrl+D   - Exit REPL
  
Features:
  - Syntax highlighting for code blocks
  - Command history
  - Multi-line input support
  - Minimal visual differences between user and AI text
        """.strip()
        
        self.add_line("", "output")
        for line in help_text.split('\n'):
            self.add_line(line, "system")
        self.add_line("", "output")
    
    def clear_output(self) -> None:
        """Clear all output."""
        output_view = self.query_one("#repl-output")
        output_view.remove_children()
        
        # Add welcome message back
        self.add_line("Loki Code REPL v0.1.0", "system")
        self.add_line("Type 'help' for help, 'exit' to quit", "system")
        self.add_line("", "output")
    
    def action_clear_input(self) -> None:
        """Clear the current input."""
        input_widget = self.query_one("#repl-input")
        input_widget.value = ""
    
    def action_focus_input(self) -> None:
        """Focus the input field."""
        self.query_one("#repl-input").focus()
    
    def action_history_prev(self) -> None:
        """Navigate to previous command in history."""
        input_widget = self.query_one("#repl-input")
        prev_cmd = self.history.get_previous()
        if prev_cmd is not None:
            input_widget.value = prev_cmd
    
    def action_history_next(self) -> None:
        """Navigate to next command in history."""
        input_widget = self.query_one("#repl-input")
        next_cmd = self.history.get_next()
        if next_cmd is not None:
            input_widget.value = next_cmd


def create_repl_app(config, use_adapter: bool = True) -> Optional[App]:
    """Factory function to create the REPL TUI app."""
    if not TEXTUAL_AVAILABLE:
        return None
    
    app = LokiREPL(config)
    
    # Optionally integrate with UI adapter
    if use_adapter:
        try:
            from .tui_adapter import create_tui_adapter
            adapter = create_tui_adapter(config)
            app.ui_adapter = adapter
            adapter.textual_app = app
        except ImportError:
            # Fallback to no adapter
            pass
    
    return app