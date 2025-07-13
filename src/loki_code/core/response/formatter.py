"""
Response formatting and presentation for Loki Code.

This module provides rich formatting capabilities for agent responses,
tool results, and conversation elements with support for terminal output,
markdown, and structured presentation.
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.rule import Rule
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .parser import ParsedResponse, ValidatedToolCall
from ...tools.types import ToolResult
from ...utils.logging import get_logger


class FormatStyle(Enum):
    """Response formatting styles."""
    RICH = "rich"           # Rich console formatting
    MARKDOWN = "markdown"   # Markdown formatting
    PLAIN = "plain"         # Plain text
    COLORED = "colored"     # ANSI colored text
    JSON = "json"           # JSON structured output


class ContentType(Enum):
    """Types of content to format."""
    AGENT_RESPONSE = "agent_response"
    TOOL_RESULT = "tool_result"
    ERROR_MESSAGE = "error_message"
    STATUS_UPDATE = "status_update"
    CONVERSATION_TURN = "conversation_turn"
    SYSTEM_MESSAGE = "system_message"
    CODE_SNIPPET = "code_snippet"
    FILE_CONTENT = "file_content"


@dataclass
class ResponseSection:
    """A section of formatted response."""
    title: str
    content: str
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    style_hints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1=highest, 10=lowest


@dataclass
class FormattedResponse:
    """Complete formatted response with all sections."""
    sections: List[ResponseSection] = field(default_factory=list)
    summary: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    style: FormatStyle = FormatStyle.RICH
    total_length: int = 0
    rendering_hints: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: ResponseSection):
        """Add a response section."""
        self.sections.append(section)
        self.total_length += len(section.content)
    
    def get_sections_by_type(self, content_type: ContentType) -> List[ResponseSection]:
        """Get all sections of a specific type."""
        return [s for s in self.sections if s.content_type == content_type]
    
    def get_plain_text(self) -> str:
        """Get all content as plain text."""
        return "\n\n".join(section.content for section in self.sections)


@dataclass
class FormattingConfig:
    """Configuration for response formatting."""
    style: FormatStyle = FormatStyle.RICH
    show_timestamps: bool = True
    show_confidence: bool = True
    show_reasoning: bool = True
    show_tool_details: bool = True
    show_metadata: bool = False
    max_text_length: int = 1000
    max_code_length: int = 500
    use_colors: bool = True
    use_rich_formatting: bool = True
    compact_mode: bool = False
    show_section_separators: bool = True
    highlight_errors: bool = True
    highlight_warnings: bool = True
    code_theme: str = "monokai"
    width: Optional[int] = None


class ResponseFormatter:
    """
    Formatter for agent responses with rich terminal output support.
    
    Handles formatting of agent responses, tool results, errors, and
    conversation elements with support for multiple output styles.
    """
    
    def __init__(self, config: Optional[FormattingConfig] = None):
        self.config = config or FormattingConfig()
        self.logger = get_logger(__name__)
        
        # Initialize console if Rich is available
        self.console = None
        if RICH_AVAILABLE and self.config.use_rich_formatting:
            self.console = Console(
                width=self.config.width,
                force_terminal=True,
                color_system="auto"
            )
        
        # Color codes for non-Rich formatting
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
        }
    
    def format_agent_response(self, parsed_response: ParsedResponse, 
                            tool_results: Optional[List[ToolResult]] = None) -> FormattedResponse:
        """Format a complete agent response with tool results.
        
        Args:
            parsed_response: Parsed LLM response
            tool_results: Results from tool executions
            
        Returns:
            FormattedResponse ready for display
        """
        formatted = FormattedResponse(style=self.config.style)
        
        # Add reasoning section if available
        if parsed_response.reasoning and self.config.show_reasoning:
            formatted.add_section(ResponseSection(
                title="Agent Reasoning",
                content=self._format_reasoning(parsed_response.reasoning),
                content_type=ContentType.AGENT_RESPONSE,
                style_hints={"icon": "🤔", "color": "cyan"}
            ))
        
        # Add main response text
        if parsed_response.text_content:
            formatted.add_section(ResponseSection(
                title="Response",
                content=self._format_text_content(parsed_response.text_content),
                content_type=ContentType.AGENT_RESPONSE,
                style_hints={"icon": "💬", "color": "white"}
            ))
        
        # Add tool call sections
        if parsed_response.tool_calls:
            for tool_call in parsed_response.tool_calls:
                if tool_call.is_valid:
                    formatted.add_section(self._format_tool_call(tool_call))
        
        # Add tool results
        if tool_results:
            for result in tool_results:
                formatted.add_section(self._format_tool_result(result))
        
        # Add confidence and metadata if requested
        if self.config.show_confidence:
            confidence_text = f"Confidence: {parsed_response.confidence_score:.1%}"
            if parsed_response.needs_clarification:
                confidence_text += " (needs clarification)"
            
            formatted.add_section(ResponseSection(
                title="Analysis",
                content=confidence_text,
                content_type=ContentType.STATUS_UPDATE,
                style_hints={"icon": "📊", "color": "yellow"}
            ))
        
        # Add errors if any
        if parsed_response.parsing_errors:
            for error in parsed_response.parsing_errors:
                formatted.add_section(ResponseSection(
                    title="Error",
                    content=error,
                    content_type=ContentType.ERROR_MESSAGE,
                    style_hints={"icon": "❌", "color": "red"}
                ))
        
        # Add clarification questions
        if parsed_response.clarification_questions:
            questions_text = "\n".join(f"• {q}" for q in parsed_response.clarification_questions)
            formatted.add_section(ResponseSection(
                title="Questions for Clarification",
                content=questions_text,
                content_type=ContentType.AGENT_RESPONSE,
                style_hints={"icon": "❓", "color": "yellow"}
            ))
        
        return formatted
    
    def format_tool_result(self, result: ToolResult) -> FormattedResponse:
        """Format a single tool result."""
        formatted = FormattedResponse(style=self.config.style)
        formatted.add_section(self._format_tool_result(result))
        return formatted
    
    def format_error(self, error: str, error_type: str = "Error") -> FormattedResponse:
        """Format an error message."""
        formatted = FormattedResponse(style=self.config.style)
        formatted.add_section(ResponseSection(
            title=error_type,
            content=error,
            content_type=ContentType.ERROR_MESSAGE,
            style_hints={"icon": "❌", "color": "red", "highlight": True}
        ))
        return formatted
    
    def format_status(self, message: str, status_type: str = "Status") -> FormattedResponse:
        """Format a status message."""
        formatted = FormattedResponse(style=self.config.style)
        formatted.add_section(ResponseSection(
            title=status_type,
            content=message,
            content_type=ContentType.STATUS_UPDATE,
            style_hints={"icon": "ℹ️", "color": "blue"}
        ))
        return formatted
    
    def render_to_console(self, formatted_response: FormattedResponse) -> str:
        """Render formatted response to console output.
        
        Args:
            formatted_response: Response to render
            
        Returns:
            Rendered string ready for console output
        """
        if self.config.style == FormatStyle.RICH and self.console:
            return self._render_rich(formatted_response)
        elif self.config.style == FormatStyle.MARKDOWN:
            return self._render_markdown(formatted_response)
        elif self.config.style == FormatStyle.COLORED:
            return self._render_colored(formatted_response)
        elif self.config.style == FormatStyle.JSON:
            return self._render_json(formatted_response)
        else:
            return self._render_plain(formatted_response)
    
    def print_to_console(self, formatted_response: FormattedResponse):
        """Print formatted response directly to console."""
        if self.config.style == FormatStyle.RICH and self.console:
            self._print_rich(formatted_response)
        else:
            output = self.render_to_console(formatted_response)
            print(output)
    
    def _format_reasoning(self, reasoning: str) -> str:
        """Format reasoning text."""
        if len(reasoning) > self.config.max_text_length:
            reasoning = reasoning[:self.config.max_text_length] + "..."
        return reasoning.strip()
    
    def _format_text_content(self, content: str) -> str:
        """Format main text content."""
        if len(content) > self.config.max_text_length:
            content = content[:self.config.max_text_length] + "..."
        return content.strip()
    
    def _format_tool_call(self, tool_call: ValidatedToolCall) -> ResponseSection:
        """Format a tool call for display."""
        if tool_call.is_valid:
            call_text = f"Tool: {tool_call.raw_call.tool_name}"
            if self.config.show_tool_details:
                input_str = json.dumps(tool_call.raw_call.input_data, indent=2)
                if len(input_str) > self.config.max_code_length:
                    input_str = input_str[:self.config.max_code_length] + "..."
                call_text += f"\nInput: {input_str}"
            
            return ResponseSection(
                title="Tool Call",
                content=call_text,
                content_type=ContentType.AGENT_RESPONSE,
                style_hints={"icon": "🔧", "color": "green"}
            )
        else:
            error_text = f"Invalid tool call: {tool_call.raw_call.tool_name}"
            if tool_call.error:
                error_text += f"\nError: {tool_call.error}"
            
            return ResponseSection(
                title="Tool Call Error",
                content=error_text,
                content_type=ContentType.ERROR_MESSAGE,
                style_hints={"icon": "⚠️", "color": "red"}
            )
    
    def _format_tool_result(self, result: ToolResult) -> ResponseSection:
        """Format a tool execution result."""
        if result.success:
            content = str(result.content)
            if len(content) > self.config.max_text_length:
                content = content[:self.config.max_text_length] + "..."
            
            return ResponseSection(
                title=f"Tool Result: {result.tool_name}",
                content=content,
                content_type=ContentType.TOOL_RESULT,
                style_hints={"icon": "✅", "color": "green"},
                metadata={"execution_time": result.execution_time}
            )
        else:
            error_text = f"Tool execution failed: {result.error}"
            if result.error_details:
                error_text += f"\nDetails: {result.error_details}"
            
            return ResponseSection(
                title=f"Tool Error: {result.tool_name}",
                content=error_text,
                content_type=ContentType.ERROR_MESSAGE,
                style_hints={"icon": "❌", "color": "red"}
            )
    
    def _render_rich(self, formatted_response: FormattedResponse) -> str:
        """Render using Rich console formatting."""
        if not self.console:
            return self._render_plain(formatted_response)
        
        # Capture console output
        with self.console.capture() as capture:
            self._print_rich(formatted_response)
        
        return capture.get()
    
    def _print_rich(self, formatted_response: FormattedResponse):
        """Print using Rich console formatting."""
        if not self.console:
            return
        
        # Print timestamp if requested
        if self.config.show_timestamps:
            timestamp = time.strftime("%H:%M:%S", time.localtime(formatted_response.timestamp))
            self.console.print(f"[dim]{timestamp}[/dim]")
        
        # Print each section
        for i, section in enumerate(formatted_response.sections):
            if i > 0 and self.config.show_section_separators:
                self.console.print()
            
            self._print_rich_section(section)
        
        # Print summary if available
        if formatted_response.summary:
            self.console.print()
            self.console.print(Rule("Summary"))
            self.console.print(formatted_response.summary)
    
    def _print_rich_section(self, section: ResponseSection):
        """Print a single section using Rich."""
        if not self.console:
            return
        
        # Get style hints
        icon = section.style_hints.get("icon", "")
        color = section.style_hints.get("color", "white")
        highlight = section.style_hints.get("highlight", False)
        
        # Create title with icon
        title = f"{icon} {section.title}" if icon else section.title
        
        # Format content based on type
        if section.content_type == ContentType.CODE_SNIPPET:
            # Syntax highlighting for code
            syntax = Syntax(section.content, "python", theme=self.config.code_theme)
            self.console.print(Panel(syntax, title=title, border_style=color))
        elif section.content_type == ContentType.ERROR_MESSAGE:
            # Error highlighting
            self.console.print(Panel(
                section.content,
                title=title,
                border_style="red",
                style="red" if highlight else None
            ))
        elif section.content_type == ContentType.TOOL_RESULT:
            # Tool result with metadata
            content = section.content
            if section.metadata.get("execution_time"):
                content += f"\n[dim]Executed in {section.metadata['execution_time']:.3f}s[/dim]"
            
            self.console.print(Panel(content, title=title, border_style=color))
        else:
            # Regular content
            if self.config.compact_mode:
                self.console.print(f"[{color}]{title}[/{color}]: {section.content}")
            else:
                self.console.print(Panel(section.content, title=title, border_style=color))
    
    def _render_markdown(self, formatted_response: FormattedResponse) -> str:
        """Render as markdown."""
        lines = []
        
        if self.config.show_timestamps:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(formatted_response.timestamp))
            lines.append(f"*{timestamp}*\n")
        
        for section in formatted_response.sections:
            icon = section.style_hints.get("icon", "")
            title = f"{icon} {section.title}" if icon else section.title
            
            lines.append(f"## {title}\n")
            
            if section.content_type == ContentType.CODE_SNIPPET:
                lines.append(f"```python\n{section.content}\n```\n")
            else:
                lines.append(f"{section.content}\n")
        
        if formatted_response.summary:
            lines.append(f"## Summary\n\n{formatted_response.summary}\n")
        
        return "\n".join(lines)
    
    def _render_colored(self, formatted_response: FormattedResponse) -> str:
        """Render with ANSI colors."""
        lines = []
        
        if self.config.show_timestamps:
            timestamp = time.strftime("%H:%M:%S", time.localtime(formatted_response.timestamp))
            lines.append(f"{self.colors['dim']}{timestamp}{self.colors['reset']}")
        
        for section in formatted_response.sections:
            icon = section.style_hints.get("icon", "")
            color = section.style_hints.get("color", "white")
            color_code = self.colors.get(color, self.colors['white'])
            
            title = f"{icon} {section.title}" if icon else section.title
            lines.append(f"\n{color_code}{self.colors['bold']}{title}{self.colors['reset']}")
            lines.append(section.content)
        
        if formatted_response.summary:
            lines.append(f"\n{self.colors['bold']}Summary{self.colors['reset']}")
            lines.append(formatted_response.summary)
        
        return "\n".join(lines)
    
    def _render_plain(self, formatted_response: FormattedResponse) -> str:
        """Render as plain text."""
        lines = []
        
        if self.config.show_timestamps:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(formatted_response.timestamp))
            lines.append(f"{timestamp}\n")
        
        for section in formatted_response.sections:
            lines.append(f"=== {section.title} ===")
            lines.append(section.content)
            lines.append("")
        
        if formatted_response.summary:
            lines.append("=== Summary ===")
            lines.append(formatted_response.summary)
        
        return "\n".join(lines)
    
    def _render_json(self, formatted_response: FormattedResponse) -> str:
        """Render as structured JSON."""
        data = {
            "timestamp": formatted_response.timestamp,
            "style": formatted_response.style.value,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "type": section.content_type.value,
                    "metadata": section.metadata,
                    "priority": section.priority
                }
                for section in formatted_response.sections
            ],
            "summary": formatted_response.summary,
            "total_length": formatted_response.total_length
        }
        
        return json.dumps(data, indent=2)
    
    def create_progress_display(self, description: str) -> Optional[Any]:
        """Create a progress display for long-running operations."""
        if self.console and RICH_AVAILABLE:
            return Status(description, console=self.console)
        return None
    
    def update_progress_display(self, progress_display: Any, description: str):
        """Update progress display with new description."""
        if progress_display and hasattr(progress_display, 'update'):
            progress_display.update(description)