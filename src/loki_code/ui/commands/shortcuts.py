"""
Simple shortcut commands and suggestions for Loki Code.

This module provides quick command shortcuts and context-aware suggestions.
"""

import re
from typing import Optional, Dict, Callable, List, Match
from dataclasses import dataclass

from .parser import ParsedInput, ConversationContext
from ...utils.logging import get_logger


@dataclass
class ShortcutResult:
    """Result from a shortcut command."""
    success: bool
    tool_call: Optional[tuple] = None  # (tool_name, tool_args)
    message: Optional[str] = None
    system_command: Optional[str] = None


class ShortcutManager:
    """Handles quick command shortcuts."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.shortcuts = self._load_shortcuts()
        self.pattern_shortcuts = self._load_pattern_shortcuts()
        
    def handle_shortcut(self, parsed: ParsedInput) -> Optional[ShortcutResult]:
        """Handle shortcut commands."""
        
        text = parsed.cleaned_text.strip().lower()
        
        # Exact shortcuts
        if text in self.shortcuts:
            return self.shortcuts[text]()
        
        # Pattern-based shortcuts
        for pattern, handler in self.pattern_shortcuts.items():
            if match := re.match(pattern, text):
                return handler(match)
        
        return None
    
    def _load_shortcuts(self) -> Dict[str, Callable]:
        """Load exact match shortcuts."""
        
        return {
            'h': lambda: ShortcutResult(True, system_command="help"),
            '?': lambda: ShortcutResult(True, system_command="help"),
            'q': lambda: ShortcutResult(True, system_command="quit"),
            'c': lambda: ShortcutResult(True, system_command="clear"),
            's': lambda: ShortcutResult(True, system_command="status"),
            'ls': lambda: ShortcutResult(True, tool_call=("directory_lister", {})),
            'pwd': lambda: ShortcutResult(True, system_command="current_directory"),
        }
    
    def _load_pattern_shortcuts(self) -> Dict[str, Callable]:
        """Load pattern-based shortcuts."""
        
        return {
            r'^r\s+(.+)$': self._read_file_shortcut,
            r'^a\s+(.+)$': self._analyze_file_shortcut,
            r'^find\s+(.+)$': self._find_shortcut,
        }
    
    def _read_file_shortcut(self, match: Match) -> ShortcutResult:
        """Quick file reading: 'r filename.py'"""
        filename = match.group(1).strip()
        return ShortcutResult(
            success=True,
            tool_call=('file_reader', {'file_path': filename})
        )
    
    def _analyze_file_shortcut(self, match: Match) -> ShortcutResult:
        """Quick file analysis: 'a filename.py'"""
        filename = match.group(1).strip()
        return ShortcutResult(
            success=True,
            tool_call=('file_reader', {'file_path': filename, 'analyze': True})
        )
    
    def _find_shortcut(self, match: Match) -> ShortcutResult:
        """Quick search: 'find pattern'"""
        query = match.group(1).strip()
        return ShortcutResult(
            success=True,
            tool_call=('file_searcher', {'query': query})
        )


class SuggestionEngine:
    """Generates context-aware suggestions."""
    
    def __init__(self, session_manager=None, tool_registry=None):
        self.session = session_manager
        self.tools = tool_registry
        self.logger = get_logger(__name__)
        
    def generate_suggestions(self, parsed: ParsedInput, 
                           context: ConversationContext) -> List[str]:
        """Generate helpful suggestions."""
        
        suggestions = []
        
        # Recent files suggestions
        if context.recent_files and not parsed.entities.get('files'):
            recent = context.recent_files[:2]
            suggestions.extend([f"analyze {file}" for file in recent])
        
        # Intent-based suggestions
        if parsed.intent.type.value == "conversation":
            suggestions.extend([
                "help - Show available commands",
                "status - Show system status", 
                "r <filename> - Read a file",
                "a <filename> - Analyze a file"
            ])
        
        # Common shortcuts
        suggestions.extend([
            "h - Help",
            "ls - List files",
            "c - Clear screen"
        ])
        
        return suggestions[:5]  # Limit to 5 suggestions