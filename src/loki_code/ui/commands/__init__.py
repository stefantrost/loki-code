"""
Intelligent command processing system for Loki Code.

This module provides natural language command interpretation, smart routing,
and context-aware suggestions for user interactions with the AI coding assistant.

Key Components:
- CommandProcessor: Main orchestrator for command processing
- CommandParser: Natural language parsing and intent recognition  
- CommandRouter: Intelligent routing logic for commands
- ShortcutManager: Quick command shortcuts and aliases
- SuggestionEngine: Context-aware command suggestions

Usage:
    from loki_code.ui.commands import CommandProcessor
    
    processor = CommandProcessor(agent, tool_registry, session)
    result = await processor.process_input("analyze auth.py", context)
"""

from .processor import (
    CommandProcessor,
    ProcessedCommand,
    ConversationContext
)

from .parser import (
    CommandParser,
    ParsedInput,
    Intent,
    IntentType
)

from .router import (
    CommandRouter,
    RoutingDecision,
    RouteType
)

from .shortcuts import (
    ShortcutManager,
    SuggestionEngine
)

__all__ = [
    # Main processor
    "CommandProcessor",
    "ProcessedCommand", 
    "ConversationContext",
    
    # Parser components
    "CommandParser",
    "ParsedInput",
    "Intent",
    "IntentType",
    
    # Router components
    "CommandRouter",
    "RoutingDecision",
    "RouteType",
    
    # Shortcuts and suggestions
    "ShortcutManager",
    "SuggestionEngine"
]

# Version info
__version__ = "0.1.0"

# Command processing information
COMMAND_SYSTEM_INFO = {
    "version": __version__,
    "features": [
        "natural_language_parsing",
        "intent_recognition",
        "smart_routing",
        "context_awareness",
        "shortcut_commands",
        "suggestion_engine"
    ],
    "supported_intents": [
        "file_analysis",
        "code_generation", 
        "code_explanation",
        "debugging",
        "refactoring",
        "tool_execution",
        "system_command",
        "conversation"
    ]
}


def get_command_system_info() -> dict:
    """Get information about the command processing system.
    
    Returns:
        Dictionary with command system information
    """
    return COMMAND_SYSTEM_INFO.copy()