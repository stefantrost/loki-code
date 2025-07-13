"""
Simple intelligent routing logic for Loki Code commands.

This module determines how to route parsed commands - either to direct tool execution,
agent conversation, or system commands.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

from .parser import ParsedInput, IntentType, ConversationContext
from ...utils.logging import get_logger


class RouteType(Enum):
    """Types of routing decisions."""
    DIRECT_TOOL = "direct_tool"
    AGENT_CONVERSATION = "agent_conversation"  
    SYSTEM_COMMAND = "system_command"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class RoutingDecision:
    """Decision about how to route a command."""
    route_type: RouteType
    confidence: float
    target_tool: Optional[str] = None
    system_command: Optional[str] = None
    clarification_questions: List[str] = None
    reasoning: str = ""


class CommandRouter:
    """Simple intelligent routing for commands."""
    
    def __init__(self, agent, tool_registry):
        self.agent = agent
        self.tools = tool_registry
        self.logger = get_logger(__name__)
        
    def determine_route(self, parsed: ParsedInput, 
                       context: ConversationContext) -> RoutingDecision:
        """Determine how to route the parsed input."""
        
        # System commands first (highest priority)
        if parsed.intent.type == IntentType.SYSTEM_COMMAND:
            return RoutingDecision(
                route_type=RouteType.SYSTEM_COMMAND,
                system_command=self._extract_system_command(parsed),
                confidence=0.95,
                reasoning="System command detected"
            )
        
        # Help commands
        if parsed.intent.type == IntentType.HELP:
            return RoutingDecision(
                route_type=RouteType.SYSTEM_COMMAND,
                system_command="help",
                confidence=0.95,
                reasoning="Help request"
            )
        
        # Simple file operations -> direct tool
        if self._is_simple_file_operation(parsed):
            tool_name = self._identify_file_tool(parsed)
            return RoutingDecision(
                route_type=RouteType.DIRECT_TOOL,
                target_tool=tool_name,
                confidence=0.85,
                reasoning=f"Simple file operation -> {tool_name}"
            )
        
        # Low confidence -> ask for clarification
        if parsed.confidence < 0.4:
            return RoutingDecision(
                route_type=RouteType.CLARIFICATION_NEEDED,
                clarification_questions=self._generate_clarification_questions(parsed),
                confidence=0.6,
                reasoning="Low confidence, need clarification"
            )
        
        # Everything else -> agent conversation
        return RoutingDecision(
            route_type=RouteType.AGENT_CONVERSATION,
            confidence=0.7,
            reasoning="Complex request, routing to agent"
        )
    
    def _is_simple_file_operation(self, parsed: ParsedInput) -> bool:
        """Check if this is a simple file operation."""
        
        # Simple patterns for direct tool execution
        simple_patterns = [
            r'^\s*(read|analyze)\s+[\w./]+\.(py|js|ts|java)',
            r'^\s*list\s+(files|directory)',
            r'^\s*show\s+.*\.(py|js|ts)'
        ]
        
        import re
        for pattern in simple_patterns:
            if re.match(pattern, parsed.cleaned_text, re.IGNORECASE):
                return True
        
        # Check if we have file entities and simple intent
        has_files = 'files' in parsed.entities and len(parsed.entities['files']) == 1
        simple_intent = parsed.intent.type in [IntentType.FILE_ANALYSIS, IntentType.SEARCH]
        
        return has_files and simple_intent and len(parsed.cleaned_text.split()) <= 4
    
    def _identify_file_tool(self, parsed: ParsedInput) -> str:
        """Identify which file tool to use."""
        
        if parsed.intent.type == IntentType.FILE_ANALYSIS:
            return "file_reader"
        elif parsed.intent.type == IntentType.SEARCH:
            return "file_searcher" 
        else:
            return "file_reader"  # Default
    
    def _extract_system_command(self, parsed: ParsedInput) -> str:
        """Extract the system command from parsed input."""
        
        # Simple command extraction
        first_word = parsed.cleaned_text.split()[0].lower()
        
        command_map = {
            'help': 'help',
            'h': 'help', 
            '?': 'help',
            'status': 'status',
            'clear': 'clear',
            'reset': 'reset',
            'quit': 'quit',
            'exit': 'quit',
            'ls': 'list_files',
            'pwd': 'current_directory'
        }
        
        return command_map.get(first_word, first_word)
    
    def _generate_clarification_questions(self, parsed: ParsedInput) -> List[str]:
        """Generate helpful clarification questions."""
        
        questions = []
        
        if not parsed.entities:
            questions.append("What file would you like me to work with?")
            questions.append("What specific task can I help you with?")
        
        if parsed.intent.type == IntentType.CONVERSATION:
            questions.extend([
                "Are you looking to analyze code, generate code, or get help?",
                "Type 'help' to see available commands"
            ])
        
        return questions[:3]  # Limit to 3 questions