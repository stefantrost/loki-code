"""
Intelligent routing logic for Loki Code commands.

This module determines how to route parsed commands - either to direct tool execution,
agent conversation, or system commands. Uses NLP-based understanding instead of regex patterns.
"""

from typing import Optional, List, Dict, Any

from .types import ParsedInput, IntentType, ConversationContext, RouteType, RoutingDecision
from ...utils.logging import get_logger


class CommandRouter:
    """Simple intelligent routing for commands."""

    def __init__(self, agent, tool_registry):
        self.agent = agent
        self.tools = tool_registry
        self.logger = get_logger(__name__)

    def determine_route(
        self, 
        parsed: ParsedInput, 
        context: ConversationContext, 
        routing_hints: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Determine how to route the parsed input."""

        # System commands first (highest priority)
        if parsed.intent.type == IntentType.SYSTEM_COMMAND:
            return RoutingDecision(
                route_type=RouteType.SYSTEM_COMMAND,
                system_command=self._extract_system_command(parsed),
                confidence=0.95,
                reasoning="System command detected",
            )

        # Help commands
        if parsed.intent.type == IntentType.HELP:
            return RoutingDecision(
                route_type=RouteType.SYSTEM_COMMAND,
                system_command="help",
                confidence=0.95,
                reasoning="Help request",
            )

        # Simple file operations -> direct tool
        if self._is_simple_file_operation(parsed, routing_hints):
            tool_name = self._identify_file_tool(parsed, routing_hints)
            
            # Apply context confidence boost
            base_confidence = 0.85
            if routing_hints and routing_hints.get("confidence_boost", 0) > 0:
                base_confidence += routing_hints["confidence_boost"]
                base_confidence = min(base_confidence, 0.95)  # Cap at 0.95
            
            return RoutingDecision(
                route_type=RouteType.DIRECT_TOOL,
                target_tool=tool_name,
                confidence=base_confidence,
                reasoning=f"Simple file operation -> {tool_name}" + 
                         (f" (context boost: +{routing_hints.get('confidence_boost', 0)})" if routing_hints else ""),
            )

        # Low confidence -> ask for clarification
        if parsed.confidence < 0.6:
            return RoutingDecision(
                route_type=RouteType.CLARIFICATION_NEEDED,
                clarification_questions=self._generate_clarification_questions(parsed),
                confidence=0.6,
                reasoning="Low confidence, need clarification",
            )

        # Everything else -> agent conversation
        return RoutingDecision(
            route_type=RouteType.AGENT_CONVERSATION,
            confidence=0.7,
            reasoning="Complex request, routing to agent",
        )

    def _is_simple_file_operation(self, parsed: ParsedInput, routing_hints: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this is a simple file operation using NLP understanding."""
        
        # Check for clear intent types that map to direct tools
        simple_intents = [
            IntentType.FILE_ANALYSIS,
            IntentType.SEARCH,
            IntentType.CODE_GENERATION,
        ]
        
        if parsed.intent.type not in simple_intents:
            return False
        
        # High confidence single-intent operations
        if parsed.intent.confidence >= 0.8:
            # File analysis with clear file target
            if parsed.intent.type == IntentType.FILE_ANALYSIS and parsed.entities.get("files"):
                return True
            
            # Code generation with clear target
            if parsed.intent.type == IntentType.CODE_GENERATION and (
                parsed.entities.get("files") or 
                parsed.entities.get("create_targets") or
                "empty" in parsed.cleaned_text.lower() or
                # Contextual file operations (referring to recently created files)
                any(phrase in parsed.cleaned_text.lower() for phrase in [
                    "in that file", "in the file", "to that file", "to the file"
                ]) or
                # Context hints suggest file context is available
                (routing_hints and routing_hints.get("file_context"))
            ):
                return True
            
            # Search with clear query
            if parsed.intent.type == IntentType.SEARCH:
                return True
        
        # Simple commands (short and focused)
        word_count = len(parsed.cleaned_text.split())
        if word_count <= 4 and parsed.intent.confidence >= 0.7:
            return any(parsed.entities.get(key, []) for key in ["files", "functions", "classes"])
        
        return False

    def _identify_file_tool(self, parsed: ParsedInput, routing_hints: Optional[Dict[str, Any]] = None) -> str:
        """Identify which file tool to use based on NLP understanding."""

        if parsed.intent.type == IntentType.FILE_ANALYSIS:
            return "file_reader"
        
        elif parsed.intent.type == IntentType.SEARCH:
            return "file_searcher"
        
        elif parsed.intent.type == IntentType.CODE_GENERATION:
            text_lower = parsed.cleaned_text.lower()
            
            # Check routing hints for suggested tool
            if routing_hints and routing_hints.get("suggested_tool"):
                return routing_hints["suggested_tool"]
            
            # Contextual file operations (writing to existing files)
            if any(phrase in text_lower for phrase in [
                "in that file", "in the file", "to that file", "to the file"
            ]):
                return "file_writer"
            
            # Use NLP entities and keywords to determine if this is file creation
            creation_indicators = [
                "create" in text_lower,
                "write" in text_lower,
                "make" in text_lower,
                "generate" in text_lower,
                "empty" in text_lower,
                bool(parsed.entities.get("create_targets")),
                "file" in text_lower,
            ]
            
            # Implementation/addition indicators (writing code)
            implementation_indicators = [
                "implement" in text_lower,
                "add" in text_lower,
                "build" in text_lower,
                "develop" in text_lower,
                "code" in text_lower,
            ]
            
            # If multiple creation indicators, likely file creation
            if sum(creation_indicators) >= 2:
                return "file_writer"
            
            # If implementation indicators, likely code writing
            if any(implementation_indicators):
                return "file_writer"
            
            # Check for specific file targets
            if parsed.entities.get("files"):
                return "file_writer"
            
            # Default to file_writer for code generation
            return "file_writer"
        
        else:
            return "file_reader"  # Default

    def _extract_system_command(self, parsed: ParsedInput) -> str:
        """Extract the system command from parsed input."""

        # Simple command extraction
        first_word = parsed.cleaned_text.split()[0].lower()

        command_map = {
            "help": "help",
            "h": "help",
            "?": "help",
            "status": "status",
            "clear": "clear",
            "reset": "reset",
            "quit": "quit",
            "exit": "quit",
            "ls": "list_files",
            "pwd": "current_directory",
        }

        return command_map.get(first_word, first_word)

    def _generate_clarification_questions(self, parsed: ParsedInput) -> List[str]:
        """Generate helpful clarification questions."""

        questions = []

        if not parsed.entities:
            questions.append("What file would you like me to work with?")
            questions.append("What specific task can I help you with?")

        if parsed.intent.type == IntentType.CONVERSATION:
            questions.extend(
                [
                    "Are you looking to analyze code, generate code, or get help?",
                    "Type 'help' to see available commands",
                ]
            )

        return questions[:3]  # Limit to 3 questions
