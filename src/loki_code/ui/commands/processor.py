"""
Main command processor for Loki Code.

This module orchestrates command parsing, routing, and execution.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

from .parser import CommandParser, ParsedInput, ConversationContext
from .router import CommandRouter, RoutingDecision, RouteType
from .shortcuts import ShortcutManager, SuggestionEngine, ShortcutResult
from ...utils.logging import get_logger


@dataclass
class ProcessedCommand:
    """Result of processing a user command."""
    success: bool
    message: str
    execution_type: str  # "direct_tool", "agent_conversation", "system_command", "shortcut"
    tool_results: List[Any] = field(default_factory=list)
    agent_response: Optional[Any] = None
    direct_tool_call: Optional[tuple] = None  # (tool_name, tool_args)
    system_command: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommandProcessor:
    """Main command processor that orchestrates natural language understanding."""
    
    def __init__(self, agent, tool_registry, session_manager=None):
        self.agent = agent
        self.tools = tool_registry
        self.session = session_manager
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.parser = CommandParser()
        self.router = CommandRouter(agent, tool_registry)
        self.shortcuts = ShortcutManager()
        self.suggestions = SuggestionEngine(session_manager, tool_registry)
        
    async def process_input(self, user_input: str, 
                          context: ConversationContext) -> ProcessedCommand:
        """Process user input and determine appropriate action."""
        
        self.logger.info(f"Processing input: {user_input[:50]}...")
        
        try:
            # Step 1: Parse the input
            parsed = self.parser.parse_input(user_input, context)
            self.logger.debug(f"Parsed intent: {parsed.intent.type.value}")
            
            # Step 2: Check for shortcuts first
            if shortcut_result := self.shortcuts.handle_shortcut(parsed):
                return await self._handle_shortcut_result(shortcut_result, parsed, context)
            
            # Step 3: Route to appropriate handler
            routing_decision = self.router.determine_route(parsed, context)
            self.logger.debug(f"Routing decision: {routing_decision.route_type.value}")
            
            # Step 4: Execute based on routing decision
            return await self._execute_routed_command(routing_decision, parsed, context)
            
        except Exception as e:
            self.logger.error(f"Error processing command: {e}", exc_info=True)
            return ProcessedCommand(
                success=False,
                message=f"Sorry, I encountered an error: {str(e)}",
                execution_type="error",
                suggestions=["Try rephrasing your request", "Type 'help' for guidance"]
            )
    
    async def _handle_shortcut_result(self, shortcut: ShortcutResult, 
                                    parsed: ParsedInput,
                                    context: ConversationContext) -> ProcessedCommand:
        """Handle a shortcut command result."""
        
        if shortcut.tool_call:
            tool_name, tool_args = shortcut.tool_call
            try:
                # Execute tool directly
                result = await self._execute_tool_direct(tool_name, tool_args)
                return ProcessedCommand(
                    success=result.get('success', True),
                    message=result.get('message', 'Tool executed successfully'),
                    execution_type="shortcut",
                    direct_tool_call=shortcut.tool_call,
                    tool_results=[result]
                )
            except Exception as e:
                return ProcessedCommand(
                    success=False,
                    message=f"Shortcut tool execution failed: {str(e)}",
                    execution_type="shortcut",
                    suggestions=[f"Try: help"]
                )
        
        elif shortcut.system_command:
            return await self._execute_system_command(shortcut.system_command, parsed, context)
        
        else:
            return ProcessedCommand(
                success=shortcut.success,
                message=shortcut.message or "Shortcut executed",
                execution_type="shortcut"
            )
    
    async def _execute_routed_command(self, routing: RoutingDecision, 
                                    parsed: ParsedInput,
                                    context: ConversationContext) -> ProcessedCommand:
        """Execute command based on routing decision."""
        
        if routing.route_type == RouteType.DIRECT_TOOL:
            return await self._execute_direct_tool(routing, parsed, context)
        
        elif routing.route_type == RouteType.AGENT_CONVERSATION:
            return await self._execute_agent_conversation(routing, parsed, context)
        
        elif routing.route_type == RouteType.SYSTEM_COMMAND:
            return await self._execute_system_command(routing.system_command, parsed, context)
        
        elif routing.route_type == RouteType.CLARIFICATION_NEEDED:
            return self._request_clarification(routing, parsed, context)
        
        else:
            return ProcessedCommand(
                success=False,
                message="Unable to understand the request",
                execution_type="unknown",
                suggestions=self.suggestions.generate_suggestions(parsed, context)
            )
    
    async def _execute_direct_tool(self, routing: RoutingDecision, 
                                 parsed: ParsedInput,
                                 context: ConversationContext) -> ProcessedCommand:
        """Execute tool directly without agent processing."""
        
        tool_name = routing.target_tool
        tool_input = self._extract_tool_input(parsed, tool_name)
        
        try:
            result = await self._execute_tool_direct(tool_name, tool_input)
            
            return ProcessedCommand(
                success=result.get('success', True),
                message=result.get('message', 'Tool executed successfully'),
                tool_results=[result],
                execution_type="direct_tool",
                direct_tool_call=(tool_name, tool_input),
                metadata={'routing_confidence': routing.confidence}
            )
        
        except Exception as e:
            return ProcessedCommand(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                execution_type="direct_tool",
                suggestions=[f"Try: help", "Check file path"]
            )
    
    async def _execute_agent_conversation(self, routing: RoutingDecision,
                                        parsed: ParsedInput, 
                                        context: ConversationContext) -> ProcessedCommand:
        """Process through intelligent agent."""
        
        try:
            # Create agent context
            from ...core.agent.types import RequestContext
            agent_context = RequestContext(
                project_path=context.project_path,
                current_file=context.current_file,
                target_files=parsed.entities.get('files', []),
                session_id=context.session_id
            )
            
            # Process with agent
            agent_response = await self.agent.process_request(
                parsed.original_text,
                context=agent_context
            )
            
            return ProcessedCommand(
                success=True,
                message=agent_response.content,
                agent_response=agent_response,
                execution_type="agent_conversation",
                metadata={
                    'routing_confidence': routing.confidence,
                    'agent_state': agent_response.state.value if hasattr(agent_response, 'state') else None
                }
            )
        
        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}", exc_info=True)
            return ProcessedCommand(
                success=False,
                message=f"I'm having trouble processing that request: {str(e)}",
                execution_type="agent_conversation",
                suggestions=["Try rephrasing your request", "Type 'help' for guidance"]
            )
    
    async def _execute_system_command(self, command: str, parsed: ParsedInput,
                                    context: ConversationContext) -> ProcessedCommand:
        """Execute system command."""
        
        if command == "help":
            help_text = self._generate_help_text()
            return ProcessedCommand(
                success=True,
                message=help_text,
                execution_type="system_command",
                system_command="help"
            )
        
        elif command == "status":
            status_text = self._generate_status_text()
            return ProcessedCommand(
                success=True,
                message=status_text,
                execution_type="system_command",
                system_command="status"
            )
        
        elif command == "clear":
            return ProcessedCommand(
                success=True,
                message="Screen cleared",
                execution_type="system_command",
                system_command="clear"
            )
        
        elif command in ["quit", "exit"]:
            return ProcessedCommand(
                success=True,
                message="Goodbye!",
                execution_type="system_command",
                system_command="quit"
            )
        
        else:
            return ProcessedCommand(
                success=False,
                message=f"Unknown system command: {command}",
                execution_type="system_command",
                suggestions=["Type 'help' for available commands"]
            )
    
    def _request_clarification(self, routing: RoutingDecision, 
                             parsed: ParsedInput,
                             context: ConversationContext) -> ProcessedCommand:
        """Request clarification from user."""
        
        questions = routing.clarification_questions or [
            "I'm not sure what you'd like me to do.",
            "Could you please be more specific?"
        ]
        
        message = "I need a bit more information:\n" + "\n".join(f"• {q}" for q in questions)
        
        return ProcessedCommand(
            success=True,
            message=message,
            execution_type="clarification_needed",
            suggestions=self.suggestions.generate_suggestions(parsed, context)
        )
    
    def _extract_tool_input(self, parsed: ParsedInput, tool_name: str) -> Dict[str, Any]:
        """Extract appropriate input for the tool."""
        
        tool_input = {}
        
        # File-based tools
        if tool_name in ['file_reader', 'file_writer']:
            if 'files' in parsed.entities and parsed.entities['files']:
                tool_input['file_path'] = parsed.entities['files'][0]
            else:
                # Try to extract file from text
                import re
                file_match = re.search(r'[^\s]+\.(py|js|ts|java|cpp|c|rs)', parsed.cleaned_text)
                if file_match:
                    tool_input['file_path'] = file_match.group()
        
        # Search tools
        elif tool_name == 'file_searcher':
            # Extract search query
            words = parsed.cleaned_text.split()
            if len(words) > 1:
                tool_input['query'] = ' '.join(words[1:])  # Skip first word (likely "find" or "search")
        
        return tool_input
    
    async def _execute_tool_direct(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool directly using the real tool system."""
        
        try:
            # Get the tool from the registry
            tool = self.tools.get_tool(tool_name)
            if not tool:
                available_tools = self.tools.list_tool_names()
                return {
                    'success': False,
                    'message': f'Tool "{tool_name}" not found. Available tools: {", ".join(available_tools)}',
                    'available_tools': available_tools
                }
            
            # Create tool context
            from ...tools.types import ToolContext
            tool_context = ToolContext(
                session_id="ui_session",
                user_id="ui_user",
                project_path=None,  # Will be set from context if available
                working_directory=None,
                security_level=tool.get_schema().security_level
            )
            
            # Execute the tool with proper error handling
            self.logger.info(f"Executing tool {tool_name} with args: {tool_args}")
            result = await self.tools.execute_tool(tool_name, tool_args, tool_context)
            
            # Convert ToolResult to dict format expected by UI
            if result.success:
                response = {
                    'success': True,
                    'message': result.message or f'Tool {tool_name} executed successfully',
                    'output': result.output,
                    'metadata': result.metadata or {},
                    'execution_time': getattr(result, 'execution_time', None)
                }
                
                # For file_reader, extract content for display
                if tool_name == "file_reader" and result.output:
                    if hasattr(result.output, 'content'):
                        response['content'] = result.output.content
                        response['file_info'] = getattr(result.output, 'file_info', None)
                        response['analysis_summary'] = getattr(result.output, 'analysis_summary', None)
                    elif isinstance(result.output, dict):
                        response['content'] = result.output.get('content', '')
                        response['file_info'] = result.output.get('file_info', {})
                        response['analysis_summary'] = result.output.get('analysis_summary', '')
                
                return response
            else:
                return {
                    'success': False,
                    'message': result.message or f'Tool {tool_name} execution failed',
                    'error': result.error_message,
                    'metadata': result.metadata or {}
                }
                
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'Error executing tool {tool_name}: {str(e)}',
                'error': str(e)
            }
    
    def _generate_help_text(self) -> str:
        """Generate help text."""
        
        return """🤖 Loki Code Help

Natural Language Commands:
• "analyze file.py" - Analyze a code file
• "explain this function" - Get code explanations  
• "create a function that..." - Generate code
• "debug this error" - Get debugging help

Quick Shortcuts:
• h or ? - Show this help
• r <file> - Read a file
• a <file> - Analyze a file
• ls - List files
• c - Clear screen
• q - Quit

Examples:
• "analyze auth.py"
• "r config.json"
• "find function login"
• "help me debug this error"

Just type naturally - I'll understand what you need!"""
    
    def _generate_status_text(self) -> str:
        """Generate status text."""
        
        tool_count = len(self.tools.list_tool_names()) if self.tools else 0
        agent_status = self.agent.get_agent_status() if hasattr(self.agent, 'get_agent_status') else {}
        
        return f"""🔧 Loki Code Status

Agent: {agent_status.get('state', 'Active')}
Tools Available: {tool_count}
Session: Active
Command Processor: Ready

Type 'help' for available commands."""