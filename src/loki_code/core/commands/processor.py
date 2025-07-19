"""
Main command processor for Loki Code.

This module orchestrates command parsing, routing, and execution using NLP-based understanding.
"""

from typing import Optional, List, Dict, Any, Union

from .nlp_parser import NLPCommandParser
from .router import CommandRouter
from .confirmation_manager import ConfirmationManager, ConfirmationStatus
from .unified_context_manager import UnifiedContextManager
from .types import OperationType
from .types import ParsedInput, ConversationContext, RoutingDecision, RouteType, ProcessedCommand, IntentType
from ...utils.logging import get_logger


class CommandProcessor:
    """Main command processor that orchestrates natural language understanding."""

    def __init__(self, agent: Any, tool_registry: Any, session_manager: Any = None, model_manager: Any = None):
        self.agent = agent
        self.tools = tool_registry
        self.session = session_manager
        self.model_manager = model_manager
        self.logger = get_logger(__name__)

        # Initialize components - use NLPCommandParser with model_manager
        self.parser = NLPCommandParser(model_manager=model_manager)
        self.router = CommandRouter(agent, tool_registry)
        self.confirmation_manager = ConfirmationManager()
        
        # Unified context manager (consolidates all context management)
        self.unified_context = UnifiedContextManager()

    async def process_input(
        self, user_input: str, context: ConversationContext
    ) -> ProcessedCommand:
        """Process user input and determine appropriate action."""

        self.logger.info(f"Processing input: {user_input[:50]}...")

        try:
            # Step 0: Clean up expired confirmations
            self.confirmation_manager.cleanup_expired_confirmations()
            
            # Step 1: Check if this is a confirmation response
            pending_confirmation, confirmation_response = self.confirmation_manager.handle_confirmation_response(
                context.session_id, user_input
            )
            
            if confirmation_response:
                # This is a confirmation response
                return await self._handle_confirmation_response(
                    pending_confirmation, confirmation_response, context
                )
            
            # Step 2: Parse the input (if not a confirmation) - now using LLM strategies
            parsed = await self.parser.parse_input(user_input, context)
            self.logger.debug(f"Parsed intent: {parsed.intent.type.value}")

            # Step 3: Check for follow-up requests
            if parsed.intent.type == IntentType.FOLLOW_UP:
                return await self._handle_follow_up_request(parsed, context)

            # Step 4: Get context-aware routing hints
            routing_hints = self.unified_context.get_contextual_routing_hints(
                context.session_id, user_input
            )
            
            # Step 5: Route to appropriate handler with context hints
            routing_decision = self.router.determine_route(parsed, context, routing_hints)
            self.logger.debug(f"Routing decision: {routing_decision.route_type.value}")

            # Step 6: Execute based on routing decision
            result = await self._execute_routed_command(routing_decision, parsed, context)
            
            # Step 7: Record successful operations in both context managers
            if result.success and result.execution_type == "direct_tool" and result.direct_tool_call:
                self._record_operation(result, parsed, context)
                
                # Record in unified context manager
                tool_name, tool_input = result.direct_tool_call
                self.unified_context.record_operation(
                    session_id=context.session_id,
                    operation_type=self._get_operation_type(parsed.intent.type, tool_name),
                    tool_name=tool_name,
                    tool_input=tool_input,
                    result=result,
                    user_input=user_input,
                    project_path=context.project_path,
                    user_id=getattr(context, 'user_id', None)
                )
            
            # Step 8: Record conversation turn
            self.unified_context.record_conversation_turn(
                session_id=context.session_id,
                user_input=user_input,
                system_response=result.message,
                intent_type=parsed.intent.type.value,
                entities=parsed.entities,
                tools_used=[result.direct_tool_call[0]] if result.direct_tool_call else [],
                files_affected=[result.direct_tool_call[1].get("file_path")] if result.direct_tool_call and result.direct_tool_call[1].get("file_path") else [],
                success=result.success
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Error processing command: {e}", exc_info=True)
            return ProcessedCommand(
                success=False,
                message=f"Sorry, I encountered an error: {str(e)}",
                execution_type="error",
                suggestions=["Try rephrasing your request", "Type 'help' for guidance"],
            )

    async def _handle_confirmation_response(
        self, 
        pending_confirmation, 
        confirmation_response, 
        context: ConversationContext
    ) -> ProcessedCommand:
        """Handle a confirmation response and execute the pending operation."""
        
        if not pending_confirmation:
            return ProcessedCommand(
                success=False,
                message="No pending operation to confirm.",
                execution_type="confirmation_error",
                suggestions=["Try your request again"],
            )
        
        # Handle denial
        if confirmation_response.response_type in ["deny", "never"]:
            return ProcessedCommand(
                success=True,
                message="Operation cancelled.",
                execution_type="confirmation_denied",
            )
        
        # Handle confirmation - execute the pending tool
        try:
            tool_name = pending_confirmation.tool_name
            tool_input = pending_confirmation.input_data
            tool_context = pending_confirmation.context
            
            # Execute the tool directly
            result = await self._execute_tool_raw(tool_name, tool_input, context)
            
            # Add confirmation metadata and enhance message formatting
            if isinstance(result, dict):
                result_message = result.get("message", "Operation completed successfully")
                
                # Enhance message with file path information
                if tool_name == "file_writer" and result.get("success", True):
                    file_path = result.get("metadata", {}).get("file_path") or tool_input.get("file_path")
                    if file_path:
                        result_message = f"✅ Created file: {file_path}"
                
                if confirmation_response.remember_choice:
                    result_message += f"\n✅ Choice remembered for future {tool_name} operations"
                
                return ProcessedCommand(
                    success=result.get("success", True),
                    message=result_message,
                    execution_type="confirmed_tool",
                    tool_results=[result],
                    direct_tool_call=(tool_name, tool_input),
                    metadata={
                        "confirmation_id": pending_confirmation.confirmation_id,
                        "response_type": confirmation_response.response_type,
                        "remembered": confirmation_response.remember_choice,
                    },
                )
            else:
                return ProcessedCommand(
                    success=True,
                    message="Operation completed successfully after confirmation",
                    execution_type="confirmed_tool",
                    metadata={
                        "confirmation_id": pending_confirmation.confirmation_id,
                        "response_type": confirmation_response.response_type,
                    },
                )
                
        except Exception as e:
            self.logger.error(f"Error executing confirmed operation: {e}", exc_info=True)
            return ProcessedCommand(
                success=False,
                message=f"Failed to execute confirmed operation: {str(e)}",
                execution_type="confirmation_error",
                suggestions=["Try the operation again"],
            )

    async def _execute_routed_command(
        self, routing: RoutingDecision, parsed: ParsedInput, context: ConversationContext
    ) -> ProcessedCommand:
        """Execute command based on routing decision."""

        if routing.route_type == RouteType.DIRECT_TOOL:
            return await self._execute_direct_tool(routing, parsed, context)

        elif routing.route_type == RouteType.AGENT_CONVERSATION:
            return await self._execute_agent_conversation(routing, parsed, context)

        elif routing.route_type == RouteType.SYSTEM_COMMAND:
            system_command = routing.system_command or "help"
            return await self._execute_system_command(system_command, parsed, context)

        elif routing.route_type == RouteType.CLARIFICATION_NEEDED:
            return self._request_clarification(routing, parsed, context)

        else:
            return ProcessedCommand(
                success=False,
                message="Unable to understand the request",
                execution_type="unknown",
                suggestions=["Try rephrasing your request", "Type 'help' for guidance"],
            )

    async def _execute_direct_tool(
        self, routing: RoutingDecision, parsed: ParsedInput, context: ConversationContext
    ) -> ProcessedCommand:
        """Execute tool directly without agent processing."""

        tool_name = routing.target_tool or "file_reader"  # Default tool
        tool_input = self._extract_tool_input(parsed, tool_name, context)

        try:
            result = await self._execute_tool_with_confirmation_check(tool_name, tool_input, context, parsed.original_text)

            # Check if result requires confirmation
            if isinstance(result, dict) and result.get("needs_confirmation"):
                # Create pending confirmation
                confirmation = self.confirmation_manager.create_confirmation(
                    tool_name=tool_name,
                    input_data=tool_input,
                    context=context,
                    user_message=parsed.original_text
                )
                
                return ProcessedCommand(
                    success=True,
                    message=confirmation.prompt_shown,
                    execution_type="confirmation_required",
                    metadata={
                        "confirmation_id": confirmation.confirmation_id,
                        "tool_name": tool_name,
                        "needs_confirmation": True,
                    },
                )

            # Enhanced message formatting with file path details
            message = result.get("message", "Tool executed successfully")
            
            # For file_writer, enhance the message with file path information
            if tool_name == "file_writer" and result.get("success", True):
                file_path = result.get("metadata", {}).get("file_path") or tool_input.get("file_path")
                if file_path:
                    message = f"✅ Created file: {file_path}"
                else:
                    message = "✅ File created successfully"
            
            # For file_reader, enhance the message with file path information
            elif tool_name == "file_reader" and result.get("success", True):
                file_path = result.get("metadata", {}).get("file_path") or tool_input.get("file_path")
                if file_path:
                    message = f"✅ Read file: {file_path}"
            
            return ProcessedCommand(
                success=result.get("success", True),
                message=message,
                tool_results=[result],
                execution_type="direct_tool",
                direct_tool_call=(tool_name, tool_input),
                metadata={"routing_confidence": routing.confidence},
            )

        except Exception as e:
            return ProcessedCommand(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                execution_type="direct_tool",
                suggestions=[f"Try: help", "Check file path"],
            )

    async def _execute_agent_conversation(
        self, routing: RoutingDecision, parsed: ParsedInput, context: ConversationContext
    ) -> ProcessedCommand:
        """Process through intelligent agent."""

        try:
            # Create enhanced agent context with unified context
            from ...core.agent.types import RequestContext

            # Get unified context for richer agent processing
            unified_context = self.unified_context.get_unified_context(context.session_id)
            
            # Extract relevant information from unified context
            session_context = unified_context["scopes"]["session"]
            conversation_context = unified_context["scopes"]["conversation"]
            file_contexts = unified_context["scopes"].get("file", {})
            
            # Extract recent operations for context
            recent_operations = []
            if "recent_operations" in session_context:
                recent_operations = [
                    {
                        "type": op.operation_type.value if hasattr(op, "operation_type") else str(op.get("operation_type", "")),
                        "file_path": getattr(op, "file_path", op.get("file_path", "")),
                        "language": getattr(op, "language", op.get("language", "")),
                        "timestamp": getattr(op, "timestamp", op.get("timestamp", 0))
                    }
                    for op in session_context["recent_operations"]
                ]
            
            # Get language consistency from recent operations
            language_consistency = None
            if recent_operations:
                recent_languages = [op.get("language") for op in recent_operations if op.get("language")]
                if recent_languages:
                    language_consistency = recent_languages[-1]  # Most recent language
            
            # Create enhanced agent context with unified context data
            agent_context = RequestContext(
                project_path=context.project_path,
                current_file=context.current_file,
                target_files=parsed.entities.get("files", []),
                session_id=context.session_id,
                conversation_history=conversation_context.get("recent_turns", []),
                recent_operations=recent_operations,
                file_contexts=file_contexts,
                language_consistency=language_consistency,
                session_metadata={
                    "unified_context_timestamp": unified_context.get("timestamp"),
                    "context_scopes": list(unified_context["scopes"].keys())
                }
            )

            # Process with agent using enhanced RequestContext
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
                    "routing_confidence": routing.confidence,
                    "agent_state": (
                        agent_response.state.value if hasattr(agent_response, "state") else None
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}", exc_info=True)
            return ProcessedCommand(
                success=False,
                message=f"I'm having trouble processing that request: {str(e)}",
                execution_type="agent_conversation",
                suggestions=["Try rephrasing your request", "Type 'help' for guidance"],
            )

    async def _execute_system_command(
        self, command: str, parsed: ParsedInput, context: ConversationContext
    ) -> ProcessedCommand:
        """Execute system command."""

        if command == "help":
            help_text = self._generate_help_text()
            return ProcessedCommand(
                success=True,
                message=help_text,
                execution_type="system_command",
                system_command="help",
            )

        elif command == "status":
            status_text = self._generate_status_text()
            return ProcessedCommand(
                success=True,
                message=status_text,
                execution_type="system_command",
                system_command="status",
            )

        elif command == "clear":
            return ProcessedCommand(
                success=True,
                message="Screen cleared",
                execution_type="system_command",
                system_command="clear",
            )

        elif command in ["quit", "exit"]:
            return ProcessedCommand(
                success=True,
                message="Goodbye!",
                execution_type="system_command",
                system_command="quit",
            )

        else:
            return ProcessedCommand(
                success=False,
                message=f"Unknown system command: {command}",
                execution_type="system_command",
                suggestions=["Type 'help' for available commands"],
            )

    def _request_clarification(
        self, routing: RoutingDecision, parsed: ParsedInput, context: ConversationContext
    ) -> ProcessedCommand:
        """Request clarification from user."""

        questions = routing.clarification_questions or [
            "I'm not sure what you'd like me to do.",
            "Could you please be more specific?",
        ]

        message = "I need a bit more information:\n" + "\n".join(f"• {q}" for q in questions)

        return ProcessedCommand(
            success=True,
            message=message,
            execution_type="clarification_needed",
            suggestions=[],  # Simplified for NLP approach
        )

    def _extract_tool_input(self, parsed: ParsedInput, tool_name: str, context: ConversationContext) -> Dict[str, Any]:
        """Extract appropriate input for the tool using NLP entities."""

        tool_input = {}

        # File-based tools - use smart linguistic entities
        if tool_name in ["file_reader", "file_writer"]:
            filename = self._extract_filename_smartly(parsed, context)
            if filename:
                tool_input["file_path"] = filename
            else:
                # Fallback to language-based defaults for file creation
                if tool_name == "file_writer":
                    if parsed.entities.get("languages"):
                        lang = parsed.entities["languages"][0]
                        if lang == "python":
                            tool_input["file_path"] = "new_file.py"
                        elif lang in ["javascript", "js"]:
                            tool_input["file_path"] = "new_file.js"
                        elif lang in ["typescript", "ts"]:
                            tool_input["file_path"] = "new_file.ts"
                        elif lang == "java":
                            tool_input["file_path"] = "new_file.java"
                        elif lang == "go":
                            tool_input["file_path"] = "new_file.go"
                        elif lang == "rust":
                            tool_input["file_path"] = "new_file.rs"
                        elif lang == "c":
                            tool_input["file_path"] = "new_file.c"
                        elif lang in ["cpp", "c++"]:
                            tool_input["file_path"] = "new_file.cpp"
                        elif lang == "ruby":
                            tool_input["file_path"] = "new_file.rb"
                        else:
                            tool_input["file_path"] = "new_file.txt"
                    else:
                        # Try to get language from context for consistency
                        context_language = self._get_language_from_context(context)
                        if context_language:
                            extension = self._get_extension_from_language([context_language])
                            tool_input["file_path"] = f"new_file{extension}"
                        else:
                            # Default based on text content - more comprehensive language detection
                            text_lower = parsed.cleaned_text.lower()
                        
                        # Language to extension mapping - order matters (longer matches first)
                        language_extensions = [
                            ("javascript", ".js"),
                            ("typescript", ".ts"), 
                            ("dockerfile", "Dockerfile"),
                            ("makefile", "Makefile"),
                            ("python", ".py"),
                            ("java", ".java"),
                            ("html", ".html"),
                            ("css", ".css"),
                            ("json", ".json"),
                            ("xml", ".xml"),
                            ("yaml", ".yaml"),
                            ("yml", ".yml"),
                            ("php", ".php"),
                            ("ruby", ".rb"),
                            ("go", ".go"),
                            ("rust", ".rs"),
                            ("swift", ".swift"),
                            ("kotlin", ".kt"),
                            ("scala", ".scala"),
                            ("perl", ".pl"),
                            ("shell", ".sh"),
                            ("bash", ".sh"),
                            ("sql", ".sql"),
                            ("cpp", ".cpp"),
                            ("c++", ".cpp"),
                            ("c", ".c"),
                            ("markdown", ".md"),
                            ("text", ".txt"),
                            ("config", ".conf"),
                        ]
                        
                        # Check for language keywords (using word boundaries to avoid substring matches)
                        import re
                        for lang, ext in language_extensions:
                            # Use word boundaries for single-letter languages like 'c'
                            if lang == "c":
                                pattern = r"\bc\b"
                            elif lang == "c++":
                                pattern = r"\bc\+\+\b"
                            else:
                                pattern = r"\b" + re.escape(lang) + r"\b"
                            
                            if re.search(pattern, text_lower):
                                # Special handling for files without extensions
                                if lang == "dockerfile":
                                    tool_input["file_path"] = "Dockerfile"
                                elif lang == "makefile":
                                    tool_input["file_path"] = "Makefile"
                                else:
                                    tool_input["file_path"] = f"new_file{ext}"
                                break
                        else:
                            # Default to .txt if no language detected
                            tool_input["file_path"] = "new_file.txt"

            # Special handling for file_writer
            if tool_name == "file_writer":
                # Use NLP understanding for content generation
                text_lower = parsed.cleaned_text.lower()
                
                if "empty" in text_lower:
                    tool_input["content"] = ""
                elif parsed.entities.get("create_targets"):
                    # For "create a function", etc.
                    target = parsed.entities["create_targets"][0]
                    tool_input["content"] = f"# TODO: Implement {target}\n"
                else:
                    # Check if this is a contextual request referring to a recently created file
                    if any(phrase in text_lower for phrase in ["in that file", "in the file", "implement", "add"]):
                        # Try unified context first, fallback to legacy context
                        unified_context = self.unified_context.get_unified_context(context.session_id)
                        last_file_context = unified_context["scopes"]["session"].get("last_file_context")
                        
                        if last_file_context:
                            # Use the file from unified context
                            tool_input["file_path"] = last_file_context["file_path"]
                            language = last_file_context["language"]
                            
                            # Generate appropriate content based on the request and language
                            content = self._generate_contextual_content(parsed.original_text, language)
                            tool_input["content"] = content
                        else:
                            # Fallback to legacy context
                            file_context = self.unified_context.get_last_file_context(context.session_id)
                            if file_context:
                                tool_input["file_path"] = file_context["file_path"]
                                language = file_context["language"]
                                content = self._generate_contextual_content(parsed.original_text, language)
                                tool_input["content"] = content
                            else:
                                tool_input["content"] = ""  # Default to empty content
                    else:
                        tool_input["content"] = ""  # Default to empty content

        # Search tools
        elif tool_name == "file_searcher":
            # Use entities for search query, fallback to text processing
            if parsed.entities.get("functions"):
                tool_input["query"] = parsed.entities["functions"][0]
            elif parsed.entities.get("classes"):
                tool_input["query"] = parsed.entities["classes"][0]
            else:
                # Extract search query from text
                words = parsed.cleaned_text.split()
                if len(words) > 1:
                    # Skip action words
                    action_words = {"find", "search", "look", "locate", "show", "list"}
                    query_words = [w for w in words if w.lower() not in action_words]
                    tool_input["query"] = " ".join(query_words) if query_words else words[-1]

        return tool_input
    
    def _extract_filename_smartly(self, parsed: ParsedInput, context: ConversationContext) -> Optional[str]:
        """
        Smart filename extraction using linguistic entities and context.
        
        Priority:
        1. Explicit filenames from entities (quoted names, etc.)
        2. Contextual file references ("that file", "the file")
        3. Generate filename from entities + language
        """
        
        # Method 1: Check for explicit filenames from smart entity extraction
        if parsed.entities.get("files"):
            filename = parsed.entities["files"][0]
            
            # If filename has no extension, add one based on detected language
            if '.' not in filename:
                extension = self._get_extension_from_language(parsed.entities.get("languages", []))
                if extension:
                    filename += extension
            
            return filename
        
        # Method 2: Check for contextual file references
        text_lower = parsed.cleaned_text.lower()
        if any(phrase in text_lower for phrase in ["in that file", "in the file", "to that file", "that file"]):
            # Try unified context first, fallback to legacy context
            unified_context = self.unified_context.get_unified_context(context.session_id)
            last_file_context = unified_context["scopes"]["session"].get("last_file_context")
            
            if last_file_context:
                return last_file_context["file_path"]
            else:
                # Fallback to legacy context
                file_context = self.unified_context.get_last_file_context(context.session_id)
                if file_context:
                    return file_context["file_path"]
        
        # Method 3: No explicit filename found
        return None
    
    def _get_extension_from_language(self, languages: List[str]) -> Optional[str]:
        """Get file extension based on detected programming language."""
        if not languages:
            return None
        
        language = languages[0].lower()
        extension_map = {
            "python": ".py",
            "javascript": ".js", 
            "typescript": ".ts",
            "java": ".java",
            "go": ".go",
            "rust": ".rs",
            "c": ".c",
            "cpp": ".cpp",
            "c++": ".cpp",
            "ruby": ".rb",
            "php": ".php",
            "swift": ".swift",
            "kotlin": ".kt",
            "scala": ".scala",
            "html": ".html",
            "css": ".css",
            "json": ".json",
            "yaml": ".yaml",
            "xml": ".xml",
            "sql": ".sql",
            "shell": ".sh",
            "bash": ".sh",
        }
        
        return extension_map.get(language, ".txt")
    
    def _get_language_from_context(self, context: ConversationContext) -> Optional[str]:
        """Get programming language from recent context for consistency."""
        try:
            # Get unified context
            unified_context = self.unified_context.get_unified_context(context.session_id)
            
            # Check last file context for language
            last_file_context = unified_context["scopes"]["session"].get("last_file_context")
            if last_file_context and last_file_context.get("language"):
                return last_file_context["language"]
            
            # Check conversation context for recent language mentions
            conversation_context = unified_context["scopes"]["conversation"]
            if conversation_context.get("recent_turns"):
                for turn in reversed(conversation_context["recent_turns"]):
                    # Check entities in recent turns for languages
                    if turn.entities and turn.entities.get("languages"):
                        return turn.entities["languages"][0]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get language from context: {e}")
            return None

    async def _execute_tool_raw(
        self, tool_name: str, tool_args: Dict[str, Any], context: ConversationContext
    ) -> Dict[str, Any]:
        """Execute a tool directly using the real tool system."""

        try:
            # Get the tool from the registry
            tool = self.tools.get_tool(tool_name)
            if not tool:
                available_tools = self.tools.list_tool_names()
                return {
                    "success": False,
                    "message": f'Tool "{tool_name}" not found. Available tools: {", ".join(available_tools)}',
                    "available_tools": available_tools,
                }

            # Create tool context
            from ...tools.types import ToolContext

            tool_context = ToolContext(
                session_id="ui_session",
                user_id="ui_user",
                project_path=context.project_path
                or ".",  # Use context project path or current directory
                working_directory=context.project_path or ".",
            )

            # Execute the tool with proper error handling
            self.logger.info(f"Executing tool {tool_name} with args: {tool_args}")
            result = await self.tools.execute_tool(tool_name, tool_args, tool_context)

            # Convert ToolResult to dict format expected by UI
            if result.success:
                response = {
                    "success": True,
                    "message": result.message or f"Tool {tool_name} executed successfully",
                    "output": result.output,
                    "metadata": result.metadata or {},
                    "execution_time": getattr(result, "execution_time", None),
                }

                # For file_reader, extract content for display
                if tool_name == "file_reader" and result.output:
                    if hasattr(result.output, "content"):
                        response["content"] = result.output.content
                        response["file_info"] = getattr(result.output, "file_info", None)
                        response["analysis_summary"] = getattr(
                            result.output, "analysis_summary", None
                        )
                    elif isinstance(result.output, dict):
                        response["content"] = result.output.get("content", "")
                        response["file_info"] = result.output.get("file_info", {})
                        response["analysis_summary"] = result.output.get("analysis_summary", "")

                return response
            else:
                return {
                    "success": False,
                    "message": result.message or f"Tool {tool_name} execution failed",
                    "error": result.message,  # Use message for error info
                    "metadata": result.metadata or {},
                }

        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error executing tool {tool_name}: {str(e)}",
                "error": str(e),
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
        agent_status = (
            self.agent.get_agent_status() if hasattr(self.agent, "get_agent_status") else {}
        )

        return f"""🔧 Loki Code Status

Agent: {agent_status.get('state', 'Active')}
Tools Available: {tool_count}
Session: Active
Command Processor: Ready

Type 'help' for available commands."""

    async def _execute_tool_with_confirmation_check(
        self, tool_name: str, tool_input: Dict[str, Any], context: ConversationContext, user_message: str = ""
    ) -> Dict[str, Any]:
        """Execute a tool and check if it requires confirmation."""
        try:
            # Execute the tool using the existing method
            result = await self._execute_tool_raw(tool_name, tool_input, context)
            return result
            
        except Exception as e:
            # Check if this is a confirmation requirement (based on the error message)
            error_message = str(e).lower()
            if "requires confirmation" in error_message or "confirmation needed" in error_message:
                # This tool requires confirmation
                return {
                    "success": False,
                    "needs_confirmation": True,
                    "message": f"Tool '{tool_name}' requires confirmation before execution",
                    "error": str(e)
                }
            else:
                # Re-raise other exceptions
                raise
    
    async def _handle_follow_up_request(self, parsed: ParsedInput, context: ConversationContext) -> ProcessedCommand:
        """Handle follow-up requests by repeating the last operation."""
        
        # Get the most recent operation from the session
        last_operation = self.unified_context.get_last_operation(context.session_id)
        
        if not last_operation:
            return ProcessedCommand(
                success=False,
                message="I don't have any recent operations to repeat. Could you please specify what you'd like me to do?",
                execution_type="follow_up",
                suggestions=["Try being more specific about what you want to create or do"],
            )
        
        # Create a summary of what we're repeating
        operation_summary = self.unified_context.get_operation_summary(last_operation)
        
        try:
            # Execute the same tool with the same parameters
            result = await self._execute_tool_raw(
                last_operation.tool_name, 
                last_operation.tool_input, 
                context
            )
            
            # Enhance the message to show it's a repeat operation
            if result.get("success", True):
                original_message = result.get("message", "Operation completed successfully")
                enhanced_message = f"🔄 Repeated operation: {operation_summary}\n\n{original_message}"
                
                # Apply the same message enhancement logic as direct tools
                if last_operation.tool_name == "file_writer":
                    file_path = result.get("metadata", {}).get("file_path") or last_operation.tool_input.get("file_path")
                    if file_path:
                        enhanced_message = f"🔄 Repeated operation: {operation_summary}\n\n✅ Created file: {file_path}"
                
                return ProcessedCommand(
                    success=True,
                    message=enhanced_message,
                    execution_type="follow_up",
                    tool_results=[result],
                    direct_tool_call=(last_operation.tool_name, last_operation.tool_input),
                    metadata={"repeated_operation": operation_summary},
                )
            else:
                return ProcessedCommand(
                    success=False,
                    message=f"Failed to repeat operation: {operation_summary}. Error: {result.get('message', 'Unknown error')}",
                    execution_type="follow_up",
                    suggestions=["Try the operation again with different parameters"],
                )
        
        except Exception as e:
            self.logger.error(f"Error repeating operation: {e}", exc_info=True)
            return ProcessedCommand(
                success=False,
                message=f"Failed to repeat operation: {operation_summary}. Error: {str(e)}",
                execution_type="follow_up",
                suggestions=["Try the operation again manually"],
            )
    
    def _record_operation(self, result: ProcessedCommand, parsed: ParsedInput, context: ConversationContext) -> None:
        """Record a successful operation for future reference."""
        
        if not result.direct_tool_call:
            return
        
        tool_name, tool_input = result.direct_tool_call
        
        # Determine operation type based on tool and intent
        operation_type = OperationType.FILE_CREATION  # Default
        
        if tool_name == "file_writer":
            operation_type = OperationType.FILE_CREATION
        elif tool_name == "file_reader":
            operation_type = OperationType.FILE_READING
        elif tool_name == "file_searcher":
            operation_type = OperationType.SEARCH
        elif parsed.intent.type == IntentType.CODE_GENERATION:
            operation_type = OperationType.CODE_GENERATION
        elif parsed.intent.type == IntentType.FILE_ANALYSIS:
            operation_type = OperationType.ANALYSIS
        
        # Record the operation
        self.unified_context.record_operation(
            session_id=context.session_id,
            operation_type=operation_type,
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
            user_input=parsed.original_text
        )
        
        self.logger.debug(f"Recorded operation: {operation_type.value} for session {context.session_id}")
    
    def _get_operation_type(self, intent_type: IntentType, tool_name: str = None) -> OperationType:
        """Map intent type to operation type, considering tool name for better accuracy."""
        # Tool-based mapping takes precedence for better accuracy
        if tool_name:
            if tool_name == "file_writer":
                return OperationType.FILE_CREATION
            elif tool_name == "file_reader":
                return OperationType.FILE_READING
            elif tool_name == "file_searcher":
                return OperationType.SEARCH
        
        # Fallback to intent-based mapping
        mapping = {
            IntentType.CODE_GENERATION: OperationType.CODE_GENERATION,
            IntentType.FILE_ANALYSIS: OperationType.ANALYSIS,
            IntentType.SEARCH: OperationType.SEARCH,
        }
        return mapping.get(intent_type, OperationType.CODE_GENERATION)
    
    def _generate_contextual_content(self, user_request: str, language: str) -> str:
        """Generate appropriate code content based on user request and target language."""
        
        request_lower = user_request.lower()
        
        # Extract what the user wants to implement
        if "fibonacci" in request_lower:
            return self._generate_fibonacci_function(language)
        elif "factorial" in request_lower:
            return self._generate_factorial_function(language)
        elif "hello world" in request_lower:
            return self._generate_hello_world(language)
        elif "function" in request_lower:
            # Generic function template
            return self._generate_generic_function(language, user_request)
        else:
            # Default: create a basic template
            return self._generate_basic_template(language, user_request)
    
    def _generate_fibonacci_function(self, language: str) -> str:
        """Generate Fibonacci function in the specified language."""
        
        if language == "go":
            return '''package main

import "fmt"

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

func main() {
    // Example usage
    fmt.Println(fibonacci(10))  // Output: 55
}
'''
        elif language == "python":
            return '''def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
if __name__ == "__main__":
    print(fibonacci(10))  # Output: 55
'''
        elif language == "javascript":
            return '''function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

// Example usage
console.log(fibonacci(10)); // Output: 55
'''
        elif language == "java":
            return '''public class Fibonacci {
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n-1) + fibonacci(n-2);
    }
    
    public static void main(String[] args) {
        System.out.println(fibonacci(10)); // Output: 55
    }
}
'''
        elif language == "rust":
            return '''fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n-1) + fibonacci(n-2)
}

fn main() {
    println!("{}", fibonacci(10)); // Output: 55
}
'''
        elif language == "c":
            return '''#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
    printf("%d\\n", fibonacci(10)); // Output: 55
    return 0;
}
'''
        else:
            return f"// Fibonacci function implementation for {language}\\n// TODO: Implement fibonacci function\\n"
    
    def _generate_factorial_function(self, language: str) -> str:
        """Generate factorial function in the specified language."""
        
        if language == "go":
            return '''package main

import "fmt"

func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n-1)
}

func main() {
    fmt.Println(factorial(5))  // Output: 120
}
'''
        elif language == "python":
            return '''def factorial(n):
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Example usage
if __name__ == "__main__":
    print(factorial(5))  # Output: 120
'''
        else:
            return f"// Factorial function implementation for {language}\\n// TODO: Implement factorial function\\n"
    
    def _generate_hello_world(self, language: str) -> str:
        """Generate hello world program in the specified language."""
        
        if language == "go":
            return '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
'''
        elif language == "python":
            return '''print("Hello, World!")
'''
        elif language == "javascript":
            return '''console.log("Hello, World!");
'''
        elif language == "java":
            return '''public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''
        else:
            return f"// Hello World program for {language}\\n// TODO: Implement hello world\\n"
    
    def _generate_generic_function(self, language: str, user_request: str) -> str:
        """Generate a generic function template."""
        
        if language == "go":
            return '''package main

import "fmt"

func myFunction() {
    // TODO: Implement function logic
    fmt.Println("Function called")
}

func main() {
    myFunction()
}
'''
        elif language == "python":
            return '''def my_function():
    """TODO: Implement function logic"""
    print("Function called")

if __name__ == "__main__":
    my_function()
'''
        else:
            return f"// Function implementation for {language}\\n// TODO: Implement function\\n"
    
    def _generate_basic_template(self, language: str, user_request: str) -> str:
        """Generate a basic template based on user request."""
        
        if language == "go":
            return '''package main

import "fmt"

func main() {
    // TODO: Implement based on request
    fmt.Println("Hello from Go!")
}
'''
        elif language == "python":
            return '''# TODO: Implement based on request
print("Hello from Python!")
'''
        elif language == "javascript":
            return '''// TODO: Implement based on request
console.log("Hello from JavaScript!");
'''
        else:
            return f"// TODO: Implement for {language}\\n"
