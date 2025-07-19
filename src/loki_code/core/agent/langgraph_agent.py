"""
LangGraph-based agent implementation for Loki Code.

This module implements a modern LangGraph ReAct agent that replaces the legacy
AgentExecutor pattern with proper termination logic and better state management.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .types import (
    AgentConfig,
    AgentResponse,
    RequestContext,
    AgentState,
    RequestUnderstanding,
)
from .permission_manager import PermissionManager, PermissionConfig
from .safety_manager import SafetyManager, SafetyConfig
from .conversation_manager import ConversationManager, ConversationConfig
from ...tools.langchain_tools import create_langchain_tools
from ...utils.logging import get_logger


class LangGraphReasoningCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to display agent reasoning steps in real-time."""

    def __init__(self, logger):
        """Initialize the callback handler.

        Args:
            logger: Logger instance to use for displaying reasoning steps
        """
        super().__init__()
        self.logger = logger
        self.step_count = 0

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Called when the agent takes an action."""
        self.step_count += 1
        self.logger.info(f"🔧 Action #{self.step_count}: {action.tool}")
        self.logger.info(f"📝 Action Input: {action.tool_input}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "Unknown Tool")
        self.logger.info(f"⚡ Executing tool: {tool_name}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Called when a tool finishes running."""
        # Truncate very long outputs for readability
        display_output = output[:200] + "..." if len(output) > 200 else output
        self.logger.info(f"👀 Observation: {display_output}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when a tool encounters an error."""
        self.logger.error(f"❌ Tool Error: {str(error)}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Called when the agent finishes."""
        self.logger.info(f"✅ Final Answer: {finish.return_values.get('output', 'No output')}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Called when LLM starts generating."""
        self.logger.info("🤔 Thinking...")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Called when LLM finishes generating."""
        # Extract the agent's reasoning if available
        if response.generations and len(response.generations) > 0:
            generation = response.generations[0][0]
            if hasattr(generation, "text"):
                text = generation.text.strip()
                # Look for thought patterns in the response
                if "Thought:" in text:
                    thought_lines = [
                        line.strip()
                        for line in text.split("\n")
                        if line.strip().startswith("Thought:")
                    ]
                    for thought in thought_lines:
                        self.logger.info(f"💭 {thought}")


class LokiLangGraphAgent:
    """
    LangGraph-based agent for Loki Code using ReAct pattern.

    This agent uses the modern langgraph.prebuilt.create_react_agent implementation
    with proper termination logic and state management.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the LangGraph agent.

        Args:
            llm: Language model to use for the agent
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.llm = llm
        self.logger = get_logger(__name__)
        self.state = AgentState.IDLE

        # Initialize managers
        self.permission_manager = PermissionManager(
            PermissionConfig(auto_grant_safe_operations=self.config.auto_approve_safe_actions)
        )
        self.safety_manager = SafetyManager(SafetyConfig())
        self.conversation_manager = ConversationManager(ConversationConfig())

        # Create LangChain tools from Loki tools
        self.tools = create_langchain_tools(
            permission_manager=self.permission_manager, safety_manager=self.safety_manager
        )

        # Validate tools before creating react agent
        self._validate_tools()

        # Bind tools to the model for proper LangGraph integration
        if hasattr(self.llm, "bind_tools") and self.tools:
            try:
                self.llm = self.llm.bind_tools(self.tools)
                self.logger.info(f"Successfully bound {len(self.tools)} tools to model")
            except Exception as e:
                self.logger.warning(f"Failed to bind tools to model: {e}")
                self.logger.info(
                    "Continuing without tool binding - create_react_agent will handle tools"
                )
        else:
            self.logger.info(
                f"Model does not support bind_tools or no tools available - create_react_agent will handle {len(self.tools)} tools"
            )

        # Create custom callback handler for reasoning display
        self.reasoning_callback = LangGraphReasoningCallbackHandler(self.logger)

        # Create memory checkpointer for state persistence
        self.memory = MemorySaver()

        # Calculate recursion limit (default: 6 for simple tasks)
        recursion_limit = min(self.config.max_steps * 2 + 1, 13)  # Cap at 13

        # Create system prompt that encourages thinking out loud (LangGraph pattern)
        system_prompt = (
            "You are Loki Code, a helpful coding assistant with access to powerful tools. "
            "Always think step by step and explain your reasoning process. "
            "When analyzing a request, first think about what you need to do, then decide if you need tools. "
            "Show your thought process before giving your final answer. "
            "When you have completed a task successfully, provide a clear final answer. "
            "Do not continue working after you have given a complete response."
        )

        # Create the LangGraph ReAct agent with string prompt
        self.agent = create_react_agent(
            model=self.llm, tools=self.tools, checkpointer=self.memory, prompt=system_prompt
        )

        # Configure the agent with recursion limit
        self.agent_with_config = self.agent.with_config(recursion_limit=recursion_limit)

        self.logger.info(f"LokiLangGraphAgent initialized with recursion_limit={recursion_limit}")

    def _validate_tools(self) -> None:
        """Validate that all tools are properly defined for LangGraph create_react_agent."""
        self.logger.info(f"Validating {len(self.tools)} tools before creating react agent")

        valid_tools = []

        for tool in self.tools:
            try:
                # Check if tool is a BaseTool instance
                if not hasattr(tool, "name"):
                    self.logger.warning(
                        f"Tool {tool} is missing required 'name' attribute - skipping"
                    )
                    continue

                if not hasattr(tool, "description"):
                    self.logger.warning(
                        f"Tool {tool.name} is missing required 'description' attribute - skipping"
                    )
                    continue

                if not hasattr(tool, "args_schema"):
                    self.logger.warning(
                        f"Tool {tool.name} is missing required 'args_schema' attribute - skipping"
                    )
                    continue

                # Validate args_schema is a BaseModel subclass
                if tool.args_schema is not None:
                    from pydantic import BaseModel

                    if not (
                        isinstance(tool.args_schema, type)
                        and issubclass(tool.args_schema, BaseModel)
                    ):
                        self.logger.warning(
                            f"Tool {tool.name} args_schema must be a BaseModel subclass, got {type(tool.args_schema)} - skipping"
                        )
                        continue

                # Check if tool has required methods
                if not hasattr(tool, "_run"):
                    self.logger.warning(
                        f"Tool {tool.name} is missing required '_run' method - skipping"
                    )
                    continue

                # If we get here, the tool is valid
                valid_tools.append(tool)
                self.logger.debug(f"✓ Tool {tool.name} validated successfully")

            except Exception as e:
                self.logger.warning(
                    f"Tool {getattr(tool, 'name', 'unknown')} validation failed: {e} - skipping"
                )
                continue

        # Update the tools list with only valid tools
        self.tools = valid_tools

        if not self.tools:
            self.logger.warning("No valid tools found - agent will run without tools")
        else:
            self.logger.info(f"Tools validation passed for {len(self.tools)} tools")

    async def process_request(
        self, user_message: str, context: Optional[RequestContext] = None
    ) -> AgentResponse:
        """
        Process a user request using the LangGraph ReAct agent.

        Args:
            user_message: The user's message/request
            context: Optional context for the request

        Returns:
            AgentResponse with the agent's response and metadata
        """
        self.logger.info(f"Processing request: {user_message[:100]}...")
        self.state = AgentState.THINKING

        try:
            # Prepare context
            context = context or RequestContext()

            # Create session config for memory persistence
            session_config = {
                "configurable": {"thread_id": context.session_id or "default_session"}
            }

            # Analyze request understanding
            understanding = await self._analyze_request(user_message, context)

            # Execute using LangGraph agent
            self.state = AgentState.EXECUTING
            result = await self._execute_with_langgraph(user_message, session_config)

            # Process response
            self.state = AgentState.COMPLETED
            response = self._create_response(understanding, result, context)

            return response

        except GraphRecursionError as e:
            self.logger.error(f"Agent reached recursion limit: {e}")
            self.state = AgentState.ERROR_RECOVERY

            step_count = getattr(self.reasoning_callback, "step_count", 0)
            if step_count == 0:
                error_message = (
                    "❌ Agent Startup Failed: The agent couldn't begin processing your request. "
                    "This is likely due to LLM connectivity issues. "
                    "Please check if your LLM provider (Ollama) is running and accessible."
                )
            else:
                error_message = (
                    f"❌ Agent Stopped: The agent completed {step_count} reasoning steps but "
                    "reached the iteration limit. The request may be too complex. "
                    "Try breaking it down into smaller parts."
                )

            return AgentResponse(
                content=error_message,
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": "GraphRecursionError",
                    "agent_step_count": step_count,
                },
            )

        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            self.state = AgentState.ERROR_RECOVERY

            # Enhanced error analysis and messaging
            error_message = self._analyze_and_format_error(e, user_message)

            return AgentResponse(
                content=error_message,
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "agent_step_count": getattr(self.reasoning_callback, "step_count", 0),
                },
            )
        finally:
            self.state = AgentState.IDLE

    async def process_request_stream(
        self, user_message: str, context: Optional[RequestContext] = None
    ):
        """
        Process a user request using the LangGraph ReAct agent with streaming.

        Args:
            user_message: The user's message/request
            context: Optional context for the request

        Yields:
            Dict[str, Any]: Streaming updates from the agent execution
        """
        self.logger.info(f"Processing streaming request: {user_message[:100]}...")
        self.state = AgentState.THINKING

        try:
            # Prepare context
            context = context or RequestContext()

            # Create session config for memory persistence
            session_config = {
                "configurable": {"thread_id": context.session_id or "default_session"}
            }

            # Analyze request understanding
            understanding = await self._analyze_request(user_message, context)

            # Execute using LangGraph agent with streaming
            self.state = AgentState.EXECUTING

            async for update in self._execute_with_langgraph_stream(user_message, session_config):
                yield update

        except GraphRecursionError as e:
            self.logger.error(f"Agent reached recursion limit: {e}")
            self.state = AgentState.ERROR_RECOVERY

            step_count = getattr(self.reasoning_callback, "step_count", 0)
            if step_count == 0:
                error_message = (
                    "❌ Agent Startup Failed: The agent couldn't begin processing your request. "
                    "This is likely due to LLM connectivity issues. "
                    "Please check if your LLM provider (Ollama) is running and accessible."
                )
            else:
                error_message = (
                    f"❌ Agent Stopped: The agent completed {step_count} reasoning steps but "
                    "reached the iteration limit. The request may be too complex. "
                    "Try breaking it down into smaller parts."
                )

            yield {
                "type": "error",
                "content": error_message,
                "state": AgentState.ERROR_RECOVERY,
                "metadata": {
                    "error": str(e),
                    "error_type": "GraphRecursionError",
                    "agent_step_count": step_count,
                },
            }

        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            self.state = AgentState.ERROR_RECOVERY

            # Enhanced error analysis and messaging
            error_message = self._analyze_and_format_error(e, user_message)

            yield {
                "type": "error",
                "content": error_message,
                "state": AgentState.ERROR_RECOVERY,
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "agent_step_count": getattr(self.reasoning_callback, "step_count", 0),
                },
            }
        finally:
            self.state = AgentState.IDLE

    async def _analyze_request(
        self, user_message: str, context: RequestContext
    ) -> RequestUnderstanding:
        """Analyze the user request to understand intent."""
        # Simple analysis for now - can be enhanced with more sophisticated NLP
        understanding = RequestUnderstanding(
            user_intent=user_message,
            confidence=0.8,  # Default confidence
            extracted_entities={},
            ambiguous_aspects=[],
            required_tools=[],
            risk_assessment="low",
            suggested_approach="Use LangGraph ReAct pattern",
        )

        # Basic tool requirement detection
        if "read" in user_message.lower() or "file" in user_message.lower():
            understanding.required_tools.append("file_reader")

        return understanding

    async def _execute_with_langgraph(
        self, user_message: str, session_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the request using LangGraph agent with proper message-based state."""
        try:
            # Prepare input in LangGraph message format using proper message objects
            from langchain_core.messages import HumanMessage

            input_messages = [HumanMessage(content=user_message)]

            # Create input state for LangGraph following documented patterns
            input_state = {"messages": input_messages}

            # Log start of execution
            self.logger.info(
                f"🚀 Starting LangGraph execution with {len(input_messages)} input messages"
            )
            self.logger.debug(f"Session config: {session_config}")
            self.logger.debug(f"Input state: {input_state}")

            # Execute agent synchronously (LangGraph pattern)
            # Most LangGraph examples show synchronous invoke, not async
            result = self.agent_with_config.invoke(input_state, session_config)

            # Log execution completion
            messages_count = len(result.get("messages", []))
            self.logger.info(
                f"✅ LangGraph execution completed with {messages_count} total messages"
            )

            return result

        except Exception as e:
            self.logger.error(f"LangGraph execution error: {e}", exc_info=True)
            raise

    async def _execute_with_langgraph_stream(
        self, user_message: str, session_config: Dict[str, Any]
    ):
        """Execute the request using LangGraph agent with streaming updates."""
        try:
            # Prepare input in LangGraph message format using proper message objects
            from langchain_core.messages import HumanMessage

            input_messages = [HumanMessage(content=user_message)]

            # Create input state for LangGraph following documented patterns
            input_state = {"messages": input_messages}

            # Log start of execution
            self.logger.info(
                f"🚀 Starting LangGraph streaming execution with {len(input_messages)} input messages"
            )

            # Yield immediate thinking step
            yield {
                "type": "reasoning",
                "step": "🤔 Starting to think about your request...",
                "details": {"stage": "initialization"},
            }

            # Track all messages for final response
            all_messages = []
            seen_messages = set()

            # Execute agent with streaming using updates mode
            # This gives us step-by-step updates as each node executes
            for update in self.agent_with_config.stream(
                input_state, session_config, stream_mode="updates"
            ):
                self.logger.debug(f"LangGraph stream update: {update}")

                # Process each update from LangGraph
                for node_name, node_result in update.items():
                    if node_name == "__start__":
                        # Skip start node
                        continue

                    # Yield node execution step
                    yield {
                        "type": "reasoning",
                        "step": f"⚡ Processing node: {node_name}",
                        "details": {"node_name": node_name, "stage": "node_execution"},
                    }

                    # Extract messages from node result
                    messages = node_result.get("messages", [])

                    # Process each message in the update
                    for message in messages:
                        # Create a unique identifier for this message
                        message_id = f"{id(message)}_{getattr(message, 'content', '')[:50]}"

                        # Skip if we've already seen this message
                        if message_id in seen_messages:
                            continue
                        seen_messages.add(message_id)

                        if hasattr(message, "__class__"):
                            message_type = message.__class__.__name__

                            # Handle AI messages (model reasoning/tool calls)
                            if "AIMessage" in message_type:
                                # Check for tool calls first
                                if hasattr(message, "tool_calls") and message.tool_calls:
                                    for tool_call in message.tool_calls:
                                        tool_name = tool_call.get("name", "unknown_tool")
                                        tool_args = tool_call.get("args", {})

                                        # Yield reasoning step for tool planning
                                        yield {
                                            "type": "reasoning",
                                            "step": f"🔧 Planning to use tool: {tool_name}",
                                            "details": {
                                                "tool_name": tool_name,
                                                "tool_args": tool_args,
                                                "message_type": message_type,
                                            },
                                        }

                                        if tool_args:
                                            args_str = (
                                                str(tool_args)[:100] + "..."
                                                if len(str(tool_args)) > 100
                                                else str(tool_args)
                                            )
                                            yield {
                                                "type": "reasoning",
                                                "step": f"📝 With arguments: {args_str}",
                                                "details": {"args": tool_args},
                                            }

                                # Show AI message content as reasoning
                                elif hasattr(message, "content") and message.content:
                                    content = message.content
                                    if content.strip():
                                        # Check if this looks like reasoning vs final answer
                                        is_reasoning = self._is_reasoning_content(content)

                                        if is_reasoning:
                                            display_content = (
                                                content[:200] + "..."
                                                if len(content) > 200
                                                else content
                                            )
                                            yield {
                                                "type": "reasoning",
                                                "step": f"💭 {display_content}",
                                                "details": {"full_content": content},
                                            }
                                        else:
                                            # This is likely the final answer
                                            all_messages.append(message)

                            # Handle Tool messages (tool results)
                            elif "ToolMessage" in message_type:
                                tool_name = getattr(message, "name", "unknown_tool")
                                tool_content = getattr(message, "content", "")

                                yield {
                                    "type": "reasoning",
                                    "step": f"🔧 Tool {tool_name} executed",
                                    "details": {
                                        "tool_name": tool_name,
                                        "message_type": message_type,
                                    },
                                }

                                if tool_content:
                                    # Show truncated tool result
                                    display_content = (
                                        tool_content[:150] + "..."
                                        if len(tool_content) > 150
                                        else tool_content
                                    )
                                    yield {
                                        "type": "reasoning",
                                        "step": f"👀 Result: {display_content}",
                                        "details": {"full_content": tool_content},
                                    }

                        # Store all messages for final response
                        all_messages.append(message)

            # Log execution completion
            self.logger.info(
                f"✅ LangGraph streaming execution completed with {len(all_messages)} total messages"
            )

            # Yield final response with all messages
            yield {
                "type": "final_response",
                "messages": all_messages,
                "total_messages": len(all_messages),
            }

        except Exception as e:
            self.logger.error(f"LangGraph streaming execution error: {e}", exc_info=True)
            raise

    def _is_reasoning_content(self, content: str) -> bool:
        """Check if content appears to be reasoning rather than a final answer."""
        content_lower = content.lower().strip()

        # Reasoning indicators
        reasoning_patterns = [
            "i need to",
            "let me",
            "first",
            "then",
            "next",
            "step",
            "analyze",
            "consider",
            "think",
            "looking at",
            "based on",
            "since",
            "because",
            "however",
            "therefore",
            "given",
            "plan",
            "approach",
            "strategy",
        ]

        # Check for reasoning patterns
        for pattern in reasoning_patterns:
            if pattern in content_lower:
                return True

        # Very short responses are likely final answers
        if len(content.split()) <= 5:
            return False

        # If it starts with a greeting, it's likely a final answer
        if content_lower.startswith(("hello", "hi", "hey", "greetings")):
            return False

        # Default to treating as reasoning to show more intermediate steps
        return True

    def _is_final_conversational_response(self, content: str) -> bool:
        """Check if the content is likely a final conversational response rather than reasoning."""
        content_lower = content.lower().strip()

        # Look for reasoning indicators - if present, it's likely reasoning
        reasoning_indicators = [
            "i need to",
            "let me think",
            "first",
            "then",
            "next",
            "because",
            "since",
            "therefore",
            "however",
            "although",
            "considering",
            "given that",
            "step by step",
            "analyze",
            "looking at",
            "based on",
            "thinking about",
        ]

        # If it contains reasoning indicators, it's likely reasoning content
        for indicator in reasoning_indicators:
            if indicator in content_lower:
                return False

        # Very short responses are likely final answers
        if len(content.split()) <= 8:
            return True

        # Common patterns for final conversational responses (only very obvious ones)
        final_response_patterns = [
            "hello!",
            "hi!",
            "how can i help you",
            "how can i assist you",
            "i'd be happy to help you",
            "feel free to ask",
        ]

        # Check if the content exactly matches these patterns
        for pattern in final_response_patterns:
            if content_lower.startswith(pattern):
                return True

        # Default to showing as reasoning if we're not sure
        return False

    def _create_response(
        self,
        understanding: RequestUnderstanding,
        langgraph_result: Dict[str, Any],
        context: RequestContext,
    ) -> AgentResponse:
        """Create an AgentResponse from LangGraph results with enhanced message processing."""

        # Extract information from LangGraph result
        messages = langgraph_result.get("messages", [])

        # Process messages using LangGraph's message types
        output = ""
        actions_taken = []
        tools_used = []
        reasoning_steps = []

        # Track conversation flow through messages
        for i, message in enumerate(messages):
            # Handle different message types properly
            if hasattr(message, "__class__"):
                message_type = message.__class__.__name__

                if "AIMessage" in message_type:
                    # Extract final AI response - get the last AI message as the final output
                    if hasattr(message, "content") and message.content:
                        output = message.content

                    # Check for tool calls in AI messages
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.get("name", "unknown_tool")
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
                            actions_taken.append(f"Called {tool_name}")

                elif "ToolMessage" in message_type:
                    # Extract tool execution results
                    tool_name = getattr(message, "name", "unknown_tool")
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)

                    tool_content = getattr(message, "content", "")
                    if tool_content:
                        # Truncate long tool outputs for readability
                        display_content = (
                            tool_content[:100] + "..." if len(tool_content) > 100 else tool_content
                        )
                        reasoning_steps.append(f"Tool {tool_name} returned: {display_content}")

                elif "HumanMessage" in message_type:
                    # Track user inputs
                    reasoning_steps.append("User input received")

        # Remove duplicates and clean up
        tools_used = list(set(tools_used))

        # Extract final output from the last AI message if output is still empty
        if not output and messages:
            for message in reversed(messages):
                if hasattr(message, "__class__") and "AIMessage" in message.__class__.__name__:
                    if hasattr(message, "content") and message.content:
                        output = message.content
                        break

        # Enhanced metadata with message analysis
        metadata = {
            "langgraph_result": langgraph_result,
            "understanding": understanding.__dict__,
            "message_count": len(messages),
            "agent_type": "LangGraph",
            "message_types": [
                msg.__class__.__name__ for msg in messages if hasattr(msg, "__class__")
            ],
            "reasoning_steps": reasoning_steps,
            "session_id": context.session_id if context else None,
        }

        return AgentResponse(
            content=output,
            state=AgentState.COMPLETED,
            actions_taken=actions_taken,
            tools_used=tools_used,
            permissions_requested=0,  # TODO: Track from permission manager
            safety_checks_passed=len(actions_taken),  # Based on successful tool executions
            confidence=understanding.confidence,
            metadata=metadata,
        )

    def _analyze_and_format_error(self, error: Exception, user_message: str) -> str:
        """Analyze the error and provide a helpful error message to the user."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        step_count = getattr(self.reasoning_callback, "step_count", 0)

        # Check for common error patterns and provide specific guidance
        if "connection" in error_str or "refused" in error_str:
            return (
                "❌ Connection Error: I couldn't connect to the LLM service. "
                "Please check if Ollama is running with: `ollama serve` "
                "or verify your LLM provider configuration."
            )

        elif "model" in error_str and ("not found" in error_str or "404" in error_str):
            return (
                "❌ Model Not Found: The specified LLM model isn't available. "
                "Please check your model configuration or run `ollama list` "
                "to see available models."
            )

        elif "permission" in error_str or "access" in error_str:
            return (
                "❌ Permission Error: I don't have permission to perform this action. "
                "Please check file permissions or your safety/permission settings."
            )

        elif "tool" in error_str:
            return (
                f"❌ Tool Error: There was an issue with one of my tools after {step_count} steps. "
                f"Error details: {str(error)}"
            )

        else:
            # Generic error with helpful context
            return (
                f"❌ Unexpected Error: I encountered an issue while processing your request "
                f"'{user_message[:50]}...'. Steps completed: {step_count}. "
                f"Error type: {error_type}. Details: {str(error)}"
            )

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "agent_type": "LangGraph",
            "memory_type": "MemorySaver",
            "state": self.state.value,
            "tools_count": len(self.tools),
        }

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        # Create new memory saver to clear state
        self.memory = MemorySaver()
        # Validate tools before recreating agent
        self._validate_tools()

        # Recreate system prompt
        system_prompt = (
            "You are Loki Code, a helpful coding assistant with access to powerful tools. "
            "When you have completed a task successfully, provide a clear final answer. "
            "Do not continue working after you have given a complete response."
        )

        # Recreate agent with new memory
        self.agent = create_react_agent(
            model=self.llm, tools=self.tools, checkpointer=self.memory, prompt=system_prompt
        )
        recursion_limit = min(self.config.max_steps * 2 + 1, 13)
        self.agent_with_config = self.agent.with_config(recursion_limit=recursion_limit)
        self.logger.info("Memory cleared and agent recreated")

    def get_conversation_summary(self) -> Optional[str]:
        """Get conversation summary if available."""
        return "LangGraph agent - summary not yet implemented"


class LokiLangGraphAgentFactory:
    """Factory for creating LangGraph agents with different LLM providers."""

    @staticmethod
    def create_with_model_config(
        model_config,  # Import will be added at the top
        provider,  # Import will be added at the top
        agent_config: Optional[AgentConfig] = None,
    ) -> LokiLangGraphAgent:
        """Create agent with model configuration and provider."""
        llm = provider.create_llm(model_config)
        return LokiLangGraphAgent(llm=llm, config=agent_config)

    @staticmethod
    def create_with_transformers(
        model_name: str = "deepseek-coder-v2-lite",
        model_path: Optional[str] = None,
        config: Optional[AgentConfig] = None,
    ) -> LokiLangGraphAgent:
        """Create agent with Transformers model."""
        from ..models import get_model_config, get_default_model_config, TransformersProvider

        # Use provided model_path or get default config
        if model_path:
            model_config = get_model_config(model_name, model_path)
        else:
            model_config = get_default_model_config()

        provider = TransformersProvider()
        return LokiLangGraphAgentFactory.create_with_model_config(model_config, provider, config)

    @staticmethod
    def create_with_openai(
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        config: Optional[AgentConfig] = None,
    ) -> LokiLangGraphAgent:
        """Create agent with OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(api_key=api_key, model=model_name)
            return LokiLangGraphAgent(llm=llm, config=config)
        except ImportError:
            raise ImportError("langchain-openai is required for OpenAI integration")

    @staticmethod
    def create_with_custom_llm(
        llm: BaseLanguageModel,
        config: Optional[AgentConfig] = None,
    ) -> LokiLangGraphAgent:
        """Create agent with custom LLM."""
        return LokiLangGraphAgent(llm=llm, config=config)


# Alias for consistency
LokiGraphAgent = LokiLangGraphAgent
