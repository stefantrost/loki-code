"""
Main response processor integrating all response handling components.

This module provides the complete response processing system that integrates
parsing, formatting, conversation management, and streaming for the Loki Code
intelligent agent system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import logging

from .parser import ResponseParser, ParsedResponse, ParsingConfig
from .formatter import ResponseFormatter, FormattedResponse, FormattingConfig
from .conversation import ConversationManager, ConversationConfig, ConversationContext
from .streaming import StreamingResponseHandler, StreamingConfig, StreamChunk, ChunkType
from ...tools.types import ToolResult
from ...core.tool_executor import ToolExecutor
from ...utils.logging import get_logger


class ProcessingPhase(Enum):
    """Phases of response processing."""
    INITIALIZATION = "initialization"
    PARSING = "parsing"
    VALIDATION = "validation"
    TOOL_EXECUTION = "tool_execution"
    FORMATTING = "formatting"
    DELIVERY = "delivery"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"


class ExecutionStrategy(Enum):
    """Strategies for tool execution."""
    SEQUENTIAL = "sequential"      # Execute tools one by one
    PARALLEL = "parallel"          # Execute independent tools in parallel
    PIPELINE = "pipeline"          # Execute tools in dependency order
    INTERACTIVE = "interactive"    # Ask user before each tool execution


@dataclass
class ExecutionContext:
    """Context for response processing and tool execution."""
    session_id: str
    user_message: str
    conversation_context: Optional[ConversationContext] = None
    project_path: Optional[str] = None
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_tool_executions: int = 5
    timeout_seconds: float = 30.0
    enable_streaming: bool = True
    require_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionResult:
    """Result of a complete agent interaction."""
    parsed_response: ParsedResponse
    formatted_response: FormattedResponse
    tool_results: List[ToolResult] = field(default_factory=list)
    execution_time: float = 0.0
    processing_phases: List[ProcessingPhase] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_tool_results(self) -> bool:
        """Check if interaction produced tool results."""
        return len(self.tool_results) > 0
    
    @property
    def successful_tool_count(self) -> int:
        """Get number of successful tool executions."""
        return sum(1 for result in self.tool_results if result.success)
    
    @property
    def failed_tool_count(self) -> int:
        """Get number of failed tool executions."""
        return sum(1 for result in self.tool_results if not result.success)


@dataclass
class ProcessedInteraction:
    """Complete processed interaction with all components."""
    execution_context: ExecutionContext
    interaction_result: InteractionResult
    conversation_turn_id: Optional[str] = None
    streaming_chunks: List[StreamChunk] = field(default_factory=list)
    processing_log: List[str] = field(default_factory=list)
    
    def add_log_entry(self, message: str):
        """Add entry to processing log."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.processing_log.append(f"[{timestamp}] {message}")


@dataclass
class ResponseProcessingConfig:
    """Configuration for the response processing system."""
    enable_async_execution: bool = True
    max_concurrent_tools: int = 3
    execution_timeout: float = 30.0
    enable_streaming: bool = True
    enable_conversation_tracking: bool = True
    enable_error_recovery: bool = True
    auto_format_responses: bool = True
    log_processing_steps: bool = True
    require_tool_confirmation: bool = False
    max_retry_attempts: int = 2
    
    # Component configurations
    parser_config: Optional[ParsingConfig] = None
    formatter_config: Optional[FormattingConfig] = None
    conversation_config: Optional[ConversationConfig] = None
    streaming_config: Optional[StreamingConfig] = None


class AgentResponseProcessor:
    """
    Complete response processor for the Loki Code agent system.
    
    Integrates parsing, formatting, conversation management, streaming,
    and tool execution into a unified response processing pipeline.
    """
    
    def __init__(self,
                 parser: ResponseParser,
                 formatter: ResponseFormatter,
                 conversation: ConversationManager,
                 streaming: StreamingResponseHandler,
                 tool_executor: Optional[ToolExecutor] = None,
                 config: Optional[ResponseProcessingConfig] = None):
        self.parser = parser
        self.formatter = formatter
        self.conversation = conversation
        self.streaming = streaming
        self.tool_executor = tool_executor
        self.config = config or ResponseProcessingConfig()
        self.logger = get_logger(__name__)
        
        # Processing state
        self.active_sessions: Dict[str, ExecutionContext] = {}
        self.processing_stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "total_tool_executions": 0,
            "average_processing_time": 0.0
        }
        
        # Set up streaming callbacks
        if self.config.enable_streaming:
            self.streaming.add_callback(self._handle_stream_chunk)
    
    async def process_agent_interaction(self, 
                                      user_message: str,
                                      agent_response: str,
                                      execution_context: ExecutionContext) -> ProcessedInteraction:
        """Process a complete agent interaction.
        
        Args:
            user_message: User's input message
            agent_response: Agent's raw response
            execution_context: Execution context and parameters
            
        Returns:
            Complete processed interaction
        """
        start_time = time.perf_counter()
        interaction = ProcessedInteraction(
            execution_context=execution_context,
            interaction_result=InteractionResult(
                parsed_response=ParsedResponse(text_content=""),
                formatted_response=FormattedResponse()
            )
        )
        
        try:
            interaction.add_log_entry("Starting agent interaction processing")
            
            # Phase 1: Parse response
            interaction.add_log_entry("Phase 1: Parsing agent response")
            parsed_response = await self._parse_response(agent_response, execution_context)
            interaction.interaction_result.parsed_response = parsed_response
            interaction.interaction_result.processing_phases.append(ProcessingPhase.PARSING)
            
            # Phase 2: Execute tools if needed
            tool_results = []
            if parsed_response.has_valid_tool_calls:
                interaction.add_log_entry(f"Phase 2: Executing {len(parsed_response.tool_calls)} tool(s)")
                tool_results = await self._execute_tools(parsed_response, execution_context, interaction)
                interaction.interaction_result.tool_results = tool_results
                interaction.interaction_result.processing_phases.append(ProcessingPhase.TOOL_EXECUTION)
            
            # Phase 3: Format response
            interaction.add_log_entry("Phase 3: Formatting response")
            formatted_response = await self._format_response(parsed_response, tool_results, execution_context)
            interaction.interaction_result.formatted_response = formatted_response
            interaction.interaction_result.processing_phases.append(ProcessingPhase.FORMATTING)
            
            # Phase 4: Update conversation
            if self.config.enable_conversation_tracking:
                interaction.add_log_entry("Phase 4: Updating conversation")
                turn_id = await self._update_conversation(
                    user_message, parsed_response, tool_results, execution_context
                )
                interaction.conversation_turn_id = turn_id
            
            # Calculate execution time
            execution_time = time.perf_counter() - start_time
            interaction.interaction_result.execution_time = execution_time
            interaction.interaction_result.processing_phases.append(ProcessingPhase.COMPLETION)
            
            # Update statistics
            self._update_processing_stats(interaction)
            
            interaction.add_log_entry(f"Interaction completed successfully in {execution_time:.3f}s")
            
            return interaction
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            interaction.interaction_result.execution_time = execution_time
            interaction.interaction_result.success = False
            interaction.interaction_result.error = str(e)
            interaction.interaction_result.processing_phases.append(ProcessingPhase.ERROR_HANDLING)
            
            interaction.add_log_entry(f"Interaction failed after {execution_time:.3f}s: {e}")
            
            if self.config.enable_error_recovery:
                await self._handle_processing_error(e, interaction)
            
            return interaction
    
    async def process_streaming_interaction(self,
                                          user_message: str,
                                          response_generator,
                                          execution_context: ExecutionContext) -> ProcessedInteraction:
        """Process a streaming agent interaction.
        
        Args:
            user_message: User's input message
            response_generator: Async generator for agent response
            execution_context: Execution context and parameters
            
        Returns:
            Complete processed interaction with streaming data
        """
        interaction = ProcessedInteraction(
            execution_context=execution_context,
            interaction_result=InteractionResult(
                parsed_response=ParsedResponse(text_content=""),
                formatted_response=FormattedResponse()
            )
        )
        
        try:
            interaction.add_log_entry("Starting streaming interaction processing")
            
            # Set up streaming collection
            collected_chunks = []
            
            def chunk_collector(chunk: StreamChunk):
                collected_chunks.append(chunk)
                interaction.streaming_chunks.append(chunk)
            
            self.streaming.add_callback(chunk_collector)
            
            # Stream and parse response
            parsed_response = await self.streaming.stream_response(response_generator)
            interaction.interaction_result.parsed_response = parsed_response
            
            # Continue with standard processing
            if parsed_response.has_valid_tool_calls:
                tool_results = await self._execute_tools(parsed_response, execution_context, interaction)
                interaction.interaction_result.tool_results = tool_results
            
            # Format and finalize
            formatted_response = await self._format_response(
                parsed_response, 
                interaction.interaction_result.tool_results, 
                execution_context
            )
            interaction.interaction_result.formatted_response = formatted_response
            
            # Clean up
            self.streaming.remove_callback(chunk_collector)
            
            interaction.add_log_entry("Streaming interaction completed successfully")
            
            return interaction
            
        except Exception as e:
            interaction.interaction_result.success = False
            interaction.interaction_result.error = str(e)
            interaction.add_log_entry(f"Streaming interaction failed: {e}")
            
            return interaction
    
    async def _parse_response(self, response: str, context: ExecutionContext) -> ParsedResponse:
        """Parse agent response."""
        conversation_context = None
        if context.conversation_context:
            conversation_context = context.conversation_context.__dict__
        
        return self.parser.parse_llm_response(response, conversation_context)
    
    async def _execute_tools(self, 
                           parsed_response: ParsedResponse,
                           context: ExecutionContext,
                           interaction: ProcessedInteraction) -> List[ToolResult]:
        """Execute tools from parsed response."""
        if not self.tool_executor:
            self.logger.warning("No tool executor available for tool execution")
            return []
        
        valid_calls = [call for call in parsed_response.tool_calls if call.is_valid]
        if not valid_calls:
            return []
        
        # Check execution limits
        if len(valid_calls) > context.max_tool_executions:
            self.logger.warning(f"Tool execution limit exceeded: {len(valid_calls)} > {context.max_tool_executions}")
            valid_calls = valid_calls[:context.max_tool_executions]
        
        tool_results = []
        
        if context.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            # Execute tools sequentially
            for tool_call in valid_calls:
                if context.require_confirmation:
                    # In a real implementation, this would prompt the user
                    interaction.add_log_entry(f"Would execute tool: {tool_call.raw_call.tool_name}")
                
                result = await self._execute_single_tool(tool_call, context)
                tool_results.append(result)
                
                # Stop on critical errors
                if not result.success and result.error and "critical" in result.error.lower():
                    break
        
        elif context.execution_strategy == ExecutionStrategy.PARALLEL:
            # Execute independent tools in parallel
            if self.config.enable_async_execution:
                tasks = []
                for tool_call in valid_calls[:self.config.max_concurrent_tools]:
                    task = asyncio.create_task(self._execute_single_tool(tool_call, context))
                    tasks.append(task)
                
                tool_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                for i, result in enumerate(tool_results):
                    if isinstance(result, Exception):
                        tool_results[i] = ToolResult(
                            tool_name=valid_calls[i].raw_call.tool_name,
                            success=False,
                            error=str(result)
                        )
            else:
                # Fall back to sequential
                for tool_call in valid_calls:
                    result = await self._execute_single_tool(tool_call, context)
                    tool_results.append(result)
        
        return tool_results
    
    async def _execute_single_tool(self, tool_call, context: ExecutionContext) -> ToolResult:
        """Execute a single tool call."""
        try:
            # Use tool executor if available
            if self.tool_executor:
                # Create execution context for tool
                tool_context = {
                    "session_id": context.session_id,
                    "project_path": context.project_path,
                    "timeout": context.timeout_seconds
                }
                
                return await self.tool_executor.execute_tool(
                    tool_call.raw_call.tool_name,
                    tool_call.raw_call.input_data,
                    tool_context
                )
            else:
                # Simulate tool execution
                await asyncio.sleep(0.1)
                return ToolResult(
                    tool_name=tool_call.raw_call.tool_name,
                    success=True,
                    content=f"Simulated execution of {tool_call.raw_call.tool_name}",
                    execution_time=0.1
                )
                
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.raw_call.tool_name,
                success=False,
                error=str(e)
            )
    
    async def _format_response(self, 
                             parsed_response: ParsedResponse,
                             tool_results: List[ToolResult],
                             context: ExecutionContext) -> FormattedResponse:
        """Format the complete response."""
        if not self.config.auto_format_responses:
            return FormattedResponse()
        
        return self.formatter.format_agent_response(parsed_response, tool_results)
    
    async def _update_conversation(self,
                                 user_message: str,
                                 parsed_response: ParsedResponse,
                                 tool_results: List[ToolResult],
                                 context: ExecutionContext) -> str:
        """Update conversation with the interaction."""
        return self.conversation.add_turn(
            context.session_id,
            user_message,
            parsed_response,
            tool_results
        )
    
    async def _handle_processing_error(self, error: Exception, interaction: ProcessedInteraction):
        """Handle processing errors with recovery attempts."""
        self.logger.error(f"Processing error: {error}")
        
        # Add error information to response
        error_response = self.formatter.format_error(str(error), "Processing Error")
        interaction.interaction_result.formatted_response = error_response
        
        # Log error details
        interaction.add_log_entry(f"Error handled: {type(error).__name__}: {error}")
    
    def _handle_stream_chunk(self, chunk: StreamChunk):
        """Handle streaming chunks."""
        if self.config.log_processing_steps:
            if chunk.chunk_type == ChunkType.STATUS:
                self.logger.debug(f"Stream status: {chunk.content}")
            elif chunk.chunk_type == ChunkType.ERROR:
                self.logger.error(f"Stream error: {chunk.content}")
    
    def _update_processing_stats(self, interaction: ProcessedInteraction):
        """Update processing statistics."""
        self.processing_stats["total_interactions"] += 1
        
        if interaction.interaction_result.success:
            self.processing_stats["successful_interactions"] += 1
        
        self.processing_stats["total_tool_executions"] += len(interaction.interaction_result.tool_results)
        
        # Update average processing time
        total_time = (self.processing_stats["average_processing_time"] * 
                     (self.processing_stats["total_interactions"] - 1) + 
                     interaction.interaction_result.execution_time)
        self.processing_stats["average_processing_time"] = total_time / self.processing_stats["total_interactions"]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        
        # Add success rate
        if stats["total_interactions"] > 0:
            stats["success_rate"] = stats["successful_interactions"] / stats["total_interactions"]
        else:
            stats["success_rate"] = 0.0
        
        # Add active sessions count
        stats["active_sessions"] = len(self.active_sessions)
        
        return stats
    
    def create_execution_context(self,
                               session_id: str,
                               user_message: str,
                               **kwargs) -> ExecutionContext:
        """Create an execution context with default values."""
        return ExecutionContext(
            session_id=session_id,
            user_message=user_message,
            project_path=kwargs.get("project_path"),
            execution_strategy=kwargs.get("execution_strategy", ExecutionStrategy.SEQUENTIAL),
            max_tool_executions=kwargs.get("max_tool_executions", 5),
            timeout_seconds=kwargs.get("timeout_seconds", self.config.execution_timeout),
            enable_streaming=kwargs.get("enable_streaming", self.config.enable_streaming),
            require_confirmation=kwargs.get("require_confirmation", self.config.require_tool_confirmation),
            metadata=kwargs.get("metadata", {})
        )
    
    async def cleanup_session(self, session_id: str):
        """Clean up resources for a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Clean up conversation
        self.conversation.end_conversation(session_id)
        
        self.logger.debug(f"Cleaned up session {session_id}")
    
    async def cleanup(self):
        """Clean up all resources."""
        # Clean up all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.cleanup_session(session_id)
        
        # Clean up streaming
        self.streaming.cleanup()
        
        # Clean up conversations
        self.conversation.cleanup_inactive_conversations()
        
        self.logger.info("Response processor cleanup completed")