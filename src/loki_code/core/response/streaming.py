"""
Streaming response handling for Loki Code.

This module provides real-time streaming capabilities for agent responses,
tool executions, and conversation updates with support for progressive
display and user interaction.
"""

import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union, Iterator
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

from .parser import ParsedResponse, RawToolCall, ToolCallPattern
from .formatter import ResponseFormatter, FormattedResponse, ResponseSection, ContentType
from ...tools.types import ToolResult
from ...utils.logging import get_logger


class StreamState(Enum):
    """States of a streaming response."""
    IDLE = "idle"
    THINKING = "thinking"
    PARSING = "parsing"
    TOOL_EXECUTION = "tool_execution"
    FORMATTING = "formatting"
    COMPLETE = "complete"
    ERROR = "error"


class ChunkType(Enum):
    """Types of streaming chunks."""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    ERROR = "error"
    METADATA = "metadata"
    COMPLETION = "completion"


@dataclass
class StreamChunk:
    """A chunk of streaming data."""
    chunk_type: ChunkType
    content: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "type": self.chunk_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "is_final": self.is_final
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming responses."""
    chunk_size: int = 100
    enable_tool_streaming: bool = True
    enable_status_updates: bool = True
    buffer_incomplete_calls: bool = True
    update_interval_ms: int = 50
    max_buffer_size: int = 1000
    enable_progressive_parsing: bool = True
    show_thinking_process: bool = True
    enable_real_time_formatting: bool = True
    typing_delay_ms: int = 20  # Simulated typing delay


# Type aliases for callbacks
StreamCallback = Callable[[StreamChunk], None]
AsyncStreamCallback = Callable[[StreamChunk], None]


class ChunkProcessor:
    """
    Processor for streaming chunks with buffering and parsing.
    
    Handles progressive parsing of tool calls and intelligent
    buffering for incomplete responses.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.logger = get_logger(__name__)
        
        # Buffering state
        self.text_buffer = ""
        self.tool_call_buffer = ""
        self.incomplete_calls: List[RawToolCall] = []
        
        # Parsing state
        self.partial_tool_calls: List[Dict[str, Any]] = []
        self.current_tool_pattern: Optional[ToolCallPattern] = None
        
        # Tool call detection patterns (simplified for streaming)
        self.tool_patterns = {
            ToolCallPattern.CODE_BLOCK: r'```tool\s*\n',
            ToolCallPattern.FUNCTION_CALL: r'\w+\s*\(',
            ToolCallPattern.JSON_CALL: r'{"tool":\s*"',
            ToolCallPattern.MARKDOWN_CALL: r'\*\*\w+\*\*:',
        }
    
    def process_text_chunk(self, text: str) -> List[StreamChunk]:
        """Process a chunk of text and extract any complete elements.
        
        Args:
            text: New text chunk
            
        Returns:
            List of processed chunks
        """
        chunks = []
        
        # Add to buffer
        self.text_buffer += text
        
        # Check for complete tool calls
        if self.config.enable_tool_streaming:
            tool_chunks = self._extract_tool_calls_from_buffer()
            chunks.extend(tool_chunks)
        
        # Extract complete text segments
        text_chunks = self._extract_text_segments()
        chunks.extend(text_chunks)
        
        # Generate status updates
        if self.config.enable_status_updates:
            status_chunk = self._generate_status_update()
            if status_chunk:
                chunks.append(status_chunk)
        
        return chunks
    
    def finalize_buffer(self) -> List[StreamChunk]:
        """Finalize any remaining content in buffers.
        
        Returns:
            Final chunks from remaining buffer content
        """
        chunks = []
        
        # Process any remaining text
        if self.text_buffer.strip():
            chunks.append(StreamChunk(
                chunk_type=ChunkType.TEXT,
                content=self.text_buffer,
                is_final=True
            ))
            self.text_buffer = ""
        
        # Process any incomplete tool calls
        for incomplete_call in self.incomplete_calls:
            chunks.append(StreamChunk(
                chunk_type=ChunkType.TOOL_CALL,
                content=incomplete_call,
                metadata={"status": "incomplete"},
                is_final=True
            ))
        
        self.incomplete_calls.clear()
        
        return chunks
    
    def reset(self):
        """Reset processor state."""
        self.text_buffer = ""
        self.tool_call_buffer = ""
        self.incomplete_calls.clear()
        self.partial_tool_calls.clear()
        self.current_tool_pattern = None
    
    def _extract_tool_calls_from_buffer(self) -> List[StreamChunk]:
        """Extract complete tool calls from buffer."""
        chunks = []
        
        # Look for tool call patterns
        for pattern_type, pattern in self.tool_patterns.items():
            import re
            matches = list(re.finditer(pattern, self.text_buffer))
            
            for match in matches:
                # Try to extract complete tool call
                tool_call = self._try_extract_complete_tool_call(
                    self.text_buffer[match.start():], pattern_type
                )
                
                if tool_call:
                    chunks.append(StreamChunk(
                        chunk_type=ChunkType.TOOL_CALL,
                        content=tool_call,
                        metadata={"pattern": pattern_type.value}
                    ))
                    
                    # Remove from buffer
                    self.text_buffer = (
                        self.text_buffer[:match.start()] + 
                        self.text_buffer[match.start() + len(tool_call.raw_text):]
                    )
        
        return chunks
    
    def _try_extract_complete_tool_call(self, text: str, 
                                      pattern_type: ToolCallPattern) -> Optional[RawToolCall]:
        """Try to extract a complete tool call from text."""
        try:
            if pattern_type == ToolCallPattern.CODE_BLOCK:
                return self._extract_code_block_call(text)
            elif pattern_type == ToolCallPattern.FUNCTION_CALL:
                return self._extract_function_call(text)
            elif pattern_type == ToolCallPattern.JSON_CALL:
                return self._extract_json_call(text)
            elif pattern_type == ToolCallPattern.MARKDOWN_CALL:
                return self._extract_markdown_call(text)
        except Exception as e:
            self.logger.debug(f"Failed to extract tool call: {e}")
        
        return None
    
    def _extract_code_block_call(self, text: str) -> Optional[RawToolCall]:
        """Extract code block format tool call."""
        import re
        pattern = r'```tool\s*\n(?:tool_name:\s*)?(\w+)\s*\n(?:input:\s*)?({.*?})\s*\n```'
        match = re.match(pattern, text, re.DOTALL)
        
        if match:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                return RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.CODE_BLOCK,
                    confidence=0.95
                )
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_function_call(self, text: str) -> Optional[RawToolCall]:
        """Extract function call format."""
        import re
        pattern = r'(\w+)\s*\(({.*?})\)'
        match = re.match(pattern, text)
        
        if match:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                return RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.FUNCTION_CALL,
                    confidence=0.8
                )
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_json_call(self, text: str) -> Optional[RawToolCall]:
        """Extract JSON format tool call."""
        import re
        pattern = r'{"tool":\s*"(\w+)",\s*"input":\s*({.*?})}'
        match = re.match(pattern, text)
        
        if match:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                return RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.JSON_CALL,
                    confidence=0.9
                )
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_markdown_call(self, text: str) -> Optional[RawToolCall]:
        """Extract markdown format tool call."""
        import re
        pattern = r'\*\*(\w+)\*\*:\s*({.*?})'
        match = re.match(pattern, text)
        
        if match:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                return RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.MARKDOWN_CALL,
                    confidence=0.75
                )
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_text_segments(self) -> List[StreamChunk]:
        """Extract complete text segments from buffer."""
        chunks = []
        
        # Simple sentence-based chunking
        sentences = self._split_into_sentences(self.text_buffer)
        
        if len(sentences) > 1:
            # Keep last incomplete sentence in buffer
            complete_text = " ".join(sentences[:-1])
            if complete_text.strip():
                chunks.append(StreamChunk(
                    chunk_type=ChunkType.TEXT,
                    content=complete_text
                ))
            
            # Update buffer with remaining text
            self.text_buffer = sentences[-1]
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_status_update(self) -> Optional[StreamChunk]:
        """Generate status update based on current buffer state."""
        status = None
        
        if self.incomplete_calls:
            status = "Processing tool calls..."
        elif len(self.text_buffer) > 100:
            status = "Generating response..."
        elif self.partial_tool_calls:
            status = "Parsing tool calls..."
        
        if status:
            return StreamChunk(
                chunk_type=ChunkType.STATUS,
                content=status,
                metadata={"buffer_size": len(self.text_buffer)}
            )
        
        return None


class StreamingResponseHandler:
    """
    Handler for streaming agent responses with real-time formatting.
    
    Provides progressive display of agent responses, tool executions,
    and conversation updates with support for user interaction.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None,
                 formatter: Optional[ResponseFormatter] = None):
        self.config = config or StreamingConfig()
        self.formatter = formatter
        self.logger = get_logger(__name__)
        
        # Processing components
        self.chunk_processor = ChunkProcessor(self.config)
        
        # Streaming state
        self.current_state = StreamState.IDLE
        self.stream_callbacks: List[StreamCallback] = []
        self.async_callbacks: List[AsyncStreamCallback] = []
        
        # Buffers and tracking
        self.response_buffer = ""
        self.formatted_sections: List[ResponseSection] = []
        self.tool_executions: Dict[str, Any] = {}
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def add_callback(self, callback: Union[StreamCallback, AsyncStreamCallback]):
        """Add a callback for stream events."""
        if asyncio.iscoroutinefunction(callback):
            self.async_callbacks.append(callback)
        else:
            self.stream_callbacks.append(callback)
    
    def remove_callback(self, callback: Union[StreamCallback, AsyncStreamCallback]):
        """Remove a callback."""
        if callback in self.stream_callbacks:
            self.stream_callbacks.remove(callback)
        if callback in self.async_callbacks:
            self.async_callbacks.remove(callback)
    
    async def stream_response(self, response_generator: AsyncGenerator[str, None]) -> ParsedResponse:
        """Stream and process an agent response.
        
        Args:
            response_generator: Async generator yielding response chunks
            
        Returns:
            Complete parsed response
        """
        self.current_state = StreamState.THINKING
        self.chunk_processor.reset()
        
        try:
            # Process chunks as they arrive
            async for chunk in response_generator:
                await self._process_chunk(chunk)
            
            # Finalize processing
            await self._finalize_streaming()
            
            # Parse complete response
            parsed_response = await self._parse_complete_response()
            
            self.current_state = StreamState.COMPLETE
            await self._emit_completion_chunk(parsed_response)
            
            return parsed_response
            
        except Exception as e:
            self.current_state = StreamState.ERROR
            await self._emit_error_chunk(str(e))
            raise
    
    async def stream_tool_execution(self, tool_name: str, tool_input: Dict[str, Any]) -> AsyncGenerator[StreamChunk, None]:
        """Stream tool execution progress.
        
        Args:
            tool_name: Name of tool being executed
            tool_input: Tool input parameters
            
        Yields:
            Progress chunks for tool execution
        """
        execution_id = f"{tool_name}_{time.time()}"
        self.tool_executions[execution_id] = {
            "tool_name": tool_name,
            "input": tool_input,
            "start_time": time.time(),
            "status": "starting"
        }
        
        # Emit start chunk
        yield StreamChunk(
            chunk_type=ChunkType.STATUS,
            content=f"Executing {tool_name}...",
            metadata={"execution_id": execution_id, "tool_name": tool_name}
        )
        
        # Simulate progress updates (in real implementation, this would be actual progress)
        progress_steps = ["initializing", "processing", "completing"]
        for step in progress_steps:
            await asyncio.sleep(0.1)  # Simulate work
            yield StreamChunk(
                chunk_type=ChunkType.STATUS,
                content=f"{tool_name}: {step}",
                metadata={"execution_id": execution_id, "step": step}
            )
        
        # Clean up
        del self.tool_executions[execution_id]
    
    def create_simulated_stream(self, text: str, delay_ms: Optional[int] = None) -> AsyncGenerator[str, None]:
        """Create a simulated stream from static text for testing.
        
        Args:
            text: Text to stream
            delay_ms: Delay between chunks in milliseconds
            
        Returns:
            Async generator that yields text chunks
        """
        return self._simulate_typing(text, delay_ms or self.config.typing_delay_ms)
    
    async def _simulate_typing(self, text: str, delay_ms: int) -> AsyncGenerator[str, None]:
        """Simulate typing effect for text."""
        words = text.split()
        current_chunk = ""
        
        for word in words:
            current_chunk += word + " "
            
            # Yield chunk at word boundaries or when chunk is large enough
            if (len(current_chunk) >= self.config.chunk_size or 
                word.endswith(('.', '!', '?', ':', ';'))):
                yield current_chunk
                current_chunk = ""
                
                # Add typing delay
                await asyncio.sleep(delay_ms / 1000.0)
        
        # Yield any remaining content
        if current_chunk.strip():
            yield current_chunk
    
    async def _process_chunk(self, chunk: str):
        """Process a single chunk of response."""
        self.response_buffer += chunk
        
        # Process through chunk processor
        processed_chunks = self.chunk_processor.process_text_chunk(chunk)
        
        # Emit processed chunks
        for processed_chunk in processed_chunks:
            await self._emit_chunk(processed_chunk)
            
            # Handle real-time formatting
            if (self.config.enable_real_time_formatting and 
                processed_chunk.chunk_type == ChunkType.TEXT):
                await self._handle_real_time_formatting(processed_chunk)
    
    async def _finalize_streaming(self):
        """Finalize streaming and process remaining buffers."""
        self.current_state = StreamState.PARSING
        
        # Process any remaining chunks
        final_chunks = self.chunk_processor.finalize_buffer()
        for chunk in final_chunks:
            await self._emit_chunk(chunk)
    
    async def _parse_complete_response(self) -> ParsedResponse:
        """Parse the complete buffered response."""
        # This would integrate with the ResponseParser
        # For now, create a basic parsed response
        return ParsedResponse(
            text_content=self.response_buffer,
            confidence_score=0.8,
            metadata={"streaming": True, "chunks_processed": len(self.formatted_sections)}
        )
    
    async def _handle_real_time_formatting(self, chunk: StreamChunk):
        """Handle real-time formatting of content."""
        if not self.formatter:
            return
        
        try:
            # Create a formatted section for the chunk
            section = ResponseSection(
                title="Live Response",
                content=str(chunk.content),
                content_type=ContentType.AGENT_RESPONSE
            )
            
            self.formatted_sections.append(section)
            
            # Emit formatted chunk
            formatted_chunk = StreamChunk(
                chunk_type=ChunkType.TEXT,
                content=section,
                metadata={"formatted": True}
            )
            await self._emit_chunk(formatted_chunk)
            
        except Exception as e:
            self.logger.warning(f"Real-time formatting failed: {e}")
    
    async def _emit_chunk(self, chunk: StreamChunk):
        """Emit a chunk to all callbacks."""
        # Synchronous callbacks
        for callback in self.stream_callbacks:
            try:
                callback(chunk)
            except Exception as e:
                self.logger.error(f"Stream callback error: {e}")
        
        # Asynchronous callbacks
        for callback in self.async_callbacks:
            try:
                await callback(chunk)
            except Exception as e:
                self.logger.error(f"Async stream callback error: {e}")
    
    async def _emit_completion_chunk(self, parsed_response: ParsedResponse):
        """Emit completion chunk."""
        completion_chunk = StreamChunk(
            chunk_type=ChunkType.COMPLETION,
            content=parsed_response,
            is_final=True,
            metadata={"total_length": len(self.response_buffer)}
        )
        await self._emit_chunk(completion_chunk)
    
    async def _emit_error_chunk(self, error: str):
        """Emit error chunk."""
        error_chunk = StreamChunk(
            chunk_type=ChunkType.ERROR,
            content=error,
            is_final=True
        )
        await self._emit_chunk(error_chunk)
    
    def get_current_state(self) -> StreamState:
        """Get current streaming state."""
        return self.current_state
    
    def get_response_preview(self) -> str:
        """Get preview of current response."""
        return self.response_buffer[:500] + ("..." if len(self.response_buffer) > 500 else "")
    
    def cleanup(self):
        """Clean up streaming resources."""
        self.stream_callbacks.clear()
        self.async_callbacks.clear()
        self.chunk_processor.reset()
        self.executor.shutdown(wait=False)