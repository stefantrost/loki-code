"""
Response processing system for Loki Code.

This package provides comprehensive response processing capabilities including
tool call parsing, result formatting, conversation management, and streaming
response handling for the intelligent agent system.

Components:
- ResponseParser: Parse and validate LLM responses with tool calls
- ResponseFormatter: Format responses for user consumption with rich styling
- ConversationManager: Manage conversation flow and context
- StreamingHandler: Handle real-time streaming responses
- AgentResponseProcessor: Complete integration with agent system

Usage:
    from loki_code.core.response import (
        ResponseParser, ResponseFormatter, ConversationManager, 
        AgentResponseProcessor, create_default_processor
    )
    
    # Create complete response processing system
    processor = create_default_processor(tool_registry)
    
    # Process agent interaction
    result = await processor.process_agent_interaction(
        user_message="Analyze the code",
        agent_response=llm_response
    )
"""

from .parser import (
    ResponseParser, ParsedResponse, RawToolCall, ValidatedToolCall,
    ParsingConfig, ToolCallPattern
)
from .formatter import (
    ResponseFormatter, FormattedResponse, ResponseSection, FormattingConfig,
    FormatStyle, ContentType
)
from .conversation import (
    ConversationManager, Conversation, ConversationTurn, ConversationContext,
    ConversationConfig, ConversationIntent, ContextWindow
)
from .streaming import (
    StreamingResponseHandler, StreamState, StreamingConfig,
    ChunkProcessor, StreamCallback
)
from .processor import (
    AgentResponseProcessor, ProcessedInteraction, ResponseProcessingConfig,
    InteractionResult, ExecutionContext
)

# Version info
__version__ = "0.1.0"

# Export all public APIs
__all__ = [
    # Parser
    "ResponseParser",
    "ParsedResponse",
    "RawToolCall", 
    "ValidatedToolCall",
    "ParsingConfig",
    "ToolCallPattern",
    
    # Formatter
    "ResponseFormatter",
    "FormattedResponse",
    "ResponseSection",
    "FormattingConfig",
    "FormatStyle",
    "ContentType",
    
    # Conversation
    "ConversationManager",
    "Conversation",
    "ConversationTurn",
    "ConversationContext",
    "ConversationConfig",
    "ConversationIntent",
    "ContextWindow",
    
    # Streaming
    "StreamingResponseHandler",
    "StreamState",
    "StreamingConfig",
    "ChunkProcessor",
    "StreamCallback",
    
    # Main Processor
    "AgentResponseProcessor",
    "ProcessedInteraction",
    "ResponseProcessingConfig",
    "InteractionResult",
    "ExecutionContext",
    
    # Convenience functions
    "create_default_processor",
    "create_default_config"
]

# System information
RESPONSE_SYSTEM_INFO = {
    "version": __version__,
    "features": [
        "tool_call_parsing",
        "response_validation",
        "rich_formatting",
        "conversation_management",
        "streaming_support",
        "error_recovery",
        "context_tracking"
    ],
    "supported_formats": [
        "json_tool_calls",
        "function_call_syntax", 
        "markdown_blocks",
        "structured_responses"
    ],
    "formatting_engines": [
        "rich_console",
        "markdown",
        "plain_text",
        "colored_terminal"
    ]
}


def get_response_system_info() -> dict:
    """Get information about the response processing system.
    
    Returns:
        Dictionary with system information
    """
    return RESPONSE_SYSTEM_INFO.copy()


def create_default_processor(tool_registry, **kwargs):
    """Create a complete response processor with default configuration.
    
    Args:
        tool_registry: Tool registry for validation
        **kwargs: Additional configuration options
        
    Returns:
        AgentResponseProcessor ready for use
    """
    # Create components with default configs
    parser_config = ParsingConfig(**kwargs.get('parser_config', {}))
    formatter_config = FormattingConfig(**kwargs.get('formatter_config', {}))
    conversation_config = ConversationConfig(**kwargs.get('conversation_config', {}))
    streaming_config = StreamingConfig(**kwargs.get('streaming_config', {}))
    
    # Create components
    parser = ResponseParser(tool_registry, parser_config)
    formatter = ResponseFormatter(formatter_config)
    conversation = ConversationManager(conversation_config)
    streaming = StreamingResponseHandler(streaming_config, formatter)
    
    # Create main processor
    processor_config = ResponseProcessingConfig(**kwargs.get('processor_config', {}))
    processor = AgentResponseProcessor(
        parser=parser,
        formatter=formatter,
        conversation=conversation,
        streaming=streaming,
        config=processor_config
    )
    
    return processor


def create_default_config(**overrides):
    """Create default configuration for response processing.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Complete configuration dictionary
    """
    config = {
        'parser_config': {
            'confidence_threshold': 0.7,
            'strict_validation': True,
            'enable_fuzzy_matching': True
        },
        'formatter_config': {
            'show_reasoning': True,
            'show_confidence': True,
            'use_rich_formatting': True,
            'max_text_length': 1000
        },
        'conversation_config': {
            'max_turns': 50,
            'context_window': 10,
            'enable_context_compression': True
        },
        'streaming_config': {
            'chunk_size': 100,
            'enable_tool_streaming': True,
            'buffer_incomplete_calls': True
        },
        'processor_config': {
            'enable_async_execution': True,
            'max_concurrent_tools': 3,
            'execution_timeout': 30.0
        }
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    return config


def validate_response_system():
    """Validate the response processing system is properly configured.
    
    Returns:
        Dictionary with validation results
    """
    try:
        # Test imports
        from rich.console import Console
        rich_available = True
    except ImportError:
        rich_available = False
    
    try:
        import json
        import re
        import asyncio
        core_deps_available = True
    except ImportError:
        core_deps_available = False
    
    return {
        "system_available": core_deps_available,
        "rich_formatting": rich_available,
        "async_support": True,
        "streaming_support": True,
        "validation_errors": [] if core_deps_available else ["Missing core dependencies"]
    }


# Auto-validate on import
_validation_result = validate_response_system()
if not _validation_result["system_available"]:
    import warnings
    warnings.warn("Response processing system not fully available: missing dependencies")