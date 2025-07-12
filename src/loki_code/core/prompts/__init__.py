"""
Prompt Template System for Loki Code.

This module provides a comprehensive prompt template system that enables
the LLM to effectively use our tool system. It includes:

- Template definitions for different agent types
- Dynamic prompt building with context
- Tool calling instructions and formatting
- Context-aware prompt generation

Usage:
    from loki_code.core.prompts import (
        PromptTemplate, PromptTemplateRegistry, PromptBuilder,
        PromptContext, ContextBuilder, ToolPromptFormatter
    )
    
    # Create prompt builder
    builder = PromptBuilder(template_registry, tool_registry)
    
    # Build context-aware prompt
    context = PromptContext(user_message="Help me analyze this file")
    prompt = builder.build_prompt("coding_agent", context)
"""

from .templates import (
    PromptTemplate,
    PromptTemplateRegistry,
    CODING_AGENT_TEMPLATE,
    CODE_REVIEW_TEMPLATE,
    DEBUGGING_TEMPLATE,
    FILE_ANALYSIS_TEMPLATE,
    PROJECT_ANALYSIS_TEMPLATE,
    create_default_template_registry
)

from .builder import (
    PromptBuilder,
    ToolPromptFormatter,
    ToolCall,
    PromptBuildError
)

from .context import (
    PromptContext,
    ContextBuilder,
    ConversationEntry,
    CodeContext,
    ProjectContext,
    ContextBuildError
)

__all__ = [
    # Template system
    "PromptTemplate",
    "PromptTemplateRegistry",
    
    # Pre-defined templates
    "CODING_AGENT_TEMPLATE",
    "CODE_REVIEW_TEMPLATE", 
    "DEBUGGING_TEMPLATE",
    "FILE_ANALYSIS_TEMPLATE",
    "PROJECT_ANALYSIS_TEMPLATE",
    "create_default_template_registry",
    
    # Prompt building
    "PromptBuilder",
    "ToolPromptFormatter",
    "ToolCall",
    "PromptBuildError",
    
    # Context management
    "PromptContext",
    "ContextBuilder",
    "ConversationEntry",
    "CodeContext", 
    "ProjectContext",
    "ContextBuildError"
]

__version__ = "0.1.0"