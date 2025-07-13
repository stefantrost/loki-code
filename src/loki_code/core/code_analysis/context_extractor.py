"""
Context extractor for code analysis.

This module provides backward compatibility while using the new simplified implementation.
"""

# Import from the new simplified implementation
from .context_extractor_core import ContextExtractor

# Import types for backward compatibility
from .context_types import (
    ContextLevel, ContextConfig, FunctionContext,
    ClassContext, FileContext, ProjectContext
)

__all__ = [
    "ContextExtractor",
    "ContextLevel",
    "ContextConfig", 
    "FunctionContext",
    "ClassContext",
    "FileContext",
    "ProjectContext"
]