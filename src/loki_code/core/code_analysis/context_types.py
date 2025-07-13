"""
Context type definitions for code analysis.

Extracted from context_extractor.py for better organization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ContextLevel(Enum):
    """Levels of context detail to extract."""
    MINIMAL = "minimal"      # Just basic info
    STANDARD = "standard"    # Standard analysis
    DETAILED = "detailed"    # Detailed with relationships
    COMPREHENSIVE = "comprehensive"  # Everything


@dataclass
class ContextConfig:
    """Configuration for context extraction."""
    level: ContextLevel = ContextLevel.STANDARD
    include_docstrings: bool = True
    include_comments: bool = False
    include_imports: bool = True
    include_relationships: bool = True
    max_depth: int = 3
    max_context_length: int = 10000
    include_examples: bool = False
    language_specific_features: bool = True


@dataclass
class FunctionContext:
    """Context information for a function."""
    name: str
    signature: str
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    line_range: tuple[int, int] = (0, 0)
    complexity_score: float = 0.0
    calls_functions: List[str] = field(default_factory=list)
    called_by_functions: List[str] = field(default_factory=list)


@dataclass
class ClassContext:
    """Context information for a class."""
    name: str
    base_classes: List[str] = field(default_factory=list)
    methods: List[FunctionContext] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    line_range: tuple[int, int] = (0, 0)
    is_abstract: bool = False


@dataclass
class FileContext:
    """Context information for a file."""
    file_path: str
    language: str
    imports: List[str] = field(default_factory=list)
    functions: List[FunctionContext] = field(default_factory=list)
    classes: List[ClassContext] = field(default_factory=list)
    global_variables: List[str] = field(default_factory=list)
    line_count: int = 0
    complexity_score: float = 0.0
    docstring: Optional[str] = None


@dataclass
class ProjectContext:
    """Context information for a project."""
    project_path: str
    files: List[FileContext] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    total_lines: int = 0
    languages: List[str] = field(default_factory=list)