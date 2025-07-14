"""
Intelligent file reader tool for Loki Code.

This tool provides rich, context-aware file reading capabilities by leveraging
Tree-sitter for code analysis and structure extraction. It can read both simple
text files and provide detailed analysis for supported programming languages.

Features:
- Multi-language code analysis using Tree-sitter
- Intelligent suggestions based on file content
- Security-conscious file access with validation
- MCP-compatible interface
- Multiple analysis levels (minimal, standard, detailed, comprehensive)
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.code_analysis import (
    TreeSitterParser, CodeAnalyzer, ContextExtractor,
    ContextLevel, ContextConfig, SupportedLanguage,
    get_language_from_extension, FileContext
)
from .base import BaseTool, SimpleFileTool
from .types import (
    ToolSchema, ToolContext, ToolResult, ToolCapability,
    SecurityLevel, ConfirmationLevel, InputValidationSchema
)
from .exceptions import ToolValidationError, ToolExecutionError
from ..utils.logging import get_logger
from ..utils.error_handling import handle_tool_execution


@dataclass
class FileReaderInput:
    """Input schema for file reader tool."""
    file_path: str
    analysis_level: str = "standard"  # minimal, standard, detailed, comprehensive
    include_context: bool = True      # Include Tree-sitter analysis
    max_size_mb: int = 10            # Safety limit
    encoding: str = "utf-8"


@dataclass  
class FileInfo:
    """Basic file information."""
    path: str
    size_bytes: int
    lines: int
    language: Optional[str]
    encoding: str
    last_modified: str
    is_binary: bool


@dataclass
class FileReaderOutput:
    """Output schema for file reader tool."""
    content: str
    file_info: FileInfo
    code_analysis: Optional[Dict[str, Any]] = None
    analysis_summary: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)


class FileReaderTool(SimpleFileTool):
    """
    Intelligent file reader with code analysis capabilities.
    
    This tool reads files and provides optional Tree-sitter based analysis
    for supported programming languages. It includes safety checks and
    intelligent suggestions based on the file content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the file reader tool."""
        super().__init__(config)
        
        # Initialize Tree-sitter components
        self.parser = TreeSitterParser()
        self.analyzer = CodeAnalyzer(self.parser) 
        self.context_extractor = ContextExtractor(self.parser)
        
        # Configuration
        self.default_max_size_mb = config.get('max_file_size_mb', 10) if config else 10
        self.logger = get_logger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the tool components."""
        await super().initialize()
        # Tree-sitter components are initialized lazily
        self.logger.debug("FileReaderTool initialized")
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema definition."""
        input_schema = InputValidationSchema.create_schema(
            properties={
                "file_path": InputValidationSchema.file_path_field(
                    "Path to the file to read"
                ),
                "analysis_level": {
                    "type": "string",
                    "enum": ["minimal", "standard", "detailed", "comprehensive"],
                    "default": "standard",
                    "description": "Level of code analysis to perform"
                },
                "include_context": InputValidationSchema.boolean_field(
                    "Include Tree-sitter code analysis", True
                ),
                "max_size_mb": InputValidationSchema.integer_field(
                    "Maximum file size in MB", minimum=1, maximum=100
                ),
                "encoding": {
                    "type": "string", 
                    "default": "utf-8",
                    "description": "File encoding to use"
                }
            },
            required=["file_path"],
            description="Input for intelligent file reading with optional code analysis"
        )
        
        output_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "File content"},
                "file_info": {
                    "type": "object",
                    "description": "Basic file information"
                },
                "code_analysis": {
                    "type": "object", 
                    "description": "Tree-sitter code analysis (if applicable)"
                },
                "analysis_summary": {
                    "type": "string",
                    "description": "Human-readable analysis summary"
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Intelligent suggestions based on file content"
                }
            },
            "required": ["content", "file_info"]
        }
        
        return ToolSchema(
            name="file_reader",
            description="Read and analyze files with intelligent Tree-sitter parsing",
            input_schema=input_schema,
            output_schema=output_schema,
            capabilities=[
                ToolCapability.READ_FILE,
                ToolCapability.CODE_ANALYSIS,
                ToolCapability.PROJECT_NAVIGATION
            ],
            security_level=SecurityLevel.SAFE,
            confirmation_level=ConfirmationLevel.NONE,
            mcp_compatible=True,
            version="1.0.0",
            tags=["file", "read", "analysis", "tree-sitter"],
            examples=[
                {
                    "input": {
                        "file_path": "example.py",
                        "analysis_level": "standard"
                    },
                    "output": {
                        "content": "# Example Python file...",
                        "analysis_summary": "Python file with 3 functions, low complexity"
                    }
                }
            ]
        )
    
    async def _custom_validate_input(self, input_data: Any) -> None:
        """Custom input validation for file reader."""
        if not isinstance(input_data, dict):
            raise ToolValidationError("Input must be a dictionary", self.get_name())
        
        # Validate file path
        file_path = input_data.get('file_path')
        if not file_path:
            raise ToolValidationError("file_path is required", self.get_name())
        
        # Validate analysis level
        analysis_level = input_data.get('analysis_level', 'standard')
        valid_levels = ['minimal', 'standard', 'detailed', 'comprehensive']
        if analysis_level not in valid_levels:
            raise ToolValidationError(
                f"analysis_level must be one of: {valid_levels}",
                self.get_name()
            )
        
        # Validate max size
        max_size_mb = input_data.get('max_size_mb', self.default_max_size_mb)
        if not isinstance(max_size_mb, int) or max_size_mb < 1 or max_size_mb > 100:
            raise ToolValidationError(
                "max_size_mb must be an integer between 1 and 100",
                self.get_name()
            )
    
    @handle_tool_execution("file_read")
    async def execute(self, input_data: Any, context: ToolContext) -> ToolResult:
        """Execute file reading with optional analysis."""
        # Step 1: Validate and parse input
        validated_input = await self._validate_and_parse_input(input_data, context)
        
        # Step 2: Read file safely
        content = await self._read_file_safely(validated_input)
        
        # Step 3: Perform code analysis
        analysis_result = await self._perform_code_analysis(content, validated_input)
        
        # Step 4: Build response
        return self._build_response(content, analysis_result, validated_input)
    
    async def _validate_and_parse_input(self, input_data: Any, context: ToolContext) -> Tuple[FileReaderInput, Path]:
        """Validate input and parse into structured data.
        
        Args:
            input_data: Raw input data
            context: Tool execution context
            
        Returns:
            Tuple of (parsed input, validated file path)
        """
        # Parse input data
        file_reader_input = self._parse_input(input_data)
        
        # Validate and resolve file path
        file_path = self._validate_file_path(file_reader_input.file_path, context)
        
        # Check file safety constraints
        self._check_file_safety(file_path, file_reader_input, context)
        
        return file_reader_input, file_path
    
    async def _read_file_safely(self, validated_input: Tuple[FileReaderInput, Path]) -> str:
        """Read file content safely with proper encoding handling.
        
        Args:
            validated_input: Tuple of (input params, file path)
            
        Returns:
            File content as string
        """
        file_reader_input, file_path = validated_input
        return await self._read_file_content(file_path, file_reader_input.encoding)
    
    async def _perform_code_analysis(self, content: str, validated_input: Tuple[FileReaderInput, Path]) -> Dict[str, Any]:
        """Perform code analysis if requested and applicable.
        
        Args:
            content: File content
            validated_input: Tuple of (input params, file path)
            
        Returns:
            Dictionary containing analysis results
        """
        file_reader_input, file_path = validated_input
        
        # Get basic file information
        file_info = self._get_file_info(file_path, content, file_reader_input.encoding)
        
        # Perform code analysis if requested and applicable
        code_analysis = None
        analysis_summary = None
        
        if file_reader_input.include_context and not file_info.is_binary:
            code_analysis = await self._analyze_code_file(
                file_path, 
                content, 
                file_reader_input.analysis_level
            )
            
            if code_analysis:
                analysis_summary = self._generate_analysis_summary(code_analysis)
        
        # Generate intelligent suggestions
        suggestions = self._generate_suggestions(file_info, code_analysis)
        
        return {
            'file_info': file_info,
            'code_analysis': code_analysis,
            'analysis_summary': analysis_summary,
            'suggestions': suggestions
        }
    
    def _build_response(self, content: str, analysis_result: Dict[str, Any], validated_input: Tuple[FileReaderInput, Path]) -> ToolResult:
        """Build the final tool response.
        
        Args:
            content: File content
            analysis_result: Analysis results dictionary
            validated_input: Tuple of (input params, file path)
            
        Returns:
            ToolResult with complete file reading response
        """
        file_reader_input, file_path = validated_input
        
        # Create output
        output = FileReaderOutput(
            content=content,
            file_info=analysis_result['file_info'],
            code_analysis=analysis_result['code_analysis'],
            analysis_summary=analysis_result['analysis_summary'],
            suggestions=analysis_result['suggestions']
        )
        
        # Build response message
        message = f"Successfully read {file_reader_input.file_path}"
        if analysis_result['analysis_summary']:
            message += f" - {analysis_result['analysis_summary']}"
        
        return ToolResult.success_result(
            output=output,
            message=message,
            suggested_next_actions=analysis_result['suggestions'][:3] if analysis_result['suggestions'] else []
        )
    
    def _parse_input(self, input_data: Any) -> FileReaderInput:
        """Parse input data into FileReaderInput object."""
        if isinstance(input_data, dict):
            return FileReaderInput(
                file_path=input_data['file_path'],
                analysis_level=input_data.get('analysis_level', 'standard'),
                include_context=input_data.get('include_context', True),
                max_size_mb=input_data.get('max_size_mb', self.default_max_size_mb),
                encoding=input_data.get('encoding', 'utf-8')
            )
        else:
            raise ToolValidationError("Input must be a dictionary", self.get_name())
    
    def _check_file_safety(
        self, 
        file_path: Path, 
        input_data: FileReaderInput, 
        context: ToolContext
    ) -> None:
        """Check file safety constraints."""
        # Check if file exists
        if not file_path.exists():
            raise ToolValidationError(f"File does not exist: {file_path}", self.get_name())
        
        if not file_path.is_file():
            raise ToolValidationError(f"Path is not a file: {file_path}", self.get_name())
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > input_data.max_size_mb:
            raise ToolValidationError(
                f"File size ({size_mb:.1f}MB) exceeds limit ({input_data.max_size_mb}MB)",
                self.get_name()
            )
        
        # Use parent class file size checking
        self._check_file_size_limit(file_path, context)
    
    async def _read_file_content(self, file_path: Path, encoding: str) -> str:
        """Read file content with encoding handling."""
        try:
            # Try to read with specified encoding
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Try common encodings if specified encoding fails
            for fallback_encoding in ['utf-8', 'latin-1', 'cp1252']:
                if fallback_encoding != encoding:
                    try:
                        self.logger.warning(
                            f"Failed to read with {encoding}, trying {fallback_encoding}"
                        )
                        return file_path.read_text(encoding=fallback_encoding)
                    except UnicodeDecodeError:
                        continue
            
            # If all encodings fail, read as binary and return info
            self.logger.warning(f"Could not decode {file_path} as text, treating as binary")
            return f"<Binary file: {file_path.stat().st_size} bytes>"
        
        except Exception as e:
            raise ToolExecutionError(f"Failed to read file: {str(e)}", self.get_name())
    
    def _get_file_info(self, file_path: Path, content: str, encoding: str) -> FileInfo:
        """Extract basic file information."""
        stat = file_path.stat()
        
        # Check if file appears to be binary
        is_binary = content.startswith("<Binary file:")
        
        # Count lines
        lines = 0 if is_binary else len(content.splitlines())
        
        # Detect language
        language = None
        try:
            lang_enum = get_language_from_extension(str(file_path))
            language = lang_enum.value if lang_enum else None
        except Exception:
            pass
        
        return FileInfo(
            path=str(file_path),
            size_bytes=stat.st_size,
            lines=lines,
            language=language,
            encoding=encoding,
            last_modified=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
            is_binary=is_binary
        )
    
    async def _analyze_code_file(
        self, 
        file_path: Path, 
        content: str, 
        analysis_level: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze code file using Tree-sitter."""
        try:
            # Skip analysis for binary files or very large files
            if content.startswith("<Binary file:") or len(content) > 500_000:
                return None
            
            # Get language for the file
            language = get_language_from_extension(str(file_path))
            if not language:
                return None
            
            # Configure analysis level
            context_level = ContextLevel(analysis_level)
            config = ContextConfig(level=context_level)
            
            # Extract context using Tree-sitter
            context: FileContext = self.context_extractor.extract_file_context(
                str(file_path), config
            )
            
            # Convert to dictionary format
            return {
                "language": context.language.value if context.language else None,
                "purpose": context.purpose,
                "functions": [
                    {
                        "name": func.name,
                        "line_start": func.line_start,
                        "line_end": func.line_end,
                        "complexity": func.complexity_score,
                        "parameters": func.parameters,
                        "docstring": func.docstring
                    }
                    for func in context.functions[:10]  # Limit to first 10
                ],
                "classes": [
                    {
                        "name": cls.name,
                        "line_start": cls.line_start,
                        "line_end": cls.line_end,
                        "methods": len(cls.methods),
                        "docstring": cls.docstring
                    }
                    for cls in context.classes[:5]  # Limit to first 5
                ],
                "imports": [
                    {
                        "module": imp.module,
                        "alias": imp.alias,
                        "is_local": imp.is_local
                    }
                    for imp in context.imports[:20]  # Limit to first 20
                ],
                "complexity_score": context.complexity_score,
                "key_concepts": context.key_concepts,
                "dependencies": context.dependencies,
                "structure_analysis": {
                    "total_functions": len(context.functions),
                    "total_classes": len(context.classes),
                    "total_imports": len(context.imports),
                    "lines_of_code": context.lines_of_code,
                    "complexity_distribution": context.complexity_distribution
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Code analysis failed for {file_path}: {str(e)}")
            return None
    
    def _generate_analysis_summary(self, code_analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary of code analysis."""
        if not code_analysis:
            return "No code analysis available"
        
        summary_parts = []
        
        # Language
        language = code_analysis.get('language', 'Unknown')
        summary_parts.append(f"{language} file")
        
        # Structure elements
        structure = code_analysis.get('structure_analysis', {})
        functions = structure.get('total_functions', 0)
        classes = structure.get('total_classes', 0)
        
        if functions > 0:
            summary_parts.append(f"{functions} function{'s' if functions != 1 else ''}")
        
        if classes > 0:
            summary_parts.append(f"{classes} class{'es' if classes != 1 else ''}")
        
        # Complexity assessment
        complexity = code_analysis.get('complexity_score', 0)
        if complexity > 0.8:
            summary_parts.append("high complexity")
        elif complexity > 0.5:
            summary_parts.append("moderate complexity")
        elif complexity > 0:
            summary_parts.append("low complexity")
        
        # Lines of code
        lines = structure.get('lines_of_code', 0)
        if lines > 0:
            summary_parts.append(f"{lines} lines")
        
        return ", ".join(summary_parts) if summary_parts else "Empty file"
    
    def _generate_suggestions(
        self, 
        file_info: FileInfo, 
        code_analysis: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate intelligent suggestions based on file analysis."""
        suggestions = []
        
        # Code-specific suggestions
        if code_analysis:
            structure = code_analysis.get('structure_analysis', {})
            functions = structure.get('total_functions', 0)
            classes = structure.get('total_classes', 0)
            complexity = code_analysis.get('complexity_score', 0)
            
            # Structure suggestions
            if functions > 20:
                suggestions.append("Consider splitting large files into smaller modules")
            
            if complexity > 0.8:
                suggestions.append("High complexity detected - consider refactoring")
            
            if functions == 0 and classes == 0 and file_info.language:
                suggestions.append("No functions or classes found - consider adding structure")
            
            # Import suggestions
            imports = len(code_analysis.get('imports', []))
            if imports > 15:
                suggestions.append("Many imports detected - review dependencies")
            
            # Language-specific suggestions
            language = code_analysis.get('language')
            if language == 'python':
                if not any('__main__' in str(imp) for imp in code_analysis.get('imports', [])):
                    if functions > 0:
                        suggestions.append("Consider adding if __name__ == '__main__': guard")
        
        # File-specific suggestions
        if file_info.lines > 1000:
            suggestions.append("Large file - consider using analysis tools for better overview")
        
        if file_info.size_bytes > 100_000:
            suggestions.append("Large file size - might need chunked processing for some operations")
        
        # Binary file suggestions
        if file_info.is_binary:
            suggestions.append("Binary file detected - content not displayed")
        
        # Generic suggestions
        if file_info.language:
            suggestions.append(f"Use language-specific tools for {file_info.language} development")
        
        return suggestions[:5]  # Limit to 5 suggestions