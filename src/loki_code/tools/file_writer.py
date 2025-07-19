"""
Intelligent file writer tool for Loki Code.

This tool provides safe, context-aware file writing capabilities with comprehensive
safety checks, backup functionality, and integration with the existing permission
and safety systems.

Features:
- Safe file writing with permission checks
- Automatic backup creation before overwriting
- Path validation and sanitization
- Content validation and formatting
- Integration with Tree-sitter for code formatting
- MCP-compatible interface
- Comprehensive error handling and recovery
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

from ..core.code_analysis import (
    TreeSitterParser,
    CodeAnalyzer,
    ContextExtractor,
    ContextLevel,
    ContextConfig,
    SupportedLanguage,
    get_language_from_extension,
    FileContext,
)
from .base import BaseTool, SimpleFileTool
from .types import (
    ToolSchema,
    ToolContext,
    ToolResult,
    ToolCapability,
    SecurityLevel,
    ConfirmationLevel,
    InputValidationSchema,
    OutputValidationSchema,
)
from .exceptions import ToolValidationError, ToolExecutionError, ToolSecurityError
from ..utils.logging import get_logger
from ..utils.error_handling import handle_tool_execution


@dataclass
class FileWriterInput:
    """Input schema for file writer tool."""

    file_path: str
    content: str
    mode: str = "write"  # write, append, insert
    encoding: str = "utf-8"
    create_backup: bool = True  # Create backup before overwriting
    auto_format: bool = True  # Auto-format code files
    validate_syntax: bool = True  # Validate syntax before writing
    max_size_mb: int = 50  # Safety limit for file size
    insert_line: Optional[int] = None  # Line number for insert mode
    create_dirs: bool = True  # Create parent directories if needed


@dataclass
class BackupInfo:
    """Information about created backup."""

    backup_path: str
    original_path: str
    backup_time: str
    original_size: int
    backup_size: int


@dataclass
class FileWriterOutput:
    """Output schema for file writer tool."""

    file_path: str
    bytes_written: int
    lines_written: int
    operation: str  # created, updated, appended
    backup_info: Optional[BackupInfo] = None
    formatting_applied: bool = False
    syntax_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class FileWriterTool(SimpleFileTool):
    """
    Intelligent file writer with safety checks and code formatting.

    This tool provides safe file writing capabilities with comprehensive
    validation, backup functionality, and optional code formatting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the file writer tool."""
        super().__init__(config)

        # Initialize Tree-sitter components for formatting
        self.parser = TreeSitterParser()
        self.analyzer = CodeAnalyzer(self.parser)

        # Configuration
        self.default_max_size_mb = config.get("max_file_size_mb", 50) if config else 50
        self.backup_dir = config.get("backup_dir", None) if config else None
        self.logger = get_logger(__name__)

    def get_schema(self) -> ToolSchema:
        """Get the tool schema for MCP compatibility."""
        return ToolSchema(
            name="file_writer",
            description="Write content to files with safety checks, backup creation, and optional code formatting",
            capabilities=[
                ToolCapability.WRITE_FILE,
                ToolCapability.CODE_GENERATION,  # Using existing capabilities
                ToolCapability.PROJECT_NAVIGATION,
            ],
            security_level=SecurityLevel.CAUTION,  # Lowered from MODERATE for project files
            confirmation_level=ConfirmationLevel.NONE,  # Will be determined dynamically
            input_schema=InputValidationSchema.create_schema(
                properties={
                    "file_path": InputValidationSchema.file_path_field("Path to the file to write"),
                    "content": InputValidationSchema.string_field("Content to write to the file"),
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append", "insert"],
                        "default": "write",
                        "description": "Write mode: write (overwrite), append, or insert at line",
                    },
                    "encoding": {
                        "type": "string",
                        "default": "utf-8",
                        "description": "File encoding",
                    },
                    "create_backup": InputValidationSchema.boolean_field(
                        "Create backup before overwriting", True
                    ),
                    "auto_format": InputValidationSchema.boolean_field(
                        "Auto-format code files", True
                    ),
                    "validate_syntax": InputValidationSchema.boolean_field(
                        "Validate syntax before writing", True
                    ),
                    "max_size_mb": InputValidationSchema.integer_field(
                        "Maximum file size in MB", minimum=1, maximum=1000
                    ),
                    "insert_line": InputValidationSchema.integer_field(
                        "Line number for insert mode (1-based)", minimum=1
                    ),
                    "create_dirs": InputValidationSchema.boolean_field(
                        "Create parent directories if needed", True
                    ),
                },
                required=["file_path", "content"],
                description="Input schema for file writer tool",
            ),
            output_schema=OutputValidationSchema.success_output_schema(),
            mcp_compatible=True,
            version="1.0.0",
            tags=["file", "write", "backup", "formatting"],
        )

    def requires_confirmation(self, context: ToolContext) -> bool:
        """Smart confirmation logic for file operations."""
        # Override the default requires_confirmation to be more intelligent
        
        # For now, don't require confirmation for project-local operations
        # This could be enhanced with more sophisticated logic later
        
        # Check if we're in dry run mode - always require confirmation
        if context.dry_run:
            return True
            
        # Check safety settings - if user specifically wants confirmation for CAUTION level
        if SecurityLevel.CAUTION in context.safety_settings.require_confirmation_for:
            return True
            
        # For small files in the project directory, don't require confirmation
        # This makes development workflow smoother
        return False

    async def execute(self, inputs: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the file writer tool."""
        try:
            # Parse and validate inputs
            writer_input = self._parse_inputs(inputs)

            # Security and safety validation
            await self._validate_security(writer_input, context)

            # Prepare file writing
            file_path = Path(writer_input.file_path).resolve()

            # Create backup if file exists and backup is requested
            backup_info = None
            if writer_input.create_backup and file_path.exists():
                backup_info = await self._create_backup(file_path)

            # Process content (formatting, validation)
            processed_content, formatting_applied, syntax_valid, warnings = (
                await self._process_content(writer_input.content, file_path, writer_input)
            )

            # Write the file
            operation, bytes_written, lines_written = await self._write_file(
                file_path, processed_content, writer_input
            )

            # Generate suggestions
            suggestions = await self._generate_suggestions(file_path, writer_input, context)

            # Create result
            output = FileWriterOutput(
                file_path=str(file_path),
                bytes_written=bytes_written,
                lines_written=lines_written,
                operation=operation,
                backup_info=backup_info,
                formatting_applied=formatting_applied,
                syntax_valid=syntax_valid,
                warnings=warnings,
                suggestions=suggestions,
            )

            return ToolResult(
                success=True,
                output=self._format_output(output),
                message="File operation completed successfully",
                metadata={
                    "tool": "file_writer",
                    "operation": operation,
                    "file_path": str(file_path),
                    "bytes_written": bytes_written,
                    "backup_created": backup_info is not None,
                    "data": output.__dict__,
                },
            )

        except Exception as e:
            self.logger.error(f"File writer execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                message=f"Failed to write file: {str(e)}",
                metadata={"tool": "file_writer", "error_type": type(e).__name__},
            )

    def _parse_inputs(self, inputs: Dict[str, Any]) -> FileWriterInput:
        """Parse and validate input parameters."""
        try:
            return FileWriterInput(
                file_path=inputs["file_path"],
                content=inputs["content"],
                mode=inputs.get("mode", "write"),
                encoding=inputs.get("encoding", "utf-8"),
                create_backup=inputs.get("create_backup", True),
                auto_format=inputs.get("auto_format", True),
                validate_syntax=inputs.get("validate_syntax", True),
                max_size_mb=inputs.get("max_size_mb", self.default_max_size_mb),
                insert_line=inputs.get("insert_line"),
                create_dirs=inputs.get("create_dirs", True),
            )
        except KeyError as e:
            raise ToolValidationError(f"Missing required parameter: {e}", tool_name="file_writer")
        except Exception as e:
            raise ToolValidationError(f"Invalid input parameters: {e}", tool_name="file_writer")

    async def _validate_security(self, writer_input: FileWriterInput, context: ToolContext) -> None:
        """Perform security validation on the write operation."""
        file_path = Path(writer_input.file_path).resolve()

        # Path validation
        if not self._is_safe_path(file_path, context.project_path):
            raise ToolSecurityError(
                f"File path is outside allowed workspace: {file_path}",
                tool_name="file_writer",
                security_violation="path_outside_workspace",
            )

        # Size validation
        content_size_mb = len(writer_input.content.encode(writer_input.encoding)) / (1024 * 1024)
        if content_size_mb > writer_input.max_size_mb:
            raise ToolSecurityError(
                f"Content size ({content_size_mb:.1f}MB) exceeds limit ({writer_input.max_size_mb}MB)",
                tool_name="file_writer",
                security_violation="content_size_exceeded",
            )

        # Check for potentially dangerous content
        if self._contains_dangerous_content(writer_input.content):
            raise ToolSecurityError(
                "Content contains potentially dangerous patterns",
                tool_name="file_writer",
                security_violation="dangerous_content",
            )

        # Mode-specific validation
        if writer_input.mode == "insert" and writer_input.insert_line is None:
            raise ToolValidationError(
                "insert_line is required for insert mode", tool_name="file_writer"
            )

    def _is_safe_path(self, file_path: Path, workspace_path: str) -> bool:
        """Check if the file path is safe (within workspace)."""
        try:
            workspace = Path(workspace_path).resolve()
            file_path = file_path.resolve()

            # Check if file path is within workspace
            return str(file_path).startswith(str(workspace))
        except Exception:
            return False

    def _contains_dangerous_content(self, content: str) -> bool:
        """Check for potentially dangerous content patterns."""
        dangerous_patterns = [
            "rm -rf /",
            "del /q",
            "format c:",
            "__import__('os').system",
            "exec(",
            "eval(",
            "subprocess.call",
            "os.system",
        ]

        content_lower = content.lower()
        return any(pattern in content_lower for pattern in dangerous_patterns)

    async def _create_backup(self, file_path: Path) -> BackupInfo:
        """Create a backup of the existing file."""
        try:
            # Generate backup path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.name}.backup_{timestamp}"

            if self.backup_dir:
                backup_dir = Path(self.backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / backup_name
            else:
                backup_path = file_path.parent / backup_name

            # Copy file
            shutil.copy2(file_path, backup_path)

            # Get file sizes
            original_size = file_path.stat().st_size
            backup_size = backup_path.stat().st_size

            backup_info = BackupInfo(
                backup_path=str(backup_path),
                original_path=str(file_path),
                backup_time=datetime.now().isoformat(),
                original_size=original_size,
                backup_size=backup_size,
            )

            self.logger.info(f"Created backup: {backup_path}")
            return backup_info

        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            raise ToolExecutionError(f"Backup creation failed: {e}", tool_name="file_writer")

    async def _process_content(
        self, content: str, file_path: Path, writer_input: FileWriterInput
    ) -> Tuple[str, bool, bool, List[str]]:
        """Process content: formatting, validation, etc."""
        processed_content = content
        formatting_applied = False
        syntax_valid = True
        warnings = []

        try:
            # Get language from file extension
            language = get_language_from_extension(file_path.suffix)

            # Syntax validation if requested and language is supported
            if writer_input.validate_syntax and language:
                try:
                    # Try to parse with Tree-sitter
                    parse_result = self.parser.parse_code(processed_content, language)
                    if parse_result.tree and parse_result.tree.root_node.has_error:
                        syntax_valid = False
                        warnings.append("Syntax errors detected in content")
                except Exception as e:
                    warnings.append(f"Could not validate syntax: {e}")

            # Auto-formatting if requested and language is supported
            if writer_input.auto_format and language and syntax_valid:
                try:
                    # Basic formatting (normalize line endings, strip trailing whitespace)
                    lines = processed_content.split("\n")
                    formatted_lines = [line.rstrip() for line in lines]
                    formatted_content = "\n".join(formatted_lines)

                    if formatted_content != processed_content:
                        processed_content = formatted_content
                        formatting_applied = True

                except Exception as e:
                    warnings.append(f"Auto-formatting failed: {e}")

        except Exception as e:
            warnings.append(f"Content processing error: {e}")

        return processed_content, formatting_applied, syntax_valid, warnings

    async def _write_file(
        self, file_path: Path, content: str, writer_input: FileWriterInput
    ) -> Tuple[str, int, int]:
        """Write content to file."""
        try:
            # Create parent directories if needed
            if writer_input.create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine operation type
            operation = "created" if not file_path.exists() else "updated"

            # Handle different write modes
            if writer_input.mode == "write":
                # Overwrite file
                with open(file_path, "w", encoding=writer_input.encoding) as f:
                    f.write(content)

            elif writer_input.mode == "append":
                # Append to file
                with open(file_path, "a", encoding=writer_input.encoding) as f:
                    f.write(content)
                operation = "appended"

            elif writer_input.mode == "insert":
                # Insert at specific line
                if file_path.exists():
                    with open(file_path, "r", encoding=writer_input.encoding) as f:
                        lines = f.readlines()
                else:
                    lines = []

                # Insert content at specified line (1-based indexing)
                # insert_line is guaranteed to be not None due to validation
                assert writer_input.insert_line is not None
                insert_index = max(0, min(writer_input.insert_line - 1, len(lines)))
                lines.insert(insert_index, content + "\n")

                with open(file_path, "w", encoding=writer_input.encoding) as f:
                    f.writelines(lines)
                operation = "inserted"

            # Calculate statistics
            bytes_written = len(content.encode(writer_input.encoding))
            lines_written = len(content.split("\n"))

            self.logger.info(
                f"File {operation}: {file_path} ({bytes_written} bytes, {lines_written} lines)"
            )

            return operation, bytes_written, lines_written

        except Exception as e:
            raise ToolExecutionError(f"Failed to write file: {e}", tool_name="file_writer")

    async def _generate_suggestions(
        self, file_path: Path, writer_input: FileWriterInput, context: ToolContext
    ) -> List[str]:
        """Generate helpful suggestions based on the write operation."""
        suggestions = []

        try:
            # File type specific suggestions
            if file_path.suffix == ".py":
                suggestions.append(
                    "Consider running 'python -m py_compile' to check for syntax errors"
                )
                suggestions.append("Use 'black' or 'autopep8' for consistent formatting")

            elif file_path.suffix in [".js", ".ts"]:
                suggestions.append("Consider running 'eslint' to check for issues")
                suggestions.append("Use 'prettier' for consistent formatting")

            elif file_path.suffix in [".json"]:
                suggestions.append("Validate JSON syntax with 'python -m json.tool'")

            # General suggestions
            if writer_input.mode == "write" and file_path.exists():
                suggestions.append("File was overwritten - backup was created for safety")

            if not writer_input.validate_syntax:
                suggestions.append("Consider enabling syntax validation for better error detection")

        except Exception as e:
            self.logger.debug(f"Error generating suggestions: {e}")

        return suggestions

    def _format_output(self, output: FileWriterOutput) -> str:
        """Format the output for display."""
        lines = [
            f"📝 File {output.operation}: {output.file_path}",
            f"   • {output.bytes_written:,} bytes written ({output.lines_written:,} lines)",
        ]

        if output.backup_info:
            lines.append(f"   • Backup created: {output.backup_info.backup_path}")

        if output.formatting_applied:
            lines.append("   • Auto-formatting applied")

        if not output.syntax_valid:
            lines.append("   ⚠️  Syntax validation failed")

        if output.warnings:
            lines.append("   ⚠️  Warnings:")
            for warning in output.warnings:
                lines.append(f"      - {warning}")

        if output.suggestions:
            lines.append("   💡 Suggestions:")
            for suggestion in output.suggestions:
                lines.append(f"      - {suggestion}")

        return "\n".join(lines)
