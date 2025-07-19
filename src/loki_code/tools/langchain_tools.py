"""
Proper LangChain tool implementations using BaseTool with Pydantic schemas.

This module provides clean BaseTool subclasses that properly expose Pydantic
schemas to LangChain for automatic JSON input validation and parsing.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from .file_reader import FileReaderTool
from .file_writer import FileWriterTool
from .types import ToolResult, ToolContext
from ..utils.logging import get_logger


class LangChainToolBase(BaseTool):
    """
    Base class for LangChain tools that eliminates code duplication.
    
    Provides common functionality for:
    - JSON input parsing from ReAct agents
    - Tool context creation
    - Error handling patterns
    """
    
    # Define logger field for Pydantic v2 compatibility
    logger: Any = Field(default=None, description="Logger instance for this tool")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger(__name__)
    
    def _parse_input(self, tool_input, tool_call_id=None):
        """Override to handle JSON string input from ReAct agents."""
        import json
        
        # If input is a string that looks like JSON, parse it
        if isinstance(tool_input, str) and tool_input.strip().startswith("{") and tool_input.strip().endswith("}"):
            try:
                parsed_input = json.loads(tool_input)
                self.logger.debug(f"Parsed JSON input: {parsed_input}")
                return parsed_input
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON input: {e}")
                # Fall back to original input
                pass
        
        # Use parent class parsing for dict inputs
        return super()._parse_input(tool_input, tool_call_id)
    
    def _create_tool_context(self) -> ToolContext:
        """Create a standardized tool context for internal tool execution."""
        return ToolContext(
            project_path=".",  # TODO: Get from config or context
            user_id="langchain_user",
            session_id="langchain_session",
            environment={},
        )
    
    def _handle_tool_result(self, result: ToolResult) -> str:
        """Standardized handling of internal tool results."""
        if result.success:
            return str(result.output)
        else:
            return f"Tool execution failed: {result.message}"
    
    def _handle_execution_error(self, e: Exception, tool_name: str) -> str:
        """Standardized error handling for tool execution."""
        self.logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
        return f"Tool execution error: {str(e)}"


class FileReaderArgsSchema(BaseModel):
    """Input schema for FileReaderTool."""
    file_path: str = Field(description="Path to the file to read")
    analysis_level: str = Field(
        default="standard",
        description="Analysis level: minimal, standard, detailed, comprehensive"
    )
    include_context: bool = Field(default=True, description="Include Tree-sitter code analysis")
    max_size_mb: int = Field(default=10, description="Maximum file size in MB")
    encoding: str = Field(default="utf-8", description="File encoding")


class FileWriterArgsSchema(BaseModel):
    """Input schema for FileWriterTool."""
    file_path: str = Field(description="Path to the file to write (REQUIRED - e.g., 'hello.go', 'main.py')")
    content: str = Field(description="Content to write to the file (REQUIRED - the actual code or text)")
    mode: str = Field(
        default="write", 
        description="Write mode: write (overwrite), append, or insert at line"
    )
    encoding: str = Field(default="utf-8", description="File encoding")
    create_backup: bool = Field(default=True, description="Create backup before overwriting")
    auto_format: bool = Field(default=True, description="Auto-format code files")
    validate_syntax: bool = Field(default=True, description="Validate syntax before writing")
    max_size_mb: int = Field(default=50, description="Maximum file size in MB")
    insert_line: Optional[int] = Field(
        default=None, 
        description="Line number for insert mode (1-based)"
    )
    create_dirs: bool = Field(default=True, description="Create parent directories if needed")


class LangChainFileReaderTool(LangChainToolBase):
    """LangChain BaseTool wrapper for FileReaderTool with proper Pydantic schema."""
    
    name: str = "file_reader"
    description: str = "Read and analyze files with intelligent code analysis using Tree-sitter"
    args_schema: Type[BaseModel] = FileReaderArgsSchema
    
    # Define internal tool field for Pydantic v2 compatibility
    file_reader_tool: Any = Field(default=None, description="Internal file reader tool instance")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the internal tool
        self.file_reader_tool = FileReaderTool()
    
    def _run(
        self,
        file_path: str,
        analysis_level: str = "standard",
        include_context: bool = True,
        max_size_mb: int = 10,
        encoding: str = "utf-8",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the file reader tool."""
        try:
            # Create tool context
            context = self._create_tool_context()
            
            # Prepare inputs
            inputs = {
                "file_path": file_path,
                "analysis_level": analysis_level,
                "include_context": include_context,
                "max_size_mb": max_size_mb,
                "encoding": encoding,
            }
            
            # Execute the tool
            result = asyncio.run(self.file_reader_tool.execute(inputs, context))
            return self._handle_tool_result(result)
                
        except Exception as e:
            return self._handle_execution_error(e, "file_reader")
    
    async def _arun(
        self,
        file_path: str,
        analysis_level: str = "standard",
        include_context: bool = True,
        max_size_mb: int = 10,
        encoding: str = "utf-8",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the file reader tool asynchronously."""
        try:
            # Create tool context
            context = self._create_tool_context()
            
            # Prepare inputs
            inputs = {
                "file_path": file_path,
                "analysis_level": analysis_level,
                "include_context": include_context,
                "max_size_mb": max_size_mb,
                "encoding": encoding,
            }
            
            # Execute the tool
            result = await self.file_reader_tool.execute(inputs, context)
            return self._handle_tool_result(result)
                
        except Exception as e:
            return self._handle_execution_error(e, "file_reader")


class LangChainFileWriterTool(LangChainToolBase):
    """LangChain BaseTool wrapper for FileWriterTool with proper Pydantic schema."""
    
    name: str = "file_writer"
    description: str = (
        "Write content to files with safety checks, backup creation, and optional code formatting. "
        "REQUIRED: Both file_path (filename) and content (actual text to write) must be provided. "
        "For code generation requests, first create the code content, then write it to an appropriate file."
    )
    args_schema: Type[BaseModel] = FileWriterArgsSchema
    
    # Define internal tool field for Pydantic v2 compatibility
    file_writer_tool: Any = Field(default=None, description="Internal file writer tool instance")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the internal tool
        self.file_writer_tool = FileWriterTool()
    
    def _run(
        self,
        file_path: str,
        content: str,
        mode: str = "write",
        encoding: str = "utf-8",
        create_backup: bool = True,
        auto_format: bool = True,
        validate_syntax: bool = True,
        max_size_mb: int = 50,
        insert_line: Optional[int] = None,
        create_dirs: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the file writer tool."""
        try:
            # Create tool context
            context = self._create_tool_context()
            
            # Prepare inputs
            inputs = {
                "file_path": file_path,
                "content": content,
                "mode": mode,
                "encoding": encoding,
                "create_backup": create_backup,
                "auto_format": auto_format,
                "validate_syntax": validate_syntax,
                "max_size_mb": max_size_mb,
                "insert_line": insert_line,
                "create_dirs": create_dirs,
            }
            
            # Execute the tool
            result = asyncio.run(self.file_writer_tool.execute(inputs, context))
            return self._handle_tool_result(result)
                
        except Exception as e:
            return self._handle_execution_error(e, "file_writer")
    
    async def _arun(
        self,
        file_path: str,
        content: str,
        mode: str = "write",
        encoding: str = "utf-8",
        create_backup: bool = True,
        auto_format: bool = True,
        validate_syntax: bool = True,
        max_size_mb: int = 50,
        insert_line: Optional[int] = None,
        create_dirs: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the file writer tool asynchronously."""
        try:
            # Create tool context
            context = self._create_tool_context()
            
            # Prepare inputs
            inputs = {
                "file_path": file_path,
                "content": content,
                "mode": mode,
                "encoding": encoding,
                "create_backup": create_backup,
                "auto_format": auto_format,
                "validate_syntax": validate_syntax,
                "max_size_mb": max_size_mb,
                "insert_line": insert_line,
                "create_dirs": create_dirs,
            }
            
            # Execute the tool
            result = await self.file_writer_tool.execute(inputs, context)
            return self._handle_tool_result(result)
                
        except Exception as e:
            return self._handle_execution_error(e, "file_writer")


def create_langchain_tools(
    permission_manager: Optional[Any] = None, 
    safety_manager: Optional[Any] = None
) -> List[BaseTool]:
    """
    Create properly implemented LangChain-compatible tools.
    
    Args:
        permission_manager: Optional permission manager for access control
        safety_manager: Optional safety manager for operation validation
        
    Returns:
        List of LangChain BaseTool instances
    """
    # Import the simple test tool
    from .test_simple_tool import SimpleTestTool
    
    tools = [
        SimpleTestTool(),
        LangChainFileReaderTool(),
        LangChainFileWriterTool(),
    ]
    
    return tools


def get_tool_names() -> List[str]:
    """Get list of available tool names."""
    return ["file_reader", "file_writer"]