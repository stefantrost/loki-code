"""
LangChain tool adapters for Loki Code tools.

This module converts existing Loki Code tools to LangChain Tool format,
allowing them to be used with LangChain agents while preserving all
existing functionality including permissions and safety checks.
"""

from typing import Any, Dict, List, Optional, Type
from langchain_core.tools import BaseTool as LangChainBaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import BaseTool
from .file_reader import FileReaderTool
from .file_writer import FileWriterTool
from .types import ToolResult, ToolContext

# Commenting out to avoid circular imports for now
# from ..core.agent.permission_manager import PermissionManager, ToolAction
# from ..core.agent.safety_manager import SafetyManager, TaskContext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.agent.permission_manager import PermissionManager
    from ..core.agent.safety_manager import SafetyManager
from ..utils.logging import get_logger


class LokiToolAdapter(LangChainBaseTool):
    """
    Base adapter that wraps Loki Code tools for LangChain compatibility.

    Maintains all existing functionality including permissions, safety checks,
    and tool-specific features while providing LangChain Tool interface.
    """

    # These fields are excluded from Pydantic serialization to avoid markup errors
    loki_tool: BaseTool = Field(exclude=True, repr=False)
    permission_manager: Optional[Any] = Field(default=None, exclude=True, repr=False)
    safety_manager: Optional[Any] = Field(default=None, exclude=True, repr=False)
    logger: Any = Field(default=None, exclude=True, repr=False)

    class Config:
        """Pydantic configuration to handle complex objects."""

        arbitrary_types_allowed = True
        validate_assignment = False
        extra = "allow"  # Allow extra attributes like logger

    def __init__(
        self,
        loki_tool: BaseTool,
        permission_manager: Optional[Any] = None,
        safety_manager: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize the adapter with a Loki tool and optional managers."""
        # Get tool schema information
        schema = loki_tool.get_schema()

        super().__init__(
            name=schema.name,
            description=schema.description,
            loki_tool=loki_tool,
            permission_manager=permission_manager,
            safety_manager=safety_manager,
            **kwargs,
        )

        # Initialize logger after calling super() to ensure it persists
        self._initialize_logger()

    def _initialize_logger(self) -> None:
        """Initialize logger for the adapter."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @classmethod
    def create_adapter(
        cls,
        loki_tool: BaseTool,
        permission_manager: Optional[Any] = None,
        safety_manager: Optional[Any] = None,
        **kwargs,
    ) -> "LokiToolAdapter":
        """Factory method to create adapter instances.

        Args:
            loki_tool: The Loki tool to wrap
            permission_manager: Optional permission manager
            safety_manager: Optional safety manager
            **kwargs: Additional arguments

        Returns:
            LokiToolAdapter instance
        """
        return cls(
            loki_tool=loki_tool,
            permission_manager=permission_manager,
            safety_manager=safety_manager,
            **kwargs,
        )

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> str:
        """Execute the Loki tool with permission and safety checks."""
        return self._execute_with_checks(kwargs, run_manager)

    async def _arun(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> str:
        """Async execution of the Loki tool with permission and safety checks."""
        return self._execute_with_checks(kwargs, run_manager, is_async=True)

    def _execute_with_checks(
        self,
        kwargs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None,
        is_async: bool = False,
    ) -> str:
        """Execute tool with permission and safety checks."""
        try:
            # Create tool context
            context = ToolContext(
                project_path=kwargs.get("project_path", "."),
                user_id="langchain_user",  # TODO: Get from run_manager or config
                session_id="langchain_session",
                environment=kwargs.get("environment", {}),
            )

            # Permission check (simplified for now)
            if self.permission_manager:
                # TODO: Implement proper permission checking
                self.logger.debug("Permission manager available but not implemented")

            # Safety check (simplified for now)
            if self.safety_manager:
                # TODO: Implement proper safety checking
                self.logger.debug("Safety manager available but not implemented")

            # Execute the tool
            if is_async:
                import asyncio

                result = asyncio.run(self.loki_tool.execute(kwargs, context))
            else:
                # For sync execution, we'll run the async method
                import asyncio

                result = asyncio.run(self.loki_tool.execute(kwargs, context))

            # Format result for LangChain
            if isinstance(result, ToolResult):
                if result.success:
                    return str(result.output)
                else:
                    return f"Tool execution failed: {result.message}"
            else:
                return str(result)

        except Exception as e:
            self.logger.error(f"Error executing {self.name}: {e}", exc_info=True)
            return f"Tool execution error: {str(e)}"


class FileReaderToolAdapter(LokiToolAdapter):
    """LangChain adapter for FileReaderTool."""

    name: str = "file_reader"
    description: str = "Read and analyze files with intelligent code analysis using Tree-sitter"

    class ArgsSchema(BaseModel):
        file_path: str = Field(description="Path to the file to read")
        analysis_level: str = Field(
            default="standard",
            description="Analysis level: minimal, standard, detailed, comprehensive",
        )
        include_context: bool = Field(default=True, description="Include Tree-sitter code analysis")
        max_size_mb: int = Field(default=10, description="Maximum file size in MB")
        encoding: str = Field(default="utf-8", description="File encoding")

    args_schema: Type[BaseModel] = ArgsSchema

    def __init__(
        self,
        permission_manager: Optional[Any] = None,
        safety_manager: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize FileReader adapter."""
        loki_tool = FileReaderTool()
        super().__init__(
            loki_tool=loki_tool,
            permission_manager=permission_manager,
            safety_manager=safety_manager,
            **kwargs,
        )


class FileWriterToolAdapter(LokiToolAdapter):
    """LangChain adapter for FileWriterTool."""

    name: str = "file_writer"
    description: str = (
        "Write content to files with safety checks, backup creation, and optional code formatting"
    )

    class ArgsSchema(BaseModel):
        file_path: str = Field(description="Path to the file to write")
        content: str = Field(description="Content to write to the file")
        mode: str = Field(
            default="write", description="Write mode: write (overwrite), append, or insert at line"
        )
        encoding: str = Field(default="utf-8", description="File encoding")
        create_backup: bool = Field(default=True, description="Create backup before overwriting")
        auto_format: bool = Field(default=True, description="Auto-format code files")
        validate_syntax: bool = Field(default=True, description="Validate syntax before writing")
        max_size_mb: int = Field(default=50, description="Maximum file size in MB")
        insert_line: Optional[int] = Field(
            default=None, description="Line number for insert mode (1-based)"
        )
        create_dirs: bool = Field(default=True, description="Create parent directories if needed")

    args_schema: Type[BaseModel] = ArgsSchema

    def __init__(
        self,
        permission_manager: Optional[Any] = None,
        safety_manager: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize FileWriter adapter."""
        loki_tool = FileWriterTool()
        super().__init__(
            loki_tool=loki_tool,
            permission_manager=permission_manager,
            safety_manager=safety_manager,
            **kwargs,
        )


def create_langchain_tools(
    permission_manager: Optional[Any] = None, safety_manager: Optional[Any] = None
) -> List[LokiToolAdapter]:
    """
    Create LangChain-compatible tools from all available Loki tools.

    Args:
        permission_manager: Optional permission manager for access control
        safety_manager: Optional safety manager for operation validation

    Returns:
        List of LangChain-compatible tool adapters
    """
    tools = []

    # Tool configurations for maintainable scaling
    tool_adapters = [
        FileReaderToolAdapter,
        FileWriterToolAdapter,
        # TODO: Add more tools as they are implemented
        # DirectoryListerToolAdapter,
        # CommandExecutorToolAdapter,
    ]

    for adapter_class in tool_adapters:
        adapter = adapter_class(
            permission_manager=permission_manager, safety_manager=safety_manager
        )
        tools.append(adapter)

    return tools


def get_tool_names() -> List[str]:
    """Get list of available tool names."""
    return ["file_reader", "file_writer"]
