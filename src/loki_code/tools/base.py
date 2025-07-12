"""
Abstract base class for all tools in Loki Code.

This module defines the core tool interface that all tools must implement,
designed for MCP compatibility from the start while providing a rich
foundation for local tool development.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
import json
from pathlib import Path

from .types import (
    ToolSchema, ToolContext, ToolResult, ToolCall, ToolExecution,
    SecurityLevel, ToolCapability, ConfirmationLevel, ToolStatus,
    InputValidationSchema, OutputValidationSchema
)
from .exceptions import (
    ToolException, ToolValidationError, ToolExecutionError,
    ToolPermissionError, ToolSecurityError, ToolTimeoutError,
    ToolConfigurationError
)
from ..utils.logging import get_logger


class BaseTool(ABC):
    """
    Abstract base class for all tools in Loki Code.
    
    This class defines the interface that all tools must implement,
    ensuring consistency and MCP compatibility across the tool ecosystem.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base tool.
        
        Args:
            config: Optional configuration dictionary for the tool
        """
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._schema: Optional[ToolSchema] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the tool (called before first use).
        
        Override this method to perform any setup required by the tool.
        This is called automatically before the first execution.
        """
        self._initialized = True
    
    @abstractmethod
    async def execute(self, input_data: Any, context: ToolContext) -> ToolResult:
        """Execute the tool with given input and context.
        
        Args:
            input_data: Input data for the tool (must match input schema)
            context: Execution context with project info and settings
            
        Returns:
            ToolResult containing the execution result
            
        Raises:
            ToolException: If execution fails
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get the tool schema definition.
        
        Returns:
            ToolSchema describing the tool's interface and capabilities
        """
        pass
    
    def get_name(self) -> str:
        """Get the tool name.
        
        Returns:
            Tool name (defaults to class name in snake_case)
        """
        schema = self.get_schema()
        return schema.name
    
    def get_description(self) -> str:
        """Get the tool description.
        
        Returns:
            Tool description from schema
        """
        schema = self.get_schema()
        return schema.description
    
    def get_capabilities(self) -> List[ToolCapability]:
        """Get the tool capabilities.
        
        Returns:
            List of capabilities this tool provides
        """
        schema = self.get_schema()
        return schema.capabilities
    
    def get_security_level(self) -> SecurityLevel:
        """Get the tool security level.
        
        Returns:
            Security level of this tool
        """
        schema = self.get_schema()
        return schema.security_level
    
    def requires_confirmation(self, context: ToolContext) -> bool:
        """Check if tool execution requires user confirmation.
        
        Args:
            context: Execution context
            
        Returns:
            True if confirmation is required, False otherwise
        """
        schema = self.get_schema()
        
        # Check if security level requires confirmation
        if schema.security_level in context.safety_settings.require_confirmation_for:
            return True
        
        # Check confirmation level
        if schema.confirmation_level in [ConfirmationLevel.PROMPT, ConfirmationLevel.DETAILED, ConfirmationLevel.EXPLICIT]:
            return True
        
        return False
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data against the tool's input schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid
            
        Raises:
            ToolValidationError: If input validation fails
        """
        schema = self.get_schema()
        
        try:
            # Basic type checking based on schema
            if not self._validate_against_schema(input_data, schema.input_schema):
                raise ToolValidationError(
                    f"Input validation failed for tool '{self.get_name()}'",
                    self.get_name(),
                    validation_errors=self._get_validation_errors(input_data, schema.input_schema)
                )
            
            # Additional custom validation
            await self._custom_validate_input(input_data)
            
            return True
            
        except ToolValidationError:
            raise
        except Exception as e:
            raise ToolValidationError(
                f"Input validation error: {str(e)}",
                self.get_name()
            ) from e
    
    async def _custom_validate_input(self, input_data: Any) -> None:
        """Custom input validation logic (override in subclasses).
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ToolValidationError: If custom validation fails
        """
        pass
    
    def validate_context(self, context: ToolContext) -> bool:
        """Validate execution context.
        
        Args:
            context: Execution context to validate
            
        Returns:
            True if context is valid
            
        Raises:
            ToolValidationError: If context validation fails
        """
        try:
            # Check required context fields
            if not context.project_path:
                raise ToolValidationError(
                    "Project path is required in context",
                    self.get_name()
                )
            
            # Validate project path exists
            if not Path(context.project_path).exists():
                raise ToolValidationError(
                    f"Project path does not exist: {context.project_path}",
                    self.get_name()
                )
            
            # Check security constraints
            self._validate_security_constraints(context)
            
            return True
            
        except ToolValidationError:
            raise
        except Exception as e:
            raise ToolValidationError(
                f"Context validation error: {str(e)}",
                self.get_name()
            ) from e
    
    def _validate_security_constraints(self, context: ToolContext) -> None:
        """Validate security constraints for the tool execution.
        
        Args:
            context: Execution context
            
        Raises:
            ToolSecurityError: If security constraints are violated
        """
        schema = self.get_schema()
        
        # Check if tool capabilities are allowed
        dangerous_capabilities = [
            ToolCapability.EXECUTE_COMMAND,
            ToolCapability.SYSTEM_MODIFICATION,
            ToolCapability.NETWORK_ACCESS
        ]
        
        for capability in schema.capabilities:
            if capability in dangerous_capabilities and schema.security_level == SecurityLevel.SAFE:
                raise ToolSecurityError(
                    f"Tool claims SAFE security level but has dangerous capability: {capability.value}",
                    self.get_name(),
                    "capability_security_mismatch"
                )
    
    async def safe_execute(
        self, 
        input_data: Any, 
        context: ToolContext,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Safely execute the tool with validation and error handling.
        
        Args:
            input_data: Input data for the tool
            context: Execution context
            timeout: Optional timeout in seconds
            
        Returns:
            ToolResult containing the execution result
        """
        start_time = time.perf_counter()
        
        try:
            # Initialize if needed
            if not self._initialized:
                await self.initialize()
            
            # Validate input and context
            await self.validate_input(input_data)
            self.validate_context(context)
            
            # Check if confirmation is required
            if self.requires_confirmation(context) and not context.dry_run:
                return ToolResult.confirmation_needed(
                    f"Tool '{self.get_name()}' requires confirmation before execution",
                    output={"input_preview": input_data}
                )
            
            # Execute with timeout if specified
            if timeout:
                result = await asyncio.wait_for(
                    self.execute(input_data, context),
                    timeout=timeout
                )
            else:
                # Use default timeout from context
                default_timeout = context.safety_settings.timeout_seconds
                result = await asyncio.wait_for(
                    self.execute(input_data, context),
                    timeout=default_timeout
                )
            
            # Add execution time to result
            execution_time = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            # Log successful execution
            self.logger.info(
                f"Tool '{self.get_name()}' executed successfully in {execution_time:.1f}ms"
            )
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Tool '{self.get_name()}' timed out after {execution_time:.1f}ms")
            raise ToolTimeoutError(
                f"Tool execution timed out after {timeout or context.safety_settings.timeout_seconds}s",
                self.get_name(),
                timeout or context.safety_settings.timeout_seconds
            )
        
        except ToolException:
            # Re-raise tool exceptions as-is
            raise
        
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Tool '{self.get_name()}' failed after {execution_time:.1f}ms: {str(e)}",
                exc_info=True
            )
            raise ToolExecutionError(
                f"Tool execution failed: {str(e)}",
                self.get_name()
            ) from e
    
    def create_tool_call(self, input_data: Any, context: ToolContext) -> ToolCall:
        """Create a ToolCall object for this execution.
        
        Args:
            input_data: Input data for the tool
            context: Execution context
            
        Returns:
            ToolCall object representing this execution
        """
        return ToolCall(
            tool_name=self.get_name(),
            input_data=input_data,
            context=context
        )
    
    def to_mcp_schema(self) -> Dict[str, Any]:
        """Convert tool schema to MCP-compatible format.
        
        Returns:
            MCP-compatible schema dictionary
        """
        schema = self.get_schema()
        return schema.to_json_schema()
    
    @classmethod
    def from_mcp_schema(cls, mcp_schema: Dict[str, Any]) -> 'BaseTool':
        """Create tool instance from MCP schema (for future MCP integration).
        
        Args:
            mcp_schema: MCP-compatible schema dictionary
            
        Returns:
            Tool instance
            
        Raises:
            NotImplementedError: This is a placeholder for future MCP integration
        """
        # This is a placeholder for future MCP integration
        raise NotImplementedError("MCP schema loading will be implemented in Phase 6")
    
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema to validate against
            
        Returns:
            True if data is valid
        """
        # Basic validation - in a full implementation, you'd use jsonschema library
        if schema.get("type") == "object" and not isinstance(data, dict):
            return False
        
        if schema.get("type") == "string" and not isinstance(data, str):
            return False
        
        if schema.get("type") == "array" and not isinstance(data, list):
            return False
        
        # Check required fields for objects
        if isinstance(data, dict) and "required" in schema:
            for required_field in schema["required"]:
                if required_field not in data:
                    return False
        
        return True
    
    def _get_validation_errors(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Get detailed validation errors.
        
        Args:
            data: Data that failed validation
            schema: Schema that was violated
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Basic error reporting
        if schema.get("type") == "object" and not isinstance(data, dict):
            errors.append(f"Expected object, got {type(data).__name__}")
        
        if schema.get("type") == "string" and not isinstance(data, str):
            errors.append(f"Expected string, got {type(data).__name__}")
        
        if schema.get("type") == "array" and not isinstance(data, list):
            errors.append(f"Expected array, got {type(data).__name__}")
        
        # Check required fields
        if isinstance(data, dict) and "required" in schema:
            for required_field in schema["required"]:
                if required_field not in data:
                    errors.append(f"Missing required field: {required_field}")
        
        return errors
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.get_name()}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the tool."""
        schema = self.get_schema()
        return (
            f"{self.__class__.__name__}("
            f"name='{schema.name}', "
            f"security_level='{schema.security_level.value}', "
            f"capabilities={[c.value for c in schema.capabilities]}"
            f")"
        )


class SimpleFileTool(BaseTool):
    """
    Base class for simple file-based tools.
    
    Provides common functionality for tools that work with files,
    including path validation and safety checks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the file tool."""
        super().__init__(config)
    
    def _validate_file_path(self, file_path: str, context: ToolContext) -> Path:
        """Validate and resolve a file path.
        
        Args:
            file_path: Path to validate
            context: Execution context
            
        Returns:
            Resolved Path object
            
        Raises:
            ToolValidationError: If path is invalid
            ToolSecurityError: If path violates security constraints
        """
        try:
            path = Path(file_path)
            
            # Convert to absolute path
            if not path.is_absolute():
                path = Path(context.working_directory) / path
            
            path = path.resolve()
            
            # Check if path is within allowed directories
            allowed_paths = [Path(p).resolve() for p in context.safety_settings.allowed_paths]
            
            path_allowed = False
            for allowed_path in allowed_paths:
                try:
                    path.relative_to(allowed_path)
                    path_allowed = True
                    break
                except ValueError:
                    continue
            
            if not path_allowed:
                raise ToolSecurityError(
                    f"Path '{file_path}' is not within allowed directories",
                    self.get_name(),
                    "path_restriction_violation",
                    str(path)
                )
            
            return path
            
        except Exception as e:
            if isinstance(e, (ToolSecurityError, ToolValidationError)):
                raise
            raise ToolValidationError(
                f"Invalid file path: {str(e)}",
                self.get_name()
            ) from e
    
    def _check_file_size_limit(self, file_path: Path, context: ToolContext) -> None:
        """Check if file size is within limits.
        
        Args:
            file_path: Path to check
            context: Execution context
            
        Raises:
            ToolValidationError: If file is too large
        """
        if file_path.exists() and file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            max_size_mb = context.safety_settings.max_file_size_mb
            
            if size_mb > max_size_mb:
                raise ToolValidationError(
                    f"File size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)",
                    self.get_name()
                )


class MCPResourceAdapter:
    """
    Adapter for converting MCP resources to our tool interface.
    
    This is a placeholder for future MCP integration in Phase 6.
    """
    
    def __init__(self):
        """Initialize the MCP adapter."""
        self.logger = get_logger(__name__)
    
    def adapt_mcp_resource(self, mcp_resource: Any) -> BaseTool:
        """Convert an MCP resource to a BaseTool implementation.
        
        Args:
            mcp_resource: MCP resource to adapt
            
        Returns:
            BaseTool implementation
            
        Raises:
            NotImplementedError: This is a placeholder for future implementation
        """
        # This will be implemented in Phase 6: MCP Integration
        raise NotImplementedError("MCP resource adaptation will be implemented in Phase 6")
    
    def convert_to_mcp_call(self, tool_call: ToolCall) -> Any:
        """Convert a tool call to MCP format.
        
        Args:
            tool_call: Tool call to convert
            
        Returns:
            MCP-compatible call format
            
        Raises:
            NotImplementedError: This is a placeholder for future implementation
        """
        # This will be implemented in Phase 6: MCP Integration
        raise NotImplementedError("MCP call conversion will be implemented in Phase 6")


def create_simple_tool_schema(
    name: str,
    description: str,
    input_properties: Dict[str, Dict[str, Any]],
    required_inputs: List[str],
    capabilities: List[ToolCapability],
    security_level: SecurityLevel = SecurityLevel.SAFE,
    confirmation_level: ConfirmationLevel = ConfirmationLevel.NONE
) -> ToolSchema:
    """Helper function to create simple tool schemas.
    
    Args:
        name: Tool name
        description: Tool description
        input_properties: Input schema properties
        required_inputs: Required input fields
        capabilities: Tool capabilities
        security_level: Security level
        confirmation_level: Confirmation level required
        
    Returns:
        ToolSchema object
    """
    input_schema = InputValidationSchema.create_schema(
        input_properties,
        required_inputs,
        f"Input schema for {name}"
    )
    
    output_schema = OutputValidationSchema.success_output_schema()
    
    return ToolSchema(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        capabilities=capabilities,
        security_level=security_level,
        confirmation_level=confirmation_level
    )