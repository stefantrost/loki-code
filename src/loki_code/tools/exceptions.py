"""
Tool-specific exceptions for Loki Code.

This module defines exception classes for tool operations, providing
structured error handling with context and recovery information.
"""

from typing import Optional, Dict, Any, List


class ToolException(Exception):
    """Base exception for all tool-related errors."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        recoverable: bool = True,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize tool exception.
        
        Args:
            message: Human-readable error message
            tool_name: Name of the tool that generated the error
            recoverable: Whether the error can be recovered from
            error_code: Machine-readable error code
            context: Additional context information
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.recoverable = recoverable
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "tool_name": self.tool_name,
            "recoverable": self.recoverable,
            "error_code": self.error_code,
            "context": self.context
        }


class ToolValidationError(ToolException):
    """Exception raised when tool input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            validation_errors: List of specific validation errors
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, **kwargs)
        self.validation_errors = validation_errors or []


class ToolExecutionError(ToolException):
    """Exception raised when tool execution fails."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        exit_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        **kwargs
    ):
        """Initialize execution error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            exit_code: Exit code from failed operation
            stdout: Standard output from failed operation
            stderr: Standard error from failed operation
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, **kwargs)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class ToolPermissionError(ToolException):
    """Exception raised when tool lacks required permissions."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        required_permission: Optional[str] = None,
        resource_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize permission error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            required_permission: The permission that was required
            resource_path: Path to the resource that couldn't be accessed
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, recoverable=False, **kwargs)
        self.required_permission = required_permission
        self.resource_path = resource_path


class ToolSecurityError(ToolException):
    """Exception raised when tool operation violates security constraints."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        security_violation: str,
        attempted_action: Optional[str] = None,
        **kwargs
    ):
        """Initialize security error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            security_violation: Type of security violation
            attempted_action: The action that was attempted
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, recoverable=False, **kwargs)
        self.security_violation = security_violation
        self.attempted_action = attempted_action


class ToolTimeoutError(ToolException):
    """Exception raised when tool operation times out."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        timeout_seconds: float,
        **kwargs
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            timeout_seconds: The timeout that was exceeded
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, **kwargs)
        self.timeout_seconds = timeout_seconds


class ToolResourceError(ToolException):
    """Exception raised when tool encounters resource-related issues."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        resource_type: str,
        resource_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize resource error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            resource_type: Type of resource (file, directory, network, etc.)
            resource_path: Path or identifier of the resource
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, **kwargs)
        self.resource_type = resource_type
        self.resource_path = resource_path


class ToolConfigurationError(ToolException):
    """Exception raised when tool configuration is invalid."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            config_key: The configuration key that was invalid
            config_value: The invalid configuration value
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, recoverable=False, **kwargs)
        self.config_key = config_key
        self.config_value = config_value


class ToolDependencyError(ToolException):
    """Exception raised when tool dependencies are not met."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        missing_dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize dependency error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            missing_dependencies: List of missing dependencies
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, recoverable=False, **kwargs)
        self.missing_dependencies = missing_dependencies or []


class ToolNotFoundError(ToolException):
    """Exception raised when a requested tool is not found."""
    
    def __init__(
        self, 
        tool_name: str,
        available_tools: Optional[List[str]] = None,
        suggested_alternatives: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize tool not found error.
        
        Args:
            tool_name: Name of the tool that was not found
            available_tools: List of available tools
            suggested_alternatives: Suggested alternative tools
            **kwargs: Additional arguments for base class
        """
        message = f"Tool '{tool_name}' not found"
        if suggested_alternatives:
            message += f". Suggested alternatives: {', '.join(suggested_alternatives)}"
        
        super().__init__(message, tool_name, recoverable=False, **kwargs)
        self.available_tools = available_tools or []
        self.suggested_alternatives = suggested_alternatives or []


class ToolRegistrationError(ToolException):
    """Exception raised when tool registration fails."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        registration_reason: Optional[str] = None,
        **kwargs
    ):
        """Initialize registration error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            registration_reason: Reason why registration failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, recoverable=False, **kwargs)
        self.registration_reason = registration_reason


class MCPToolError(ToolException):
    """Exception raised when MCP tool integration fails."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str,
        mcp_server: Optional[str] = None,
        mcp_error_code: Optional[str] = None,
        **kwargs
    ):
        """Initialize MCP tool error.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            mcp_server: MCP server that failed
            mcp_error_code: MCP-specific error code
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, tool_name, **kwargs)
        self.mcp_server = mcp_server
        self.mcp_error_code = mcp_error_code


# Utility functions for error handling

def handle_tool_exception(exception: ToolException) -> Dict[str, Any]:
    """Convert tool exception to structured error response.
    
    Args:
        exception: The tool exception to handle
        
    Returns:
        Structured error response dictionary
    """
    return {
        "error": True,
        "error_type": exception.__class__.__name__,
        "message": str(exception),
        "tool_name": exception.tool_name,
        "recoverable": exception.recoverable,
        "error_code": exception.error_code,
        "context": exception.context,
        "suggestions": _generate_error_suggestions(exception)
    }


def _generate_error_suggestions(exception: ToolException) -> List[str]:
    """Generate helpful suggestions based on the exception type.
    
    Args:
        exception: The tool exception
        
    Returns:
        List of helpful suggestions
    """
    suggestions = []
    
    if isinstance(exception, ToolValidationError):
        suggestions.append("Check the input parameters and try again")
        if exception.validation_errors:
            suggestions.extend(f"Fix: {error}" for error in exception.validation_errors[:3])
    
    elif isinstance(exception, ToolPermissionError):
        suggestions.append("Check file/directory permissions")
        if exception.resource_path:
            suggestions.append(f"Verify access to: {exception.resource_path}")
    
    elif isinstance(exception, ToolSecurityError):
        suggestions.append("Review security settings and constraints")
        suggestions.append("Consider using a different approach")
    
    elif isinstance(exception, ToolTimeoutError):
        suggestions.append("Try reducing the scope of the operation")
        suggestions.append("Increase timeout settings if appropriate")
    
    elif isinstance(exception, ToolDependencyError):
        suggestions.append("Install missing dependencies")
        if exception.missing_dependencies:
            for dep in exception.missing_dependencies[:3]:
                suggestions.append(f"Install: {dep}")
    
    elif isinstance(exception, ToolNotFoundError):
        if exception.suggested_alternatives:
            suggestions.extend(f"Try: {alt}" for alt in exception.suggested_alternatives[:3])
        suggestions.append("Use --list-tools to see available tools")
    
    elif isinstance(exception, ToolResourceError):
        suggestions.append("Check if the resource exists and is accessible")
        if exception.resource_path:
            suggestions.append(f"Verify: {exception.resource_path}")
    
    else:
        suggestions.append("Check the tool documentation")
        if exception.recoverable:
            suggestions.append("Try the operation again")
    
    return suggestions


def is_recoverable_error(exception: Exception) -> bool:
    """Check if an exception represents a recoverable error.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is recoverable, False otherwise
    """
    if isinstance(exception, ToolException):
        return exception.recoverable
    
    # Non-tool exceptions are generally not recoverable in tool context
    return False


def get_error_severity(exception: ToolException) -> str:
    """Get the severity level of a tool exception.
    
    Args:
        exception: The tool exception
        
    Returns:
        Severity level as string (low, medium, high, critical)
    """
    if isinstance(exception, (ToolSecurityError, ToolPermissionError)):
        return "critical"
    elif isinstance(exception, (ToolExecutionError, ToolResourceError)):
        return "high"
    elif isinstance(exception, (ToolValidationError, ToolTimeoutError)):
        return "medium"
    else:
        return "low"