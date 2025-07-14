"""
Unified error handling utilities for Loki Code.

This module provides decorators and utilities to standardize error handling
across the codebase, reducing duplication and ensuring consistent behavior.
"""

import functools
import asyncio
import logging
from typing import Any, Callable, Optional, Type, Union, Dict
from ..utils.logging import get_logger


class LokiCodeError(Exception):
    """Base exception for all Loki Code errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ToolExecutionError(LokiCodeError):
    """Error during tool execution."""
    pass


class ConfigurationError(LokiCodeError):
    """Configuration-related error."""
    pass


class ProviderError(LokiCodeError):
    """LLM provider-related error."""
    pass


class ValidationError(LokiCodeError):
    """Input validation error."""
    pass


def handle_tool_execution(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to standardize tool execution error handling.
    
    Args:
        operation_name: Human-readable name of the operation
        logger: Optional logger instance (defaults to operation-specific logger)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            _logger = logger or get_logger(f"loki_code.tools.{operation_name}")
            
            try:
                _logger.debug(f"Starting {operation_name}")
                result = await func(*args, **kwargs)
                _logger.info(f"{operation_name} completed successfully")
                return result
                
            except ToolExecutionError:
                # Already a tool error, just re-raise
                raise
                
            except (FileNotFoundError, PermissionError, OSError) as e:
                _logger.error(f"{operation_name} failed - file system error: {e}")
                raise ToolExecutionError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "filesystem", "original_error": str(e)}
                ) from e
                
            except (ValueError, TypeError) as e:
                _logger.error(f"{operation_name} failed - validation error: {e}")
                raise ValidationError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "validation", "original_error": str(e)}
                ) from e
                
            except Exception as e:
                _logger.error(f"{operation_name} failed - unexpected error: {e}", exc_info=True)
                raise ToolExecutionError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "unexpected", "original_error": str(e)}
                ) from e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            _logger = logger or get_logger(f"loki_code.tools.{operation_name}")
            
            try:
                _logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                _logger.info(f"{operation_name} completed successfully")
                return result
                
            except ToolExecutionError:
                # Already a tool error, just re-raise
                raise
                
            except (FileNotFoundError, PermissionError, OSError) as e:
                _logger.error(f"{operation_name} failed - file system error: {e}")
                raise ToolExecutionError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "filesystem", "original_error": str(e)}
                ) from e
                
            except (ValueError, TypeError) as e:
                _logger.error(f"{operation_name} failed - validation error: {e}")
                raise ValidationError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "validation", "original_error": str(e)}
                ) from e
                
            except Exception as e:
                _logger.error(f"{operation_name} failed - unexpected error: {e}", exc_info=True)
                raise ToolExecutionError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "unexpected", "original_error": str(e)}
                ) from e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_provider_operation(operation_name: str, timeout: Optional[float] = None):
    """
    Decorator to standardize LLM provider error handling.
    
    Args:
        operation_name: Human-readable name of the operation
        timeout: Optional timeout for the operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"loki_code.providers.{operation_name}")
            
            try:
                logger.debug(f"Starting {operation_name}")
                
                if timeout and asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = await func(*args, **kwargs)
                
                logger.info(f"{operation_name} completed successfully")
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"{operation_name} timed out after {timeout}s")
                raise ProviderError(
                    f"{operation_name} timed out",
                    details={"error_type": "timeout", "timeout_seconds": timeout}
                )
                
            except ConnectionError as e:
                logger.error(f"{operation_name} failed - connection error: {e}")
                raise ProviderError(
                    f"{operation_name} failed: Connection error",
                    details={"error_type": "connection", "original_error": str(e)}
                ) from e
                
            except Exception as e:
                logger.error(f"{operation_name} failed - unexpected error: {e}", exc_info=True)
                raise ProviderError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "unexpected", "original_error": str(e)}
                ) from e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"loki_code.providers.{operation_name}")
            
            try:
                logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.info(f"{operation_name} completed successfully")
                return result
                
            except ConnectionError as e:
                logger.error(f"{operation_name} failed - connection error: {e}")
                raise ProviderError(
                    f"{operation_name} failed: Connection error",
                    details={"error_type": "connection", "original_error": str(e)}
                ) from e
                
            except Exception as e:
                logger.error(f"{operation_name} failed - unexpected error: {e}", exc_info=True)
                raise ProviderError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "unexpected", "original_error": str(e)}
                ) from e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_configuration_operation(operation_name: str):
    """
    Decorator to standardize configuration operation error handling.
    
    Args:
        operation_name: Human-readable name of the operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"loki_code.config.{operation_name}")
            
            try:
                logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.info(f"{operation_name} completed successfully")
                return result
                
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"{operation_name} failed - file access error: {e}")
                raise ConfigurationError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "file_access", "original_error": str(e)}
                ) from e
                
            except (ValueError, TypeError) as e:
                logger.error(f"{operation_name} failed - validation error: {e}")
                raise ConfigurationError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "validation", "original_error": str(e)}
                ) from e
                
            except Exception as e:
                logger.error(f"{operation_name} failed - unexpected error: {e}", exc_info=True)
                raise ConfigurationError(
                    f"{operation_name} failed: {e}",
                    details={"error_type": "unexpected", "original_error": str(e)}
                ) from e
        
        return wrapper
    
    return decorator


class AsyncProviderMixin:
    """Mixin to provide async execution utilities for providers."""
    
    async def run_sync_in_executor(
        self, 
        func: Callable, 
        *args, 
        timeout: Optional[float] = None,
        executor=None
    ) -> Any:
        """
        Run a synchronous function in an executor with optional timeout.
        
        Args:
            func: The synchronous function to run
            *args: Arguments to pass to the function
            timeout: Optional timeout in seconds
            executor: Optional executor to use (defaults to thread pool)
        """
        loop = asyncio.get_event_loop()
        
        try:
            if timeout:
                return await asyncio.wait_for(
                    loop.run_in_executor(executor, func, *args), 
                    timeout=timeout
                )
            else:
                return await loop.run_in_executor(executor, func, *args)
                
        except asyncio.TimeoutError:
            raise ProviderError(
                f"Operation timed out after {timeout}s",
                details={"error_type": "timeout", "timeout_seconds": timeout}
            )


def validate_input(
    data: Any, 
    field_name: str, 
    expected_type: Type = None,
    required: bool = True,
    validator: Optional[Callable] = None
) -> Any:
    """
    Standardized input validation utility.
    
    Args:
        data: The data to validate
        field_name: Name of the field being validated
        expected_type: Expected type of the data
        required: Whether the field is required
        validator: Optional custom validator function
    
    Returns:
        The validated data
        
    Raises:
        ValidationError: If validation fails
    """
    if required and data is None:
        raise ValidationError(f"{field_name} is required")
    
    if data is not None and expected_type and not isinstance(data, expected_type):
        raise ValidationError(
            f"{field_name} must be of type {expected_type.__name__}, got {type(data).__name__}"
        )
    
    if validator:
        try:
            return validator(data)
        except Exception as e:
            raise ValidationError(f"{field_name} validation failed: {e}") from e
    
    return data