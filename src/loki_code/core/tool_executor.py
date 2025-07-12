"""
Tool Execution Engine for Loki Code.

This module provides the ToolExecutor class that handles tool execution
with comprehensive error handling, security validation, and execution tracking.
Designed to work seamlessly with both local tools and future MCP tools.

Features:
- Secure tool execution with validation
- Comprehensive error handling and recovery
- Execution tracking and analytics
- Performance monitoring
- Security constraint enforcement
- Future MCP execution support
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
import logging

from .tool_registry import ToolRegistry, ToolExecutionRecord, get_global_registry
from ..tools.base import BaseTool
from ..tools.types import (
    ToolSchema, ToolContext, ToolResult, ToolCall, ToolExecution,
    SecurityLevel, ToolCapability, ConfirmationLevel, ToolStatus,
    SafetySettings
)
from ..tools.exceptions import (
    ToolException, ToolNotFoundError, ToolRegistrationError,
    ToolValidationError, ToolExecutionError, ToolSecurityError,
    ToolTimeoutError, handle_tool_exception
)
from ..utils.logging import get_logger


@dataclass
class ExecutionConfig:
    """Configuration for tool execution."""
    default_timeout: float = 30.0
    max_concurrent_executions: int = 3
    enable_execution_tracking: bool = True
    enable_performance_monitoring: bool = True
    retry_failed_executions: bool = False
    max_retries: int = 2
    retry_delay: float = 1.0


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    fastest_execution: float = float('inf')
    slowest_execution: float = 0.0
    total_execution_time: float = 0.0
    
    def update(self, execution_time: float, success: bool) -> None:
        """Update metrics with a new execution."""
        self.total_executions += 1
        self.total_execution_time += execution_time
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        self.fastest_execution = min(self.fastest_execution, execution_time)
        self.slowest_execution = max(self.slowest_execution, execution_time)
        self.average_execution_time = self.total_execution_time / self.total_executions
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "average_execution_time_ms": self.average_execution_time * 1000,
            "fastest_execution_ms": self.fastest_execution * 1000 if self.fastest_execution != float('inf') else None,
            "slowest_execution_ms": self.slowest_execution * 1000,
            "total_execution_time_ms": self.total_execution_time * 1000
        }


class ExecutionContext:
    """Context manager for tool execution."""
    
    def __init__(
        self, 
        executor: 'ToolExecutor',
        tool_name: str,
        input_data: Any,
        context: ToolContext
    ):
        self.executor = executor
        self.tool_name = tool_name
        self.input_data = input_data
        self.context = context
        self.start_time: Optional[float] = None
        self.execution_record: Optional[ToolExecutionRecord] = None
        self.logger = get_logger(__name__)
    
    async def __aenter__(self) -> 'ExecutionContext':
        """Enter execution context."""
        self.start_time = time.perf_counter()
        self.execution_record = ToolExecutionRecord.start(
            self.tool_name, 
            self.input_data, 
            self.context
        )
        
        # Check execution limits
        if not await self.executor._check_execution_limits():
            raise ToolExecutionError(
                "Maximum concurrent executions reached",
                self.tool_name
            )
        
        # Register active execution
        self.executor._active_executions[self.execution_record.execution_id] = self
        
        self.logger.debug(f"Started execution of {self.tool_name} [{self.execution_record.execution_id}]")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit execution context."""
        if self.execution_record and self.start_time:
            execution_time = time.perf_counter() - self.start_time
            
            if exc_type is None:
                # Successful execution
                self.executor._update_metrics(self.tool_name, execution_time, True)
            else:
                # Failed execution
                self.execution_record.fail(str(exc_val) if exc_val else "Unknown error")
                self.executor._update_metrics(self.tool_name, execution_time, False)
            
            # Add to registry history
            if self.executor.config.enable_execution_tracking:
                self.executor.registry.add_execution_record(self.execution_record)
        
        # Remove from active executions
        if self.execution_record:
            self.executor._active_executions.pop(self.execution_record.execution_id, None)
            
            self.logger.debug(
                f"Completed execution of {self.tool_name} "
                f"[{self.execution_record.execution_id}] "
                f"in {execution_time*1000:.1f}ms"
            )


class ToolExecutor:
    """
    Advanced tool execution engine with comprehensive error handling,
    security validation, and performance monitoring.
    
    Supports both local tools and future MCP tool execution with
    unified interface and consistent behavior.
    """
    
    def __init__(
        self, 
        registry: Optional[ToolRegistry] = None,
        config: Optional[ExecutionConfig] = None
    ):
        """Initialize the tool executor."""
        self.registry = registry or get_global_registry()
        self.config = config or ExecutionConfig()
        self.logger = get_logger(__name__)
        
        # Execution tracking
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._tool_metrics: Dict[str, ExecutionMetrics] = {}
        
        # Security and validation
        self._security_validators: List[Callable] = []
        self._input_validators: List[Callable] = []
        self._output_validators: List[Callable] = []
        
        # Performance monitoring
        self._performance_thresholds: Dict[str, float] = {
            "slow_execution_threshold": 5.0,  # seconds
            "very_slow_execution_threshold": 30.0  # seconds
        }
        
        self.logger.info("ToolExecutor initialized")
    
    async def execute_tool(
        self,
        tool_name: str,
        input_data: Any,
        context: ToolContext,
        timeout: Optional[float] = None,
        retry_on_failure: bool = False
    ) -> ToolResult:
        """
        Execute a tool with comprehensive error handling and validation.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for the tool
            context: Execution context
            timeout: Optional timeout in seconds
            retry_on_failure: Whether to retry on failure
            
        Returns:
            ToolResult from the execution
            
        Raises:
            ToolNotFoundError: If tool is not found
            ToolExecutionError: If execution fails
            ToolSecurityError: If security constraints are violated
            ToolTimeoutError: If execution times out
        """
        # Validate tool exists
        if tool_name not in self.registry:
            available_tools = list(self.registry.list_tool_names())
            suggested = self.registry._suggest_similar_tools(tool_name)
            
            raise ToolNotFoundError(
                tool_name,
                available_tools=available_tools,
                suggested_alternatives=suggested
            )
        
        # Get tool and validate it's enabled
        registration = self.registry.get_tool_registration(tool_name)
        if not registration or not registration.enabled:
            raise ToolExecutionError(
                f"Tool '{tool_name}' is disabled",
                tool_name
            )
        
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            raise ToolExecutionError(
                f"Failed to get tool instance for '{tool_name}'",
                tool_name
            )
        
        # Execute with retry logic if configured
        if retry_on_failure or self.config.retry_failed_executions:
            return await self._execute_with_retry(
                tool, tool_name, input_data, context, timeout
            )
        else:
            return await self._execute_single(
                tool, tool_name, input_data, context, timeout
            )
    
    async def _execute_single(
        self,
        tool: BaseTool,
        tool_name: str,
        input_data: Any,
        context: ToolContext,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Execute a tool once with full validation and monitoring."""
        timeout = timeout or self.config.default_timeout
        
        async with ExecutionContext(self, tool_name, input_data, context) as exec_ctx:
            try:
                # Pre-execution validation
                await self._validate_execution(tool, input_data, context)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    tool.safe_execute(input_data, context),
                    timeout=timeout
                )
                
                # Post-execution validation
                await self._validate_result(result, tool_name)
                
                # Update execution record
                if exec_ctx.execution_record:
                    exec_ctx.execution_record.complete(result)
                    exec_ctx.execution_record.security_level = tool.get_security_level()
                
                # Performance monitoring
                execution_time = time.perf_counter() - exec_ctx.start_time
                await self._monitor_performance(tool_name, execution_time, result)
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Tool execution timed out after {timeout}s"
                self.logger.error(f"{error_msg} for tool '{tool_name}'")
                raise ToolTimeoutError(error_msg, tool_name, timeout)
            
            except ToolException:
                # Re-raise tool exceptions as-is
                raise
            
            except Exception as e:
                error_msg = f"Unexpected error during tool execution: {str(e)}"
                self.logger.error(f"{error_msg} for tool '{tool_name}'", exc_info=True)
                raise ToolExecutionError(error_msg, tool_name) from e
    
    async def _execute_with_retry(
        self,
        tool: BaseTool,
        tool_name: str,
        input_data: Any,
        context: ToolContext,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Execute a tool with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying tool '{tool_name}' (attempt {attempt + 1})")
                    await asyncio.sleep(self.config.retry_delay * attempt)
                
                return await self._execute_single(tool, tool_name, input_data, context, timeout)
                
            except (ToolTimeoutError, ToolSecurityError, ToolValidationError):
                # Don't retry these types of errors
                raise
            
            except ToolException as e:
                last_exception = e
                if attempt == self.config.max_retries:
                    break
                
                self.logger.warning(
                    f"Tool '{tool_name}' failed on attempt {attempt + 1}: {str(e)}"
                )
        
        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise ToolExecutionError(f"Tool '{tool_name}' failed after {self.config.max_retries} retries", tool_name)
    
    async def execute_multiple_tools(
        self,
        tool_requests: List[Dict[str, Any]],
        context: ToolContext,
        concurrent: bool = False
    ) -> List[ToolResult]:
        """
        Execute multiple tools either sequentially or concurrently.
        
        Args:
            tool_requests: List of tool execution requests
            context: Execution context
            concurrent: Whether to execute concurrently
            
        Returns:
            List of ToolResults in the same order as requests
        """
        if concurrent:
            return await self._execute_concurrent(tool_requests, context)
        else:
            return await self._execute_sequential(tool_requests, context)
    
    async def _execute_sequential(
        self,
        tool_requests: List[Dict[str, Any]],
        context: ToolContext
    ) -> List[ToolResult]:
        """Execute tools sequentially."""
        results = []
        
        for request in tool_requests:
            try:
                result = await self.execute_tool(
                    tool_name=request['tool_name'],
                    input_data=request['input_data'],
                    context=context,
                    timeout=request.get('timeout'),
                    retry_on_failure=request.get('retry_on_failure', False)
                )
                results.append(result)
                
            except Exception as e:
                # Create error result
                error_result = ToolResult.failure_result(
                    f"Sequential execution failed: {str(e)}"
                )
                results.append(error_result)
        
        return results
    
    async def _execute_concurrent(
        self,
        tool_requests: List[Dict[str, Any]],
        context: ToolContext
    ) -> List[ToolResult]:
        """Execute tools concurrently."""
        # Create tasks
        tasks = []
        for request in tool_requests:
            task = asyncio.create_task(
                self.execute_tool(
                    tool_name=request['tool_name'],
                    input_data=request['input_data'],
                    context=context,
                    timeout=request.get('timeout'),
                    retry_on_failure=request.get('retry_on_failure', False)
                )
            )
            tasks.append(task)
        
        # Execute all tasks
        results = []
        for i, task in enumerate(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = ToolResult.failure_result(
                    f"Concurrent execution failed: {str(e)}"
                )
                results.append(error_result)
        
        return results
    
    async def validate_tool_call(
        self,
        tool_name: str,
        input_data: Any,
        context: ToolContext
    ) -> bool:
        """
        Validate a tool call before execution.
        
        Args:
            tool_name: Name of the tool
            input_data: Input data for the tool
            context: Execution context
            
        Returns:
            True if valid
            
        Raises:
            ToolValidationError: If validation fails
        """
        # Check tool exists
        if tool_name not in self.registry:
            raise ToolValidationError(f"Tool '{tool_name}' not found", tool_name)
        
        # Get tool
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            raise ToolValidationError(f"Failed to get tool instance for '{tool_name}'", tool_name)
        
        # Validate input
        try:
            await tool.validate_input(input_data)
        except Exception as e:
            raise ToolValidationError(f"Input validation failed: {str(e)}", tool_name) from e
        
        # Validate context
        try:
            tool.validate_context(context)
        except Exception as e:
            raise ToolValidationError(f"Context validation failed: {str(e)}", tool_name) from e
        
        # Security validation
        await self._validate_security_constraints(tool, context)
        
        return True
    
    async def _validate_execution(
        self,
        tool: BaseTool,
        input_data: Any,
        context: ToolContext
    ) -> None:
        """Validate tool execution pre-conditions."""
        # Input validation
        await tool.validate_input(input_data)
        
        # Context validation
        tool.validate_context(context)
        
        # Security validation
        await self._validate_security_constraints(tool, context)
        
        # Custom validators
        for validator in self._input_validators:
            await validator(tool, input_data, context)
    
    async def _validate_security_constraints(
        self,
        tool: BaseTool,
        context: ToolContext
    ) -> None:
        """Validate security constraints for tool execution."""
        schema = tool.get_schema()
        
        # Check if tool requires confirmation
        if tool.requires_confirmation(context) and not context.dry_run:
            raise ToolSecurityError(
                f"Tool '{schema.name}' requires confirmation before execution",
                schema.name,
                "confirmation_required"
            )
        
        # Check security level constraints
        if schema.security_level == SecurityLevel.DANGEROUS:
            if SecurityLevel.DANGEROUS not in context.safety_settings.require_confirmation_for:
                self.logger.warning(
                    f"Dangerous tool '{schema.name}' executed without confirmation requirement"
                )
        
        # Custom security validators
        for validator in self._security_validators:
            await validator(tool, context)
    
    async def _validate_result(self, result: ToolResult, tool_name: str) -> None:
        """Validate tool execution result."""
        # Basic result validation
        if result is None:
            raise ToolExecutionError(f"Tool '{tool_name}' returned None result", tool_name)
        
        # Custom output validators
        for validator in self._output_validators:
            await validator(result, tool_name)
    
    async def _check_execution_limits(self) -> bool:
        """Check if execution limits allow new execution."""
        active_count = len(self._active_executions)
        if active_count >= self.config.max_concurrent_executions:
            self.logger.warning(
                f"Maximum concurrent executions ({self.config.max_concurrent_executions}) reached. "
                f"Active: {active_count}"
            )
            return False
        return True
    
    async def _monitor_performance(
        self,
        tool_name: str,
        execution_time: float,
        result: ToolResult
    ) -> None:
        """Monitor and log performance metrics."""
        if not self.config.enable_performance_monitoring:
            return
        
        # Update metrics
        self._update_metrics(tool_name, execution_time, result.success)
        
        # Performance warnings
        slow_threshold = self._performance_thresholds["slow_execution_threshold"]
        very_slow_threshold = self._performance_thresholds["very_slow_execution_threshold"]
        
        if execution_time > very_slow_threshold:
            self.logger.warning(
                f"Very slow execution: {tool_name} took {execution_time:.2f}s "
                f"(threshold: {very_slow_threshold}s)"
            )
        elif execution_time > slow_threshold:
            self.logger.info(
                f"Slow execution: {tool_name} took {execution_time:.2f}s "
                f"(threshold: {slow_threshold}s)"
            )
    
    def _update_metrics(self, tool_name: str, execution_time: float, success: bool) -> None:
        """Update execution metrics for a tool."""
        if tool_name not in self._tool_metrics:
            self._tool_metrics[tool_name] = ExecutionMetrics()
        
        self._tool_metrics[tool_name].update(execution_time, success)
    
    def get_execution_metrics(self, tool_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get execution metrics for a tool or all tools."""
        if tool_name:
            metrics = self._tool_metrics.get(tool_name)
            return metrics.to_dict() if metrics else {}
        else:
            return {
                name: metrics.to_dict()
                for name, metrics in self._tool_metrics.items()
            }
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get information about currently active executions."""
        return [
            {
                "execution_id": exec_ctx.execution_record.execution_id,
                "tool_name": exec_ctx.tool_name,
                "start_time": exec_ctx.execution_record.start_time.isoformat(),
                "elapsed_time": (datetime.now() - exec_ctx.execution_record.start_time).total_seconds()
            }
            for exec_ctx in self._active_executions.values()
            if exec_ctx.execution_record
        ]
    
    def add_security_validator(self, validator: Callable) -> None:
        """Add a custom security validator."""
        self._security_validators.append(validator)
    
    def add_input_validator(self, validator: Callable) -> None:
        """Add a custom input validator."""
        self._input_validators.append(validator)
    
    def add_output_validator(self, validator: Callable) -> None:
        """Add a custom output validator."""
        self._output_validators.append(validator)
    
    def clear_metrics(self) -> None:
        """Clear all execution metrics."""
        self._tool_metrics.clear()
        self.logger.info("Cleared execution metrics")
    
    def get_executor_stats(self) -> Dict[str, Any]:
        """Get overall executor statistics."""
        total_executions = sum(metrics.total_executions for metrics in self._tool_metrics.values())
        successful_executions = sum(metrics.successful_executions for metrics in self._tool_metrics.values())
        
        return {
            "active_executions": len(self._active_executions),
            "total_tools_executed": len(self._tool_metrics),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "overall_success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
            "config": {
                "max_concurrent_executions": self.config.max_concurrent_executions,
                "default_timeout": self.config.default_timeout,
                "retry_enabled": self.config.retry_failed_executions,
                "max_retries": self.config.max_retries
            }
        }


# Global executor instance (can be replaced with dependency injection)
_global_executor: Optional[ToolExecutor] = None


def get_global_executor() -> ToolExecutor:
    """Get the global tool executor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = ToolExecutor()
    return _global_executor


def set_global_executor(executor: ToolExecutor) -> None:
    """Set the global tool executor instance."""
    global _global_executor
    _global_executor = executor