"""
Test suite for error handling utilities.

This module tests the standardized error handling decorators, exception
mapping, and error reporting functionality.
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

from loki_code.utils.error_handling import (
    handle_tool_execution, handle_provider_operation, 
    AsyncProviderMixin, get_logger
)
from loki_code.tools.exceptions import ToolExecutionError, ToolValidationError
from loki_code.core.providers.exceptions import (
    ProviderConnectionError, ProviderModelError, ProviderTimeoutError
)


class TestHandleToolExecution:
    """Test the handle_tool_execution decorator."""
    
    def test_successful_sync_execution(self):
        """Test decorator with successful synchronous function."""
        mock_logger = Mock()
        
        @handle_tool_execution("test_operation", mock_logger)
        def test_function(value):
            return f"success: {value}"
        
        result = test_function("test")
        
        assert result == "success: test"
        mock_logger.debug.assert_called_with("Starting test_operation")
        mock_logger.info.assert_called_with("test_operation completed successfully")
    
    @pytest.mark.asyncio
    async def test_successful_async_execution(self):
        """Test decorator with successful asynchronous function."""
        mock_logger = Mock()
        
        @handle_tool_execution("async_test", mock_logger)
        async def async_test_function(value):
            await asyncio.sleep(0.01)  # Simulate async work
            return f"async success: {value}"
        
        result = await async_test_function("test")
        
        assert result == "async success: test"
        mock_logger.debug.assert_called_with("Starting async_test")
        mock_logger.info.assert_called_with("async_test completed successfully")
    
    def test_sync_function_exception_handling(self):
        """Test decorator handling exceptions in sync functions."""
        mock_logger = Mock()
        
        @handle_tool_execution("failing_operation", mock_logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ToolExecutionError) as exc_info:
            failing_function()
        
        # Check that original exception is wrapped
        assert "failing_operation failed: Test error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)
        
        # Check logging
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert "failing_operation failed: Test error" in error_call[0][0]
        assert error_call[1]["exc_info"] is True
    
    @pytest.mark.asyncio
    async def test_async_function_exception_handling(self):
        """Test decorator handling exceptions in async functions."""
        mock_logger = Mock()
        
        @handle_tool_execution("async_failing", mock_logger)
        async def async_failing_function():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async test error")
        
        with pytest.raises(ToolExecutionError) as exc_info:
            await async_failing_function()
        
        # Check that original exception is wrapped
        assert "async_failing failed: Async test error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        
        # Check logging
        mock_logger.error.assert_called_once()
    
    def test_default_logger_creation(self):
        """Test that default logger is created when none provided."""
        @handle_tool_execution("test_default_logger")
        def test_function():
            return "success"
        
        # Should not raise an error
        result = test_function()
        assert result == "success"
    
    def test_function_metadata_preservation(self):
        """Test that original function metadata is preserved."""
        mock_logger = Mock()
        
        @handle_tool_execution("preserve_test", mock_logger)
        def original_function(arg1, arg2="default"):
            """Original function docstring."""
            return f"{arg1}-{arg2}"
        
        # Check that function name and docstring are preserved
        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original function docstring."
        
        # Check that function still works correctly
        result = original_function("test", arg2="value")
        assert result == "test-value"
    
    def test_exception_chaining(self):
        """Test that exception chaining preserves original context."""
        @handle_tool_execution("chain_test")
        def nested_exception_function():
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Nested error") from e
        
        with pytest.raises(ToolExecutionError) as exc_info:
            nested_exception_function()
        
        # Should preserve the nested exception chain
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert isinstance(exc_info.value.__cause__.__cause__, ValueError)


class TestHandleProviderOperation:
    """Test the handle_provider_operation decorator."""
    
    def test_successful_provider_operation(self):
        """Test decorator with successful provider operation."""
        mock_logger = Mock()
        
        @handle_provider_operation("provider_test", mock_logger)
        def provider_function(model_name):
            return f"Generated with {model_name}"
        
        result = provider_function("test-model")
        
        assert result == "Generated with test-model"
        mock_logger.debug.assert_called_with("Starting provider_test")
        mock_logger.info.assert_called_with("provider_test completed successfully")
    
    @pytest.mark.asyncio
    async def test_async_provider_operation(self):
        """Test decorator with async provider operation."""
        mock_logger = Mock()
        
        @handle_provider_operation("async_provider", mock_logger)
        async def async_provider_function():
            await asyncio.sleep(0.01)
            return "async result"
        
        result = await async_provider_function()
        
        assert result == "async result"
        mock_logger.debug.assert_called_with("Starting async_provider")
        mock_logger.info.assert_called_with("async_provider completed successfully")
    
    def test_connection_error_mapping(self):
        """Test that connection errors are properly mapped."""
        @handle_provider_operation("connection_test")
        def connection_failing_function():
            raise ConnectionError("Connection failed")
        
        with pytest.raises(ProviderConnectionError) as exc_info:
            connection_failing_function()
        
        assert "connection_test failed: Connection failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ConnectionError)
    
    def test_timeout_error_mapping(self):
        """Test that timeout errors are properly mapped."""
        @handle_provider_operation("timeout_test")
        def timeout_failing_function():
            raise TimeoutError("Operation timed out")
        
        with pytest.raises(ProviderTimeoutError) as exc_info:
            timeout_failing_function()
        
        assert "timeout_test failed: Operation timed out" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, TimeoutError)
    
    def test_generic_error_mapping(self):
        """Test that generic errors are mapped to ProviderModelError."""
        @handle_provider_operation("generic_test")
        def generic_failing_function():
            raise ValueError("Generic error")
        
        with pytest.raises(ProviderModelError) as exc_info:
            generic_failing_function()
        
        assert "generic_test failed: Generic error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)
    
    def test_multiple_error_types(self):
        """Test handling of different error types in one decorator."""
        @handle_provider_operation("multi_error_test")
        def multi_error_function(error_type):
            if error_type == "connection":
                raise ConnectionError("Connection issue")
            elif error_type == "timeout":
                raise TimeoutError("Timeout issue")
            else:
                raise ValueError("Generic issue")
        
        # Test connection error
        with pytest.raises(ProviderConnectionError):
            multi_error_function("connection")
        
        # Test timeout error
        with pytest.raises(ProviderTimeoutError):
            multi_error_function("timeout")
        
        # Test generic error
        with pytest.raises(ProviderModelError):
            multi_error_function("other")


class TestAsyncProviderMixin:
    """Test the AsyncProviderMixin error handling."""
    
    def test_mixin_method_creation(self):
        """Test that mixin creates sync wrapper methods."""
        class TestProvider(AsyncProviderMixin):
            async def async_generate(self, prompt):
                return f"Generated: {prompt}"
            
            async def async_health_check(self):
                return True
        
        provider = TestProvider()
        
        # Should have sync wrapper methods
        assert hasattr(provider, 'generate')
        assert hasattr(provider, 'health_check')
        
        # Sync methods should work
        result = provider.generate("test prompt")
        assert result == "Generated: test prompt"
        
        health = provider.health_check()
        assert health is True
    
    def test_mixin_error_handling(self):
        """Test that mixin properly handles async errors in sync wrappers."""
        class FailingProvider(AsyncProviderMixin):
            async def async_operation(self):
                raise RuntimeError("Async error")
        
        provider = FailingProvider()
        
        # Error should be properly propagated through sync wrapper
        with pytest.raises(RuntimeError) as exc_info:
            provider.operation()
        
        assert "Async error" in str(exc_info.value)
    
    def test_mixin_preserves_return_values(self):
        """Test that mixin preserves return values correctly."""
        class DataProvider(AsyncProviderMixin):
            async def async_get_data(self, key):
                data = {"key": key, "value": f"data_for_{key}"}
                return data
            
            async def async_get_list(self):
                return [1, 2, 3, 4, 5]
            
            async def async_get_none(self):
                return None
        
        provider = DataProvider()
        
        # Test dict return
        data = provider.get_data("test")
        assert data == {"key": "test", "value": "data_for_test"}
        
        # Test list return
        list_data = provider.get_list()
        assert list_data == [1, 2, 3, 4, 5]
        
        # Test None return
        none_data = provider.get_none()
        assert none_data is None
    
    @pytest.mark.asyncio
    async def test_mixin_with_async_context(self):
        """Test that mixin works correctly in async context."""
        class AsyncContextProvider(AsyncProviderMixin):
            async def async_complex_operation(self, steps):
                results = []
                for step in steps:
                    await asyncio.sleep(0.001)  # Simulate async work
                    results.append(f"completed_{step}")
                return results
        
        provider = AsyncContextProvider()
        
        # Should work in both sync and async contexts
        steps = ["step1", "step2", "step3"]
        
        # Test sync wrapper
        sync_result = provider.complex_operation(steps)
        assert sync_result == ["completed_step1", "completed_step2", "completed_step3"]
        
        # Test original async method
        async_result = await provider.async_complex_operation(steps)
        assert async_result == ["completed_step1", "completed_step2", "completed_step3"]


class TestLoggingIntegration:
    """Test logging integration in error handling."""
    
    def test_get_logger_function(self):
        """Test the get_logger utility function."""
        logger = get_logger("test.module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
    
    def test_logger_hierarchy(self):
        """Test that loggers follow proper hierarchy."""
        parent_logger = get_logger("loki_code.test")
        child_logger = get_logger("loki_code.test.child")
        
        assert parent_logger.name == "loki_code.test"
        assert child_logger.name == "loki_code.test.child"
        assert child_logger.parent == parent_logger
    
    @patch('loki_code.utils.error_handling.logging.getLogger')
    def test_decorator_logger_usage(self, mock_get_logger):
        """Test that decorators use loggers correctly."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @handle_tool_execution("logger_test")
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        mock_get_logger.assert_called_with("loki_code.tools.logger_test")
        mock_logger.debug.assert_called_with("Starting logger_test")
        mock_logger.info.assert_called_with("logger_test completed successfully")
    
    def test_error_logging_with_context(self):
        """Test that errors are logged with proper context."""
        mock_logger = Mock()
        
        @handle_tool_execution("context_test", mock_logger)
        def failing_function_with_context():
            # Create some local context
            operation_id = "op_12345"
            data = {"key": "value"}
            raise ValueError(f"Error in operation {operation_id}")
        
        with pytest.raises(ToolExecutionError):
            failing_function_with_context()
        
        # Check that error was logged with exc_info
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert "context_test failed" in error_call[0][0]
        assert error_call[1]["exc_info"] is True


class TestErrorHandlingEdgeCases:
    """Test edge cases and error handling robustness."""
    
    def test_nested_decorator_application(self):
        """Test that decorators can be nested without conflicts."""
        mock_logger = Mock()
        
        @handle_provider_operation("outer_operation", mock_logger)
        @handle_tool_execution("inner_operation", mock_logger)
        def nested_decorated_function(value):
            if value == "fail":
                raise ValueError("Nested failure")
            return f"nested success: {value}"
        
        # Test success case
        result = nested_decorated_function("test")
        assert result == "nested success: test"
        
        # Test failure case - should be wrapped by outer decorator
        with pytest.raises(ProviderModelError) as exc_info:
            nested_decorated_function("fail")
        
        # Should be wrapped by provider decorator (outer)
        assert "outer_operation failed" in str(exc_info.value)
        # But the inner ToolExecutionError should be the immediate cause
        assert isinstance(exc_info.value.__cause__, ToolExecutionError)
    
    def test_decorator_with_complex_signatures(self):
        """Test decorators with functions having complex signatures."""
        @handle_tool_execution("complex_sig")
        def complex_function(pos_arg, *args, keyword=None, **kwargs):
            return {
                "pos_arg": pos_arg,
                "args": args,
                "keyword": keyword,
                "kwargs": kwargs
            }
        
        result = complex_function(
            "position", 
            "extra1", "extra2",
            keyword="test",
            extra_kw1="value1",
            extra_kw2="value2"
        )
        
        expected = {
            "pos_arg": "position",
            "args": ("extra1", "extra2"),
            "keyword": "test",
            "kwargs": {"extra_kw1": "value1", "extra_kw2": "value2"}
        }
        
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_async_generator_handling(self):
        """Test that decorators work with async generators."""
        @handle_tool_execution("async_gen_test")
        async def async_generator_function():
            for i in range(3):
                await asyncio.sleep(0.001)
                yield f"item_{i}"
        
        # Should work as async generator
        results = []
        async for item in async_generator_function():
            results.append(item)
        
        assert results == ["item_0", "item_1", "item_2"]
    
    def test_exception_message_preservation(self):
        """Test that original exception messages are preserved."""
        original_msg = "Very specific error message with details"
        
        @handle_tool_execution("message_test")
        def specific_error_function():
            raise ValueError(original_msg)
        
        with pytest.raises(ToolExecutionError) as exc_info:
            specific_error_function()
        
        # Original message should be preserved in the wrapper
        assert original_msg in str(exc_info.value)
        # And in the original exception
        assert str(exc_info.value.__cause__) == original_msg
    
    def test_concurrent_decorated_functions(self):
        """Test that decorators work correctly with concurrent execution."""
        import threading
        import time
        
        results = {}
        errors = {}
        
        @handle_tool_execution("concurrent_test")
        def concurrent_function(thread_id, should_fail=False):
            time.sleep(0.01)  # Simulate work
            if should_fail:
                raise ValueError(f"Error in thread {thread_id}")
            return f"Success from thread {thread_id}"
        
        def worker(thread_id, should_fail=False):
            try:
                result = concurrent_function(thread_id, should_fail)
                results[thread_id] = result
            except Exception as e:
                errors[thread_id] = e
        
        # Start multiple threads
        threads = []
        for i in range(5):
            should_fail = (i == 2)  # Make thread 2 fail
            thread = threading.Thread(target=worker, args=(i, should_fail))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 4  # 4 successful threads
        assert len(errors) == 1   # 1 failed thread
        
        assert results[0] == "Success from thread 0"
        assert results[1] == "Success from thread 1"
        assert results[3] == "Success from thread 3"
        assert results[4] == "Success from thread 4"
        
        assert isinstance(errors[2], ToolExecutionError)
        assert "Error in thread 2" in str(errors[2])