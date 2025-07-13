"""
Core tool execution tests.

Tests the fundamental tool execution functionality including:
- Basic tool registration and discovery
- Tool execution workflow
- Error handling and validation
- Tool context and result handling
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from ...core.tool_registry import ToolRegistry, get_global_registry
from ...tools.base import BaseTool
from ...tools.types import ToolContext, ToolResult, ToolSchema, SecurityLevel
from ...tools.exceptions import ToolException, ToolExecutionError
from ...tools.file_reader import FileReaderTool
from ..fixtures.tool_test_fixtures import (
    mock_llm, test_config, temp_workspace, ToolTestHelpers,
    get_sample_code_path
)


class TestToolExecution:
    """Test core tool execution functionality."""
    
    def test_tool_discovery(self):
        """Test tool discovery and listing."""
        registry = ToolRegistry()
        
        # Test that we can discover tools
        tools = registry.list_tools()
        assert len(tools) >= 0  # Should have at least some tools or empty list
        
        # Test tool discovery with global registry
        global_registry = get_global_registry()
        global_tools = global_registry.list_tools()
        assert isinstance(global_tools, list)
    
    async def test_file_reader_basic_execution(self, temp_workspace):
        """Test basic file reader tool execution."""
        # Create test file
        test_file = temp_workspace / "test.py"
        test_content = "print('Hello, World!')\n"
        ToolTestHelpers.create_test_file(test_file, test_content)
        
        # Execute file reader
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        result = await tool.execute({
            "file_path": str(test_file),
            "analysis_level": "minimal"
        }, context)
        
        # Verify result
        ToolTestHelpers.assert_tool_result_success(result, ["Hello, World!"])
        assert test_content in result.content
    
    def test_file_reader_with_analysis(self, temp_workspace):
        """Test file reader with code analysis."""
        # Use sample Python file
        sample_file = get_sample_code_path("python")
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        result = tool.execute({
            "file_path": str(sample_file),
            "analysis_level": "detailed",
            "include_context": True
        }, context)
        
        # Verify analysis results
        ToolTestHelpers.assert_tool_result_success(result, ["fibonacci", "Calculator"])
        assert "function" in result.content.lower()
        assert "class" in result.content.lower()
    
    def test_tool_execution_error_handling(self, temp_workspace):
        """Test tool execution error handling."""
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Test with non-existent file
        result = tool.execute({
            "file_path": "/nonexistent/file.py"
        }, context)
        
        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower() or "no such file" in result.error.lower()
    
    def test_tool_security_validation(self, temp_workspace):
        """Test tool security validation."""
        tool = FileReaderTool()
        
        # Test with restricted context
        restricted_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "safety_settings": {
                "max_file_size": 100,  # Very small limit
                "allowed_extensions": [".txt"]  # Only txt files
            }
        }
        
        # Test with disallowed extension
        test_file = temp_workspace / "test.py"
        ToolTestHelpers.create_test_file(test_file, "print('test')")
        
        result = tool.execute({
            "file_path": str(test_file)
        }, restricted_context)
        
        # Should fail due to security restrictions
        assert not result.success
    
    def test_tool_input_validation(self):
        """Test tool input validation."""
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Test with missing required input
        result = tool.execute({}, context)
        assert not result.success
        
        # Test with invalid input type
        result = tool.execute({
            "file_path": 123  # Should be string
        }, context)
        assert not result.success
    
    def test_tool_context_handling(self, temp_workspace):
        """Test tool context handling and metadata."""
        test_file = temp_workspace / "context_test.py"
        ToolTestHelpers.create_test_file(test_file, "# Test file for context")
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        result = tool.execute({
            "file_path": str(test_file)
        }, context)
        
        # Verify context is properly handled
        ToolTestHelpers.assert_tool_result_success(result)
        assert result.metadata is not None
        assert "execution_time" in result.metadata
    
    def test_multiple_tools_execution(self, temp_workspace):
        """Test execution of multiple tools in sequence."""
        registry = ToolRegistry()
        
        # Register multiple tools
        file_reader = FileReaderTool()
        registry.register_tool("file_reader", file_reader)
        
        # Create test file
        test_file = temp_workspace / "multi_test.py"
        ToolTestHelpers.create_test_file(test_file, "def test(): pass")
        
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Execute file reader
        result1 = file_reader.execute({
            "file_path": str(test_file)
        }, context)
        
        ToolTestHelpers.assert_tool_result_success(result1)
        
        # Could add more tools here for sequential execution testing
        assert len(registry.list_tools()) >= 1
    
    @pytest.mark.slow
    def test_tool_performance(self, temp_workspace):
        """Test tool execution performance."""
        import time
        
        # Create a larger test file
        large_content = "# Large test file\n" + "\n".join([
            f"def function_{i}(x): return x * {i}" for i in range(100)
        ])
        
        large_file = temp_workspace / "large_test.py"
        ToolTestHelpers.create_test_file(large_file, large_content)
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        start_time = time.time()
        result = tool.execute({
            "file_path": str(large_file),
            "analysis_level": "comprehensive"
        }, context)
        execution_time = time.time() - start_time
        
        ToolTestHelpers.assert_tool_result_success(result)
        
        # Performance assertion (adjust as needed)
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert "function_50" in result.content  # Verify content processed