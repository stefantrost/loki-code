"""
Simplified core tool execution tests.

Basic tests to validate the new test structure works correctly.
"""

import pytest
from pathlib import Path

from ...tools.file_reader import FileReaderTool
from ..fixtures.tool_test_fixtures import (
    temp_workspace, ToolTestHelpers, get_sample_code_path
)


class TestToolExecutionSimple:
    """Simplified test class for validating test structure."""
    
    def test_basic_imports(self):
        """Test that all imports work correctly."""
        # Test tool imports
        tool = FileReaderTool()
        assert tool is not None
        
        # Test helper imports
        helpers = ToolTestHelpers()
        assert helpers is not None
    
    def test_sample_files_exist(self):
        """Test that sample code files exist."""
        languages = ["python", "javascript", "typescript", "rust"]
        
        for lang in languages:
            sample_path = get_sample_code_path(lang)
            assert sample_path.exists(), f"Sample file for {lang} not found"
            assert sample_path.stat().st_size > 0, f"Sample file for {lang} is empty"
    
    def test_temp_workspace_fixture(self, temp_workspace):
        """Test that temp workspace fixture works."""
        assert temp_workspace.exists()
        assert temp_workspace.is_dir()
        
        # Test creating files in workspace
        test_file = temp_workspace / "test.txt"
        ToolTestHelpers.create_test_file(test_file, "test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_tool_helper_methods(self):
        """Test tool helper methods."""
        # Test context creation
        context = ToolTestHelpers.create_mock_tool_context()
        assert context is not None
        assert "project_path" in context
        assert "permissions" in context
        
        # Test result assertion helper
        from ...tools.types import ToolResult, ToolStatus
        
        success_result = ToolResult(
            success=True,
            output="test content with expected",
            message="Success",
            status=ToolStatus.SUCCESS
        )
        
        # Should not raise
        ToolTestHelpers.assert_tool_result_success(success_result, ["expected"])
        
        # Test failure case
        failure_result = ToolResult(
            success=False,
            output=None,
            message="Test error",
            status=ToolStatus.FAILURE
        )
        
        with pytest.raises(AssertionError):
            ToolTestHelpers.assert_tool_result_success(failure_result)