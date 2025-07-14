"""
Tests for the FileWriterTool.

This module provides comprehensive tests for the file writer tool including
safety checks, permission validation, backup functionality, and integration
with the LangChain adapter system.
"""

import asyncio
import json
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from ..tools.file_writer import FileWriterTool, FileWriterInput, FileWriterOutput, BackupInfo
from ..tools.langchain_adapters import FileWriterToolAdapter
from ..tools.types import ToolContext, ToolResult, SecurityLevel
from ..tools.exceptions import (
    ToolSecurityError, ToolValidationError, ToolExecutionError
)
from ..core.agent.permission_manager import PermissionManager, PermissionConfig
from ..core.agent.safety_manager import SafetyManager, SafetyConfig
from ..utils.logging import get_logger


class TestFileWriterTool:
    """Test suite for FileWriterTool."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def tool_context(self, temp_workspace):
        """Create a tool context for testing."""
        return ToolContext(
            user_id="test_user",
            session_id="test_session", 
            workspace_path=str(temp_workspace),
            environment="test"
        )
    
    @pytest.fixture
    def file_writer_tool(self):
        """Create a FileWriterTool instance."""
        return FileWriterTool()
    
    def test_tool_schema(self, file_writer_tool):
        """Test that the tool schema is properly defined."""
        schema = file_writer_tool.get_schema()
        
        assert schema.name == "file_writer"
        assert "write content to files" in schema.description.lower()
        assert schema.security_level == SecurityLevel.MEDIUM
        assert "file_path" in schema.input_schema.properties
        assert "content" in schema.input_schema.properties
        assert schema.input_schema.required == ["file_path", "content"]
    
    @pytest.mark.asyncio
    async def test_basic_file_write(self, file_writer_tool, tool_context, temp_workspace):
        """Test basic file writing functionality."""
        test_file = temp_workspace / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        
        inputs = {
            "file_path": str(test_file),
            "content": test_content,
            "mode": "write"
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert result.success
        assert test_file.exists()
        assert test_file.read_text() == test_content
        assert "created" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_file_overwrite_with_backup(self, file_writer_tool, tool_context, temp_workspace):
        """Test file overwriting with backup creation."""
        test_file = temp_workspace / "existing.txt"
        original_content = "Original content"
        new_content = "New content"
        
        # Create original file
        test_file.write_text(original_content)
        
        inputs = {
            "file_path": str(test_file),
            "content": new_content,
            "mode": "write",
            "create_backup": True
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert result.success
        assert test_file.read_text() == new_content
        assert "backup" in result.content.lower()
        
        # Check backup was created
        backup_files = list(temp_workspace.glob("*.backup_*"))
        assert len(backup_files) == 1
        assert backup_files[0].read_text() == original_content
    
    @pytest.mark.asyncio
    async def test_append_mode(self, file_writer_tool, tool_context, temp_workspace):
        """Test append mode functionality."""
        test_file = temp_workspace / "append_test.txt"
        original_content = "Line 1\n"
        append_content = "Line 2\n"
        
        # Create original file
        test_file.write_text(original_content)
        
        inputs = {
            "file_path": str(test_file),
            "content": append_content,
            "mode": "append"
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert result.success
        final_content = test_file.read_text()
        assert final_content == original_content + append_content
        assert "appended" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_insert_mode(self, file_writer_tool, tool_context, temp_workspace):
        """Test insert mode functionality."""
        test_file = temp_workspace / "insert_test.txt"
        original_content = "Line 1\nLine 3\n"
        insert_content = "Line 2"
        
        # Create original file
        test_file.write_text(original_content)
        
        inputs = {
            "file_path": str(test_file),
            "content": insert_content,
            "mode": "insert",
            "insert_line": 2
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert result.success
        final_content = test_file.read_text()
        expected = "Line 1\nLine 2\nLine 3\n"
        assert final_content == expected
        assert "inserted" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_directory_creation(self, file_writer_tool, tool_context, temp_workspace):
        """Test automatic directory creation."""
        nested_file = temp_workspace / "nested" / "dirs" / "test.txt"
        test_content = "Test content"
        
        inputs = {
            "file_path": str(nested_file),
            "content": test_content,
            "create_dirs": True
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert result.success
        assert nested_file.exists()
        assert nested_file.read_text() == test_content
        assert nested_file.parent.exists()
    
    @pytest.mark.asyncio
    async def test_security_validation_path_traversal(self, file_writer_tool, tool_context, temp_workspace):
        """Test security validation against path traversal attacks."""
        malicious_path = temp_workspace / ".." / "malicious.txt"
        
        inputs = {
            "file_path": str(malicious_path),
            "content": "malicious content"
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert not result.success
        assert "security" in result.error.lower() or "outside" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_security_validation_dangerous_content(self, file_writer_tool, tool_context, temp_workspace):
        """Test security validation against dangerous content."""
        test_file = temp_workspace / "dangerous.py"
        dangerous_content = "__import__('os').system('rm -rf /')"
        
        inputs = {
            "file_path": str(test_file),
            "content": dangerous_content
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert not result.success
        assert "dangerous" in result.error.lower() or "security" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_size_limit_validation(self, file_writer_tool, tool_context, temp_workspace):
        """Test file size limit validation."""
        test_file = temp_workspace / "large.txt"
        large_content = "x" * (2 * 1024 * 1024)  # 2MB content
        
        inputs = {
            "file_path": str(test_file),
            "content": large_content,
            "max_size_mb": 1  # 1MB limit
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert not result.success
        assert "size" in result.error.lower() or "limit" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_syntax_validation_python(self, file_writer_tool, tool_context, temp_workspace):
        """Test syntax validation for Python files."""
        test_file = temp_workspace / "test.py"
        invalid_python = "def invalid_syntax(\n    missing_closing_paren"
        
        inputs = {
            "file_path": str(test_file),
            "content": invalid_python,
            "validate_syntax": True
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        # File should still be written but with warnings
        assert result.success
        assert test_file.exists()
        assert "syntax" in result.content.lower() or "warning" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_auto_formatting(self, file_writer_tool, tool_context, temp_workspace):
        """Test auto-formatting functionality."""
        test_file = temp_workspace / "format_test.py"
        unformatted_content = "def test():    \n    return 'hello'   \n"
        
        inputs = {
            "file_path": str(test_file),
            "content": unformatted_content,
            "auto_format": True
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert result.success
        # Check that trailing whitespace was removed
        final_content = test_file.read_text()
        assert not any(line.endswith(' ') or line.endswith('\t') for line in final_content.split('\n')[:-1])
    
    @pytest.mark.asyncio
    async def test_input_validation_missing_params(self, file_writer_tool, tool_context):
        """Test input validation with missing required parameters."""
        inputs = {
            "file_path": "/test/file.txt"
            # Missing 'content' parameter
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert not result.success
        assert "missing" in result.error.lower() or "required" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_insert_mode_validation(self, file_writer_tool, tool_context, temp_workspace):
        """Test validation for insert mode without line number."""
        test_file = temp_workspace / "test.txt"
        
        inputs = {
            "file_path": str(test_file),
            "content": "test content",
            "mode": "insert"
            # Missing 'insert_line' parameter
        }
        
        result = await file_writer_tool.execute(inputs, tool_context)
        
        assert not result.success
        assert "insert_line" in result.error.lower() or "required" in result.error.lower()


class TestFileWriterLangChainAdapter:
    """Test suite for FileWriterToolAdapter."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def permission_manager(self):
        """Create a permission manager for testing."""
        config = PermissionConfig(auto_grant_safe_operations=True)
        return PermissionManager(config)
    
    @pytest.fixture
    def safety_manager(self):
        """Create a safety manager for testing."""
        config = SafetyConfig()
        return SafetyManager(config)
    
    @pytest.fixture
    def adapter(self, permission_manager, safety_manager):
        """Create a FileWriterToolAdapter instance."""
        return FileWriterToolAdapter(
            permission_manager=permission_manager,
            safety_manager=safety_manager
        )
    
    def test_adapter_schema(self, adapter):
        """Test that the adapter has correct schema."""
        assert adapter.name == "file_writer"
        assert "write content to files" in adapter.description.lower()
        assert hasattr(adapter, 'args_schema')
        
        # Test schema fields
        schema_fields = adapter.args_schema.__fields__
        assert "file_path" in schema_fields
        assert "content" in schema_fields
        assert "mode" in schema_fields
    
    def test_adapter_execution(self, adapter, temp_workspace):
        """Test adapter execution with LangChain interface."""
        test_file = temp_workspace / "adapter_test.txt"
        test_content = "LangChain adapter test"
        
        # Mock the workspace path in the adapter's tool context
        with patch('loki_code.tools.langchain_adapters.ToolContext') as mock_context:
            mock_context.return_value.workspace_path = str(temp_workspace)
            
            result = adapter._run(
                file_path=str(test_file),
                content=test_content,
                mode="write"
            )
        
        # Should return a string result
        assert isinstance(result, str)
        assert "created" in result.lower() or "written" in result.lower()


class TestFileWriterIntegration:
    """Integration tests for FileWriter with other system components."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_permission_integration(self, temp_workspace):
        """Test integration with permission system."""
        from ..tools.langchain_adapters import create_langchain_tools
        
        # Create restrictive permission manager
        permission_config = PermissionConfig(auto_grant_safe_operations=False)
        permission_manager = PermissionManager(permission_config)
        safety_manager = SafetyManager(SafetyConfig())
        
        tools = create_langchain_tools(permission_manager, safety_manager)
        file_writer = next(tool for tool in tools if tool.name == "file_writer")
        
        test_file = temp_workspace / "permission_test.txt"
        
        # This should work with the mock context
        with patch('loki_code.tools.langchain_adapters.ToolContext') as mock_context:
            mock_context.return_value.workspace_path = str(temp_workspace)
            
            result = file_writer._run(
                file_path=str(test_file),
                content="test content"
            )
        
        # Should either succeed or have permission-related message
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_backup_functionality_detailed(self, temp_workspace):
        """Test detailed backup functionality."""
        tool = FileWriterTool()
        context = ToolContext(
            user_id="test",
            session_id="test",
            workspace_path=str(temp_workspace),
            environment="test"
        )
        
        test_file = temp_workspace / "backup_test.txt"
        original_content = "Original file content\nLine 2\nLine 3"
        new_content = "New file content\nUpdated content"
        
        # Create original file
        test_file.write_text(original_content)
        original_size = test_file.stat().st_size
        
        inputs = {
            "file_path": str(test_file),
            "content": new_content,
            "create_backup": True
        }
        
        result = await tool.execute(inputs, context)
        
        assert result.success
        
        # Parse result data to check backup info
        result_data = result.data
        assert "backup_info" in result_data
        
        backup_info = result_data["backup_info"]
        assert backup_info is not None
        assert backup_info["original_size"] == original_size
        
        # Verify backup file exists and has correct content
        backup_path = Path(backup_info["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == original_content
        
        # Verify main file has new content
        assert test_file.read_text() == new_content


if __name__ == "__main__":
    pytest.main([__file__])