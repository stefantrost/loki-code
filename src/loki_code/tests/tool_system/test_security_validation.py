"""
Security and permissions validation tests.

Tests the security aspects of the tool system:
- Permission-based access control
- File size and type restrictions
- Security level validation
- Safety settings enforcement
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from ...tools.base import BaseTool
from ...tools.types import (
    ToolContext, ToolResult, SecurityLevel, 
    SafetySettings, ConfirmationLevel
)
from ...tools.exceptions import ToolSecurityError, ToolValidationError
from ...tools.file_reader import FileReaderTool
from ..fixtures.tool_test_fixtures import (
    test_config, temp_workspace, ToolTestHelpers
)


class TestToolSecurity:
    """Test tool security and permission system."""
    
    def test_file_size_restrictions(self, temp_workspace):
        """Test file size restrictions."""
        tool = FileReaderTool()
        
        # Create a file that exceeds size limit
        large_content = "x" * 2000  # 2KB content
        large_file = temp_workspace / "large.py"
        ToolTestHelpers.create_test_file(large_file, large_content)
        
        # Context with strict size limit
        restricted_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "safety_settings": {
                "max_file_size": 1000,  # 1KB limit
                "allowed_extensions": [".py", ".js", ".ts", ".rs"]
            }
        }
        
        result = tool.execute({
            "file_path": str(large_file)
        }, restricted_context)
        
        # Should fail due to size restriction
        assert not result.success
        assert "size" in result.error.lower()
    
    def test_file_extension_restrictions(self, temp_workspace):
        """Test file extension restrictions."""
        tool = FileReaderTool()
        
        # Create files with different extensions
        test_files = {
            "allowed.py": "print('allowed')",
            "forbidden.exe": "binary content",
            "config.json": '{"key": "value"}',
        }
        
        for filename, content in test_files.items():
            file_path = temp_workspace / filename
            ToolTestHelpers.create_test_file(file_path, content)
        
        # Context allowing only Python files
        restricted_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "safety_settings": {
                "max_file_size": 10240,
                "allowed_extensions": [".py"]  # Only Python files
            }
        }
        
        # Test allowed file
        result_allowed = tool.execute({
            "file_path": str(temp_workspace / "allowed.py")
        }, restricted_context)
        ToolTestHelpers.assert_tool_result_success(result_allowed)
        
        # Test forbidden executable
        result_exe = tool.execute({
            "file_path": str(temp_workspace / "forbidden.exe")
        }, restricted_context)
        assert not result_exe.success
        assert "extension" in result_exe.error.lower() or "not allowed" in result_exe.error.lower()
        
        # Test JSON file (not in allowed list)
        result_json = tool.execute({
            "file_path": str(temp_workspace / "config.json")
        }, restricted_context)
        assert not result_json.success
    
    def test_permission_requirements(self, temp_workspace):
        """Test permission-based access control."""
        tool = FileReaderTool()
        
        test_file = temp_workspace / "test.py"
        ToolTestHelpers.create_test_file(test_file, "print('test')")
        
        # Context without required permissions
        no_permission_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "permissions": []  # No permissions
        }
        
        result = tool.execute({
            "file_path": str(test_file)
        }, no_permission_context)
        
        # Should fail due to missing permissions
        assert not result.success
        assert "permission" in result.error.lower()
    
    def test_path_traversal_protection(self, temp_workspace):
        """Test protection against path traversal attacks."""
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Test various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\SAM",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            temp_workspace / ".." / ".." / "sensitive_file.txt"
        ]
        
        for malicious_path in malicious_paths:
            result = tool.execute({
                "file_path": str(malicious_path)
            }, context)
            
            # Should either fail or only access allowed files
            if result.success:
                # If it succeeds, ensure it's not accessing system files
                assert not any(sensitive in result.content.lower() 
                             for sensitive in ["root:", "administrator", "password"])
    
    def test_security_level_enforcement(self, temp_workspace):
        """Test security level enforcement."""
        # This would test different security levels if implemented
        # For now, test basic security validation
        
        tool = FileReaderTool()
        test_file = temp_workspace / "test.py"
        ToolTestHelpers.create_test_file(test_file, "print('test')")
        
        # High security context
        high_security_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "security_level": "high",
            "safety_settings": {
                "max_file_size": 1024,
                "allowed_extensions": [".py"],
                "require_confirmation": True
            }
        }
        
        result = tool.execute({
            "file_path": str(test_file)
        }, high_security_context)
        
        # Should work with proper security context
        ToolTestHelpers.assert_tool_result_success(result)
    
    def test_confirmation_requirements(self, temp_workspace):
        """Test confirmation requirements for sensitive operations."""
        # Mock a tool that requires confirmation
        class ConfirmationTool(BaseTool):
            def execute(self, inputs, context=None):
                # Check if confirmation is required and provided
                safety_settings = context.get("safety_settings", {})
                require_confirmation = safety_settings.get("require_confirmation", False)
                confirmation_provided = inputs.get("confirmed", False)
                
                if require_confirmation and not confirmation_provided:
                    return ToolResult(
                        success=False,
                        error="Confirmation required for this operation",
                        content=None
                    )
                
                return ToolResult(
                    success=True,
                    content="Operation completed",
                    metadata={"confirmed": confirmation_provided}
                )
        
        tool = ConfirmationTool()
        
        # Context requiring confirmation
        confirmation_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "safety_settings": {"require_confirmation": True}
        }
        
        # Test without confirmation
        result_no_confirm = tool.execute({
            "operation": "sensitive_action"
        }, confirmation_context)
        assert not result_no_confirm.success
        assert "confirmation" in result_no_confirm.error.lower()
        
        # Test with confirmation
        result_with_confirm = tool.execute({
            "operation": "sensitive_action",
            "confirmed": True
        }, confirmation_context)
        ToolTestHelpers.assert_tool_result_success(result_with_confirm)
    
    def test_input_sanitization(self, temp_workspace):
        """Test input sanitization and validation."""
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Test various potentially malicious inputs
        malicious_inputs = [
            {"file_path": ""},  # Empty path
            {"file_path": None},  # Null path
            {"file_path": 123},  # Wrong type
            {"file_path": "file.py\x00.txt"},  # Null byte injection
            {"file_path": "file.py; rm -rf /"},  # Command injection attempt
        ]
        
        for malicious_input in malicious_inputs:
            result = tool.execute(malicious_input, context)
            
            # Should fail with validation error, not crash
            assert not result.success
            assert result.error is not None
    
    def test_resource_limits(self, temp_workspace):
        """Test resource usage limits."""
        tool = FileReaderTool()
        
        # Create a file with many lines
        many_lines = "\n".join([f"line_{i}" for i in range(1000)])
        large_file = temp_workspace / "many_lines.py"
        ToolTestHelpers.create_test_file(large_file, many_lines)
        
        # Context with resource limits
        limited_context = {
            **ToolTestHelpers.create_mock_tool_context(),
            "safety_settings": {
                "max_file_size": len(many_lines.encode()) + 100,  # Just enough
                "max_processing_time": 5.0  # 5 seconds
            }
        }
        
        result = tool.execute({
            "file_path": str(large_file),
            "analysis_level": "comprehensive"
        }, limited_context)
        
        # Should complete within resource limits
        ToolTestHelpers.assert_tool_result_success(result)
    
    def test_concurrent_execution_safety(self, temp_workspace):
        """Test safety of concurrent tool execution."""
        import threading
        import time
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        test_file = temp_workspace / "concurrent.py"
        ToolTestHelpers.create_test_file(test_file, "print('concurrent test')")
        
        results = []
        errors = []
        
        def execute_tool():
            try:
                result = tool.execute({
                    "file_path": str(test_file)
                }, context)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=execute_tool)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All executions should succeed without interference
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        assert len(results) == 5
        
        for result in results:
            ToolTestHelpers.assert_tool_result_success(result)