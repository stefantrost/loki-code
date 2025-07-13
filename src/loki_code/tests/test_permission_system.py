"""
Comprehensive permission system tests for Loki Code.

Tests all aspects of the permission management system including
different permission types, persistence, and safety boundaries.
"""

import asyncio
import pytest
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from ..core.agent.permission_manager import (
    PermissionManager, PermissionConfig, ToolAction, PermissionResult,
    PermissionLevel, PermissionScope
)
from ..core.agent.safety_manager import SafetyManager, TaskContext
from ..tools.types import SecurityLevel, ToolCapability, ToolSecurityLevel
from .fixtures.agent_test_scenarios import (
    create_test_agent_config, create_test_context, TestProjectManager
)


class TestPermissionTypes:
    """Test different permission response types and their behavior."""
    
    @pytest.fixture
    def permission_manager(self):
        """Create a test permission manager."""
        config = PermissionConfig(
            auto_grant_safe_operations=False,
            remember_session_choices=True,
            remember_permanent_choices=True,
            require_confirmation_for_destructive=True
        )
        return PermissionManager(config)
    
    @pytest.fixture
    def test_action(self):
        """Create a test tool action."""
        return ToolAction(
            tool_name="file_writer",
            description="Write content to a file",
            input_data={"file_path": "test.py", "content": "print('hello')"},
            file_paths=["test.py"],
            is_destructive=False,
            security_level=ToolSecurityLevel.SAFE
        )
    
    async def test_yes_once_permission(self, permission_manager, test_action):
        """Test 'yes once' permission - should ask again for similar action."""
        
        # Mock user input for "yes once"
        with patch.object(permission_manager, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                scope=PermissionScope.SPECIFIC,
                reason="User granted permission for single use"
            )
            
            # First request - should ask permission
            result1 = await permission_manager.request_permission(test_action, "First request")
            assert result1.granted
            assert result1.scope == PermissionScope.SPECIFIC
            assert mock_ask.call_count == 1
            
            # Second identical request - should ask again
            result2 = await permission_manager.request_permission(test_action, "Second request")
            assert result2.granted
            assert result2.level == PermissionLevel.ONCE
            assert mock_ask.call_count == 2  # Asked again
    
    async def test_yes_session_permission(self, permission_manager, test_action):
        """Test 'yes session' permission - should not ask again this session."""
        
        with patch.object(permission_manager, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                level=PermissionLevel.SESSION,
                reason="User granted permission for entire session"
            )
            
            # First request - should ask permission
            result1 = await permission_manager.request_permission(test_action, "First request")
            assert result1.granted
            assert result1.level == PermissionLevel.SESSION
            assert mock_ask.call_count == 1
            
            # Second identical request - should NOT ask again
            result2 = await permission_manager.request_permission(test_action, "Second request")
            assert result2.granted
            assert result2.level == PermissionLevel.SESSION
            assert mock_ask.call_count == 1  # Not asked again
    
    async def test_yes_always_permission(self, permission_manager, test_action):
        """Test 'yes always' permission - should remember permanently."""
        
        with patch.object(permission_manager, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                level=PermissionLevel.ALWAYS,
                reason="User granted permanent permission"
            )
            
            # First request - should ask permission
            result1 = await permission_manager.request_permission(test_action, "First request")
            assert result1.granted
            assert result1.level == PermissionLevel.ALWAYS
            assert mock_ask.call_count == 1
            
            # Create new permission manager (simulates restart)
            new_manager = PermissionManager(permission_manager.config)
            new_manager.permanent_permissions = permission_manager.permanent_permissions.copy()
            
            # Request with new manager - should NOT ask
            result2 = await new_manager.request_permission(test_action, "After restart")
            assert result2.granted
            assert result2.level == PermissionLevel.ALWAYS
            # mock_ask should still be 1 (not called by new manager)
    
    async def test_permission_denied(self, permission_manager, test_action):
        """Test permission denial and retry behavior."""
        
        with patch.object(permission_manager, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=False,
                level=PermissionLevel.DENIED,
                reason="User denied permission"
            )
            
            # Request permission - should be denied
            result = await permission_manager.request_permission(test_action, "Denied request")
            assert not result.granted
            assert result.level == PermissionLevel.DENIED
            assert "denied" in result.reason.lower()
            
            # Subsequent identical request - should still ask (user might change mind)
            result2 = await permission_manager.request_permission(test_action, "Retry request")
            assert not result2.granted
            assert mock_ask.call_count == 2  # Asked again


class TestSafetyOverrides:
    """Test that safety boundaries cannot be overridden by permissions."""
    
    @pytest.fixture
    def permission_manager(self):
        config = PermissionConfig(auto_grant_safe_operations=True)
        return PermissionManager(config)
    
    @pytest.fixture
    def safety_manager(self):
        from ..core.agent.safety_manager import SafetyConfig
        config = SafetyConfig(
            immutable_rules_enabled=True,
            project_boundary_enforcement=True,
            resource_limit_enforcement=True
        )
        return SafetyManager(config)
    
    async def test_dangerous_action_always_asks(self, permission_manager):
        """Test that dangerous actions always require permission, even with auto-grant."""
        
        # Create dangerous action
        dangerous_action = ToolAction(
            tool_name="file_manager",
            description="Delete all files",
            input_data={"pattern": "*.py", "action": "delete"},
            file_paths=["*.py"],
            is_destructive=True,
            security_level=ToolSecurityLevel.DANGEROUS
        )
        
        # Set permanent permission for this action
        action_key = permission_manager._get_action_key(dangerous_action)
        permission_manager.permanent_permissions[action_key] = PermissionLevel.ALWAYS
        
        with patch.object(permission_manager, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                scope=PermissionScope.SPECIFIC,
                reason="User confirmed dangerous operation"
            )
            
            # Should still ask despite permanent permission
            result = await permission_manager.request_permission(dangerous_action, "Dangerous op")
            assert mock_ask.called
            assert "dangerous" in result.reason.lower()
    
    async def test_system_file_access_blocked(self, permission_manager):
        """Test that system file access is blocked regardless of permissions."""
        
        system_action = ToolAction(
            tool_name="file_reader",
            description="Read system file",
            input_data={"file_path": "/etc/passwd"},
            file_paths=["/etc/passwd"],
            is_destructive=False,
            security_level=ToolSecurityLevel.SAFE
        )
        
        # Grant permanent permission
        action_key = permission_manager._get_action_key(system_action)
        permission_manager.permanent_permissions[action_key] = PermissionLevel.ALWAYS
        
        # Should still be blocked by safety check
        with patch.object(permission_manager, '_check_safety_boundaries') as mock_safety:
            mock_safety.return_value = False  # Safety check fails
            
            result = await permission_manager.request_permission(system_action, "System access")
            assert not result.granted
            assert "safety" in result.reason.lower()
    
    async def test_resource_limits_enforced(self, permission_manager):
        """Test that resource limits cannot be overridden by permissions."""
        
        large_file_action = ToolAction(
            tool_name="file_reader",
            description="Read large file",
            input_data={"file_path": "huge_file.log", "size_mb": 1000},
            file_paths=["huge_file.log"],
            is_destructive=False,
            security_level=ToolSecurityLevel.SAFE
        )
        
        # Mock resource limit check
        with patch.object(permission_manager, '_check_resource_limits') as mock_limits:
            mock_limits.return_value = False  # Exceeds limits
            
            result = await permission_manager.request_permission(large_file_action, "Large file")
            assert not result.granted
            assert "resource" in result.reason.lower() or "limit" in result.reason.lower()


class TestPermissionPersistence:
    """Test permission persistence across sessions and restarts."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file.close()
        yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)
    
    async def test_permanent_permissions_persist(self, temp_storage_path):
        """Test that permanent permissions are saved and loaded correctly."""
        
        # Create first manager
        config1 = PermissionConfig(
            permanent_storage_path=temp_storage_path,
            remember_permanent_choices=True
        )
        manager1 = PermissionManager(config1)
        
        test_action = ToolAction(
            tool_name="file_reader",
            description="Read file",
            input_data={"file_path": "test.py"},
            file_paths=["test.py"]
        )
        
        # Grant permanent permission
        with patch.object(manager1, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                level=PermissionLevel.ALWAYS,
                reason="Permanent permission granted"
            )
            
            result1 = await manager1.request_permission(test_action, "First request")
            assert result1.granted
            assert result1.level == PermissionLevel.ALWAYS
        
        # Save permissions
        await manager1.save_permanent_permissions()
        
        # Create second manager (simulates restart)
        config2 = PermissionConfig(
            permanent_storage_path=temp_storage_path,
            remember_permanent_choices=True
        )
        manager2 = PermissionManager(config2)
        await manager2.load_permanent_permissions()
        
        # Should not ask permission (already granted permanently)
        with patch.object(manager2, '_ask_user_permission') as mock_ask2:
            result2 = await manager2.request_permission(test_action, "After restart")
            assert result2.granted
            assert result2.level == PermissionLevel.ALWAYS
            assert not mock_ask2.called  # Should not ask again
    
    async def test_session_permissions_dont_persist(self, temp_storage_path):
        """Test that session permissions are cleared on restart."""
        
        config = PermissionConfig(
            permanent_storage_path=temp_storage_path,
            remember_session_choices=True,
            remember_permanent_choices=True
        )
        manager1 = PermissionManager(config)
        
        test_action = ToolAction(
            tool_name="file_writer",
            description="Write file",
            input_data={"file_path": "test.py", "content": "test"},
            file_paths=["test.py"]
        )
        
        # Grant session permission
        with patch.object(manager1, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                level=PermissionLevel.SESSION,
                reason="Session permission granted"
            )
            
            result1 = await manager1.request_permission(test_action, "Session request")
            assert result1.granted
            assert result1.level == PermissionLevel.SESSION
        
        # Create new manager (simulates restart)
        manager2 = PermissionManager(config)
        
        # Should ask permission again (session permissions don't persist)
        with patch.object(manager2, '_ask_user_permission') as mock_ask2:
            mock_ask2.return_value = PermissionResult(
                granted=True,
                scope=PermissionScope.SPECIFIC,
                reason="Permission granted after restart"
            )
            
            result2 = await manager2.request_permission(test_action, "After restart")
            assert result2.granted
            assert mock_ask2.called  # Should ask again


class TestPermissionIntegration:
    """Test permission system integration with other components."""
    
    @pytest.fixture
    def project_manager(self):
        """Create test project manager."""
        manager = TestProjectManager()
        yield manager
        manager.cleanup_all()
    
    async def test_permission_with_safety_integration(self, project_manager):
        """Test permission system working with safety manager."""
        
        from .fixtures.agent_test_scenarios import TestProjectFactory
        
        # Create test project
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        # Create managers
        permission_config = PermissionConfig(auto_grant_safe_operations=False)
        permission_manager = PermissionManager(permission_config)
        
        from ..core.agent.safety_manager import SafetyConfig
        safety_config = SafetyConfig(project_boundary_enforcement=True)
        safety_manager = SafetyManager(safety_config)
        
        # Test action within project boundaries
        safe_action = ToolAction(
            tool_name="file_reader",
            description="Read project file",
            input_data={"file_path": f"{project_path}/auth.py"},
            file_paths=[f"{project_path}/auth.py"]
        )
        
        task_context = TaskContext(
            project_path=project_path,
            current_file=None,
            target_files=[f"{project_path}/auth.py"],
            operation_type="file_read"
        )
        
        # Safety check should pass
        safety_result = safety_manager.validate_action(safe_action, task_context)
        assert safety_result.approved
        
        # Permission should be askable
        with patch.object(permission_manager, '_ask_user_permission') as mock_ask:
            mock_ask.return_value = PermissionResult(
                granted=True,
                scope=PermissionScope.SPECIFIC,
                reason="User granted permission"
            )
            
            permission_result = await permission_manager.request_permission(
                safe_action, "Read project file"
            )
            assert permission_result.granted
    
    async def test_permission_denied_for_unsafe_action(self, project_manager):
        """Test permission system blocks unsafe actions."""
        
        from .fixtures.agent_test_scenarios import TestProjectFactory
        
        # Create test project
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        permission_config = PermissionConfig(auto_grant_safe_operations=False)
        permission_manager = PermissionManager(permission_config)
        
        # Test action outside project boundaries
        unsafe_action = ToolAction(
            tool_name="file_reader",
            description="Read system file",
            input_data={"file_path": "/etc/passwd"},
            file_paths=["/etc/passwd"]
        )
        
        # Should be blocked by safety check
        with patch.object(permission_manager, '_check_safety_boundaries') as mock_safety:
            mock_safety.return_value = False
            
            result = await permission_manager.request_permission(unsafe_action, "Unsafe action")
            assert not result.granted
            assert "safety" in result.reason.lower()


class TestPermissionUI:
    """Test permission user interface and interaction."""
    
    @pytest.fixture
    def permission_manager(self):
        config = PermissionConfig()
        return PermissionManager(config)
    
    async def test_permission_request_formatting(self, permission_manager):
        """Test permission request is formatted clearly for user."""
        
        action = ToolAction(
            tool_name="file_writer",
            description="Create new Python module",
            input_data={
                "file_path": "new_module.py",
                "content": "# New module\nprint('Hello')"
            },
            file_paths=["new_module.py"],
            is_destructive=False
        )
        
        with patch.object(permission_manager, '_display_permission_request') as mock_display:
            with patch.object(permission_manager, '_get_user_choice') as mock_choice:
                mock_choice.return_value = "once"
                
                await permission_manager.request_permission(action, "Create module")
                
                # Verify permission request was formatted properly
                mock_display.assert_called_once()
                call_args = mock_display.call_args[0]
                request_text = call_args[0]
                
                assert "file_writer" in request_text
                assert "new_module.py" in request_text
                assert "Create new Python module" in request_text
    
    async def test_permission_choices_presented(self, permission_manager):
        """Test all permission choices are presented to user."""
        
        action = ToolAction(
            tool_name="test_tool",
            description="Test action",
            input_data={},
            file_paths=[]
        )
        
        with patch.object(permission_manager, '_get_user_choice') as mock_choice:
            mock_choice.return_value = "session"
            
            with patch('builtins.print') as mock_print:
                await permission_manager.request_permission(action, "Test")
                
                # Verify choice options were presented
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                choice_text = " ".join(print_calls)
                
                assert "once" in choice_text.lower()
                assert "session" in choice_text.lower()
                assert "always" in choice_text.lower()
                assert "deny" in choice_text.lower() or "no" in choice_text.lower()


class TestPermissionPerformance:
    """Test permission system performance and efficiency."""
    
    async def test_permission_lookup_performance(self):
        """Test permission lookup is fast even with many stored permissions."""
        
        config = PermissionConfig(remember_permanent_choices=True)
        manager = PermissionManager(config)
        
        # Add many permanent permissions
        for i in range(1000):
            action_key = f"tool_{i}:action_{i}"
            manager.permanent_permissions[action_key] = PermissionLevel.ALWAYS
        
        # Test lookup performance
        import time
        
        test_action = ToolAction(
            tool_name="tool_500",
            description="Test action 500",
            input_data={"action_id": 500},
            file_paths=[]
        )
        
        start_time = time.perf_counter()
        result = await manager.request_permission(test_action, "Performance test")
        lookup_time = time.perf_counter() - start_time
        
        assert result.granted  # Should find permission
        assert lookup_time < 0.1  # Should be fast
    
    async def test_concurrent_permission_requests(self):
        """Test permission system handles concurrent requests properly."""
        
        config = PermissionConfig()
        manager = PermissionManager(config)
        
        async def request_permission(action_id: int):
            action = ToolAction(
                tool_name=f"tool_{action_id}",
                description=f"Action {action_id}",
                input_data={"id": action_id},
                file_paths=[]
            )
            
            with patch.object(manager, '_ask_user_permission') as mock_ask:
                mock_ask.return_value = PermissionResult(
                    granted=True,
                    scope=PermissionScope.SPECIFIC,
                    reason=f"Permission for action {action_id}"
                )
                
                return await manager.request_permission(action, f"Concurrent test {action_id}")
        
        # Run multiple concurrent permission requests
        tasks = [request_permission(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result.granted for result in results)
        assert len(set(result.reason for result in results)) == 10  # All unique


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])