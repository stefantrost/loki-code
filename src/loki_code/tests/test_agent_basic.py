"""
Basic agent integration tests for Loki Code.

Simplified tests to validate the core agent functionality works
without complex async setup or extensive mocking.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from ..core.agent.loki_agent import LokiCodeAgent, AgentConfig, RequestContext, AgentState
from ..core.agent.permission_manager import PermissionManager, PermissionConfig, ToolAction, PermissionResult
from .fixtures.agent_test_scenarios import create_test_agent_config, TestProjectManager, TestProjectFactory


class TestBasicAgentFunctionality:
    """Basic tests for agent functionality."""
    
    def test_agent_creation(self):
        """Test agent can be created with valid config."""
        config = create_test_agent_config()
        agent = LokiCodeAgent(config)
        
        assert agent is not None
        assert agent.config == config
        assert agent.current_state == AgentState.IDLE
    
    def test_agent_status(self):
        """Test agent provides status information."""
        config = create_test_agent_config()
        agent = LokiCodeAgent(config)
        
        status = agent.get_agent_status()
        
        assert isinstance(status, dict)
        assert "state" in status
        assert "tools_available" in status
        assert "config" in status
        assert status["state"] == "idle"
    
    @pytest.mark.asyncio
    async def test_agent_simple_workflow(self):
        """Test agent can process a simple request (mocked)."""
        config = create_test_agent_config()
        agent = LokiCodeAgent(config)
        
        # Mock the complex dependencies
        with patch.object(agent, '_analyze_request') as mock_analyze:
            mock_analyze.return_value = Mock(
                user_intent="test request",
                confidence=0.9,
                ambiguous_aspects=[],
                required_tools=["test_tool"],
                risk_assessment="low"
            )
            
            with patch.object(agent, '_create_execution_plan') as mock_plan:
                mock_plan.return_value = Mock(
                    steps=[{"action": "test", "tool": "test_tool"}],
                    required_permissions=[],
                    safety_considerations=[]
                )
                
                with patch.object(agent, '_execute_plan_safely') as mock_execute:
                    mock_execute.return_value = Mock(
                        content="Test completed successfully",
                        state=AgentState.COMPLETED,
                        actions_taken=["test action"],
                        tools_used=["test_tool"],
                        confidence=0.9
                    )
                    
                    # Test request processing
                    context = RequestContext(
                        project_path="./test",
                        session_id="test_session"
                    )
                    
                    response = await agent.process_request("Test request", context)
                    
                    assert response is not None
                    assert mock_analyze.called
                    assert mock_plan.called
                    assert mock_execute.called


class TestPermissionManagerBasic:
    """Basic tests for permission manager."""
    
    def test_permission_manager_creation(self):
        """Test permission manager can be created."""
        config = PermissionConfig()
        manager = PermissionManager(config)
        
        assert manager is not None
        assert manager.config == config
    
    def test_tool_action_creation(self):
        """Test tool action can be created."""
        from ..tools.types import ToolSecurityLevel
        
        action = ToolAction(
            tool_name="test_tool",
            description="Test action",
            input_data={"test": "data"},
            security_level=ToolSecurityLevel.SAFE
        )
        
        assert action.tool_name == "test_tool"
        assert action.description == "Test action"
        assert action.input_data == {"test": "data"}
        assert action.security_level == ToolSecurityLevel.SAFE
    
    def test_permission_result_creation(self):
        """Test permission result can be created."""
        result = PermissionResult(
            granted=True,
            reason="Test permission granted"
        )
        
        assert result.granted is True
        assert result.reason == "Test permission granted"


class TestProjectSetup:
    """Test project setup and management."""
    
    def test_project_manager_creation(self):
        """Test project manager can create test projects."""
        manager = TestProjectManager()
        
        # Test basic functionality
        assert manager is not None
        assert manager.created_projects == []
        
        # Cleanup
        manager.cleanup_all()
    
    def test_test_project_creation(self):
        """Test test project can be created."""
        project_factory = TestProjectFactory()
        project = project_factory.create_basic_python_project()
        
        assert project is not None
        assert len(project.files) > 0
        assert "auth.py" in project.files
        assert "config.py" in project.files
    
    def test_temp_project_creation(self):
        """Test temporary project creation on filesystem."""
        manager = TestProjectManager()
        project = TestProjectFactory.create_basic_python_project()
        
        try:
            project_path = manager.create_project(project)
            
            assert Path(project_path).exists()
            assert Path(project_path, "auth.py").exists()
            assert Path(project_path, "config.py").exists()
            
            # Test file content
            auth_content = Path(project_path, "auth.py").read_text()
            assert "def authenticate_user" in auth_content
            
        finally:
            manager.cleanup_all()


class TestAgentComponents:
    """Test individual agent components work."""
    
    def test_agent_config_validation(self):
        """Test agent config has reasonable defaults."""
        config = create_test_agent_config()
        
        # Test simplified config fields
        assert config.max_steps == 10
        assert config.timeout_seconds == 30
        assert config.model_name == "test_model"
        assert config.auto_approve_safe_actions == False
        assert config.debug_mode == True
    
    @pytest.mark.asyncio
    async def test_request_understanding_simulation(self):
        """Test request analyzer component works."""
        config = create_test_agent_config()
        agent = LokiCodeAgent(config)
        
        # Test that the agent has the expected components
        assert agent.request_analyzer is not None
        assert agent.execution_planner is not None
        assert agent.permission_manager is not None
        assert agent.safety_manager is not None
        assert agent.conversation_manager is not None
        
        # Test basic component configuration
        assert hasattr(agent.request_analyzer, 'config')
        assert hasattr(agent.execution_planner, 'config')


# Integration test to verify basic workflow without mocking everything
class TestMinimalIntegration:
    """Minimal integration tests."""
    
    @pytest.mark.asyncio
    async def test_agent_reset(self):
        """Test agent can be reset to initial state."""
        config = create_test_agent_config()
        agent = LokiCodeAgent(config)
        
        # Change state
        agent.current_state = AgentState.THINKING
        
        # Reset
        await agent.reset_session()
        
        assert agent.current_state == AgentState.IDLE
        # Verify the agent is back to initial state
        assert agent.config is not None
        assert agent.request_analyzer is not None


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])