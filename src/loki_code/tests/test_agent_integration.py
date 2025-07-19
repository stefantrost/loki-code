"""
Comprehensive end-to-end agent integration tests for Loki Code.

Tests the complete flow from user request to intelligent action with
permission handling, safety validation, and error recovery.
"""

import asyncio
import pytest
import time
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from typing import List, Dict, Any

from ..core.agent import (
    LokiCodeAgent, AgentConfig, RequestContext, AgentResponse, AgentState,
    RequestUnderstanding, ExecutionPlan
)
from ..core.agent.permission_manager import PermissionLevel, PermissionResult
from ..core.agent.safety_manager import SafetyResult, SafetyViolation, RecoveryPlan
from ..core.agent.conversation_manager import ConversationManager
from ..core.tool_registry import ToolRegistry
from ..tools.types import ToolResult, SecurityLevel
from .fixtures.agent_test_scenarios import (
    AgentTestScenarios, TestProjectFactory, TestProjectManager,
    create_test_context, create_test_agent_config
)


class TestEndToEndAgentWorkflow:
    """Test complete agent workflow from request to response."""
    
    @pytest.fixture
    def project_manager(self):
        """Create test project manager."""
        manager = TestProjectManager()
        yield manager
        manager.cleanup_all()
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        registry.list_tools.return_value = []
        registry.get_tool_schema.return_value = None
        return registry
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return create_test_agent_config()
    
    async def test_complete_agent_workflow_simple_read(self, project_manager, agent_config):
        """Test: User request → Agent reasoning → Tool execution → Response"""
        
        # Setup: Create test project with known files
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        # Create agent
        agent = LokiCodeAgent(agent_config)
        
        # Mock tool execution to return successful result
        with patch.object(agent, '_execute_tool') as mock_execute:
            mock_execute.return_value = "File content: auth.py contains authentication logic"
            
            # Mock permission manager to auto-grant
            with patch.object(agent.permission_manager, 'request_permission') as mock_permission:
                mock_permission.return_value = PermissionResult(
                    granted=True,
                    level=PermissionLevel.ONCE,
                    reason="Auto-granted for test"
                )
                
                # Step 1: User request
                user_request = "Analyze the auth.py file and show me its structure"
                context = create_test_context(project_path, f"{project_path}/auth.py")
                
                # Step 2: Agent processing
                response = await agent.process_request(user_request, context)
                
                # Step 3: Validate agent behavior
                assert response.state == AgentState.COMPLETED
                assert len(response.actions_taken) > 0
                assert "file_reader" in response.tools_used
                assert response.confidence > 0.5
                
                # Step 4: Validate tool integration
                assert mock_execute.called
                assert "auth.py" in response.content
    
    async def test_agent_workflow_with_clarification(self, project_manager, agent_config):
        """Test agent asks for clarification on ambiguous requests."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock conversation manager to simulate clarification
        with patch.object(agent.conversation_manager, 'ask_clarification') as mock_clarify:
            mock_clarify.return_value = "I want to see the function definitions in the file"
            
            # Ambiguous request
            user_request = "Fix the authentication"
            context = create_test_context(project_path)
            
            response = await agent.process_request(user_request, context)
            
            # Should have asked for clarification
            assert mock_clarify.called
            clarify_args = mock_clarify.call_args[0][0]
            assert clarify_args.user_intent == user_request
            assert len(clarify_args.ambiguous_aspects) > 0
    
    async def test_agent_workflow_permission_flow(self, project_manager, agent_config):
        """Test agent properly handles permission requests."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock permission manager to require permission
        permission_call_count = 0
        async def mock_permission_request(action, description):
            nonlocal permission_call_count
            permission_call_count += 1
            return PermissionResult(
                granted=True,
                level=PermissionLevel.ONCE,
                reason="User granted permission for file modification"
            )
        
        with patch.object(agent.permission_manager, 'request_permission', side_effect=mock_permission_request):
            with patch.object(agent, '_execute_tool') as mock_execute:
                mock_execute.return_value = "File successfully modified"
                
                # Request that requires permission
                user_request = "Modify the config.py file to add better logging"
                context = create_test_context(project_path, f"{project_path}/config.py")
                
                response = await agent.process_request(user_request, context)
                
                # Should have requested permission
                assert permission_call_count > 0
                assert response.permissions_requested > 0
                assert response.state == AgentState.COMPLETED
    
    async def test_agent_workflow_safety_violation(self, project_manager, agent_config):
        """Test agent respects safety boundaries."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock safety manager to block dangerous operation
        with patch.object(agent.safety_manager, 'validate_action') as mock_safety:
            mock_safety.return_value = SafetyResult(
                approved=False,
                violations=[SafetyViolation(
                    rule="project_boundary",
                    message="Attempted to access file outside project",
                    severity="high"
                )],
                suggested_alternatives=["Use files within the project directory"]
            )
            
            # Request that violates safety boundaries
            user_request = "Read the /etc/passwd file"
            context = create_test_context(project_path)
            
            response = await agent.process_request(user_request, context)
            
            # Should be blocked by safety check
            assert response.state == AgentState.ERROR_RECOVERY
            assert not response.content.startswith("✅")
            assert "safety" in response.content.lower()
    
    async def test_agent_performance_reasonable_time(self, project_manager, agent_config):
        """Test agent responds within reasonable time limits."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock fast tool execution
        with patch.object(agent, '_execute_tool') as mock_execute:
            mock_execute.return_value = "Quick response"
            
            with patch.object(agent.permission_manager, 'request_permission') as mock_permission:
                mock_permission.return_value = PermissionResult(granted=True, level=PermissionLevel.ONCE)
                
                start_time = time.perf_counter()
                
                user_request = "What functions are in auth.py?"
                context = create_test_context(project_path, f"{project_path}/auth.py")
                
                response = await agent.process_request(user_request, context)
                
                response_time = time.perf_counter() - start_time
                
                # Should respond quickly (under 10 seconds for simple requests)
                assert response_time < 10.0
                assert response.state == AgentState.COMPLETED


class TestIntelligentReasoning:
    """Test agent's intelligent reasoning capabilities."""
    
    @pytest.fixture
    def agent_config(self):
        return create_test_agent_config()
    
    async def test_clarification_when_ambiguous(self, agent_config):
        """Test agent asks clarification for ambiguous requests."""
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock clarification flow
        with patch.object(agent.conversation_manager, 'ask_clarification') as mock_clarify:
            mock_clarify.return_value = "I want to add better error handling to login failures"
            
            # Ambiguous request
            user_request = "Fix the authentication"
            context = create_test_context("./test_project")
            
            response = await agent.process_request(user_request, context)
            
            # Should have asked for clarification
            assert mock_clarify.called
            clarify_context = mock_clarify.call_args[0][0]
            assert clarify_context.user_intent == user_request
            assert clarify_context.confidence_level < 0.7  # Below threshold
    
    async def test_high_confidence_no_clarification(self, agent_config):
        """Test agent proceeds without clarification when confident."""
        
        agent = LokiCodeAgent(agent_config)
        
        with patch.object(agent, '_execute_tool') as mock_execute:
            mock_execute.return_value = "Functions: authenticate_user, hash_password, create_session_token"
            
            with patch.object(agent.conversation_manager, 'ask_clarification') as mock_clarify:
                with patch.object(agent.permission_manager, 'request_permission') as mock_permission:
                    mock_permission.return_value = PermissionResult(granted=True, level=PermissionLevel.ONCE)
                    
                    # Clear, specific request
                    user_request = "Read the auth.py file and show me all function names"
                    context = create_test_context("./test_project", "auth.py")
                    
                    response = await agent.process_request(user_request, context)
                    
                    # Should NOT have asked for clarification
                    assert not mock_clarify.called
                    assert response.confidence > 0.7
                    assert response.state == AgentState.COMPLETED
    
    async def test_request_understanding_accuracy(self, agent_config):
        """Test agent correctly understands different types of requests."""
        
        agent = LokiCodeAgent(agent_config)
        
        test_cases = [
            {
                "request": "Read the auth.py file",
                "expected_tools": ["file_reader"],
                "expected_confidence": 0.9,
                "expected_risk": "low"
            },
            {
                "request": "Delete all temporary files",
                "expected_tools": ["file_manager"],
                "expected_confidence": 0.8,
                "expected_risk": "high"
            },
            {
                "request": "Analyze the code quality",
                "expected_tools": ["code_analyzer"],
                "expected_confidence": 0.7,
                "expected_risk": "low"
            }
        ]
        
        for test_case in test_cases:
            context = create_test_context("./test_project")
            understanding = await agent._analyze_request(test_case["request"], context)
            
            assert understanding.confidence >= test_case["expected_confidence"] - 0.2
            assert understanding.risk_assessment == test_case["expected_risk"]
            for expected_tool in test_case["expected_tools"]:
                assert expected_tool in understanding.required_tools


class TestSafetyBoundaries:
    """Test agent respects safety boundaries and cannot override them."""
    
    @pytest.fixture
    def agent_config(self):
        return create_test_agent_config()
    
    @pytest.fixture 
    def project_manager(self):
        manager = TestProjectManager()
        yield manager
        manager.cleanup_all()
    
    async def test_project_boundary_enforcement(self, project_manager, agent_config):
        """Test agent respects project boundaries."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Try to access file outside project
        user_request = "Read /etc/passwd"
        context = create_test_context(project_path)
        
        response = await agent.process_request(user_request, context)
        
        assert response.state == AgentState.ERROR_RECOVERY
        assert "outside project" in response.content.lower() or "safety" in response.content.lower()
    
    async def test_resource_limit_enforcement(self, project_manager, agent_config):
        """Test agent respects resource limits."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock safety manager to reject large file
        with patch.object(agent.safety_manager, 'validate_action') as mock_safety:
            mock_safety.return_value = SafetyResult(
                approved=False,
                violations=[SafetyViolation(
                    rule="resource_limit",
                    message="File too large for processing",
                    severity="medium"
                )],
                suggested_alternatives=["Process file in smaller chunks"]
            )
            
            user_request = "Analyze the 500MB log file"
            context = create_test_context(project_path)
            
            response = await agent.process_request(user_request, context)
            
            assert response.state == AgentState.ERROR_RECOVERY
            assert "too large" in response.content.lower() or "limit" in response.content.lower()
    
    async def test_immutable_safety_rules(self, project_manager, agent_config):
        """Test that core safety rules cannot be overridden."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        # Create agent with modified config that tries to disable safety
        unsafe_config = agent_config
        unsafe_config.safety_mode = "disabled"  # Attempt to disable safety
        
        agent = LokiCodeAgent(unsafe_config)
        
        # Safety should still be enforced
        user_request = "Delete /usr/bin/python"
        context = create_test_context(project_path)
        
        response = await agent.process_request(user_request, context)
        
        # Should still be blocked despite config
        assert response.state == AgentState.ERROR_RECOVERY
        assert not response.content.startswith("✅")


class TestErrorRecovery:
    """Test agent error recovery mechanisms."""
    
    @pytest.fixture
    def agent_config(self):
        return create_test_agent_config()
    
    @pytest.fixture
    def project_manager(self):
        manager = TestProjectManager()
        yield manager
        manager.cleanup_all()
    
    async def test_file_not_found_recovery(self, project_manager, agent_config):
        """Test intelligent recovery from file not found errors."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock tool execution to raise file not found
        with patch.object(agent, '_execute_tool') as mock_execute:
            mock_execute.side_effect = FileNotFoundError("user_auth.py not found")
            
            # Mock safety manager to provide recovery suggestions
            with patch.object(agent.safety_manager, 'handle_error') as mock_recovery:
                mock_recovery.return_value = RecoveryPlan(
                    strategy="suggest_alternatives",
                    message="File not found. Did you mean: auth.py, authentication.py?",
                    suggested_actions=["Try 'auth.py'", "List available files"],
                    user_input_needed=True
                )
                
                user_request = "Analyze user_auth.py"
                context = create_test_context(project_path)
                
                response = await agent.process_request(user_request, context)
                
                assert response.state == AgentState.ERROR_RECOVERY
                assert "not found" in response.content.lower()
                assert "auth.py" in response.content  # Should suggest alternative
    
    async def test_tool_failure_graceful_degradation(self, project_manager, agent_config):
        """Test graceful degradation when tools fail."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock tool execution to fail
        with patch.object(agent, '_execute_tool') as mock_execute:
            mock_execute.side_effect = Exception("Tool execution failed")
            
            with patch.object(agent.safety_manager, 'handle_error') as mock_recovery:
                mock_recovery.return_value = RecoveryPlan(
                    strategy="graceful_degradation",
                    message="Tool failed, but I can provide basic information about the request",
                    suggested_actions=["Try a different approach", "Check tool status"],
                    user_input_needed=False
                )
                
                user_request = "Analyze auth.py"
                context = create_test_context(project_path, f"{project_path}/auth.py")
                
                response = await agent.process_request(user_request, context)
                
                assert response.state == AgentState.ERROR_RECOVERY
                assert "failed" in response.content.lower()
                assert len(response.metadata.get("suggested_actions", [])) > 0
    
    async def test_llm_failure_recovery(self, project_manager, agent_config):
        """Test recovery when LLM calls fail."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock LLM failure in request analysis
        with patch.object(agent, '_analyze_request') as mock_analyze:
            mock_analyze.side_effect = Exception("LLM service unavailable")
            
            user_request = "Help me with coding"
            context = create_test_context(project_path)
            
            response = await agent.process_request(user_request, context)
            
            assert response.state == AgentState.ERROR_RECOVERY
            assert "unavailable" in response.content.lower() or "error" in response.content.lower()
            assert response.confidence < 0.5


class TestConcurrentAgentRequests:
    """Test agent handling of concurrent requests."""
    
    @pytest.fixture
    def agent_config(self):
        return create_test_agent_config()
    
    @pytest.fixture
    def project_manager(self):
        manager = TestProjectManager()
        yield manager
        manager.cleanup_all()
    
    async def test_concurrent_agent_requests(self, project_manager, agent_config):
        """Test multiple concurrent agent conversations."""
        
        project = TestProjectFactory.create_basic_python_project()
        project_path = project_manager.create_project(project)
        
        agent = LokiCodeAgent(agent_config)
        
        # Mock tool execution with different responses
        async def mock_execute_tool(action):
            await asyncio.sleep(0.1)  # Simulate some processing time
            return f"Processed {action.tool_name} for {action.input_data.get('file_path', 'unknown')}"
        
        with patch.object(agent, '_execute_tool', side_effect=mock_execute_tool):
            with patch.object(agent.permission_manager, 'request_permission') as mock_permission:
                mock_permission.return_value = PermissionResult(granted=True, level=PermissionLevel.ONCE)
                
                # Create multiple simultaneous requests
                async def make_request(file_name: str):
                    request = f"Analyze {file_name}"
                    context = create_test_context(project_path, f"{project_path}/{file_name}")
                    return await agent.process_request(request, context)
                
                tasks = [
                    make_request("auth.py"),
                    make_request("config.py"),
                    make_request("utils.py")
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All should succeed or gracefully handle concurrency
                assert len(results) == 3
                assert all(not isinstance(r, Exception) for r in results)
                assert all(r.state in [AgentState.COMPLETED, AgentState.ERROR_RECOVERY] for r in results)


class TestAgentMetricsAndLogging:
    """Test agent metrics collection and logging."""
    
    @pytest.fixture
    def agent_config(self):
        return create_test_agent_config()
    
    async def test_agent_status_reporting(self, agent_config):
        """Test agent provides accurate status information."""
        
        agent = LokiCodeAgent(agent_config)
        
        status = agent.get_agent_status()
        
        assert "state" in status
        assert "tools_available" in status
        assert "conversation_entries" in status
        assert "permissions" in status
        assert "safety" in status
        assert "config" in status
        
        assert status["state"] == "idle"
        assert isinstance(status["tools_available"], int)
    
    async def test_conversation_history_tracking(self, agent_config):
        """Test agent tracks conversation history properly."""
        
        agent = LokiCodeAgent(agent_config)
        
        initial_entries = len(agent.conversation_manager.conversation_history)
        
        with patch.object(agent, '_execute_tool') as mock_execute:
            mock_execute.return_value = "Test response"
            
            with patch.object(agent.permission_manager, 'request_permission') as mock_permission:
                mock_permission.return_value = PermissionResult(granted=True, level=PermissionLevel.ONCE)
                
                context = create_test_context("./test_project")
                await agent.process_request("Test request", context)
                
                # Should have added entries to conversation history
                final_entries = len(agent.conversation_manager.conversation_history)
                assert final_entries > initial_entries


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])