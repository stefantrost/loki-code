"""
Comprehensive test and validation system for the Loki Code agent.

This module provides tests and validation utilities to ensure the agent
system works correctly with all its components.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any

from .loki_agent import LokiCodeAgent, AgentConfig, RequestContext, AgentState
from .permission_manager import PermissionManager, PermissionConfig, ToolAction, PermissionLevel
from .safety_manager import SafetyManager, SafetyConfig, TaskContext
from .conversation_manager import (
    ConversationManager, ConversationConfig, UserPreferences, InteractionType,
    ExplanationLevel, PersonalityStyle
)
from ...tools.types import ToolSecurityLevel
from ...utils.logging import get_logger


class AgentSystemValidator:
    """Validates the complete agent system functionality."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.temp_dir = None
        
    async def validate_complete_system(self) -> Dict[str, Any]:
        """Run comprehensive validation of the agent system."""
        
        print("🤖 Testing Loki Code Agent System")
        print("=" * 50)
        
        results = {}
        
        # Test each component
        results["permission_system"] = await self.test_permission_system()
        results["safety_system"] = await self.test_safety_system()
        results["conversation_system"] = await self.test_conversation_system()
        results["agent_core"] = await self.test_agent_core()
        results["integration"] = await self.test_system_integration()
        
        # Overall success
        results["overall_success"] = all(
            result.get("success", False) for result in results.values()
        )
        
        return results
    
    async def test_permission_system(self) -> Dict[str, Any]:
        """Test the permission-based autonomy system."""
        
        print("🔐 Testing Permission System...")
        
        try:
            config = PermissionConfig(
                auto_grant_safe_operations=True,
                remember_session_choices=True
            )
            
            permission_manager = PermissionManager(config)
            
            # Test 1: Auto-grant safe operations
            safe_action = ToolAction(
                tool_name="file_reader",
                description="Read a file safely",
                input_data={"file_path": "test.py"},
                file_paths=["test.py"],
                security_level=ToolSecurityLevel.SAFE
            )
            
            result1 = await permission_manager.request_permission(safe_action, "Testing safe operation")
            assert result1.granted, "Safe operation should be auto-granted"
            
            # Test 2: Session permission memory
            risky_action = ToolAction(
                tool_name="file_writer",
                description="Write to a file",
                input_data={"file_path": "test.py", "content": "test"},
                file_paths=["test.py"],
                security_level=ToolSecurityLevel.DANGEROUS,
                is_destructive=True
            )
            
            # Mock user response for testing
            async def mock_permission_response(prompt):
                return "2"  # Choose "similar actions"
            
            permission_manager._get_user_response = mock_permission_response
            
            result2 = await permission_manager.request_permission(risky_action, "Testing risky operation")
            
            # Test permission summary
            summary = permission_manager.get_permission_summary()
            
            return {
                "success": True,
                "auto_grant_test": result1.granted,
                "session_permissions": summary["session_permissions"],
                "permanent_permissions": summary["permanent_permissions"]
            }
            
        except Exception as e:
            self.logger.error(f"Permission system test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_safety_system(self) -> Dict[str, Any]:
        """Test the safety-first validation system."""
        
        print("🛡️ Testing Safety System...")
        
        try:
            config = SafetyConfig(
                immutable_rules_enabled=True,
                project_boundary_enforcement=True
            )
            
            safety_manager = SafetyManager(config)
            
            # Test 1: Safe operation approval
            safe_action = ToolAction(
                tool_name="file_reader",
                description="Read project file",
                input_data={"file_path": "./project_file.py"},
                file_paths=["./project_file.py"],
                security_level=ToolSecurityLevel.SAFE
            )
            
            context = TaskContext(
                project_path="./",
                current_file="project_file.py"
            )
            
            result1 = safety_manager.validate_action(safe_action, context)
            
            # Test 2: Path traversal detection
            dangerous_action = ToolAction(
                tool_name="file_reader",
                description="Read system file",
                input_data={"file_path": "../../../etc/passwd"},
                file_paths=["../../../etc/passwd"],
                security_level=ToolSecurityLevel.DANGEROUS
            )
            
            result2 = safety_manager.validate_action(dangerous_action, context)
            
            # Test 3: Error recovery
            test_error = FileNotFoundError("test.py not found")
            recovery_plan = await safety_manager.handle_error(test_error, context)
            
            return {
                "success": True,
                "safe_action_approved": result1.approved,
                "dangerous_action_blocked": not result2.approved,
                "path_traversal_detected": len(result2.violations) > 0,
                "error_recovery_strategy": recovery_plan.strategy.value,
                "safety_rules_count": len(safety_manager.immutable_rules)
            }
            
        except Exception as e:
            self.logger.error(f"Safety system test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_conversation_system(self) -> Dict[str, Any]:
        """Test the intelligent conversation management."""
        
        print("💬 Testing Conversation System...")
        
        try:
            config = ConversationConfig(
                adapt_to_user_style=True,
                learning_enabled=True
            )
            
            preferences = UserPreferences(
                explanation_level=ExplanationLevel.DETAILED,
                personality_style=PersonalityStyle.HELPFUL
            )
            
            conversation_manager = ConversationManager(config, preferences)
            
            # Test 1: Basic interaction
            response1 = await conversation_manager.interact_with_user(
                "I'll help you analyze this code",
                InteractionType.GENERAL
            )
            
            # Test 2: Clarification handling
            from .conversation_manager import InteractionContext
            
            clarification_context = InteractionContext(
                user_intent="Analyze code",
                confidence_level=0.5,
                ambiguous_aspects=["file_target", "analysis_scope"]
            )
            
            # Mock user input for testing
            async def mock_user_response(prompt):
                return "Focus on the main.py file for security analysis"
            
            conversation_manager._get_user_response = mock_user_response
            
            clarification = await conversation_manager.ask_clarification(clarification_context)
            
            # Test conversation summary
            summary = conversation_manager.get_conversation_summary()
            
            return {
                "success": True,
                "basic_interaction": len(response1) > 0,
                "clarification_received": clarification is not None,
                "conversation_entries": summary["total_entries"],
                "user_preferences_applied": True
            }
            
        except Exception as e:
            self.logger.error(f"Conversation system test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_agent_core(self) -> Dict[str, Any]:
        """Test the core agent functionality."""
        
        print("🧠 Testing Agent Core...")
        
        try:
            config = AgentConfig(
                reasoning_strategy="intelligent_react",
                clarification_threshold=0.7,
                max_execution_steps=5
            )
            
            agent = LokiCodeAgent(config)
            
            # Test 1: Agent initialization
            status = agent.get_agent_status()
            
            # Test 2: Request understanding
            context = RequestContext(
                project_path="./",
                current_file="test.py"
            )
            
            understanding = await agent._analyze_request(
                "Help me analyze this Python file for potential issues",
                context
            )
            
            # Test 3: Execution plan creation
            plan = await agent._create_execution_plan(understanding, context)
            
            # Test 4: State management
            initial_state = agent.current_state
            agent.current_state = AgentState.THINKING
            thinking_state = agent.current_state
            
            return {
                "success": True,
                "agent_initialized": status["state"] == "idle",
                "tools_available": status["tools_available"] > 0,
                "understanding_confidence": understanding.confidence,
                "plan_steps": len(plan.steps),
                "state_management": initial_state != thinking_state,
                "langchain_available": status["langchain_available"]
            }
            
        except Exception as e:
            self.logger.error(f"Agent core test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_system_integration(self) -> Dict[str, Any]:
        """Test complete system integration."""
        
        print("🔗 Testing System Integration...")
        
        try:
            # Create a temporary test environment
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                
                # Create test files
                test_file = Path(temp_dir) / "test.py"
                test_file.write_text("def hello():\\n    print('Hello, World!')")
                
                # Configure agent
                config = AgentConfig(
                    reasoning_strategy="intelligent_react",
                    permission_mode="ask_permission",
                    safety_mode="strict"
                )
                
                agent = LokiCodeAgent(config)
                
                # Mock user interactions for automated testing
                async def mock_permission_response(prompt):
                    return "1"  # Always grant once
                
                async def mock_conversation_response(prompt):
                    return "Analyze the main function"
                
                agent.permission_manager._get_user_response = mock_permission_response
                agent.conversation_manager._get_user_response = mock_conversation_response
                
                # Test complete request processing
                context = RequestContext(
                    project_path=temp_dir,
                    current_file=str(test_file),
                    target_files=[str(test_file)]
                )
                
                response = await agent.process_request(
                    "Analyze this Python file and tell me what it does",
                    context
                )
                
                return {
                    "success": True,
                    "request_processed": response.state in [AgentState.COMPLETED, AgentState.ERROR_RECOVERY],
                    "response_content_length": len(response.content),
                    "tools_used": len(response.tools_used),
                    "actions_taken": len(response.actions_taken),
                    "safety_checks": response.safety_checks_passed,
                    "confidence": response.confidence
                }
                
        except Exception as e:
            self.logger.error(f"System integration test failed: {e}")
            return {"success": False, "error": str(e)}


async def run_agent_system_tests():
    """Run comprehensive tests of the agent system."""
    
    validator = AgentSystemValidator()
    results = await validator.validate_complete_system()
    
    # Print detailed results
    print("\\n📊 Test Results")
    print("=" * 30)
    
    for test_category, result in results.items():
        if test_category == "overall_success":
            continue
            
        status = "✅" if result.get("success", False) else "❌"
        print(f"{status} {test_category.replace('_', ' ').title()}")
        
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            print(f"   Error: {error}")
        else:
            # Print key metrics
            for key, value in result.items():
                if key != "success" and not key.endswith("_test"):
                    print(f"   {key}: {value}")
    
    print()
    overall_status = "✅" if results["overall_success"] else "❌"
    print(f"{overall_status} Overall Agent System: {'PASSED' if results['overall_success'] else 'FAILED'}")
    
    return results


def test_agent_usage_examples():
    """Test agent usage with practical examples."""
    
    print("\\n📝 Agent Usage Examples")
    print("-" * 40)
    
    # Example 1: Code Analysis
    print("Example 1: Code Analysis Request")
    print("User: 'Analyze this Python file for security issues'")
    print("Expected: Agent asks for clarification, requests permission, analyzes safely")
    print()
    
    # Example 2: File Modification
    print("Example 2: File Modification Request")
    print("User: 'Add error handling to this function'")
    print("Expected: Agent plans changes, asks permission, shows safety considerations")
    print()
    
    # Example 3: Complex Workflow
    print("Example 3: Complex Multi-step Workflow")
    print("User: 'Refactor this module and add tests'")
    print("Expected: Agent breaks down into steps, requests permissions progressively")
    print()


async def demonstrate_agent_capabilities():
    """Demonstrate key agent capabilities."""
    
    print("\\n🚀 Agent Capabilities Demo")
    print("-" * 35)
    
    config = AgentConfig(
        reasoning_strategy="intelligent_react",
        clarification_threshold=0.8,
        show_reasoning=True
    )
    
    agent = LokiCodeAgent(config)
    
    print(f"🤖 Agent Status: {agent.current_state.value}")
    print(f"🔧 Tools Available: {len(agent.tool_registry.list_tools())}")
    print(f"🛡️ Safety Rules: {len(agent.safety_manager.immutable_rules)}")
    print(f"🔐 Permission System: {agent.config.permission_mode}")
    print(f"💬 Personality: {agent.config.personality}")
    
    status = agent.get_agent_status()
    print(f"\\n📈 Agent Metrics:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run comprehensive tests
    results = asyncio.run(run_agent_system_tests())
    
    # Show usage examples
    test_agent_usage_examples()
    
    # Demonstrate capabilities
    asyncio.run(demonstrate_agent_capabilities())
    
    # Exit with appropriate code
    exit(0 if results["overall_success"] else 1)