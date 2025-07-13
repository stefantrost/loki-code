"""
Core agent orchestrator for Loki Code.

Simplified main agent class that coordinates between specialized components.
"""

import asyncio
from typing import Dict, Any, Optional

from .types import (
    AgentConfig, AgentResponse, RequestContext, AgentState,
    RequestUnderstanding, ExecutionPlan
)
from .request_analyzer import RequestAnalyzer
from .execution_planner import ExecutionPlanner
from .permission_manager import PermissionManager, ToolAction
from .safety_manager import SafetyManager, TaskContext
from .conversation_manager import ConversationManager
from ..tool_registry import ToolRegistry
from ...utils.logging import get_logger


class LokiCodeAgent:
    """
    Simplified core agent for Loki Code.
    
    Coordinates between specialized components rather than handling everything directly.
    This is much simpler than the original 807-line monolithic class.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        self.state = AgentState.IDLE
        
        # Initialize specialized components
        self.request_analyzer = RequestAnalyzer(config.__dict__)
        self.execution_planner = ExecutionPlanner(config.__dict__)
        
        # Create component configs from agent config
        from .permission_manager import PermissionConfig
        from .safety_manager import SafetyConfig
        from .conversation_manager import ConversationConfig
        
        permission_config = PermissionConfig(
            auto_grant_safe_operations=config.auto_approve_safe_actions
        )
        safety_config = SafetyConfig()
        conversation_config = ConversationConfig()
        
        self.permission_manager = PermissionManager(permission_config)
        self.safety_manager = SafetyManager(safety_config)
        self.conversation_manager = ConversationManager(conversation_config)
        self.tool_registry = ToolRegistry()
        
        self.logger.info("LokiCodeAgent initialized")
    
    async def process_request(self, user_message: str, context: RequestContext) -> AgentResponse:
        """
        Process a user request and return a response.
        
        This is the main entry point for agent interactions.
        """
        self.logger.info(f"Processing request: {user_message[:100]}...")
        self.state = AgentState.THINKING
        
        try:
            # Step 1: Analyze the request
            understanding = await self._analyze_request(user_message, context)
            
            # Step 2: Create execution plan
            self.state = AgentState.PLANNING
            plan = await self._create_execution_plan(understanding, context)
            
            # Step 3: Execute with safeguards
            self.state = AgentState.EXECUTING
            result = await self._execute_plan_safely(plan, understanding, context)
            
            # Step 4: Generate response
            self.state = AgentState.COMPLETED
            response = await self._generate_response(understanding, plan, result, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            self.state = AgentState.ERROR_RECOVERY
            
            return AgentResponse(
                content=f"I encountered an error while processing your request: {str(e)}",
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={"error": str(e)}
            )
        finally:
            self.state = AgentState.IDLE
    
    async def _analyze_request(self, user_message: str, context: RequestContext) -> RequestUnderstanding:
        """Analyze the user request to understand intent and requirements."""
        return await self.request_analyzer.analyze_request(user_message, context)
    
    async def _create_execution_plan(
        self, 
        understanding: RequestUnderstanding, 
        context: RequestContext
    ) -> ExecutionPlan:
        """Create a detailed execution plan."""
        return await self.execution_planner.create_execution_plan(understanding, context)
    
    async def _execute_plan_safely(
        self, 
        plan: ExecutionPlan, 
        understanding: RequestUnderstanding,
        context: RequestContext
    ) -> Dict[str, Any]:
        """Execute the plan with safety checks and permission management."""
        results = {}
        
        for i, step in enumerate(plan.steps):
            step_name = f"step_{i}_{step.get('type', 'unknown')}"
            self.logger.debug(f"Executing {step_name}: {step.get('description', 'No description')}")
            
            try:
                # Check permissions before execution
                if await self._check_step_permissions(step, understanding):
                    result = await self._execute_step(step, context)
                    results[step_name] = result
                else:
                    self.logger.warning(f"Permission denied for step: {step_name}")
                    results[step_name] = {"error": "Permission denied"}
                    
            except Exception as e:
                self.logger.error(f"Error executing {step_name}: {e}")
                results[step_name] = {"error": str(e)}
                
                # Decide whether to continue or abort
                if step.get("critical", False):
                    raise e
        
        return results
    
    async def _check_step_permissions(self, step: Dict[str, Any], understanding: RequestUnderstanding) -> bool:
        """Check if the step has required permissions."""
        permission_level = step.get("permission_level", "none")
        
        if permission_level == "none":
            return True
        
        # Create tool action for permission check
        tool_action = ToolAction(
            tool_name=step.get("tool", "unknown"),
            action_type=step.get("type", "unknown"),
            parameters=step.get("input", {}),
            risk_level=understanding.risk_assessment,
            description=step.get("description", "")
        )
        
        # Check with permission manager
        return await self.permission_manager.check_permission(tool_action)
    
    async def _execute_step(self, step: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Execute a single step of the plan."""
        step_type = step.get("type", "unknown")
        tool_name = step.get("tool", None)
        
        if not tool_name:
            return {"error": "No tool specified for step"}
        
        # Handle different step types
        if step_type == "file_operation":
            return await self._execute_file_operation(step, context)
        elif step_type == "content_generation":
            return await self._execute_content_generation(step, context)
        elif step_type == "analysis_summary":
            return await self._execute_analysis_summary(step, context)
        elif step_type == "generate_response":
            return await self._execute_response_generation(step, context)
        else:
            # Generic tool execution
            return await self._execute_generic_tool(step, context)
    
    async def _execute_file_operation(self, step: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Execute a file operation step."""
        tool_name = step.get("tool")
        tool_input = step.get("input", {})
        
        # Get tool from registry and execute
        # This would integrate with the actual tool system
        return {
            "tool": tool_name,
            "input": tool_input,
            "result": f"Simulated {tool_name} execution",
            "success": True
        }
    
    async def _execute_content_generation(self, step: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Execute content generation step."""
        return {
            "tool": "llm_generator",
            "result": "Generated content based on request",
            "success": True
        }
    
    async def _execute_analysis_summary(self, step: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Execute analysis summary step."""
        return {
            "tool": "analyzer",
            "result": "Analysis summary generated",
            "success": True
        }
    
    async def _execute_response_generation(self, step: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Execute response generation step."""
        return {
            "tool": "response_generator",
            "result": "Response generated based on execution results",
            "success": True
        }
    
    async def _execute_generic_tool(self, step: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Execute a generic tool step."""
        tool_name = step.get("tool")
        return {
            "tool": tool_name,
            "result": f"Generic execution of {tool_name}",
            "success": True
        }
    
    async def _generate_response(
        self,
        understanding: RequestUnderstanding,
        plan: ExecutionPlan,
        execution_results: Dict[str, Any],
        context: RequestContext
    ) -> AgentResponse:
        """Generate the final response based on execution results."""
        
        # Count successful operations
        successful_steps = sum(1 for result in execution_results.values() 
                             if isinstance(result, dict) and result.get("success", False))
        total_steps = len(execution_results)
        
        # Generate response content
        if successful_steps == total_steps:
            content = f"I've successfully completed your request for {understanding.user_intent}."
        elif successful_steps > 0:
            content = f"I've partially completed your request ({successful_steps}/{total_steps} steps successful)."
        else:
            content = "I encountered issues completing your request. Please check the details below."
        
        # Add execution details if in debug mode
        if self.config.debug_mode:
            content += f"\n\nExecution details:\n"
            for step_name, result in execution_results.items():
                status = "✓" if result.get("success", False) else "✗"
                content += f"{status} {step_name}: {result.get('result', 'No result')}\n"
        
        response = AgentResponse(
            content=content,
            state=AgentState.COMPLETED,
            actions_taken=[step.get("description", "") for step in plan.steps],
            tools_used=[step.get("tool", "") for step in plan.steps],
            confidence=understanding.confidence,
            metadata={
                "understanding": understanding.__dict__,
                "plan_steps": len(plan.steps),
                "successful_steps": successful_steps,
                "execution_results": execution_results
            }
        )
        
        return response
    
    async def reset_session(self):
        """Reset the agent session."""
        self.state = AgentState.IDLE
        # Reset any internal state that needs resetting
        # Note: ConversationManager doesn't have reset_session method in simplified version
        self.logger.info("Agent session reset")
    
    @property
    def current_state(self) -> AgentState:
        """Get the current agent state."""
        return self.state
    
    @current_state.setter
    def current_state(self, value: AgentState) -> None:
        """Set the current agent state."""
        self.state = value
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status information."""
        available_tools = self.tool_registry.list_tool_names()
        
        return {
            "state": self.state.value,
            "tools_available": available_tools,
            "config": {
                "max_steps": self.config.max_steps,
                "timeout_seconds": self.config.timeout_seconds,
                "model_name": self.config.model_name,
                "debug_mode": self.config.debug_mode
            },
            "components": {
                "request_analyzer": self.request_analyzer is not None,
                "execution_planner": self.execution_planner is not None,
                "permission_manager": self.permission_manager is not None,
                "safety_manager": self.safety_manager is not None,
                "conversation_manager": self.conversation_manager is not None,
                "tool_registry": self.tool_registry is not None
            }
        }