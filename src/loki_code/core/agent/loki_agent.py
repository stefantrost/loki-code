"""
Main LangChain-based intelligent agent for Loki Code.

This module implements the core agent that combines intelligent reasoning,
permission-based autonomy, safety validation, and progressive user interaction
to create a true coding assistant.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from pathlib import Path

# LangChain imports
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool as LangChainTool
    from langchain.schema import AgentAction, AgentFinish
    from langchain.prompts import PromptTemplate as LangChainPromptTemplate
    from langchain.callbacks.base import BaseCallbackHandler
    langchain_available = True
except ImportError:
    # Fallback for development/testing without LangChain
    langchain_available = False
    AgentExecutor = object
    LangChainTool = object
    BaseCallbackHandler = object

from .permission_manager import PermissionManager, ToolAction, PermissionLevel
from .safety_manager import SafetyManager, SafetyResult, TaskContext
from .conversation_manager import ConversationManager, InteractionType, UserPreferences
from ..prompts import PromptBuilder, PromptContext, create_default_template_registry
from ..tool_registry import ToolRegistry
from ...utils.logging import get_logger


class AgentState(Enum):
    """Current state of the agent."""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_PERMISSION = "waiting_for_permission"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"


@dataclass
class RequestUnderstanding:
    """Agent's understanding of a user request."""
    user_intent: str
    confidence: float
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    ambiguous_aspects: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    risk_assessment: str = "low"
    suggested_approach: str = ""


@dataclass
class ExecutionPlan:
    """Plan for executing a user request."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration: float = 0.0
    required_permissions: List[str] = field(default_factory=list)
    safety_considerations: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from the agent."""
    content: str
    state: AgentState = AgentState.COMPLETED
    actions_taken: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    permissions_requested: int = 0
    safety_checks_passed: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Context for processing a request."""
    project_path: Optional[str] = None
    current_file: Optional[str] = None
    target_files: List[str] = field(default_factory=list)
    user_preferences: Optional[UserPreferences] = None
    session_id: str = ""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for the Loki Code agent."""
    # Core settings
    reasoning_strategy: str = "intelligent_react"
    clarification_threshold: float = 0.7
    max_planning_depth: int = 5
    max_execution_steps: int = 20
    
    # LLM settings  
    llm_provider: str = "ollama"
    model_name: str = "codellama:7b"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    # Safety and permissions
    permission_mode: str = "ask_permission"
    safety_mode: str = "strict"
    auto_grant_safe_operations: bool = True
    
    # Interaction settings
    explanation_level: str = "detailed"
    personality: str = "helpful"
    proactive_suggestions: bool = True
    show_reasoning: bool = True
    
    # Performance settings
    timeout_seconds: float = 300.0
    max_retries: int = 3
    enable_caching: bool = True


class LokiAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for the Loki agent."""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        self.logger = get_logger(__name__)
    
    def on_agent_action(self, action: "AgentAction", **kwargs) -> Any:
        """Called when agent takes an action."""
        self.logger.debug(f"Agent action: {action.tool} with input: {action.tool_input}")
    
    def on_agent_finish(self, finish: "AgentFinish", **kwargs) -> Any:
        """Called when agent finishes."""
        self.logger.debug(f"Agent finished: {finish.return_values}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> Any:
        """Called when tool execution starts."""
        tool_name = serialized.get("name", "unknown")
        self.logger.debug(f"Starting tool: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs) -> Any:
        """Called when tool execution ends."""
        self.logger.debug(f"Tool completed with output length: {len(output)}")
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        """Called when tool execution errors."""
        self.logger.error(f"Tool error: {error}")


class LokiCodeAgent:
    """
    Main LangChain-based intelligent agent for Loki Code.
    
    This agent combines:
    - Intelligent reasoning with LangChain
    - Permission-based autonomy
    - Safety-first validation
    - Progressive user interaction
    - Tool integration and orchestration
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize core components
        self.tool_registry = ToolRegistry()
        self.permission_manager = PermissionManager(self._create_permission_config())
        self.safety_manager = SafetyManager(self._create_safety_config())
        self.conversation_manager = ConversationManager(
            self._create_conversation_config(),
            self._create_user_preferences()
        )
        
        # Initialize prompt system
        self.template_registry = create_default_template_registry()
        self.prompt_builder = PromptBuilder(
            self.template_registry,
            self.tool_registry
        )
        
        # Agent state
        self.current_state = AgentState.IDLE
        self.current_plan: Optional[ExecutionPlan] = None
        self.current_context: Optional[RequestContext] = None
        
        # LangChain integration
        self.langchain_tools: List[LangChainTool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        
        # Initialize LangChain components
        if langchain_available:
            self._initialize_langchain_components()
        else:
            self.logger.warning("LangChain not available, using fallback implementation")
    
    async def process_request(self, user_message: str, context: RequestContext) -> AgentResponse:
        """
        Main entry point for processing user requests.
        
        This method implements the complete agent reasoning loop:
        1. Understand the request
        2. Check for clarification needs
        3. Plan the approach
        4. Execute with permission and safety checks
        """
        try:
            self.current_context = context
            self.current_state = AgentState.THINKING
            
            # Step 1: Understand the request
            self.logger.info(f"Processing request: {user_message[:100]}...")
            understanding = await self._analyze_request(user_message, context)
            
            # Step 2: Check if clarification is needed
            if understanding.confidence < self.config.clarification_threshold:
                self.current_state = AgentState.WAITING_FOR_CLARIFICATION
                clarification = await self.conversation_manager.ask_clarification(
                    self._create_interaction_context(understanding)
                )
                
                if clarification:
                    # Re-analyze with clarification
                    enhanced_message = f"{user_message}\\n\\nClarification: {clarification}"
                    understanding = await self._analyze_request(enhanced_message, context)
            
            # Step 3: Create execution plan
            self.current_state = AgentState.PLANNING
            plan = await self._create_execution_plan(understanding, context)
            self.current_plan = plan
            
            # Step 4: Execute the plan with safety and permission checks
            self.current_state = AgentState.EXECUTING
            response = await self._execute_plan_with_safeguards(plan, understanding, context)
            
            self.current_state = AgentState.COMPLETED
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            self.current_state = AgentState.ERROR_RECOVERY
            
            # Use safety manager for error recovery
            task_context = self._create_task_context(context)
            recovery_plan = await self.safety_manager.handle_error(e, task_context)
            
            return AgentResponse(
                content=recovery_plan.message,
                state=AgentState.ERROR_RECOVERY,
                confidence=0.5,
                metadata={"error": str(e), "recovery_strategy": recovery_plan.strategy.value}
            )
    
    async def _analyze_request(self, user_message: str, context: RequestContext) -> RequestUnderstanding:
        """Analyze and understand the user's request using LLM."""
        
        try:
            # Create prompt context for analysis
            prompt_context = PromptContext(
                user_message=user_message,
                project_path=context.project_path,
                current_file=context.current_file,
                target_files=context.target_files
            )
            
            # Use debugging template for analysis (it's good at understanding complex requests)
            analysis_prompt = await self.prompt_builder.build_prompt("debugging", prompt_context)
            
            # Add specific analysis instructions
            analysis_instructions = """
            Analyze this user request and extract:
            1. Primary intent and goals
            2. Required tools and operations
            3. Files or components involved
            4. Risk level (low/medium/high)
            5. Ambiguous aspects that need clarification
            6. Confidence in understanding (0.0-1.0)
            
            Respond in JSON format:
            {
                "intent": "description of what user wants",
                "confidence": 0.8,
                "entities": {"files": [], "operations": [], "goals": []},
                "ambiguous_aspects": ["aspect1", "aspect2"],
                "required_tools": ["tool1", "tool2"],
                "risk_assessment": "low",
                "suggested_approach": "how to proceed"
            }
            """
            
            full_prompt = f"{analysis_prompt.system_prompt}\\n\\n{analysis_instructions}\\n\\nUser request: {user_message}"
            
            # For now, create a simulated understanding (in real implementation, this would call LLM)
            understanding = self._simulate_request_analysis(user_message, context)
            
            return understanding
            
        except Exception as e:
            self.logger.error(f"Error analyzing request: {e}")
            # Fallback understanding
            return RequestUnderstanding(
                user_intent=user_message,
                confidence=0.5,
                ambiguous_aspects=["Unable to fully analyze request"],
                risk_assessment="medium"
            )
    
    def _simulate_request_analysis(self, user_message: str, context: RequestContext) -> RequestUnderstanding:
        """Simulate request analysis (placeholder for actual LLM call)."""
        
        # Simple heuristic analysis
        confidence = 0.8
        ambiguous_aspects = []
        required_tools = []
        risk_assessment = "low"
        
        message_lower = user_message.lower()
        
        # Detect file operations
        if any(word in message_lower for word in ["read", "analyze", "examine", "show"]):
            required_tools.append("file_reader")
        
        if any(word in message_lower for word in ["write", "create", "modify", "update"]):
            required_tools.append("file_writer")
            risk_assessment = "medium"
        
        if any(word in message_lower for word in ["delete", "remove", "rm"]):
            risk_assessment = "high"
            ambiguous_aspects.append("destructive_operation_confirmation")
        
        # Check for ambiguity
        if "this" in message_lower and not context.current_file:
            ambiguous_aspects.append("file_target")
            confidence -= 0.2
        
        if any(word in message_lower for word in ["best", "should", "recommend"]):
            ambiguous_aspects.append("approach_preference")
            confidence -= 0.1
        
        return RequestUnderstanding(
            user_intent=user_message,
            confidence=max(0.1, confidence),
            ambiguous_aspects=ambiguous_aspects,
            required_tools=required_tools,
            risk_assessment=risk_assessment,
            suggested_approach="Use available tools to analyze and assist with the request"
        )
    
    async def _create_execution_plan(self, understanding: RequestUnderstanding, 
                                   context: RequestContext) -> ExecutionPlan:
        """Create a detailed execution plan based on understanding."""
        
        steps = []
        required_permissions = []
        safety_considerations = []
        
        # Plan steps based on required tools
        for tool_name in understanding.required_tools:
            step = {
                "action": f"use_tool",
                "tool": tool_name,
                "description": f"Execute {tool_name} tool",
                "requires_permission": tool_name in ["file_writer", "command_executor"],
                "safety_critical": understanding.risk_assessment in ["medium", "high"]
            }
            steps.append(step)
            
            if step["requires_permission"]:
                required_permissions.append(tool_name)
        
        # Add safety considerations
        if understanding.risk_assessment != "low":
            safety_considerations.append(f"High risk operation: {understanding.risk_assessment}")
        
        if context.target_files:
            safety_considerations.append(f"Affects {len(context.target_files)} files")
        
        return ExecutionPlan(
            steps=steps,
            estimated_duration=len(steps) * 10.0,  # Rough estimate
            required_permissions=required_permissions,
            safety_considerations=safety_considerations,
            alternative_approaches=[
                "Use read-only analysis first",
                "Process files in smaller batches",
                "Create backup before modifications"
            ]
        )
    
    async def _execute_plan_with_safeguards(self, plan: ExecutionPlan, understanding: RequestUnderstanding,
                                          context: RequestContext) -> AgentResponse:
        """Execute the plan with comprehensive safety and permission checks."""
        
        actions_taken = []
        tools_used = []
        permissions_requested = 0
        safety_checks_passed = 0
        
        try:
            # Show plan to user if it's complex
            if len(plan.steps) > 3 or plan.safety_considerations:
                await self._show_execution_plan(plan)
            
            # Execute each step
            for i, step in enumerate(plan.steps):
                await self.conversation_manager.show_progress(
                    step["description"], len(plan.steps), i + 1
                )
                
                # Create tool action for this step
                tool_action = ToolAction(
                    tool_name=step["tool"],
                    description=step["description"],
                    input_data=self._create_tool_input(step, understanding, context),
                    file_paths=context.target_files,
                    is_destructive=understanding.risk_assessment == "high"
                )
                
                # Safety check
                task_context = self._create_task_context(context)
                safety_result = self.safety_manager.validate_action(tool_action, task_context)
                
                if not safety_result.approved:
                    violation_messages = [v.message for v in safety_result.violations]
                    return AgentResponse(
                        content=f"❌ Safety check failed: {'; '.join(violation_messages)}",
                        state=AgentState.ERROR_RECOVERY,
                        safety_checks_passed=safety_checks_passed
                    )
                
                safety_checks_passed += 1
                
                # Permission check
                if step.get("requires_permission", False):
                    self.current_state = AgentState.WAITING_FOR_PERMISSION
                    permission_result = await self.permission_manager.request_permission(
                        tool_action,
                        f"Step {i+1}: {step['description']}"
                    )
                    
                    permissions_requested += 1
                    
                    if not permission_result.granted:
                        return AgentResponse(
                            content=f"⛔ Permission denied: {permission_result.reason}",
                            state=AgentState.WAITING_FOR_PERMISSION,
                            permissions_requested=permissions_requested
                        )
                
                # Execute the tool
                try:
                    result = await self._execute_tool(tool_action)
                    actions_taken.append(f"Executed {step['tool']}: {result[:100]}...")
                    tools_used.append(step["tool"])
                    
                except Exception as e:
                    # Handle tool execution error
                    task_context = self._create_task_context(context)
                    recovery_plan = await self.safety_manager.handle_error(e, task_context)
                    
                    if recovery_plan.user_input_needed:
                        return AgentResponse(
                            content=recovery_plan.message,
                            state=AgentState.ERROR_RECOVERY,
                            actions_taken=actions_taken,
                            tools_used=tools_used
                        )
            
            # Plan completed successfully
            completion_message = await self._generate_completion_message(
                understanding, actions_taken, tools_used
            )
            
            return AgentResponse(
                content=completion_message,
                state=AgentState.COMPLETED,
                actions_taken=actions_taken,
                tools_used=tools_used,
                permissions_requested=permissions_requested,
                safety_checks_passed=safety_checks_passed,
                confidence=understanding.confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error executing plan: {e}")
            return AgentResponse(
                content=f"❌ Execution failed: {str(e)}",
                state=AgentState.ERROR_RECOVERY,
                actions_taken=actions_taken,
                tools_used=tools_used,
                confidence=0.3
            )
    
    async def _execute_tool(self, tool_action: ToolAction) -> str:
        """Execute a tool action safely."""
        
        # Get the tool from registry
        tool_schema = self.tool_registry.get_tool_schema(tool_action.tool_name)
        if not tool_schema:
            raise ValueError(f"Tool not found: {tool_action.tool_name}")
        
        # Execute the tool (placeholder implementation)
        if tool_action.tool_name == "file_reader":
            return self._simulate_file_read(tool_action.input_data)
        elif tool_action.tool_name == "file_writer":
            return self._simulate_file_write(tool_action.input_data)
        else:
            return f"Executed {tool_action.tool_name} successfully"
    
    def _simulate_file_read(self, input_data: Dict[str, Any]) -> str:
        """Simulate file reading operation."""
        file_path = input_data.get("file_path", "unknown")
        return f"Read file content from {file_path} (simulated)"
    
    def _simulate_file_write(self, input_data: Dict[str, Any]) -> str:
        """Simulate file writing operation."""
        file_path = input_data.get("file_path", "unknown")
        return f"Wrote content to {file_path} (simulated)"
    
    def _create_tool_input(self, step: Dict[str, Any], understanding: RequestUnderstanding,
                          context: RequestContext) -> Dict[str, Any]:
        """Create tool input based on step and context."""
        
        base_input = {}
        
        if step["tool"] == "file_reader":
            base_input = {
                "file_path": context.current_file or (context.target_files[0] if context.target_files else ""),
                "analysis_level": "standard"
            }
        elif step["tool"] == "file_writer":
            base_input = {
                "file_path": context.current_file or (context.target_files[0] if context.target_files else ""),
                "content": "# Generated content",
                "create_backup": True
            }
        
        return base_input
    
    async def _show_execution_plan(self, plan: ExecutionPlan):
        """Show the execution plan to the user."""
        
        plan_message = "📋 **Execution Plan**\\n\\n"
        
        for i, step in enumerate(plan.steps, 1):
            plan_message += f"{i}. {step['description']}\\n"
        
        if plan.safety_considerations:
            plan_message += f"\\n🛡️ **Safety Notes**: {'; '.join(plan.safety_considerations)}\\n"
        
        if plan.required_permissions:
            plan_message += f"\\n🔐 **Permissions Needed**: {', '.join(plan.required_permissions)}\\n"
        
        plan_message += f"\\n⏱️ **Estimated Duration**: {plan.estimated_duration:.1f} seconds"
        
        await self.conversation_manager.interact_with_user(
            plan_message, InteractionType.CONFIRMATION
        )
    
    async def _generate_completion_message(self, understanding: RequestUnderstanding,
                                         actions_taken: List[str], tools_used: List[str]) -> str:
        """Generate a completion message for the user."""
        
        message = f"✅ **Completed**: {understanding.user_intent}\\n\\n"
        
        if actions_taken:
            message += "**Actions Performed**:\\n"
            for action in actions_taken:
                message += f"• {action}\\n"
        
        if tools_used:
            message += f"\\n**Tools Used**: {', '.join(set(tools_used))}\\n"
        
        if self.config.proactive_suggestions:
            suggestions = self._generate_follow_up_suggestions(understanding, tools_used)
            if suggestions:
                message += f"\\n💡 **Suggestions**: {'; '.join(suggestions)}"
        
        return message
    
    def _generate_follow_up_suggestions(self, understanding: RequestUnderstanding,
                                      tools_used: List[str]) -> List[str]:
        """Generate proactive follow-up suggestions."""
        
        suggestions = []
        
        if "file_reader" in tools_used:
            suggestions.append("Would you like me to analyze any other files?")
        
        if understanding.risk_assessment == "high":
            suggestions.append("Consider creating a backup before making changes")
        
        if len(understanding.required_tools) > 1:
            suggestions.append("I can break this down into smaller steps if preferred")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _create_interaction_context(self, understanding: RequestUnderstanding):
        """Create interaction context for conversation manager."""
        from .conversation_manager import InteractionContext
        
        return InteractionContext(
            user_intent=understanding.user_intent,
            confidence_level=understanding.confidence,
            ambiguous_aspects=understanding.ambiguous_aspects,
            risk_level=understanding.risk_assessment
        )
    
    def _create_task_context(self, context: RequestContext) -> TaskContext:
        """Create task context for safety manager."""
        
        return TaskContext(
            project_path=context.project_path,
            current_file=context.current_file,
            target_files=context.target_files,
            operation_type="agent_execution"
        )
    
    def _initialize_langchain_components(self):
        """Initialize LangChain agent components."""
        
        if not langchain_available:
            return
        
        try:
            # Convert tools to LangChain format
            self.langchain_tools = self._convert_tools_to_langchain()
            
            # Create custom callback handler
            callback_handler = LokiAgentCallbackHandler(self.conversation_manager)
            
            # For now, create a basic setup (would be enhanced with actual LLM)
            self.logger.info(f"Initialized LangChain agent with {len(self.langchain_tools)} tools")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain components: {e}")
    
    def _convert_tools_to_langchain(self) -> List[LangChainTool]:
        """Convert our tools to LangChain format."""
        
        if not langchain_available:
            return []
        
        langchain_tools = []
        
        for tool_schema in self.tool_registry.list_tools():
            # Create a wrapper function for each tool
            tool_wrapper = self._create_tool_wrapper(tool_schema.name)
            
            langchain_tool = LangChainTool(
                name=tool_schema.name,
                description=tool_schema.description,
                func=tool_wrapper
            )
            
            langchain_tools.append(langchain_tool)
        
        return langchain_tools
    
    def _create_tool_wrapper(self, tool_name: str) -> Callable:
        """Create a wrapper function for LangChain tool integration."""
        
        async def tool_wrapper(input_str: str) -> str:
            try:
                # Parse input
                input_data = json.loads(input_str) if input_str.startswith('{') else {"query": input_str}
                
                # Create tool action
                tool_action = ToolAction(
                    tool_name=tool_name,
                    description=f"Execute {tool_name}",
                    input_data=input_data,
                    file_paths=list(input_data.get("file_paths", []))
                )
                
                # Check permissions
                permission_result = await self.permission_manager.request_permission(
                    tool_action,
                    f"LangChain tool execution: {tool_name}"
                )
                
                if not permission_result.granted:
                    return f"Permission denied: {permission_result.reason}"
                
                # Execute tool
                result = await self._execute_tool(tool_action)
                return result
                
            except Exception as e:
                self.logger.error(f"Tool wrapper error for {tool_name}: {e}")
                return f"Tool execution failed: {str(e)}"
        
        # Convert async to sync for LangChain compatibility
        def sync_wrapper(input_str: str) -> str:
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(tool_wrapper(input_str))
            except RuntimeError:
                # No event loop running
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(tool_wrapper(input_str))
                finally:
                    loop.close()
        
        return sync_wrapper
    
    def _create_permission_config(self):
        """Create permission manager configuration."""
        from .permission_manager import PermissionConfig
        
        return PermissionConfig(
            auto_grant_safe_operations=self.config.auto_grant_safe_operations,
            remember_session_choices=True,
            remember_permanent_choices=True
        )
    
    def _create_safety_config(self):
        """Create safety manager configuration."""
        from .safety_manager import SafetyConfig
        
        return SafetyConfig(
            immutable_rules_enabled=True,
            project_boundary_enforcement=True,
            resource_limit_enforcement=True,
            path_traversal_protection=True
        )
    
    def _create_conversation_config(self):
        """Create conversation manager configuration."""
        from .conversation_manager import ConversationConfig
        
        return ConversationConfig(
            max_history_entries=50,
            adapt_to_user_style=True,
            learning_enabled=True,
            use_markdown=True,
            use_emojis=self.config.personality in ["friendly", "helpful"]
        )
    
    def _create_user_preferences(self):
        """Create default user preferences."""
        from .conversation_manager import UserPreferences, ExplanationLevel, PersonalityStyle
        
        explanation_map = {
            "minimal": ExplanationLevel.MINIMAL,
            "standard": ExplanationLevel.STANDARD,
            "detailed": ExplanationLevel.DETAILED,
            "verbose": ExplanationLevel.VERBOSE
        }
        
        personality_map = {
            "professional": PersonalityStyle.PROFESSIONAL,
            "friendly": PersonalityStyle.FRIENDLY,
            "helpful": PersonalityStyle.HELPFUL,
            "concise": PersonalityStyle.CONCISE,
            "analytical": PersonalityStyle.ANALYTICAL
        }
        
        return UserPreferences(
            explanation_level=explanation_map.get(self.config.explanation_level, ExplanationLevel.DETAILED),
            personality_style=personality_map.get(self.config.personality, PersonalityStyle.HELPFUL),
            show_reasoning=self.config.show_reasoning,
            show_progress=True,
            ask_before_major_changes=True
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        
        return {
            "state": self.current_state.value,
            "langchain_available": langchain_available,
            "tools_available": len(self.tool_registry.list_tools()),
            "conversation_entries": len(self.conversation_manager.conversation_history),
            "permissions": self.permission_manager.get_permission_summary(),
            "safety": self.safety_manager.get_safety_summary(),
            "config": {
                "reasoning_strategy": self.config.reasoning_strategy,
                "clarification_threshold": self.config.clarification_threshold,
                "permission_mode": self.config.permission_mode,
                "safety_mode": self.config.safety_mode
            }
        }
    
    async def reset_session(self):
        """Reset agent session state."""
        
        self.current_state = AgentState.IDLE
        self.current_plan = None
        self.current_context = None
        
        # Clear session-based permissions
        self.permission_manager.clear_session_permissions()
        
        # Reset conversation if needed
        self.conversation_manager.conversation_history.clear()
        
        self.logger.info("Agent session reset")