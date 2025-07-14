"""
LangChain-based agent implementation for Loki Code.

This module implements a proper LangChain ReAct agent that integrates with
the existing Loki Code tool system, permission management, and safety systems.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForChainRun

from .types import (
    AgentConfig, AgentResponse, RequestContext, AgentState,
    RequestUnderstanding, ExecutionPlan
)
from .permission_manager import PermissionManager, PermissionConfig
from .safety_manager import SafetyManager, SafetyConfig
from .conversation_manager import ConversationManager, ConversationConfig
from .memory_manager import LangChainMemoryManager, MemoryStrategy
from ..tool_system.tool_registry_core import ToolRegistry
from ...tools.langchain_adapters import create_langchain_tools
from ...utils.logging import get_logger


class LokiLangChainAgent:
    """
    LangChain-based agent for Loki Code using ReAct pattern.
    
    This agent integrates LangChain's ReAct implementation with Loki Code's
    existing tool system, permission management, and safety checks.
    """
    
    # ReAct prompt template for the agent
    REACT_PROMPT = PromptTemplate.from_template("""
You are Loki Code, a helpful coding assistant with access to powerful tools.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Always think before acting
- Use tools to gather information when needed
- Be helpful and accurate
- Follow safety and permission protocols
- Explain your reasoning clearly

Question: {input}
{agent_scratchpad}
""")
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 config: Optional[AgentConfig] = None,
                 memory_strategy: MemoryStrategy = MemoryStrategy.SUMMARY_BUFFER):
        """
        Initialize the LangChain agent.
        
        Args:
            llm: Language model to use for the agent
            config: Agent configuration
            memory_strategy: Memory management strategy
        """
        self.config = config or AgentConfig()
        self.llm = llm
        self.logger = get_logger(__name__)
        self.state = AgentState.IDLE
        
        # Initialize managers
        self.permission_manager = PermissionManager(PermissionConfig(
            auto_grant_safe_operations=self.config.auto_approve_safe_actions
        ))
        self.safety_manager = SafetyManager(SafetyConfig())
        self.conversation_manager = ConversationManager(ConversationConfig())
        
        # Initialize LangChain memory manager
        self.memory_manager = LangChainMemoryManager(
            llm=self.llm,
            strategy=memory_strategy,
            config=ConversationConfig()
        )
        
        # Create LangChain tools from Loki tools
        self.tools = create_langchain_tools(
            permission_manager=self.permission_manager,
            safety_manager=self.safety_manager
        )
        
        # Create the ReAct agent
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.REACT_PROMPT
        )
        
        # Create agent executor with memory
        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            memory=self.memory_manager.memory,
            verbose=True,
            max_iterations=self.config.max_steps,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        self.logger.info("LokiLangChainAgent initialized with ReAct pattern")
    
    async def process_request(self, 
                            user_message: str, 
                            context: Optional[RequestContext] = None) -> AgentResponse:
        """
        Process a user request using the LangChain ReAct agent.
        
        Args:
            user_message: The user's message/request
            context: Optional context for the request
            
        Returns:
            AgentResponse with the agent's response and metadata
        """
        self.logger.info(f"Processing request: {user_message[:100]}...")
        self.state = AgentState.THINKING
        
        try:
            # Prepare context
            context = context or RequestContext()
            
            # Start memory session if needed
            if context.session_id and not self.memory_manager.session_id:
                self.memory_manager.start_session(context.session_id)
            
            # Analyze request understanding
            understanding = await self._analyze_request(user_message, context)
            
            # Execute using LangChain agent
            self.state = AgentState.EXECUTING
            result = await self._execute_with_langchain(user_message, context)
            
            # Add conversation to memory
            ai_response = result.get("output", "")
            if ai_response:
                self.memory_manager.add_message(user_message, ai_response)
            
            # Process response
            self.state = AgentState.COMPLETED
            response = self._create_response(understanding, result, context)
            
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
    
    async def _analyze_request(self, 
                             user_message: str, 
                             context: RequestContext) -> RequestUnderstanding:
        """Analyze the user request to understand intent."""
        # Simple analysis for now - can be enhanced with more sophisticated NLP
        understanding = RequestUnderstanding(
            user_intent=user_message,
            confidence=0.8,  # Default confidence
            extracted_entities={},
            ambiguous_aspects=[],
            required_tools=[],
            risk_assessment="low",
            suggested_approach="Use LangChain ReAct pattern"
        )
        
        # Basic tool requirement detection
        if "read" in user_message.lower() or "file" in user_message.lower():
            understanding.required_tools.append("file_reader")
        
        return understanding
    
    async def _execute_with_langchain(self, 
                                    user_message: str, 
                                    context: RequestContext) -> Dict[str, Any]:
        """Execute the request using LangChain agent."""
        try:
            # Run the agent executor
            # Note: LangChain's AgentExecutor doesn't have async methods by default
            # so we run it in a thread pool
            loop = asyncio.get_event_loop()
            # Get conversation context from memory manager
            conversation_context = self.memory_manager.get_conversation_context()
            
            result = await loop.run_in_executor(
                None,
                lambda: self.agent_executor.invoke({
                    "input": user_message,
                    "chat_history": conversation_context
                })
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"LangChain execution error: {e}", exc_info=True)
            raise
    
    def _format_chat_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return ""
        
        formatted = []
        for item in history[-5:]:  # Last 5 messages for context
            if item.get("role") == "user":
                formatted.append(f"Human: {item.get('content', '')}")
            elif item.get("role") == "assistant":
                formatted.append(f"Assistant: {item.get('content', '')}")
        
        return "\n".join(formatted)
    
    def _create_response(self, 
                        understanding: RequestUnderstanding,
                        langchain_result: Dict[str, Any],
                        context: RequestContext) -> AgentResponse:
        """Create an AgentResponse from LangChain results."""
        
        # Extract information from LangChain result
        output = langchain_result.get("output", "")
        intermediate_steps = langchain_result.get("intermediate_steps", [])
        
        # Extract tools used and actions taken
        tools_used = []
        actions_taken = []
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action, observation = step[0], step[1]
                if hasattr(action, 'tool'):
                    tools_used.append(action.tool)
                    actions_taken.append(f"Used {action.tool}: {action.tool_input}")
        
        # Add memory statistics to metadata
        memory_stats = self.memory_manager.get_memory_stats()
        
        return AgentResponse(
            content=output,
            state=AgentState.COMPLETED,
            actions_taken=actions_taken,
            tools_used=list(set(tools_used)),  # Remove duplicates
            permissions_requested=0,  # TODO: Track from permission manager
            safety_checks_passed=len(intermediate_steps),  # Rough estimate
            confidence=understanding.confidence,
            metadata={
                "langchain_result": langchain_result,
                "understanding": understanding.__dict__,
                "intermediate_steps_count": len(intermediate_steps),
                "memory_stats": memory_stats
            }
        )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self.memory_manager.get_memory_stats()
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory_manager.clear_memory()
    
    def get_conversation_summary(self) -> Optional[str]:
        """Get conversation summary if available."""
        return self.memory_manager.get_memory_summary()


class LokiLangChainAgentFactory:
    """Factory for creating LangChain agents with different LLM providers."""
    
    @staticmethod
    def create_with_ollama(model_name: str = "llama3.1", 
                          config: Optional[AgentConfig] = None,
                          memory_strategy: MemoryStrategy = MemoryStrategy.SUMMARY_BUFFER) -> LokiLangChainAgent:
        """Create agent with Ollama LLM."""
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(model=model_name)
            return LokiLangChainAgent(llm=llm, config=config, memory_strategy=memory_strategy)
        except ImportError:
            raise ImportError("langchain-community is required for Ollama integration")
    
    @staticmethod
    def create_with_openai(api_key: str,
                          model_name: str = "gpt-3.5-turbo",
                          config: Optional[AgentConfig] = None,
                          memory_strategy: MemoryStrategy = MemoryStrategy.SUMMARY_BUFFER) -> LokiLangChainAgent:
        """Create agent with OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(api_key=api_key, model=model_name)
            return LokiLangChainAgent(llm=llm, config=config, memory_strategy=memory_strategy)
        except ImportError:
            raise ImportError("langchain-openai is required for OpenAI integration")
    
    @staticmethod
    def create_with_custom_llm(llm: BaseLanguageModel,
                              config: Optional[AgentConfig] = None,
                              memory_strategy: MemoryStrategy = MemoryStrategy.SUMMARY_BUFFER) -> LokiLangChainAgent:
        """Create agent with custom LLM."""
        return LokiLangChainAgent(llm=llm, config=config, memory_strategy=memory_strategy)


# Backward compatibility alias
LokiCodeAgent = LokiLangChainAgent