"""
Unified agent service for both CLI and TUI interfaces.

This service provides a single point of access for agent functionality,
eliminating code duplication between different UI interfaces.
"""

import asyncio
from typing import Optional, Dict, Any, Union
from pathlib import Path

from ..agent import LokiCodeAgentFactory
from ..agent.types import AgentConfig, AgentResponse, RequestContext
from ..commands.types import ConversationContext
from ..tool_registry import get_global_registry
from ...config import LokiCodeConfig
from ...utils.logging import get_logger


class AgentService:
    """
    Unified service for agent operations across all UI interfaces.
    
    This service handles:
    - Agent creation and configuration
    - Message processing
    - Session management
    - Error handling
    """
    
    def __init__(self, config: LokiCodeConfig):
        """Initialize the agent service with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        self.agent = None
        self.session_id = None
        
    async def initialize(self, session_id: Optional[str] = None) -> bool:
        """
        Initialize the agent service.
        
        Args:
            session_id: Optional session ID for conversation persistence
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.session_id = session_id or "default_session"
            
            # Create agent configuration
            agent_config = AgentConfig(
                model_name=getattr(self.config, 'default_model', self.config.llm.model),
                max_steps=getattr(self.config, 'max_steps', 10),
                auto_approve_safe_actions=getattr(self.config, 'auto_approve_safe_actions', True)
            )
            
            # Create agent using the factory  
            self.agent = LokiCodeAgentFactory.create_with_transformers(
                model_name=self.config.llm.model,
                model_path=None,  # Use default path
                config=agent_config
            )
            
            # Validate agent
            if not self.agent:
                raise RuntimeError("Agent factory returned None")
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Agent initialization failed: {e}", exc_info=True)
            
            # Detailed error analysis
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            self.logger.error(f"🔍 Error Details:")
            self.logger.error(f"   - Error Type: {error_type}")
            self.logger.error(f"   - Error Message: {str(e)}")
            self.logger.error(f"   - Session ID: {self.session_id}")
            
            # Categorize and provide specific guidance
            if "tool" in error_str and "validation" in error_str:
                self.logger.error(
                    "❌ CATEGORY: Tool validation failed. This indicates improper tool definitions. "
                    "Check that all tools have proper Pydantic schemas and required methods."
                )
            elif "model" in error_str and ("not found" in error_str or "404" in error_str):
                self.logger.error(
                    "❌ CATEGORY: Model not found. Check that the model path exists and is accessible. "
                    f"Default model path: {getattr(self.config, 'default_model_path', 'not configured')}"
                )
            elif "connection" in error_str or "refused" in error_str:
                self.logger.error(
                    "❌ CATEGORY: Connection error. Check that your LLM provider is running. "
                    "For Ollama, run: ollama serve"
                )
            elif "transformers" in error_str or "huggingface" in error_str:
                self.logger.error(
                    "❌ CATEGORY: Transformers/HuggingFace error. Check that all required dependencies are installed. "
                    "Try: pip install transformers torch"
                )
            elif "cuda" in error_str or "bitsandbytes" in error_str:
                self.logger.error(
                    "❌ CATEGORY: CUDA/Quantization error. Check CUDA installation or disable quantization."
                )
            elif "memory" in error_str or "out of memory" in error_str:
                self.logger.error(
                    "❌ CATEGORY: Memory error. Insufficient memory to load model. Close other applications."
                )
            elif "permission" in error_str or "access" in error_str:
                self.logger.error(
                    "❌ CATEGORY: Permission error. Check file and directory permissions."
                )
            else:
                self.logger.error(f"❌ CATEGORY: Unknown error: {e}")
            
            # Log system state
            try:
                import psutil
                memory = psutil.virtual_memory()
                self.logger.error(f"🖥️  System State:")
                self.logger.error(f"   - Memory Available: {memory.available / (1024**3):.2f} GB")
                self.logger.error(f"   - Memory Used: {memory.percent}%")
                self.logger.error(f"   - Agent Object: {self.agent}")
            except:
                pass
            
            return False
    
    async def process_message(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a user message through the agent.
        
        Args:
            user_message: The user's input message
            context: Optional context information
            
        Returns:
            AgentResponse: The agent's response
        """
        if not self.agent:
            raise RuntimeError("Agent service not initialized. Call initialize() first.")
        
        try:
            # Create request context
            request_context = RequestContext(
                session_id=self.session_id,
                project_path=getattr(self.config, 'project_path', None),
                session_metadata=context or {}
            )
            
            # Process the message
            self.logger.info(f"Processing message: {user_message[:100]}...")
            response = await self.agent.process_request(user_message, request_context)
            
            self.logger.info(f"Message processed successfully. State: {response.state}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            
            # Return error response
            from ..agent.types import AgentState
            return AgentResponse(
                content=f"I encountered an error while processing your request: {str(e)}",
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def process_message_stream(self, user_message: str, context: Optional[Dict[str, Any]] = None):
        """
        Process a user message with streaming response from the agent.
        
        Args:
            user_message: The user's input message
            context: Optional context information
            
        Yields:
            Dict[str, Any]: Streaming updates from the agent
        """
        if not self.agent:
            raise RuntimeError("Agent service not initialized. Call initialize() first.")
        
        try:
            # Create request context
            request_context = RequestContext(
                session_id=self.session_id,
                project_path=getattr(self.config, 'project_path', None),
                session_metadata=context or {}
            )
            
            self.logger.info(f"Processing streaming message: {user_message[:100]}...")
            
            # Check if agent supports streaming
            if hasattr(self.agent, 'process_request_stream'):
                # Use native streaming if available
                async for update in self.agent.process_request_stream(user_message, request_context):
                    yield update
            else:
                # Fallback to non-streaming with simulated streaming
                response = await self.agent.process_request(user_message, request_context)
                
                # Yield the response in simulated streaming format
                yield {
                    "type": "final_response",
                    "response": response,
                    "content": response.content
                }
                
        except Exception as e:
            self.logger.error(f"Error in streaming message: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent."""
        if not self.agent:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "agent_type": type(self.agent).__name__,
            "available_tools": self.agent.get_available_tools(),
            "state": self.agent.get_state().value if hasattr(self.agent.get_state(), 'value') else str(self.agent.get_state()),
            "session_id": self.session_id
        }
    
    def clear_memory(self) -> None:
        """Clear the agent's conversation memory."""
        if self.agent and hasattr(self.agent, 'clear_memory'):
            self.agent.clear_memory()
            self.logger.info("Agent memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.agent:
            return {"status": "not_initialized"}
        
        if hasattr(self.agent, 'get_memory_stats'):
            return self.agent.get_memory_stats()
        else:
            return {"status": "memory_stats_not_available"}


# Global service instance (lazy initialization) with retry support
_agent_service: Optional[AgentService] = None
_initialization_attempts = 0
_max_initialization_attempts = 3


def _is_service_initialized(service) -> bool:
    """Check if a service is properly initialized."""
    if service is None:
        return False
    
    # For HttpAgentService, check _initialized flag
    if hasattr(service, '_initialized'):
        return service._initialized
    
    # For regular AgentService, check agent attribute
    return hasattr(service, 'agent') and service.agent is not None


async def get_agent_service(config: LokiCodeConfig, session_id: Optional[str] = None, retry: bool = True) -> Union[AgentService, 'HttpAgentService']:
    """
    Get or create the global agent service instance with retry logic.
    
    Args:
        config: Application configuration
        session_id: Optional session ID
        retry: Whether to retry on failure
        
    Returns:
        AgentService: Initialized agent service (HttpAgentService or traditional AgentService)
        
    Raises:
        RuntimeError: If initialization fails after max attempts
    """
    global _agent_service, _initialization_attempts
    logger = get_logger(__name__)
    
    # Check if HTTP mode is enabled
    use_http_server = getattr(config.llm, 'use_llm_server', False)
    
    if _agent_service is None or not _is_service_initialized(_agent_service):
        _initialization_attempts += 1
        
        try:
            # Create appropriate service type
            if use_http_server:
                from .http_agent_service import HttpAgentService
                _agent_service = HttpAgentService(config)
            else:
                _agent_service = AgentService(config)
            
            success = await _agent_service.initialize(session_id)
            
            if not success:
                error_msg = f"Failed to initialize agent service (attempt {_initialization_attempts}/{_max_initialization_attempts})"
                
                if retry and _initialization_attempts < _max_initialization_attempts:
                    # Reset for retry
                    _agent_service = None
                    # Recursive retry with exponential backoff
                    await asyncio.sleep(2 ** _initialization_attempts)
                    return await get_agent_service(config, session_id, retry)
                else:
                    # Final failure
                    _agent_service = None
                    raise RuntimeError(f"{error_msg}. Check logs for detailed error information.")
            
            # Reset attempt counter on success
            _initialization_attempts = 0
            
        except Exception as e:
            error_msg = f"Agent service initialization failed (attempt {_initialization_attempts}/{_max_initialization_attempts}): {e}"
            
            if retry and _initialization_attempts < _max_initialization_attempts:
                # Reset for retry
                _agent_service = None
                # Recursive retry with exponential backoff
                await asyncio.sleep(2 ** _initialization_attempts)
                return await get_agent_service(config, session_id, retry)
            else:
                # Final failure
                _agent_service = None
                raise RuntimeError(f"{error_msg}")
    
    return _agent_service


def reset_agent_service() -> None:
    """Reset the global agent service (useful for testing)."""
    global _agent_service, _initialization_attempts
    _agent_service = None
    _initialization_attempts = 0


def get_agent_service_status() -> Dict[str, Any]:
    """Get the current status of the agent service."""
    global _agent_service, _initialization_attempts
    
    if _agent_service is None:
        return {
            "status": "not_initialized",
            "attempts": _initialization_attempts,
            "max_attempts": _max_initialization_attempts
        }
    
    return {
        "status": "initialized" if _is_service_initialized(_agent_service) else "failed",
        "attempts": _initialization_attempts,
        "max_attempts": _max_initialization_attempts,
        "agent_info": _agent_service.get_agent_info() if _is_service_initialized(_agent_service) else None
    }