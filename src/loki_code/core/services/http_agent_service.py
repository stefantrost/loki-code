"""
HTTP client-based agent service for communicating with Loki LLM Server.

This service replaces the direct LLM integration with HTTP communication
to the independent LLM server.
"""

import asyncio
from typing import Optional, Dict, Any

from ..llm_client import LLMClient, LLMConnectionError, LLMServerError
from ..agent.types import AgentResponse, AgentState
from ...config import LokiCodeConfig
from ...utils.logging import get_logger


class HttpAgentService:
    """
    HTTP-based agent service that communicates with Loki LLM Server.
    
    This service provides the same interface as the original AgentService
    but uses HTTP communication instead of direct LLM integration.
    """
    
    def __init__(self, config: LokiCodeConfig):
        """Initialize the HTTP agent service with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        self.session_id = None
        self.llm_client = None
        self._initialized = False
        
        # Get LLM server configuration
        self.server_url = getattr(config.llm, 'llm_server_url', 'http://localhost:8765')
        self.server_timeout = getattr(config.llm, 'llm_server_timeout', 30.0)
        self.max_retries = getattr(config.llm, 'llm_server_retries', 3)
        
    async def initialize(self, session_id: Optional[str] = None) -> bool:
        """
        Initialize the HTTP agent service.
        
        Args:
            session_id: Optional session ID for conversation persistence
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.session_id = session_id or "default_session"
            self.logger.info(f"🚀 Starting HTTP agent initialization (session: {self.session_id})")
            self.logger.info(f"📡 Connecting to LLM server at: {self.server_url}")
            
            # Create LLM client
            self.logger.info("🔧 Step 1: Creating LLM client")
            self.llm_client = LLMClient(
                base_url=self.server_url,
                timeout=self.server_timeout,
                max_retries=self.max_retries
            )
            
            # Test connection and health
            self.logger.info("🏥 Step 2: Testing server health")
            try:
                health_status = await self.llm_client.health_check()
                self.logger.info(f"✅ Server health check passed: {health_status.get('status', 'unknown')}")
                
                # Log server information
                if 'models' in health_status:
                    model_info = health_status['models']
                    self.logger.info(f"📊 Server model status: {model_info}")
                
                if 'application' in health_status:
                    app_info = health_status['application']
                    uptime = app_info.get('uptime_seconds', 0)
                    self.logger.info(f"⏰ Server uptime: {uptime:.1f}s")
                
            except LLMConnectionError as e:
                self.logger.error(f"❌ Cannot connect to LLM server: {e}")
                self.logger.error("🔍 Troubleshooting:")
                self.logger.error("   - Check if LLM server is running: loki-llm-server")
                self.logger.error(f"   - Verify server URL: {self.server_url}")
                self.logger.error("   - Check network connectivity")
                return False
                
            except LLMServerError as e:
                self.logger.error(f"❌ LLM server returned error: {e}")
                return False
            
            # Test inference capability
            self.logger.info("🧪 Step 3: Testing inference capability")
            try:
                test_response = await self.llm_client.inference(
                    message="Hello, this is a test message. Please respond briefly.",
                    config={"max_tokens": 20}
                )
                self.logger.info(f"✅ Inference test successful: {len(test_response.content)} characters")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Inference test failed (server may be loading): {e}")
                # Don't fail initialization for this - server might be loading models
            
            self._initialized = True
            self.logger.info("🎉 HTTP agent service initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 HTTP agent initialization failed: {e}", exc_info=True)
            
            # Cleanup on failure
            if self.llm_client:
                await self.llm_client.close()
                self.llm_client = None
            
            return False
    
    async def process_message(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a user message through the HTTP LLM server.
        
        Args:
            user_message: The user's input message
            context: Optional context information
            
        Returns:
            AgentResponse: The agent's response
        """
        if not self._initialized or not self.llm_client:
            raise RuntimeError("HTTP agent service not initialized. Call initialize() first.")
        
        try:
            self.logger.info(f"📨 Processing message via HTTP: {user_message[:100]}...")
            
            # Prepare inference configuration
            inference_config = {
                "max_tokens": getattr(self.config.llm, 'max_tokens', 512),
                "temperature": getattr(self.config.llm, 'temperature', 0.7),
                "top_p": getattr(self.config.llm, 'top_p', 0.9),
            }
            
            # Add session context
            request_context = {
                "session_id": self.session_id,
                "project_path": str(getattr(self.config, 'project_path', '')),
                **(context or {})
            }
            
            # Send request to LLM server
            response = await self.llm_client.inference(
                message=user_message,
                context=request_context,
                config=inference_config
            )
            
            self.logger.info(f"✅ Message processed successfully via HTTP: {len(response.content)} characters")
            return response
            
        except LLMConnectionError as e:
            self.logger.error(f"🔌 Connection error: {e}")
            return AgentResponse(
                content="I'm having trouble connecting to the language model server. Please check if the server is running and try again.",
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={"error": str(e), "error_type": "connection"}
            )
            
        except LLMServerError as e:
            self.logger.error(f"🖥️ Server error: {e}")
            return AgentResponse(
                content="The language model server returned an error. Please try again or contact support if the problem persists.",
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={"error": str(e), "error_type": "server", "status_code": getattr(e, 'status_code', None)}
            )
            
        except Exception as e:
            self.logger.error(f"💥 Error processing message: {e}", exc_info=True)
            return AgentResponse(
                content=f"I encountered an unexpected error while processing your request: {str(e)}",
                state=AgentState.ERROR_RECOVERY,
                confidence=0.0,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def process_message_stream(self, user_message: str, context: Optional[Dict[str, Any]] = None):
        """
        Process a user message with streaming response from the HTTP LLM server.
        
        Args:
            user_message: The user's input message
            context: Optional context information
            
        Yields:
            str: Chunks of the response as they arrive
        """
        if not self._initialized or not self.llm_client:
            raise RuntimeError("HTTP agent service not initialized. Call initialize() first.")
        
        try:
            self.logger.info(f"📨 Processing streaming message via HTTP: {user_message[:100]}...")
            
            # Prepare inference configuration
            inference_config = {
                "max_tokens": getattr(self.config.llm, 'max_tokens', 512),
                "temperature": getattr(self.config.llm, 'temperature', 0.7),
                "top_p": getattr(self.config.llm, 'top_p', 0.9),
            }
            
            # Add session context
            request_context = {
                "session_id": self.session_id,
                "project_path": str(getattr(self.config, 'project_path', '')),
                **(context or {})
            }
            
            # Send streaming request to LLM server
            async for chunk in self.llm_client.inference_stream(
                message=user_message,
                context=request_context,
                config=inference_config
            ):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"💥 Error in streaming message: {e}", exc_info=True)
            yield f"Error: {str(e)}"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current HTTP agent."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "agent_type": "HttpAgentService",
            "session_id": self.session_id,
            "server_url": self.server_url,
            "server_timeout": self.server_timeout,
            "max_retries": self.max_retries,
            "connected": self.llm_client.is_connected if self.llm_client else False
        }
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get available models from the LLM server."""
        if not self._initialized or not self.llm_client:
            raise RuntimeError("HTTP agent service not initialized.")
        
        try:
            return await self.llm_client.get_models()
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return {"error": str(e)}
    
    async def load_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load a specific model on the LLM server."""
        if not self._initialized or not self.llm_client:
            raise RuntimeError("HTTP agent service not initialized.")
        
        try:
            return await self.llm_client.load_model(model_name, config)
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the HTTP agent service."""
        self.logger.info("🔄 Shutting down HTTP agent service...")
        
        if self.llm_client:
            await self.llm_client.close()
            self.llm_client = None
        
        self._initialized = False
        self.logger.info("✅ HTTP agent service shutdown complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        if hasattr(self, 'llm_client') and self.llm_client:
            # Note: This is not ideal as __del__ can't be async
            # Better to explicitly call shutdown()
            try:
                asyncio.create_task(self.llm_client.close())
            except:
                pass