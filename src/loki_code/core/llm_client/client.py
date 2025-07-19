"""
HTTP client for communicating with Loki LLM Server.

This module provides a clean interface for sending requests to the independent
LLM server and handling responses.
"""

import asyncio
import json
from typing import Dict, Any, Optional, AsyncIterator

import httpx
from loguru import logger

from .exceptions import LLMClientError, LLMServerError, LLMConnectionError, LLMTimeoutError
from ..agent.types import AgentResponse


class LLMClient:
    """
    HTTP client for Loki LLM Server.
    
    Handles communication with the independent LLM server including
    regular inference, streaming responses, and error handling.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL of the LLM server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # HTTP client configuration
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the LLM server is healthy and ready.
        
        Returns:
            Dict containing health status information
            
        Raises:
            LLMConnectionError: If unable to connect to server
            LLMServerError: If server returns error status
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Unable to connect to LLM server at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMServerError(f"LLM server health check failed: {e}", e.response.status_code)
        except Exception as e:
            raise LLMClientError(f"Unexpected error during health check: {e}")
    
    async def inference(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> AgentResponse:
        """
        Send an inference request to the LLM server.
        
        Args:
            message: The message to send to the LLM
            context: Optional context information
            config: Optional inference configuration
            model_name: Optional specific model to use
            
        Returns:
            AgentResponse containing the LLM's response
            
        Raises:
            LLMConnectionError: If unable to connect to server
            LLMServerError: If server returns error status
            LLMTimeoutError: If request times out
        """
        request_data = {
            "message": message,
            "context": context,
            "config": config,
            "model_name": model_name,
            "stream": False
        }
        
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Sending inference request (attempt {attempt + 1}): {message[:100]}...")
                
                response = await self.client.post(
                    "/api/v1/inference",
                    json=request_data
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                # Convert to AgentResponse format
                agent_response = AgentResponse(
                    content=response_data["content"],
                    metadata={
                        "model_name": response_data.get("model_name"),
                        "processing_time": response_data.get("processing_time"),
                        "tokens_generated": response_data.get("tokens_generated"),
                        "finish_reason": response_data.get("finish_reason"),
                        **response_data.get("metadata", {})
                    },
                    tools_used=response_data.get("tools_used", [])
                )
                
                logger.debug(f"Received response: {len(response_data['content'])} characters")
                return agent_response
                
            except httpx.ConnectError as e:
                if attempt < self.max_retries:
                    logger.warning(f"Connection failed (attempt {attempt + 1}), retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise LLMConnectionError(f"Unable to connect to LLM server after {self.max_retries + 1} attempts: {e}")
            
            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    logger.warning(f"Request timeout (attempt {attempt + 1}), retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise LLMTimeoutError(f"Request timed out after {self.max_retries + 1} attempts")
            
            except httpx.HTTPStatusError as e:
                error_msg = f"LLM server returned error {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f": {error_data.get('detail', 'Unknown error')}"
                except:
                    error_msg += f": {e.response.text}"
                
                raise LLMServerError(error_msg, e.response.status_code)
            
            except Exception as e:
                raise LLMClientError(f"Unexpected error during inference: {e}")
    
    async def inference_stream(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Send a streaming inference request to the LLM server.
        
        Args:
            message: The message to send to the LLM
            context: Optional context information
            config: Optional inference configuration
            model_name: Optional specific model to use
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            LLMConnectionError: If unable to connect to server
            LLMServerError: If server returns error status
            LLMTimeoutError: If request times out
        """
        request_data = {
            "message": message,
            "context": context,
            "config": config,
            "model_name": model_name,
            "stream": True
        }
        
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        try:
            logger.debug(f"Sending streaming inference request: {message[:100]}...")
            
            async with self.client.stream(
                "POST",
                "/api/v1/inference/stream",
                json=request_data
            ) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_text():
                    if chunk.strip():  # Skip empty chunks
                        yield chunk
                        
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Unable to connect to LLM server: {e}")
        
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Streaming request timed out: {e}")
        
        except httpx.HTTPStatusError as e:
            error_msg = f"LLM server returned error {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f": {error_data.get('detail', 'Unknown error')}"
            except:
                error_msg += f": {e.response.text}"
            
            raise LLMServerError(error_msg, e.response.status_code)
        
        except Exception as e:
            raise LLMClientError(f"Unexpected error during streaming inference: {e}")
    
    async def get_models(self) -> Dict[str, Any]:
        """
        Get available models from the LLM server.
        
        Returns:
            Dict containing model information
            
        Raises:
            LLMConnectionError: If unable to connect to server
            LLMServerError: If server returns error status
        """
        try:
            response = await self.client.get("/api/v1/models")
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Unable to connect to LLM server: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMServerError(f"Failed to get models: {e}", e.response.status_code)
        except Exception as e:
            raise LLMClientError(f"Unexpected error getting models: {e}")
    
    async def load_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a specific model on the LLM server.
        
        Args:
            model_name: Name of the model to load
            config: Optional model configuration
            
        Returns:
            Dict containing load result
            
        Raises:
            LLMConnectionError: If unable to connect to server
            LLMServerError: If server returns error status
        """
        request_data = {
            "model_name": model_name,
            "config": config
        }
        
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        try:
            response = await self.client.post("/api/v1/models/load", json=request_data)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Unable to connect to LLM server: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMServerError(f"Failed to load model: {e}", e.response.status_code)
        except Exception as e:
            raise LLMClientError(f"Unexpected error loading model: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected (basic check)."""
        return not self.client.is_closed