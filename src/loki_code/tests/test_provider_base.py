"""
Test suite for provider base classes and interfaces.

This module tests the BaseLLMProvider interface, GenerationRequest/Response
handling, and provider abstraction layer functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, AsyncGenerator
from abc import ABC

from loki_code.core.providers.base import (
    BaseLLMProvider, GenerationRequest, GenerationResponse,
    ProviderCapability, ModelInfo
)
from loki_code.core.providers.exceptions import (
    ProviderConnectionError, ProviderModelError, ProviderTimeoutError,
    ProviderValidationError
)
from loki_code.config.models import LokiCodeConfig


class TestGenerationRequest:
    """Test the GenerationRequest data class."""
    
    def test_generation_request_creation(self):
        """Test creating GenerationRequest with required fields."""
        request = GenerationRequest(
            prompt="Test prompt",
            model="test-model"
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "test-model"
        assert request.temperature is None
        assert request.max_tokens is None
        assert request.stream is False
        assert request.context == {}
    
    def test_generation_request_with_all_fields(self):
        """Test creating GenerationRequest with all fields."""
        context = {"session_id": "test123", "user_id": "user456"}
        
        request = GenerationRequest(
            prompt="Detailed prompt",
            model="advanced-model",
            temperature=0.8,
            max_tokens=2048,
            stream=True,
            context=context
        )
        
        assert request.prompt == "Detailed prompt"
        assert request.model == "advanced-model"
        assert request.temperature == 0.8
        assert request.max_tokens == 2048
        assert request.stream is True
        assert request.context == context
    
    def test_generation_request_validation(self):
        """Test GenerationRequest field validation."""
        # Empty prompt should be invalid
        with pytest.raises(ValueError):
            GenerationRequest(prompt="", model="test-model")
        
        # Empty model should be invalid
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Test", model="")
        
        # Invalid temperature range
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Test", model="test", temperature=-0.1)
        
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Test", model="test", temperature=2.1)
        
        # Invalid max_tokens
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Test", model="test", max_tokens=0)
    
    def test_generation_request_serialization(self):
        """Test GenerationRequest serialization to dict."""
        request = GenerationRequest(
            prompt="Test prompt",
            model="test-model",
            temperature=0.5,
            max_tokens=1024
        )
        
        request_dict = request.to_dict()
        
        expected = {
            "prompt": "Test prompt",
            "model": "test-model", 
            "temperature": 0.5,
            "max_tokens": 1024,
            "stream": False,
            "context": {}
        }
        
        assert request_dict == expected
    
    def test_generation_request_from_dict(self):
        """Test creating GenerationRequest from dictionary."""
        request_dict = {
            "prompt": "Dict prompt",
            "model": "dict-model",
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True,
            "context": {"key": "value"}
        }
        
        request = GenerationRequest.from_dict(request_dict)
        
        assert request.prompt == "Dict prompt"
        assert request.model == "dict-model"
        assert request.temperature == 0.7
        assert request.max_tokens == 512
        assert request.stream is True
        assert request.context == {"key": "value"}


class TestGenerationResponse:
    """Test the GenerationResponse data class."""
    
    def test_generation_response_creation(self):
        """Test creating GenerationResponse with required fields."""
        response = GenerationResponse(
            content="Generated content",
            model="test-model"
        )
        
        assert response.content == "Generated content"
        assert response.model == "test-model"
        assert response.finish_reason is None
        assert response.usage == {}
        assert response.metadata == {}
    
    def test_generation_response_with_all_fields(self):
        """Test creating GenerationResponse with all fields."""
        usage = {"total_tokens": 150, "prompt_tokens": 50, "completion_tokens": 100}
        metadata = {"generation_time": 2.5, "model_version": "1.0"}
        
        response = GenerationResponse(
            content="Complete response",
            model="advanced-model",
            finish_reason="stop",
            usage=usage,
            metadata=metadata
        )
        
        assert response.content == "Complete response"
        assert response.model == "advanced-model"
        assert response.finish_reason == "stop"
        assert response.usage == usage
        assert response.metadata == metadata
    
    def test_generation_response_serialization(self):
        """Test GenerationResponse serialization to dict."""
        usage = {"total_tokens": 100}
        metadata = {"time": 1.0}
        
        response = GenerationResponse(
            content="Test content",
            model="test-model",
            finish_reason="length",
            usage=usage,
            metadata=metadata
        )
        
        response_dict = response.to_dict()
        
        expected = {
            "content": "Test content",
            "model": "test-model",
            "finish_reason": "length",
            "usage": usage,
            "metadata": metadata
        }
        
        assert response_dict == expected
    
    def test_generation_response_from_dict(self):
        """Test creating GenerationResponse from dictionary."""
        response_dict = {
            "content": "Dict content",
            "model": "dict-model",
            "finish_reason": "stop",
            "usage": {"tokens": 200},
            "metadata": {"source": "test"}
        }
        
        response = GenerationResponse.from_dict(response_dict)
        
        assert response.content == "Dict content"
        assert response.model == "dict-model"
        assert response.finish_reason == "stop"
        assert response.usage == {"tokens": 200}
        assert response.metadata == {"source": "test"}


class TestModelInfo:
    """Test the ModelInfo data class."""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo with required fields."""
        model_info = ModelInfo(
            name="test-model",
            capabilities=[ProviderCapability.TEXT_GENERATION]
        )
        
        assert model_info.name == "test-model"
        assert model_info.capabilities == [ProviderCapability.TEXT_GENERATION]
        assert model_info.description is None
        assert model_info.context_length is None
        assert model_info.parameters == {}
    
    def test_model_info_with_all_fields(self):
        """Test creating ModelInfo with all fields."""
        capabilities = [
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.STREAMING,
            ProviderCapability.CHAT_COMPLETION
        ]
        parameters = {"size": "7B", "type": "causal"}
        
        model_info = ModelInfo(
            name="advanced-model",
            capabilities=capabilities,
            description="Advanced language model",
            context_length=4096,
            parameters=parameters
        )
        
        assert model_info.name == "advanced-model"
        assert model_info.capabilities == capabilities
        assert model_info.description == "Advanced language model"
        assert model_info.context_length == 4096
        assert model_info.parameters == parameters
    
    def test_model_info_capability_checking(self):
        """Test checking if model has specific capabilities."""
        model_info = ModelInfo(
            name="test-model",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.STREAMING
            ]
        )
        
        assert model_info.has_capability(ProviderCapability.TEXT_GENERATION)
        assert model_info.has_capability(ProviderCapability.STREAMING)
        assert not model_info.has_capability(ProviderCapability.CHAT_COMPLETION)
        assert not model_info.has_capability(ProviderCapability.FUNCTION_CALLING)


class ConcreteTestProvider(BaseLLMProvider):
    """Concrete implementation of BaseLLMProvider for testing."""
    
    def __init__(self, config: LokiCodeConfig):
        super().__init__(config)
        self._models = {
            "test-model": ModelInfo(
                name="test-model",
                capabilities=[ProviderCapability.TEXT_GENERATION],
                description="Test model",
                context_length=2048
            ),
            "streaming-model": ModelInfo(
                name="streaming-model", 
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.STREAMING
                ],
                context_length=4096
            )
        }
        self._connection_status = True
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Mock generate method."""
        if not self._connection_status:
            raise ProviderConnectionError("Not connected")
        
        if request.model not in self._models:
            raise ProviderModelError(f"Model {request.model} not found")
        
        # Simulate generation
        content = f"Generated response for: {request.prompt[:50]}..."
        
        return GenerationResponse(
            content=content,
            model=request.model,
            finish_reason="stop",
            usage={"total_tokens": len(content)},
            metadata={"test": True}
        )
    
    async def stream_generate(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Mock streaming generation."""
        if not self._connection_status:
            raise ProviderConnectionError("Not connected")
        
        model_info = self._models.get(request.model)
        if not model_info:
            raise ProviderModelError(f"Model {request.model} not found")
        
        if not model_info.has_capability(ProviderCapability.STREAMING):
            raise ProviderModelError(f"Model {request.model} doesn't support streaming")
        
        # Simulate streaming tokens
        tokens = ["Token", " 1", " Token", " 2", " Token", " 3"]
        for token in tokens:
            yield token
            await asyncio.sleep(0.001)  # Simulate delay
    
    async def list_models(self) -> list[ModelInfo]:
        """Mock list models method."""
        if not self._connection_status:
            raise ProviderConnectionError("Not connected")
        return list(self._models.values())
    
    async def health_check(self) -> bool:
        """Mock health check method."""
        return self._connection_status
    
    def set_connection_status(self, status: bool):
        """Helper method for testing connection issues."""
        self._connection_status = status


class TestBaseLLMProvider:
    """Test the BaseLLMProvider abstract base class."""
    
    def test_provider_creation(self):
        """Test creating a concrete provider instance."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.config == config
        assert provider.provider_name == "ConcreteTestProvider"
    
    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful text generation."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        request = GenerationRequest(
            prompt="Test prompt for generation",
            model="test-model"
        )
        
        response = await provider.generate(request)
        
        assert isinstance(response, GenerationResponse)
        assert response.model == "test-model"
        assert "Test prompt for generation" in response.content
        assert response.finish_reason == "stop"
        assert "total_tokens" in response.usage
        assert response.metadata["test"] is True
    
    @pytest.mark.asyncio
    async def test_generation_with_invalid_model(self):
        """Test generation with non-existent model."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        request = GenerationRequest(
            prompt="Test prompt",
            model="nonexistent-model"
        )
        
        with pytest.raises(ProviderModelError) as exc_info:
            await provider.generate(request)
        
        assert "Model nonexistent-model not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generation_with_connection_error(self):
        """Test generation when provider is disconnected."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        provider.set_connection_status(False)
        
        request = GenerationRequest(
            prompt="Test prompt",
            model="test-model"
        )
        
        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.generate(request)
        
        assert "Not connected" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self):
        """Test streaming text generation."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        request = GenerationRequest(
            prompt="Test streaming prompt",
            model="streaming-model",
            stream=True
        )
        
        tokens = []
        async for token in provider.stream_generate(request):
            tokens.append(token)
        
        assert len(tokens) == 6
        assert tokens == ["Token", " 1", " Token", " 2", " Token", " 3"]
    
    @pytest.mark.asyncio
    async def test_streaming_with_non_streaming_model(self):
        """Test streaming with model that doesn't support it."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        request = GenerationRequest(
            prompt="Test prompt",
            model="test-model",  # This model doesn't support streaming
            stream=True
        )
        
        with pytest.raises(ProviderModelError) as exc_info:
            async for token in provider.stream_generate(request):
                pass
        
        assert "doesn't support streaming" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        models = await provider.list_models()
        
        assert len(models) == 2
        model_names = [model.name for model in models]
        assert "test-model" in model_names
        assert "streaming-model" in model_names
        
        # Check model capabilities
        test_model = next(m for m in models if m.name == "test-model")
        assert test_model.has_capability(ProviderCapability.TEXT_GENERATION)
        assert not test_model.has_capability(ProviderCapability.STREAMING)
        
        streaming_model = next(m for m in models if m.name == "streaming-model")
        assert streaming_model.has_capability(ProviderCapability.TEXT_GENERATION)
        assert streaming_model.has_capability(ProviderCapability.STREAMING)
    
    @pytest.mark.asyncio
    async def test_list_models_connection_error(self):
        """Test listing models when provider is disconnected."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        provider.set_connection_status(False)
        
        with pytest.raises(ProviderConnectionError):
            await provider.list_models()
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test provider health check."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        # Should be healthy initially
        is_healthy = await provider.health_check()
        assert is_healthy is True
        
        # Disconnect and check again
        provider.set_connection_status(False)
        is_healthy = await provider.health_check()
        assert is_healthy is False
    
    def test_provider_name_property(self):
        """Test that provider name is correctly derived from class name."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        assert provider.provider_name == "ConcreteTestProvider"
    
    def test_provider_config_access(self):
        """Test that provider can access configuration."""
        config = LokiCodeConfig()
        config.llm.model = "custom-model"
        config.llm.temperature = 0.8
        
        provider = ConcreteTestProvider(config)
        
        assert provider.config.llm.model == "custom-model"
        assert provider.config.llm.temperature == 0.8
    
    @pytest.mark.asyncio
    async def test_provider_request_validation(self):
        """Test that provider validates requests appropriately."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        # Test with various invalid requests
        invalid_requests = [
            GenerationRequest(prompt="", model="test-model"),  # Empty prompt
            GenerationRequest(prompt="Test", model=""),        # Empty model
        ]
        
        for request in invalid_requests:
            with pytest.raises((ValueError, ProviderValidationError)):
                await provider.generate(request)


class TestProviderCapability:
    """Test the ProviderCapability enum."""
    
    def test_provider_capability_values(self):
        """Test ProviderCapability enum values."""
        assert ProviderCapability.TEXT_GENERATION.value == "text_generation"
        assert ProviderCapability.CHAT_COMPLETION.value == "chat_completion"
        assert ProviderCapability.STREAMING.value == "streaming"
        assert ProviderCapability.FUNCTION_CALLING.value == "function_calling"
        assert ProviderCapability.EMBEDDING.value == "embedding"
        assert ProviderCapability.IMAGE_GENERATION.value == "image_generation"
    
    def test_capability_membership(self):
        """Test checking capability membership in collections."""
        capabilities = [
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.STREAMING
        ]
        
        assert ProviderCapability.TEXT_GENERATION in capabilities
        assert ProviderCapability.STREAMING in capabilities
        assert ProviderCapability.CHAT_COMPLETION not in capabilities
        assert ProviderCapability.FUNCTION_CALLING not in capabilities


@pytest.mark.integration
class TestProviderIntegration:
    """Integration tests for provider system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_generation_flow(self):
        """Test complete generation flow from request to response."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        # Create request
        request = GenerationRequest(
            prompt="Write a simple function to add two numbers",
            model="test-model",
            temperature=0.7,
            max_tokens=512
        )
        
        # Generate response
        response = await provider.generate(request)
        
        # Verify response
        assert isinstance(response, GenerationResponse)
        assert response.model == "test-model"
        assert "Write a simple function" in response.content
        assert isinstance(response.usage["total_tokens"], int)
        assert response.metadata["test"] is True
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self):
        """Test streaming integration flow."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        request = GenerationRequest(
            prompt="Generate a story",
            model="streaming-model",
            stream=True
        )
        
        # Collect streaming tokens
        full_content = ""
        token_count = 0
        
        async for token in provider.stream_generate(request):
            full_content += token
            token_count += 1
        
        assert token_count == 6
        assert full_content == "Token 1 Token 2 Token 3"
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test integrated error handling across provider operations."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        # Test connection error propagation
        provider.set_connection_status(False)
        
        with pytest.raises(ProviderConnectionError):
            await provider.generate(GenerationRequest("test", "test-model"))
        
        with pytest.raises(ProviderConnectionError):
            await provider.list_models()
        
        # Reconnect and test model error
        provider.set_connection_status(True)
        
        with pytest.raises(ProviderModelError):
            await provider.generate(GenerationRequest("test", "invalid-model"))
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent provider operations."""
        config = LokiCodeConfig()
        provider = ConcreteTestProvider(config)
        
        # Create multiple concurrent requests
        requests = [
            GenerationRequest(f"Prompt {i}", "test-model")
            for i in range(5)
        ]
        
        # Execute concurrently
        tasks = [provider.generate(request) for request in requests]
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert isinstance(response, GenerationResponse)
            assert f"Prompt {i}" in response.content
            assert response.model == "test-model"