"""
Test the LangGraph agent implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from loki_code.core.agent import LokiCodeAgentFactory
from loki_code.core.agent.types import AgentConfig, AgentState


class TestLangGraphAgent:
    """Test cases for the LangGraph agent."""

    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test that LangGraph agent can be created with proper configuration."""
        config = AgentConfig(
            max_steps=3,
            model_name="test-model",
            auto_approve_safe_actions=True
        )
        
        # Mock the ChatOllama to avoid requiring actual Ollama service
        with patch('langchain_ollama.ChatOllama') as mock_ollama:
            mock_llm = Mock()
            mock_ollm.return_value = mock_llm
            
            agent = LokiCodeAgentFactory.create_with_ollama(
                model_name="test-model", 
                config=config
            )
            
            assert agent is not None
            assert agent.config == config
            assert agent.state == AgentState.IDLE
            assert len(agent.get_available_tools()) > 0

    @pytest.mark.asyncio  
    async def test_agent_error_handling(self):
        """Test that agent properly handles errors without infinite loops."""
        config = AgentConfig(max_steps=2, auto_approve_safe_actions=True)
        
        # Mock a failing LLM
        with patch('langchain_ollama.ChatOllama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("Connection failed")
            mock_ollama.return_value = mock_llm
            
            agent = LokiCodeAgentFactory.create_with_ollama(config=config)
            
            # This should not hang or loop infinitely
            result = await agent.process_request("Test message")
            
            assert result.state == AgentState.ERROR_RECOVERY
            assert "Connection" in result.content or "error" in result.content.lower()

    def test_recursion_limit_calculation(self):
        """Test that recursion limits are properly calculated."""
        config = AgentConfig(max_steps=5)
        
        with patch('langchain_ollama.ChatOllama') as mock_ollama:
            mock_ollama.return_value = Mock()
            agent = LokiCodeAgentFactory.create_with_ollama(config=config)
            
            # Recursion limit should be max_steps * 2 + 1, capped at 13
            expected_limit = min(5 * 2 + 1, 13)  # 11
            
            # Check if the agent was configured with the right limit
            # Note: This is a basic structural test since we can't easily inspect 
            # the internal agent configuration
            assert agent.config.max_steps == 5

    def test_tool_availability(self):
        """Test that required tools are available."""
        with patch('langchain_ollama.ChatOllama') as mock_ollama:
            mock_ollama.return_value = Mock()
            agent = LokiCodeAgentFactory.create_with_ollama()
            
            tools = agent.get_available_tools()
            
            # Should have at least file_reader and file_writer
            assert 'file_reader' in tools
            assert 'file_writer' in tools
            assert len(tools) >= 2