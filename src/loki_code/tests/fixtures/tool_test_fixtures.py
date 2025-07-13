"""
Test fixtures for tool system testing.

Extracted from the original monolithic test file for better organization.
"""

import pytest
from unittest.mock import Mock
from pathlib import Path
from typing import Dict, Any


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        
    def generate_sync(self, request):
        """Simulate LLM generation."""
        self.call_count += 1
        self.last_prompt = request.prompt
        
        # Mock response based on prompt
        if "hello" in request.prompt.lower():
            return Mock(content="Hello! How can I help you?")
        elif "error" in request.prompt.lower():
            raise Exception("Simulated LLM error")
        else:
            return Mock(content="This is a mock response from the LLM.")


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing."""
    return MockLLMProvider()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    from ...config.models import LokiCodeConfig, AppConfig, LLMConfig
    
    return LokiCodeConfig(
        app=AppConfig(
            debug=True,
            log_level="DEBUG"
        ),
        llm=LLMConfig(
            provider="mock",
            model="test-model"
        )
    )


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def get_sample_code_path(language: str) -> Path:
    """Get path to sample code file for a language."""
    fixtures_dir = Path(__file__).parent / "sample_code"
    
    extensions = {
        "python": "py",
        "javascript": "js", 
        "typescript": "ts",
        "rust": "rs"
    }
    
    extension = extensions.get(language, "txt")
    return fixtures_dir / f"test_sample.{extension}"


class ToolTestHelpers:
    """Helper methods for tool testing."""
    
    @staticmethod
    def create_test_file(path: Path, content: str) -> Path:
        """Create a test file with content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    
    @staticmethod
    def assert_tool_result_success(result, expected_content_parts=None):
        """Assert that a tool result is successful."""
        assert result.success, f"Tool execution failed: {result.message}"
        assert result.output is not None
        
        if expected_content_parts:
            output_str = str(result.output)
            for part in expected_content_parts:
                assert part in output_str, f"Expected '{part}' in result output"
    
    @staticmethod
    def create_mock_tool_context() -> Dict[str, Any]:
        """Create a mock tool context for testing."""
        return {
            "project_path": "/test/project",
            "user_id": "test_user",
            "session_id": "test_session",
            "permissions": ["read_file", "analyze_code"],
            "safety_settings": {
                "max_file_size": 1024 * 1024,  # 1MB
                "allowed_extensions": [".py", ".js", ".ts", ".rs"]
            }
        }