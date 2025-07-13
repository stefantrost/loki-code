"""
Shared pytest configuration for Loki Code tests.

This file provides shared fixtures and configuration for all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# Import shared fixtures from the fixtures module
from .fixtures.tool_test_fixtures import (
    mock_llm, test_config, temp_workspace, ToolTestHelpers
)

# Re-export fixtures so they're available to all test modules
__all__ = ["mock_llm", "test_config", "temp_workspace", "ToolTestHelpers"]


@pytest.fixture(scope="session")
def sample_code_dir():
    """Provide the sample code directory for tests."""
    return Path(__file__).parent / "fixtures" / "sample_code"


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    from ..config.models import LokiCodeConfig, AppConfig, LLMConfig, UIConfig, ToolsConfig
    
    return LokiCodeConfig(
        app=AppConfig(
            debug=True,
            log_level="DEBUG"
        ),
        llm=LLMConfig(
            provider="mock",
            model="test-model",
            base_url="http://localhost:11434"
        ),
        ui=UIConfig(
            theme="default",
            enable_color=True
        ),
        tools=ToolsConfig(
            enabled=["file_reader"],
            max_execution_time=30.0
        )
    )


@pytest.fixture
def mock_tool_registry():
    """Provide a mock tool registry for testing."""
    from ..core.tool_registry import ToolRegistry
    from ..tools.file_reader import FileReaderTool
    
    registry = ToolRegistry()
    registry.register_tool("file_reader", FileReaderTool())
    return registry


# Configure pytest markers
pytest_plugins = []

# Test markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-related"
    )


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Cleanup test environment after each test."""
    yield
    # Any cleanup logic can go here