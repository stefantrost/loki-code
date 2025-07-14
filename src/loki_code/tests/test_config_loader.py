"""
Test suite for configuration loading system.

This module tests the configuration loading, environment variable overrides,
YAML file parsing, and configuration discovery functionality.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from loki_code.config.loader import (
    ConfigLoader, load_config, get_config, reload_config,
    validate_config_file, ConfigurationError
)
from loki_code.config.models import LokiCodeConfig


class TestConfigLoader:
    """Test the ConfigLoader class functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ConfigLoader()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Clear any cached config
        if hasattr(get_config, '_cached_config'):
            delattr(get_config, '_cached_config')
    
    def test_load_default_config(self):
        """Test loading default configuration when no files exist."""
        config = self.loader.load_config()
        
        assert isinstance(config, LokiCodeConfig)
        assert config.app.name == "Loki Code"
        assert config.llm.provider.value == "ollama"
        assert config.llm.model == "codellama:7b"
        assert config.tools.auto_discover_builtin is True
    
    def test_load_config_from_yaml_file(self):
        """Test loading configuration from a YAML file."""
        config_content = """
app:
  name: "Custom Loki"
  version: "2.0.0"
  
llm:
  model: "custom-model:latest"
  temperature: 0.5
  
tools:
  auto_discover_builtin: false
"""
        config_file = Path(self.temp_dir) / "config.yaml"
        config_file.write_text(config_content)
        
        config = self.loader.load_config(str(config_file))
        
        assert config.app.name == "Custom Loki"
        assert config.app.version == "2.0.0"
        assert config.llm.model == "custom-model:latest"
        assert config.llm.temperature == 0.5
        assert config.tools.auto_discover_builtin is False
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override config file values."""
        config_content = """
app:
  name: "Original Name"
  
llm:
  model: "original-model"
  temperature: 0.1
"""
        config_file = Path(self.temp_dir) / "config.yaml"
        config_file.write_text(config_content)
        
        env_vars = {
            'LOKI_APP_NAME': 'Env Override Name',
            'LOKI_LLM_MODEL': 'env-model:latest',
            'LOKI_LLM_TEMPERATURE': '0.8'
        }
        
        with patch.dict(os.environ, env_vars):
            config = self.loader.load_config(str(config_file))
        
        assert config.app.name == "Env Override Name"
        assert config.llm.model == "env-model:latest"
        assert config.llm.temperature == 0.8
    
    def test_environment_variable_type_conversion(self):
        """Test that environment variables are properly type-converted."""
        env_vars = {
            'LOKI_APP_DEBUG': 'true',
            'LOKI_LLM_MAX_TOKENS': '2048',
            'LOKI_LLM_TEMPERATURE': '0.7',
            'LOKI_TOOLS_AUTO_DISCOVER_BUILTIN': 'false',
            'LOKI_PERFORMANCE_MAX_FILE_SIZE_MB': '50'
        }
        
        with patch.dict(os.environ, env_vars):
            config = self.loader.load_config()
        
        assert config.app.debug is True
        assert config.llm.max_tokens == 2048
        assert config.llm.temperature == 0.7
        assert config.tools.auto_discover_builtin is False
        assert config.performance.max_file_size_mb == 50
    
    def test_malformed_yaml_handling(self):
        """Test graceful handling of malformed YAML files."""
        malformed_content = """
app:
  name: "Test
  invalid: yaml: content
    missing: quotes and proper structure
"""
        config_file = Path(self.temp_dir) / "bad_config.yaml"
        config_file.write_text(malformed_content)
        
        with pytest.raises(ConfigurationError) as exc_info:
            self.loader.load_config(config_file=str(config_file))
        
        assert "Failed to parse YAML" in str(exc_info.value)
    
    def test_missing_config_file_fallback(self):
        """Test fallback to defaults when config file is missing."""
        non_existent_file = Path(self.temp_dir) / "nonexistent.yaml"
        
        # Should not raise error, should fall back to defaults
        config = self.loader.load_config(config_file=str(non_existent_file))
        
        assert isinstance(config, LokiCodeConfig)
        assert config.app.name == "Loki Code"  # Default value
    
    def test_partial_config_merge(self):
        """Test that partial config files merge with defaults."""
        partial_config = """
llm:
  temperature: 0.9
  
ui:
  theme: "dark"
"""
        config_file = Path(self.temp_dir) / "partial.yaml"
        config_file.write_text(partial_config)
        
        config = self.loader.load_config(str(config_file))
        
        # Changed values
        assert config.llm.temperature == 0.9
        assert config.ui.theme.value == "dark"
        
        # Default values should still be present
        assert config.app.name == "Loki Code"
        assert config.llm.provider.value == "ollama"
        assert config.tools.auto_discover_builtin is True
    
    def test_config_file_discovery(self):
        """Test automatic discovery of config files in multiple locations."""
        # Create config file in temp directory
        config_content = """
app:
  name: "Discovered Config"
"""
        config_file = Path(self.temp_dir) / "loki_code.yaml"
        config_file.write_text(config_content)
        
        # Mock the discovery paths to include our temp directory
        discovery_paths = [
            str(Path(self.temp_dir) / "loki_code.yaml"),
            str(Path(self.temp_dir) / "config.yaml"),
            "./loki_code.yaml"
        ]
        
        with patch.object(self.loader, '_get_discovery_paths', return_value=discovery_paths):
            config = self.loader.load_config()
        
        assert config.app.name == "Discovered Config"
    
    def test_deep_merge_functionality(self):
        """Test deep merging of nested configuration structures."""
        config_content = """
llm:
  model: "custom-model"
  # temperature not specified, should use default
  
tools:
  timeout_seconds: 60
  # Other tool settings should use defaults
  
performance:
  max_file_size_mb: 200
  # Other performance settings should use defaults
"""
        config_file = Path(self.temp_dir) / "config.yaml"
        config_file.write_text(config_content)
        
        config = self.loader.load_config(str(config_file))
        
        # Custom values
        assert config.llm.model == "custom-model"
        assert config.tools.timeout_seconds == 60
        assert config.performance.max_file_size_mb == 200
        
        # Default values should still be present
        assert config.llm.temperature == 0.7  # Default
        assert config.performance.cleanup_interval_seconds == 300  # Default
    
    def test_invalid_environment_variable_handling(self):
        """Test handling of invalid environment variable values."""
        env_vars = {
            'LOKI_LLM_TEMPERATURE': 'not_a_number',
            'LOKI_APP_DEBUG': 'not_a_boolean',
            'LOKI_TOOLS_ENABLED': 'invalid_json'
        }
        
        with patch.dict(os.environ, env_vars):
            # Should not crash, should fall back to defaults or raise clear error
            with pytest.raises(ConfigurationError) as exc_info:
                self.loader.load_config()
            
            assert "environment variable" in str(exc_info.value).lower()


class TestConfigLoaderFunctions:
    """Test the module-level configuration functions."""
    
    def teardown_method(self):
        """Clear cached config after each test."""
        if hasattr(get_config, '_cached_config'):
            delattr(get_config, '_cached_config')
    
    def test_load_config_function(self):
        """Test the load_config module function."""
        config = load_config()
        assert isinstance(config, LokiCodeConfig)
    
    def test_get_config_caching(self):
        """Test that get_config caches the configuration."""
        config1 = get_config()
        config2 = get_config()
        
        # Should return the same instance (cached)
        assert config1 is config2
    
    def test_reload_config_clears_cache(self):
        """Test that reload_config clears the cache."""
        config1 = get_config()
        
        # Reload should clear cache
        config2 = reload_config()
        
        # Should be different instances
        assert config1 is not config2
        assert isinstance(config2, LokiCodeConfig)
    
    def test_validate_config_file_valid(self):
        """Test validation of a valid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
app:
  name: "Valid Config"
  
llm:
  model: "test-model"
""")
            config_file = f.name
        
        try:
            # Should not raise exception
            is_valid, error_msg = validate_config_file(config_file)
            assert is_valid is True
            assert error_msg is None
        finally:
            os.unlink(config_file)
    
    def test_validate_config_file_invalid(self):
        """Test validation of an invalid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
invalid: yaml: content
  missing: proper structure
""")
            config_file = f.name
        
        try:
            is_valid, error_msg = validate_config_file(config_file)
            assert is_valid is False
            assert error_msg is not None
        finally:
            os.unlink(config_file)
    
    def test_validate_nonexistent_config_file(self):
        """Test validation of a nonexistent config file."""
        is_valid, error_msg = validate_config_file("/nonexistent/config.yaml")
        assert is_valid is False
        assert "not found" in error_msg or "does not exist" in error_msg


class TestConfigurationError:
    """Test the ConfigurationError exception."""
    
    def test_configuration_error_creation(self):
        """Test creating ConfigurationError with message and details."""
        error = ConfigurationError("Test error", {"key": "value"})
        
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}
    
    def test_configuration_error_without_details(self):
        """Test creating ConfigurationError without details."""
        error = ConfigurationError("Test error")
        
        assert str(error) == "Test error"
        assert error.details == {}


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def teardown_method(self):
        """Clear cached config after each test."""
        if hasattr(get_config, '_cached_config'):
            delattr(get_config, '_cached_config')
    
    def test_end_to_end_config_loading(self):
        """Test complete config loading pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text("""
app:
  name: "Integration Test"
  debug: true
  
llm:
  provider: "ollama"
  model: "integration-model"
  temperature: 0.6
  
tools:
  timeout_seconds: 45
  
ui:
  theme: "light"
  
performance:
  max_file_size_mb: 150
""")
            
            # Set environment overrides
            env_vars = {
                'LOKI_APP_DEBUG': 'false',
                'LOKI_LLM_TEMPERATURE': '0.8'
            }
            
            with patch.dict(os.environ, env_vars):
                config = load_config(str(config_file))
            
            # File values
            assert config.app.name == "Integration Test"
            assert config.llm.model == "integration-model"
            assert config.tools.timeout_seconds == 45
            assert config.ui.theme.value == "light"
            assert config.performance.max_file_size_mb == 150
            
            # Environment overrides
            assert config.app.debug is False  # Overridden
            assert config.llm.temperature == 0.8  # Overridden
            
            # Defaults
            assert config.llm.provider.value == "ollama"  # From file but matches default
    
    def test_config_loading_priority_order(self):
        """Test that configuration priority works: CLI > ENV > FILE > DEFAULTS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text("""
app:
  name: "File Name"
  debug: true
  
llm:
  model: "file-model"
  temperature: 0.3
""")
            
            # Set environment variables
            env_vars = {
                'LOKI_APP_NAME': 'Env Name',
                'LOKI_LLM_MODEL': 'env-model'
            }
            
            with patch.dict(os.environ, env_vars):
                config = load_config(str(config_file))
            
            # Environment should override file
            assert config.app.name == "Env Name"  # ENV > FILE
            assert config.llm.model == "env-model"  # ENV > FILE
            
            # File should override defaults
            assert config.app.debug is True  # FILE > DEFAULT
            assert config.llm.temperature == 0.3  # FILE > DEFAULT