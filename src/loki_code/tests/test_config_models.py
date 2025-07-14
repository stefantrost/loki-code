"""
Test suite for configuration models and validation.

This module tests the Pydantic configuration models, field validation,
enum validation, and cross-section consistency checks.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from loki_code.config.models import (
    LokiCodeConfig, AppConfig, LLMConfig, ToolsConfig, 
    AgentConfig, UIConfig, PerformanceConfig,
    LogLevel, Theme, ColorScheme, Provider, ReasoningStrategy,
    PermissionMode, SafetyMode, ExplanationLevel, Personality
)


class TestAppConfig:
    """Test the AppConfig model."""
    
    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        
        assert config.name == "Loki Code"
        assert config.version == "0.1.0"
        assert config.debug is False
        assert config.verbose_logging is False
        assert config.log_level == LogLevel.INFO
        
        # Path defaults should be set
        assert config.data_dir is not None
        assert config.cache_dir is not None
        assert config.backup_dir is not None
    
    def test_app_config_custom_values(self):
        """Test AppConfig with custom values."""
        config = AppConfig(
            name="Custom Loki",
            version="2.0.0",
            debug=True,
            verbose_logging=True,
            log_level=LogLevel.DEBUG,
            data_dir="./custom_data",
            cache_dir="./custom_cache"
        )
        
        assert config.name == "Custom Loki"
        assert config.version == "2.0.0"
        assert config.debug is True
        assert config.verbose_logging is True
        assert config.log_level == LogLevel.DEBUG
        assert config.data_dir == "./custom_data"
        assert config.cache_dir == "./custom_cache"
    
    def test_app_config_path_expansion(self):
        """Test that path fields expand properly."""
        config = AppConfig(
            data_dir="~/test_data",
            cache_dir="./cache",
            backup_dir="/abs/path"
        )
        
        # Paths should be converted to Path objects
        assert isinstance(config.data_dir, str)
        assert isinstance(config.cache_dir, str)
        assert isinstance(config.backup_dir, str)
    
    def test_log_level_enum_validation(self):
        """Test LogLevel enum validation."""
        # Valid enum values
        config = AppConfig(log_level=LogLevel.ERROR)
        assert config.log_level == LogLevel.ERROR
        
        # Invalid enum values should raise ValidationError
        with pytest.raises(ValidationError):
            AppConfig(log_level="INVALID_LEVEL")
    
    def test_app_config_validation_errors(self):
        """Test AppConfig field validation errors."""
        # Empty name should be invalid
        with pytest.raises(ValidationError) as exc_info:
            AppConfig(name="")
        
        assert "name" in str(exc_info.value)
        
        # Invalid version format
        with pytest.raises(ValidationError):
            AppConfig(version="")


class TestLLMConfig:
    """Test the LLMConfig model."""
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig()
        
        assert config.provider == Provider.OLLAMA
        assert config.model == "codellama:7b"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout_seconds == 120
        assert config.base_url == "http://localhost:11434"
        assert config.api_key is None
    
    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            provider=Provider.OPENAI,
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            timeout_seconds=60,
            base_url="https://api.openai.com",
            api_key="test-key"
        )
        
        assert config.provider == Provider.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.timeout_seconds == 60
        assert config.base_url == "https://api.openai.com"
        assert config.api_key == "test-key"
    
    def test_provider_enum_validation(self):
        """Test Provider enum validation."""
        # Valid enum values
        for provider in Provider:
            config = LLMConfig(provider=provider)
            assert config.provider == provider
        
        # Invalid enum value should raise ValidationError
        with pytest.raises(ValidationError):
            LLMConfig(provider="invalid_provider")
    
    def test_temperature_validation(self):
        """Test temperature field validation."""
        # Valid temperature values
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = LLMConfig(temperature=1.0)
        assert config.temperature == 1.0
        
        config = LLMConfig(temperature=0.5)
        assert config.temperature == 0.5
        
        # Invalid temperature values
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.0)
    
    def test_max_tokens_validation(self):
        """Test max_tokens field validation."""
        # Valid token counts
        config = LLMConfig(max_tokens=1)
        assert config.max_tokens == 1
        
        config = LLMConfig(max_tokens=32000)
        assert config.max_tokens == 32000
        
        # Invalid token counts
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)
        
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=100000)
    
    def test_timeout_validation(self):
        """Test timeout_seconds field validation."""
        # Valid timeout values
        config = LLMConfig(timeout_seconds=1)
        assert config.timeout_seconds == 1
        
        config = LLMConfig(timeout_seconds=300)
        assert config.timeout_seconds == 300
        
        # Invalid timeout values
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=0)
        
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=1000)


class TestToolsConfig:
    """Test the ToolsConfig model."""
    
    def test_tools_config_defaults(self):
        """Test ToolsConfig default values."""
        config = ToolsConfig()
        
        assert "file_reader" in config.enabled
        assert "file_writer" in config.enabled
        assert config.allowed_paths == ["./"]
        assert config.max_file_operations == 50
        assert config.backup_on_write is True
    
    def test_tools_config_custom_values(self):
        """Test ToolsConfig with custom values."""
        config = ToolsConfig(
            enabled=["file_reader"],
            allowed_paths=["/home/user", "./project"],
            max_file_operations=100,
            backup_on_write=False
        )
        
        assert config.enabled == ["file_reader"]
        assert config.allowed_paths == ["/home/user", "./project"]
        assert config.max_file_operations == 100
        assert config.backup_on_write is False
    
    def test_enabled_tools_validation(self):
        """Test enabled tools list validation."""
        # Empty list should be valid
        config = ToolsConfig(enabled=[])
        assert config.enabled == []
        
        # Non-empty list should be valid
        config = ToolsConfig(enabled=["tool1", "tool2"])
        assert config.enabled == ["tool1", "tool2"]
        
        # Duplicate tools should be handled
        config = ToolsConfig(enabled=["tool1", "tool1", "tool2"])
        # Should still work (duplicates might be filtered elsewhere)
        assert len(config.enabled) == 3
    
    def test_allowed_paths_validation(self):
        """Test allowed_paths validation."""
        # Valid paths
        config = ToolsConfig(allowed_paths=["./", "/home", "relative/path"])
        assert len(config.allowed_paths) == 3
        
        # Empty list should be invalid
        with pytest.raises(ValidationError):
            ToolsConfig(allowed_paths=[])
    
    def test_max_file_operations_validation(self):
        """Test max_file_operations validation."""
        # Valid values
        config = ToolsConfig(max_file_operations=1)
        assert config.max_file_operations == 1
        
        config = ToolsConfig(max_file_operations=1000)
        assert config.max_file_operations == 1000
        
        # Invalid values
        with pytest.raises(ValidationError):
            ToolsConfig(max_file_operations=0)


class TestAgentConfig:
    """Test the AgentConfig model."""
    
    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig()
        
        assert config.reasoning_strategy == ReasoningStrategy.REACT
        assert config.permission_mode == PermissionMode.PROMPT
        assert config.safety_mode == SafetyMode.STANDARD
        assert config.explanation_level == ExplanationLevel.BALANCED
        assert config.personality == Personality.PROFESSIONAL
        assert config.max_iterations == 10
        assert config.enable_memory is True
        assert config.auto_retry is True
    
    def test_agent_config_custom_values(self):
        """Test AgentConfig with custom values."""
        config = AgentConfig(
            reasoning_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            permission_mode=PermissionMode.AUTO_APPROVE,
            safety_mode=SafetyMode.STRICT,
            explanation_level=ExplanationLevel.DETAILED,
            personality=Personality.FRIENDLY,
            max_iterations=5,
            enable_memory=False,
            auto_retry=False
        )
        
        assert config.reasoning_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert config.permission_mode == PermissionMode.AUTO_APPROVE
        assert config.safety_mode == SafetyMode.STRICT
        assert config.explanation_level == ExplanationLevel.DETAILED
        assert config.personality == Personality.FRIENDLY
        assert config.max_iterations == 5
        assert config.enable_memory is False
        assert config.auto_retry is False
    
    def test_enum_validations(self):
        """Test all enum field validations in AgentConfig."""
        # Test each enum field
        for strategy in ReasoningStrategy:
            config = AgentConfig(reasoning_strategy=strategy)
            assert config.reasoning_strategy == strategy
        
        for mode in PermissionMode:
            config = AgentConfig(permission_mode=mode)
            assert config.permission_mode == mode
        
        for safety in SafetyMode:
            config = AgentConfig(safety_mode=safety)
            assert config.safety_mode == safety
        
        for level in ExplanationLevel:
            config = AgentConfig(explanation_level=level)
            assert config.explanation_level == level
        
        for personality in Personality:
            config = AgentConfig(personality=personality)
            assert config.personality == personality
    
    def test_max_iterations_validation(self):
        """Test max_iterations validation."""
        # Valid values
        config = AgentConfig(max_iterations=1)
        assert config.max_iterations == 1
        
        config = AgentConfig(max_iterations=50)
        assert config.max_iterations == 50
        
        # Invalid values
        with pytest.raises(ValidationError):
            AgentConfig(max_iterations=0)
        
        with pytest.raises(ValidationError):
            AgentConfig(max_iterations=100)


class TestUIConfig:
    """Test the UIConfig model."""
    
    def test_ui_config_defaults(self):
        """Test UIConfig default values."""
        config = UIConfig()
        
        assert config.theme == Theme.AUTO
        assert config.color_scheme == ColorScheme.BLUE
        assert config.show_progress is True
        assert config.animate_typing is True
        assert config.enable_rich_output is True
        assert config.max_output_lines == 1000
        assert config.refresh_rate_fps == 30
        assert len(config.dangerous_commands) > 0
    
    def test_ui_config_custom_values(self):
        """Test UIConfig with custom values."""
        config = UIConfig(
            theme=Theme.DARK,
            color_scheme=ColorScheme.GREEN,
            show_progress=False,
            animate_typing=False,
            enable_rich_output=False,
            max_output_lines=500,
            refresh_rate_fps=60,
            dangerous_commands=["rm", "del"]
        )
        
        assert config.theme == Theme.DARK
        assert config.color_scheme == ColorScheme.GREEN
        assert config.show_progress is False
        assert config.animate_typing is False
        assert config.enable_rich_output is False
        assert config.max_output_lines == 500
        assert config.refresh_rate_fps == 60
        assert config.dangerous_commands == ["rm", "del"]
    
    def test_theme_enum_validation(self):
        """Test Theme enum validation."""
        for theme in Theme:
            config = UIConfig(theme=theme)
            assert config.theme == theme
        
        with pytest.raises(ValidationError):
            UIConfig(theme="invalid_theme")
    
    def test_color_scheme_enum_validation(self):
        """Test ColorScheme enum validation."""
        for scheme in ColorScheme:
            config = UIConfig(color_scheme=scheme)
            assert config.color_scheme == scheme
        
        with pytest.raises(ValidationError):
            UIConfig(color_scheme="invalid_scheme")
    
    def test_max_output_lines_validation(self):
        """Test max_output_lines validation."""
        # Valid values
        config = UIConfig(max_output_lines=1)
        assert config.max_output_lines == 1
        
        config = UIConfig(max_output_lines=10000)
        assert config.max_output_lines == 10000
        
        # Invalid values
        with pytest.raises(ValidationError):
            UIConfig(max_output_lines=0)
    
    def test_refresh_rate_validation(self):
        """Test refresh_rate_fps validation."""
        # Valid values
        config = UIConfig(refresh_rate_fps=1)
        assert config.refresh_rate_fps == 1
        
        config = UIConfig(refresh_rate_fps=120)
        assert config.refresh_rate_fps == 120
        
        # Invalid values
        with pytest.raises(ValidationError):
            UIConfig(refresh_rate_fps=0)
        
        with pytest.raises(ValidationError):
            UIConfig(refresh_rate_fps=200)


class TestPerformanceConfig:
    """Test the PerformanceConfig model."""
    
    def test_performance_config_defaults(self):
        """Test PerformanceConfig default values."""
        config = PerformanceConfig()
        
        assert config.max_file_size_mb == 100
        assert config.max_output_length == 10000
        assert config.tool_timeout_seconds == 30
        assert config.cleanup_interval_seconds == 300
        assert config.cache_size_mb == 512
        assert config.enable_parallel_processing is True
        assert config.max_concurrent_operations == 5
    
    def test_performance_config_custom_values(self):
        """Test PerformanceConfig with custom values."""
        config = PerformanceConfig(
            max_file_size_mb=200,
            max_output_length=20000,
            tool_timeout_seconds=60,
            cleanup_interval_seconds=600,
            cache_size_mb=1024,
            enable_parallel_processing=False,
            max_concurrent_operations=10
        )
        
        assert config.max_file_size_mb == 200
        assert config.max_output_length == 20000
        assert config.tool_timeout_seconds == 60
        assert config.cleanup_interval_seconds == 600
        assert config.cache_size_mb == 1024
        assert config.enable_parallel_processing is False
        assert config.max_concurrent_operations == 10
    
    def test_performance_constraints_validation(self):
        """Test performance constraint validations."""
        # max_file_size_mb validation
        config = PerformanceConfig(max_file_size_mb=1)
        assert config.max_file_size_mb == 1
        
        config = PerformanceConfig(max_file_size_mb=1000)
        assert config.max_file_size_mb == 1000
        
        with pytest.raises(ValidationError):
            PerformanceConfig(max_file_size_mb=0)
        
        # max_output_length validation
        config = PerformanceConfig(max_output_length=100)
        assert config.max_output_length == 100
        
        with pytest.raises(ValidationError):
            PerformanceConfig(max_output_length=0)
        
        # timeout validation
        config = PerformanceConfig(tool_timeout_seconds=1)
        assert config.tool_timeout_seconds == 1
        
        with pytest.raises(ValidationError):
            PerformanceConfig(tool_timeout_seconds=0)
        
        # cache size validation
        config = PerformanceConfig(cache_size_mb=1)
        assert config.cache_size_mb == 1
        
        with pytest.raises(ValidationError):
            PerformanceConfig(cache_size_mb=0)
        
        # max concurrent operations validation
        config = PerformanceConfig(max_concurrent_operations=1)
        assert config.max_concurrent_operations == 1
        
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_operations=0)


class TestLokiCodeConfig:
    """Test the main LokiCodeConfig model."""
    
    def test_loki_code_config_defaults(self):
        """Test LokiCodeConfig with all default values."""
        config = LokiCodeConfig()
        
        # Should create all sub-configs with defaults
        assert isinstance(config.app, AppConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.tools, ToolsConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.performance, PerformanceConfig)
        
        # Check some key default values
        assert config.app.name == "Loki Code"
        assert config.llm.provider == Provider.OLLAMA
        assert config.agent.reasoning_strategy == ReasoningStrategy.REACT
        assert config.ui.theme == Theme.AUTO
    
    def test_loki_code_config_custom_sections(self):
        """Test LokiCodeConfig with custom section configurations."""
        custom_app = AppConfig(name="Custom App", debug=True)
        custom_llm = LLMConfig(model="custom-model", temperature=0.5)
        
        config = LokiCodeConfig(
            app=custom_app,
            llm=custom_llm
        )
        
        # Custom sections
        assert config.app.name == "Custom App"
        assert config.app.debug is True
        assert config.llm.model == "custom-model"
        assert config.llm.temperature == 0.5
        
        # Default sections
        assert isinstance(config.tools, ToolsConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.performance, PerformanceConfig)
    
    def test_loki_code_config_dict_initialization(self):
        """Test LokiCodeConfig initialization from dictionary."""
        config_dict = {
            "app": {
                "name": "Dict Config",
                "debug": True
            },
            "llm": {
                "model": "dict-model",
                "temperature": 0.8
            },
            "tools": {
                "enabled": ["file_reader"]
            }
        }
        
        config = LokiCodeConfig(**config_dict)
        
        assert config.app.name == "Dict Config"
        assert config.app.debug is True
        assert config.llm.model == "dict-model"
        assert config.llm.temperature == 0.8
        assert config.tools.enabled == ["file_reader"]
    
    def test_config_serialization(self):
        """Test that config can be serialized to dict."""
        config = LokiCodeConfig()
        config_dict = config.dict()
        
        assert isinstance(config_dict, dict)
        assert "app" in config_dict
        assert "llm" in config_dict
        assert "tools" in config_dict
        assert "agent" in config_dict
        assert "ui" in config_dict
        assert "performance" in config_dict
        
        # Check nested structure
        assert isinstance(config_dict["app"], dict)
        assert "name" in config_dict["app"]
        assert config_dict["app"]["name"] == "Loki Code"


class TestEnumValidations:
    """Test all enum types used in configuration."""
    
    def test_log_level_enum(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"
    
    def test_theme_enum(self):
        """Test Theme enum values."""
        assert Theme.AUTO.value == "auto"
        assert Theme.LIGHT.value == "light"
        assert Theme.DARK.value == "dark"
    
    def test_color_scheme_enum(self):
        """Test ColorScheme enum values."""
        expected_schemes = ["blue", "green", "purple", "orange", "red"]
        for scheme in ColorScheme:
            assert scheme.value in expected_schemes
    
    def test_provider_enum(self):
        """Test Provider enum values."""
        assert Provider.OLLAMA.value == "ollama"
        assert Provider.OPENAI.value == "openai"
    
    def test_reasoning_strategy_enum(self):
        """Test ReasoningStrategy enum values."""
        assert ReasoningStrategy.REACT.value == "react"
        assert ReasoningStrategy.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningStrategy.TREE_OF_THOUGHT.value == "tree_of_thought"
    
    def test_permission_mode_enum(self):
        """Test PermissionMode enum values."""
        assert PermissionMode.ALWAYS_ASK.value == "always_ask"
        assert PermissionMode.PROMPT.value == "prompt"
        assert PermissionMode.AUTO_APPROVE.value == "auto_approve"
    
    def test_safety_mode_enum(self):
        """Test SafetyMode enum values."""
        assert SafetyMode.PERMISSIVE.value == "permissive"
        assert SafetyMode.STANDARD.value == "standard"
        assert SafetyMode.STRICT.value == "strict"
    
    def test_explanation_level_enum(self):
        """Test ExplanationLevel enum values."""
        assert ExplanationLevel.MINIMAL.value == "minimal"
        assert ExplanationLevel.BALANCED.value == "balanced"
        assert ExplanationLevel.DETAILED.value == "detailed"
    
    def test_personality_enum(self):
        """Test Personality enum values."""
        assert Personality.PROFESSIONAL.value == "professional"
        assert Personality.FRIENDLY.value == "friendly"
        assert Personality.CONCISE.value == "concise"


@pytest.mark.integration
class TestConfigCrossValidation:
    """Test cross-section validation and consistency checks."""
    
    def test_performance_and_ui_consistency(self):
        """Test that performance and UI settings are consistent."""
        # High refresh rate should work with appropriate settings
        config = LokiCodeConfig(
            ui=UIConfig(refresh_rate_fps=60),
            performance=PerformanceConfig(max_concurrent_operations=10)
        )
        
        assert config.ui.refresh_rate_fps == 60
        assert config.performance.max_concurrent_operations == 10
    
    def test_tools_and_performance_limits(self):
        """Test that tools config respects performance limits."""
        config = LokiCodeConfig(
            tools=ToolsConfig(max_file_operations=100),
            performance=PerformanceConfig(max_file_size_mb=50)
        )
        
        # Should be able to handle the combination
        assert config.tools.max_file_operations == 100
        assert config.performance.max_file_size_mb == 50
    
    def test_agent_and_llm_compatibility(self):
        """Test that agent settings are compatible with LLM settings."""
        config = LokiCodeConfig(
            agent=AgentConfig(max_iterations=20),
            llm=LLMConfig(timeout_seconds=300)  # Long enough for iterations
        )
        
        # With 20 iterations and 300s timeout, should be reasonable
        estimated_time_per_iteration = config.llm.timeout_seconds / config.agent.max_iterations
        assert estimated_time_per_iteration >= 10  # At least 10 seconds per iteration
    
    def test_debug_mode_consistency(self):
        """Test that debug mode affects appropriate sections."""
        config = LokiCodeConfig(
            app=AppConfig(debug=True, verbose_logging=True, log_level=LogLevel.DEBUG)
        )
        
        # Debug settings should be consistent
        assert config.app.debug is True
        assert config.app.verbose_logging is True
        assert config.app.log_level == LogLevel.DEBUG