"""
Simplified Pydantic models for Loki Code configuration validation.

This module provides a streamlined configuration system that eliminates over-engineering
while preserving all functionality that is actually used in the codebase.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import os


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Theme(str, Enum):
    """Supported UI themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"


class ColorScheme(str, Enum):
    """Supported color schemes."""
    AUTO = "auto"
    DARK = "dark"
    LIGHT = "light"


class Provider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ReasoningStrategy(str, Enum):
    """Agent reasoning strategies."""
    INTELLIGENT_REACT = "intelligent_react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    CONVERSATIONAL = "conversational"
    TOOL_CALLING = "tool_calling"


class PermissionMode(str, Enum):
    """Permission modes."""
    AUTO_GRANT = "auto_grant"
    ASK_PERMISSION = "ask_permission"
    STRICT = "strict"


class SafetyMode(str, Enum):
    """Safety modes."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"


class ExplanationLevel(str, Enum):
    """Explanation levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    VERBOSE = "verbose"


class Personality(str, Enum):
    """Agent personalities."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    HELPFUL = "helpful"
    CONCISE = "concise"
    ANALYTICAL = "analytical"


class AppConfig(BaseModel):
    """Application-level configuration settings."""
    
    name: str = Field(default="Loki Code", description="Application display name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    data_dir: str = Field(default="~/.loki-code", description="Application data directory")
    cache_dir: str = Field(default="~/.loki-code/cache", description="Cache directory")
    
    # Development settings (merged from DevelopmentConfig)
    auto_reload: bool = Field(default=False, description="Auto-reload code changes")
    mock_llm: bool = Field(default=False, description="Use mock LLM responses")
    verbose_logging: bool = Field(default=False, description="Enable verbose debug logging")
    test_mode: bool = Field(default=False, description="Enable test mode behavior")
    
    # Logging settings (merged from LoggingConfig)
    log_file: str = Field(default="~/.loki-code/logs/loki-code.log", description="Main log file location")
    max_log_size_mb: int = Field(default=10, ge=1, le=1000, description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, le=100, description="Number of backup log files")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    @field_validator('data_dir', 'cache_dir', 'log_file')
    @classmethod
    def expand_path(cls, v):
        """Expand user home directory in paths."""
        return str(Path(v).expanduser())


# =============================================================================
# SINGLE SOURCE OF TRUTH FOR DEFAULT MODEL
# =============================================================================
# This is the ONLY place where the default model should be defined.
# All other code should reference this constant or use the configuration system.
# 
# To change the default model:
# 1. Update DEFAULT_MODEL below
# 2. Update configs/default.yaml if needed
# 3. Do NOT hardcode models anywhere else in the codebase
DEFAULT_MODEL = "qwen3:32b"

class LLMConfig(BaseModel):
    """Large Language Model provider configuration."""
    
    provider: Provider = Field(default=Provider.OLLAMA, description="LLM provider")
    model: str = Field(default=DEFAULT_MODEL, description="Model name")
    base_url: str = Field(default="http://localhost:11434", description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API key if required")
    
    timeout: int = Field(default=300, ge=1, le=600, description="Request timeout in seconds (5 minutes for model loading)")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="Maximum tokens per response")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Response randomness")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model name")
    
    context_window: int = Field(default=4096, ge=512, le=128000, description="Context window size")
    preserve_context: bool = Field(default=True, description="Preserve conversation context")
    
    # HTTP LLM Server configuration
    use_llm_server: bool = Field(default=False, description="Use HTTP LLM server instead of direct LLM")
    llm_server_url: str = Field(default="http://localhost:8765", description="LLM server base URL")
    llm_server_timeout: float = Field(default=180.0, ge=1.0, le=600.0, description="LLM server timeout in seconds")
    llm_server_retries: int = Field(default=3, ge=0, le=10, description="LLM server retry attempts")


class ToolsConfig(BaseModel):
    """Tool system configuration - flattened from 9 nested classes."""
    
    # Discovery settings
    auto_discover_builtin: bool = Field(default=True, description="Automatically discover built-in tools")
    plugin_directories: List[str] = Field(default=[], description="Additional directories for plugin tools")
    
    # Execution settings
    timeout_seconds: float = Field(default=180.0, ge=1.0, le=600.0, description="Default timeout for tool execution (3 minutes)")
    max_concurrent_tools: int = Field(default=3, ge=1, le=20, description="Maximum concurrent tool executions")
    max_retries: int = Field(default=2, ge=1, le=10, description="Maximum number of retries")
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")
    
    # Security settings
    allowed_paths: List[str] = Field(
        default=["./", "~/projects/", "~/Documents/"],
        description="Paths where tools are allowed to operate"
    )
    max_file_size_mb: int = Field(default=10, ge=1, le=1000, description="Maximum file size for operations")
    
    # File operations
    allowed_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".md", ".txt", ".yaml", ".yml", ".json"],
        description="Allowed file extensions"
    )
    create_backups: bool = Field(default=True, description="Create backups before modifications")
    backup_dir: str = Field(default="~/.loki-code/backups", description="Backup directory")
    max_backups: int = Field(default=50, ge=1, le=1000, description="Maximum number of backups")
    
    # Performance settings
    cache_analysis_results: bool = Field(default=True, description="Cache analysis results")
    cache_ttl_seconds: int = Field(default=1800, ge=300, le=86400, description="Cache TTL")
    slow_execution_threshold_seconds: float = Field(default=5.0, ge=1.0, description="Slow execution warning threshold")
    
    @field_validator('backup_dir')
    @classmethod
    def expand_backup_dir(cls, v):
        """Expand user home directory in backup path."""
        return str(Path(v).expanduser())
    
    @field_validator('allowed_paths')
    @classmethod
    def expand_paths(cls, v):
        """Expand user home directory in paths."""
        return [str(Path(path).expanduser()) for path in v]
    
    @field_validator('allowed_extensions')
    @classmethod
    def validate_extensions(cls, v):
        """Ensure all extensions start with a dot."""
        validated = []
        for ext in v:
            if not ext.startswith('.'):
                ext = '.' + ext
            validated.append(ext.lower())
        return validated


class AgentConfig(BaseModel):
    """Agent system configuration."""
    
    # Core reasoning settings
    reasoning_strategy: ReasoningStrategy = Field(default=ReasoningStrategy.INTELLIGENT_REACT, description="Agent reasoning strategy")
    clarification_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for asking clarification")
    max_planning_depth: int = Field(default=5, ge=1, le=20, description="Maximum planning depth")
    max_execution_steps: int = Field(default=20, ge=1, le=100, description="Maximum execution steps per request")
    
    # Permission system settings
    permission_mode: PermissionMode = Field(default=PermissionMode.ASK_PERMISSION, description="Permission mode")
    auto_grant_safe_operations: bool = Field(default=True, description="Automatically grant safe operations")
    remember_session_choices: bool = Field(default=True, description="Remember permission choices for session")
    
    # Safety system settings
    safety_mode: SafetyMode = Field(default=SafetyMode.STRICT, description="Safety mode")
    project_boundary_enforcement: bool = Field(default=True, description="Enforce project boundary restrictions")
    
    # Interaction settings
    explanation_level: ExplanationLevel = Field(default=ExplanationLevel.DETAILED, description="Level of explanation")
    personality: Personality = Field(default=Personality.HELPFUL, description="Agent personality")
    proactive_suggestions: bool = Field(default=True, description="Provide proactive suggestions")
    show_reasoning: bool = Field(default=True, description="Show agent reasoning process")
    show_progress: bool = Field(default=True, description="Show progress updates")
    
    # Performance settings
    timeout_seconds: float = Field(default=300.0, ge=1.0, le=3600.0, description="Timeout for agent operations")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries for failed operations")
    enable_caching: bool = Field(default=True, description="Enable response caching")


class UIConfig(BaseModel):
    """User interface configuration - merged UI, safety, and other settings."""
    
    # Theme and display
    theme: Theme = Field(default=Theme.DEFAULT, description="UI theme")
    color_scheme: ColorScheme = Field(default=ColorScheme.AUTO, description="Color scheme")
    show_timestamps: bool = Field(default=True, description="Show timestamps in output")
    show_token_usage: bool = Field(default=True, description="Display token usage statistics")
    progress_bar: bool = Field(default=True, description="Show progress bars")
    syntax_highlighting: bool = Field(default=True, description="Enable syntax highlighting")
    
    # Session and history
    max_history_items: int = Field(default=100, ge=1, le=10000, description="Maximum history items")
    save_session: bool = Field(default=True, description="Save session between restarts")
    auto_save_interval: int = Field(default=300, ge=30, le=3600, description="Auto-save interval in seconds")
    
    # Output settings
    max_output_lines: int = Field(default=1000, ge=10, le=100000, description="Maximum output lines")
    wrap_long_lines: bool = Field(default=True, description="Wrap long lines in output")
    
    # Interaction settings
    confirm_destructive_operations: bool = Field(default=True, description="Confirm destructive operations")
    auto_complete: bool = Field(default=True, description="Enable auto-completion")
    keyboard_shortcuts: bool = Field(default=True, description="Enable keyboard shortcuts")
    
    # Safety settings (merged from SafetyConfig)
    restricted_paths: List[str] = Field(
        default=["/etc", "/usr/bin", "/system32", "/Windows", "/System", "/private"],
        description="Paths that tools cannot access"
    )
    dangerous_commands: List[str] = Field(
        default=["rm -rf", "sudo rm", "format", "del /q", "shutdown", "halt", "reboot"],
        description="Commands that are never allowed"
    )
    max_files_per_operation: int = Field(default=100, ge=1, le=10000, description="Maximum files per operation")
    max_directory_depth: int = Field(default=10, ge=1, le=100, description="Maximum directory traversal depth")


class PerformanceConfig(BaseModel):
    """Performance and resource management configuration."""
    
    # Cache and cleanup intervals
    cache_ttl_seconds: int = Field(default=300, ge=60, le=86400, description="Default cache TTL")
    cleanup_interval_seconds: int = Field(default=300, ge=60, le=3600, description="Cleanup interval")
    max_history_entries: int = Field(default=100, ge=10, le=10000, description="Maximum history entries")
    
    # Timeout settings
    default_timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0, description="Default timeout")
    tool_timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0, description="Tool execution timeout")
    agent_timeout_seconds: float = Field(default=300.0, ge=10.0, le=3600.0, description="Agent operation timeout")
    
    # Resource limits
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size")
    max_output_length: int = Field(default=10000, ge=1000, le=100000, description="Maximum output length")
    max_concurrent_operations: int = Field(default=3, ge=1, le=20, description="Maximum concurrent operations")
    
    # Performance thresholds
    slow_execution_threshold_seconds: float = Field(default=5.0, ge=1.0, le=60.0, description="Slow execution warning threshold")
    memory_limit_mb: int = Field(default=512, ge=64, le=4096, description="Memory limit per operation")


class LokiCodeConfig(BaseModel):
    """Main configuration model - simplified from 25 classes to 6."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Allow additional fields for forward compatibility
    class Config:
        extra = "allow"
        validate_assignment = True
        
    @model_validator(mode='after')
    def validate_config_consistency(self):
        """Validate cross-section configuration consistency."""
        # Ensure cache directory is under data directory if both are specified
        if hasattr(self.app, 'data_dir') and hasattr(self.app, 'cache_dir'):
            data_dir = Path(self.app.data_dir)
            cache_dir = Path(self.app.cache_dir)
            
            # If cache_dir is not under data_dir, issue a warning (don't fail)
            try:
                cache_dir.relative_to(data_dir)
            except ValueError:
                # This is fine, just a recommendation
                pass
        
        return self