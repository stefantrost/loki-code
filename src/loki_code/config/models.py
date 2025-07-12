"""
Pydantic models for Loki Code configuration validation.

This module defines the structure and validation rules for all configuration
sections used by Loki Code. Each model corresponds to a section in the YAML
configuration files.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import os


class AppConfig(BaseModel):
    """Application-level configuration settings."""
    
    name: str = Field(default="Loki Code", description="Application display name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: str = Field(default="~/.loki-code", description="Application data directory")
    cache_dir: str = Field(default="~/.loki-code/cache", description="Cache directory")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is one of the allowed values."""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'log_level must be one of: {", ".join(allowed_levels)}')
        return v.upper()
    
    @field_validator('data_dir', 'cache_dir')
    @classmethod
    def expand_path(cls, v):
        """Expand user home directory in paths."""
        return str(Path(v).expanduser())


class LLMConfig(BaseModel):
    """Large Language Model provider configuration."""
    
    provider: str = Field(default="ollama", description="LLM provider")
    model: str = Field(default="codellama:7b", description="Model name")
    base_url: str = Field(default="http://localhost:11434", description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API key if required")
    
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="Maximum tokens per response")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Response randomness")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model name")
    
    context_window: int = Field(default=4096, ge=512, le=128000, description="Context window size")
    preserve_context: bool = Field(default=True, description="Preserve conversation context")
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        """Validate provider is supported."""
        allowed_providers = ['ollama', 'openai', 'anthropic', 'local']
        if v.lower() not in allowed_providers:
            raise ValueError(f'provider must be one of: {", ".join(allowed_providers)}')
        return v.lower()


class FileOperationsConfig(BaseModel):
    """File operation settings."""
    
    max_file_size_mb: int = Field(default=10, ge=1, le=1000, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".md", ".txt", ".yaml", ".yml", ".json"],
        description="Allowed file extensions"
    )
    create_backups: bool = Field(default=True, description="Create backups before modifications")
    backup_dir: str = Field(default="~/.loki-code/backups", description="Backup directory")
    max_backups: int = Field(default=50, ge=1, le=1000, description="Maximum number of backups")
    
    @field_validator('backup_dir')
    @classmethod
    def expand_backup_dir(cls, v):
        """Expand user home directory in backup path."""
        return str(Path(v).expanduser())
    
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


class CommandExecutionConfig(BaseModel):
    """Command execution settings."""
    
    enabled: bool = Field(default=False, description="Enable command execution")
    timeout_seconds: int = Field(default=30, ge=1, le=600, description="Command timeout")
    allowed_commands: List[str] = Field(default=[], description="Whitelist of allowed commands")
    forbidden_commands: List[str] = Field(
        default=["rm -rf", "sudo rm", "format", "del /q", "shutdown", "reboot"],
        description="Blacklist of forbidden commands"
    )


class ToolsConfig(BaseModel):
    """Tools configuration."""
    
    enabled: List[str] = Field(
        default=["file_reader", "file_writer", "directory_lister"],
        description="List of enabled tools"
    )
    file_operations: FileOperationsConfig = Field(default_factory=FileOperationsConfig)
    command_execution: CommandExecutionConfig = Field(default_factory=CommandExecutionConfig)


class UIConfig(BaseModel):
    """User interface configuration."""
    
    theme: str = Field(default="default", description="UI theme")
    color_scheme: str = Field(default="auto", description="Color scheme")
    show_timestamps: bool = Field(default=True, description="Show timestamps in output")
    show_token_usage: bool = Field(default=True, description="Display token usage statistics")
    progress_bar: bool = Field(default=True, description="Show progress bars")
    
    max_history_items: int = Field(default=100, ge=1, le=10000, description="Maximum history items")
    save_session: bool = Field(default=True, description="Save session between restarts")
    auto_save_interval: int = Field(default=300, ge=30, le=3600, description="Auto-save interval in seconds")
    
    max_output_lines: int = Field(default=1000, ge=10, le=100000, description="Maximum output lines")
    wrap_long_lines: bool = Field(default=True, description="Wrap long lines in output")
    syntax_highlighting: bool = Field(default=True, description="Enable syntax highlighting")
    
    confirm_destructive_operations: bool = Field(default=True, description="Confirm destructive operations")
    auto_complete: bool = Field(default=True, description="Enable auto-completion")
    keyboard_shortcuts: bool = Field(default=True, description="Enable keyboard shortcuts")
    
    @field_validator('theme')
    @classmethod
    def validate_theme(cls, v):
        """Validate theme is supported."""
        allowed_themes = ['default', 'dark', 'light']
        if v.lower() not in allowed_themes:
            raise ValueError(f'theme must be one of: {", ".join(allowed_themes)}')
        return v.lower()
    
    @field_validator('color_scheme')
    @classmethod
    def validate_color_scheme(cls, v):
        """Validate color scheme is supported."""
        allowed_schemes = ['auto', 'dark', 'light']
        if v.lower() not in allowed_schemes:
            raise ValueError(f'color_scheme must be one of: {", ".join(allowed_schemes)}')
        return v.lower()


class SafetyConfig(BaseModel):
    """Safety and security configuration."""
    
    restricted_paths: List[str] = Field(
        default=["/etc", "/usr/bin", "/system32", "/Windows", "/System", "/private"],
        description="Paths that tools cannot access"
    )
    dangerous_commands: List[str] = Field(
        default=["rm -rf", "sudo rm", "format", "del /q", "shutdown", "halt", "reboot", "mkfs", "dd if="],
        description="Commands that are never allowed"
    )
    max_files_per_operation: int = Field(default=100, ge=1, le=10000, description="Maximum files per operation")
    max_directory_depth: int = Field(default=10, ge=1, le=100, description="Maximum directory traversal depth")
    require_confirmation_for: List[str] = Field(
        default=["file_deletion", "directory_deletion", "system_modification", "large_file_operations"],
        description="Operations requiring confirmation"
    )


class DevelopmentConfig(BaseModel):
    """Development and debugging configuration."""
    
    auto_reload: bool = Field(default=False, description="Auto-reload code changes")
    mock_llm: bool = Field(default=False, description="Use mock LLM responses")
    verbose_logging: bool = Field(default=False, description="Enable verbose debug logging")
    
    test_mode: bool = Field(default=False, description="Enable test mode behavior")
    fixture_dir: str = Field(default="./tests/fixtures", description="Test fixtures directory")
    
    profile_performance: bool = Field(default=False, description="Enable performance profiling")
    log_api_calls: bool = Field(default=False, description="Log all API calls")
    
    enable_hot_reload: bool = Field(default=False, description="Hot reload for UI components")
    debug_tools: bool = Field(default=False, description="Enable debug tools in UI")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    log_file: str = Field(default="~/.loki-code/logs/loki-code.log", description="Main log file location")
    max_log_size_mb: int = Field(default=10, ge=1, le=1000, description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, le=100, description="Number of backup log files")
    
    loggers: Dict[str, str] = Field(
        default={
            "loki_code.core": "INFO",
            "loki_code.llm": "INFO",
            "loki_code.tools": "INFO",
            "loki_code.ui": "WARNING",
            "loki_code.config": "INFO",
        },
        description="Log levels for different components"
    )
    
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Date format for logs")
    
    @field_validator('log_file')
    @classmethod
    def expand_log_file(cls, v):
        """Expand user home directory in log file path."""
        return str(Path(v).expanduser())


class PluginsConfig(BaseModel):
    """Plugin system configuration."""
    
    enabled: List[str] = Field(default=[], description="List of enabled plugins")
    plugin_dir: str = Field(default="~/.loki-code/plugins", description="Plugin directory")
    auto_update: bool = Field(default=False, description="Auto-update plugins")
    allow_remote_plugins: bool = Field(default=False, description="Allow remote plugins")
    verify_signatures: bool = Field(default=True, description="Verify plugin signatures")
    
    @field_validator('plugin_dir')
    @classmethod
    def expand_plugin_dir(cls, v):
        """Expand user home directory in plugin path."""
        return str(Path(v).expanduser())


class CacheConfig(BaseModel):
    """Cache configuration."""
    
    enabled: bool = Field(default=True, description="Enable caching system")
    ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Default cache TTL")
    max_cache_size_mb: int = Field(default=100, ge=1, le=10000, description="Maximum cache size")
    
    llm_responses: Dict[str, Union[bool, int]] = Field(
        default={"enabled": True, "ttl_seconds": 7200},
        description="LLM response cache settings"
    )
    file_analysis: Dict[str, Union[bool, int]] = Field(
        default={"enabled": True, "ttl_seconds": 1800},
        description="File analysis cache settings"
    )


class IntegrationsConfig(BaseModel):
    """External integrations configuration."""
    
    git: Dict[str, bool] = Field(
        default={"auto_detect": True, "show_status": True},
        description="Git integration settings"
    )
    vscode: Dict[str, bool] = Field(
        default={"integration": False},
        description="VS Code integration settings"
    )
    package_managers: Dict[str, bool] = Field(
        default={"auto_detect": True, "suggest_installs": True},
        description="Package manager integration settings"
    )


class LokiCodeConfig(BaseModel):
    """Main configuration model that combines all sections."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    
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