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


class ToolDiscoveryConfig(BaseModel):
    """Tool discovery configuration."""
    
    auto_discover_builtin: bool = Field(default=True, description="Automatically discover built-in tools")
    plugin_directories: List[str] = Field(default=[], description="Additional directories to scan for plugin tools")
    scan_on_startup: bool = Field(default=True, description="Scan for tools during application startup")
    rescan_interval: int = Field(default=300, ge=60, le=3600, description="Rescan interval in seconds")


class ToolExecutionConfig(BaseModel):
    """Tool execution configuration."""
    
    default_timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0, description="Default timeout for tool execution")
    max_concurrent_tools: int = Field(default=3, ge=1, le=20, description="Maximum number of tools executing concurrently")
    retry_failed_executions: bool = Field(default=False, description="Whether to retry failed tool executions")
    max_retries: int = Field(default=2, ge=1, le=10, description="Maximum number of retries")
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")
    enable_execution_tracking: bool = Field(default=True, description="Track tool execution history and metrics")
    enable_performance_monitoring: bool = Field(default=True, description="Monitor tool performance")
    max_execution_history: int = Field(default=1000, ge=100, le=10000, description="Maximum execution records to keep")


class ToolSecurityConfig(BaseModel):
    """Tool security configuration."""
    
    allowed_paths: List[str] = Field(
        default=["./", "~/projects/", "~/Documents/"],
        description="Paths where tools are allowed to operate"
    )
    restricted_paths: List[str] = Field(
        default=["/etc", "/usr/bin", "/System", "/Windows", "/private"],
        description="Paths that tools cannot access"
    )
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size for tool operations")
    require_confirmation_for: List[str] = Field(
        default=["dangerous"],
        description="Security levels requiring confirmation"
    )
    
    @field_validator('allowed_paths', 'restricted_paths')
    @classmethod
    def expand_paths(cls, v):
        """Expand user home directory in paths."""
        return [str(Path(path).expanduser()) for path in v]


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) integration configuration."""
    
    enabled: bool = Field(default=False, description="Enable MCP tool integration")
    auto_discover_servers: bool = Field(default=False, description="Automatically discover MCP servers")
    trusted_servers: List[str] = Field(default=[], description="List of trusted MCP server URLs")
    server_timeout_seconds: int = Field(default=10, ge=1, le=60, description="Timeout for MCP server communication")
    max_tools_per_server: int = Field(default=50, ge=1, le=500, description="Maximum tools to load from each server")


class FileReaderConfig(BaseModel):
    """File reader tool configuration."""
    
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum file size for file reader tool")
    default_analysis_level: str = Field(default="standard", description="Default analysis level")
    supported_encodings: List[str] = Field(
        default=["utf-8", "latin-1", "cp1252"],
        description="Supported file encodings"
    )
    fallback_encoding: str = Field(default="utf-8", description="Fallback encoding when detection fails")
    
    @field_validator('default_analysis_level')
    @classmethod
    def validate_analysis_level(cls, v):
        """Validate analysis level."""
        allowed_levels = ['minimal', 'standard', 'detailed', 'comprehensive']
        if v not in allowed_levels:
            raise ValueError(f'analysis_level must be one of: {", ".join(allowed_levels)}')
        return v


class FileWriterConfig(BaseModel):
    """File writer tool configuration."""
    
    create_backups: bool = Field(default=True, description="Create backups before modifying files")
    backup_dir: str = Field(default="~/.loki-code/backups", description="Directory for backup files")
    max_backups: int = Field(default=50, ge=1, le=1000, description="Maximum number of backups")
    atomic_writes: bool = Field(default=True, description="Use atomic writes for safety")
    
    @field_validator('backup_dir')
    @classmethod
    def expand_backup_dir(cls, v):
        """Expand user home directory in backup path."""
        return str(Path(v).expanduser())


class CodeAnalyzerConfig(BaseModel):
    """Code analyzer tool configuration."""
    
    cache_analysis_results: bool = Field(default=True, description="Cache analysis results for performance")
    cache_ttl_seconds: int = Field(default=1800, ge=300, le=86400, description="Cache TTL")
    max_analysis_depth: int = Field(default=10, ge=1, le=50, description="Maximum directory traversal depth")


class ToolPerformanceConfig(BaseModel):
    """Tool performance monitoring configuration."""
    
    slow_execution_threshold_seconds: float = Field(default=5.0, ge=1.0, description="Threshold for slow execution warnings")
    very_slow_execution_threshold_seconds: float = Field(default=30.0, ge=10.0, description="Threshold for very slow execution warnings")
    memory_limit_mb: int = Field(default=512, ge=64, le=4096, description="Memory limit for tool execution")


class ToolRegistryConfig(BaseModel):
    """Tool registry configuration."""
    
    enable_tool_discovery_cache: bool = Field(default=True, description="Cache discovered tools")
    tool_discovery_cache_ttl: int = Field(default=3600, ge=300, le=86400, description="Cache TTL for tool discovery")
    validate_tools_on_startup: bool = Field(default=True, description="Validate all registered tools during startup")
    auto_register_builtin_tools: bool = Field(default=True, description="Automatically register built-in tools")


class ToolsConfig(BaseModel):
    """Comprehensive tool management configuration."""
    
    discovery: ToolDiscoveryConfig = Field(default_factory=ToolDiscoveryConfig)
    execution: ToolExecutionConfig = Field(default_factory=ToolExecutionConfig)
    security: ToolSecurityConfig = Field(default_factory=ToolSecurityConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    
    # Tool-specific configurations
    file_reader: FileReaderConfig = Field(default_factory=FileReaderConfig)
    file_writer: FileWriterConfig = Field(default_factory=FileWriterConfig)
    code_analyzer: CodeAnalyzerConfig = Field(default_factory=CodeAnalyzerConfig)
    
    # Legacy configurations (for backward compatibility)
    command_execution: CommandExecutionConfig = Field(default_factory=CommandExecutionConfig)
    
    # Performance and registry configurations
    performance: ToolPerformanceConfig = Field(default_factory=ToolPerformanceConfig)
    registry: ToolRegistryConfig = Field(default_factory=ToolRegistryConfig)


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


class PromptTemplateConfig(BaseModel):
    """Configuration for individual prompt templates."""
    
    enabled: bool = Field(default=True, description="Whether this template is enabled")
    personality: str = Field(default="helpful", description="Agent personality for this template")
    verbosity: str = Field(default="detailed", description="Verbosity level: concise, detailed, verbose")
    max_context_tokens: int = Field(default=4000, ge=1000, le=32000, description="Maximum context tokens")
    include_examples: bool = Field(default=True, description="Include usage examples in tool descriptions")
    focus_areas: List[str] = Field(default_factory=list, description="Specific focus areas for specialized templates")
    
    @field_validator('personality')
    @classmethod
    def validate_personality(cls, v):
        """Validate personality is supported."""
        allowed_personalities = ['helpful', 'concise', 'detailed', 'formal', 'casual', 'expert']
        if v.lower() not in allowed_personalities:
            raise ValueError(f'personality must be one of: {", ".join(allowed_personalities)}')
        return v.lower()
    
    @field_validator('verbosity')
    @classmethod
    def validate_verbosity(cls, v):
        """Validate verbosity level."""
        allowed_levels = ['concise', 'detailed', 'verbose']
        if v.lower() not in allowed_levels:
            raise ValueError(f'verbosity must be one of: {", ".join(allowed_levels)}')
        return v.lower()


class AgentConfig(BaseModel):
    """Configuration for the intelligent agent system."""
    
    # Core reasoning settings
    reasoning_strategy: str = Field(default="intelligent_react", description="Agent reasoning strategy")
    clarification_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for asking clarification")
    max_planning_depth: int = Field(default=5, ge=1, le=20, description="Maximum planning depth for complex tasks")
    max_execution_steps: int = Field(default=20, ge=1, le=100, description="Maximum execution steps per request")
    
    # Permission system settings
    permission_mode: str = Field(default="ask_permission", description="Permission mode: auto_grant, ask_permission, strict")
    auto_grant_safe_operations: bool = Field(default=True, description="Automatically grant safe operations")
    remember_session_choices: bool = Field(default=True, description="Remember permission choices for session")
    remember_permanent_choices: bool = Field(default=True, description="Remember permanent permission choices")
    
    # Safety system settings
    safety_mode: str = Field(default="strict", description="Safety mode: permissive, standard, strict")
    immutable_rules_enabled: bool = Field(default=True, description="Enable immutable safety rules")
    project_boundary_enforcement: bool = Field(default=True, description="Enforce project boundary restrictions")
    resource_limit_enforcement: bool = Field(default=True, description="Enforce resource limits")
    
    # Interaction settings
    explanation_level: str = Field(default="detailed", description="Level of explanation: minimal, standard, detailed, verbose")
    personality: str = Field(default="helpful", description="Agent personality: professional, friendly, helpful, concise, analytical")
    proactive_suggestions: bool = Field(default=True, description="Provide proactive suggestions")
    show_reasoning: bool = Field(default=True, description="Show agent reasoning process")
    show_progress: bool = Field(default=True, description="Show progress updates during execution")
    
    # Performance settings
    timeout_seconds: float = Field(default=300.0, ge=1.0, le=3600.0, description="Timeout for agent operations")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries for failed operations")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    
    @field_validator('reasoning_strategy')
    @classmethod
    def validate_reasoning_strategy(cls, v):
        """Validate reasoning strategy."""
        allowed_strategies = ['intelligent_react', 'plan_and_execute', 'conversational', 'tool_calling']
        if v not in allowed_strategies:
            raise ValueError(f'reasoning_strategy must be one of: {", ".join(allowed_strategies)}')
        return v
    
    @field_validator('permission_mode')
    @classmethod
    def validate_permission_mode(cls, v):
        """Validate permission mode."""
        allowed_modes = ['auto_grant', 'ask_permission', 'strict']
        if v not in allowed_modes:
            raise ValueError(f'permission_mode must be one of: {", ".join(allowed_modes)}')
        return v
    
    @field_validator('safety_mode')
    @classmethod
    def validate_safety_mode(cls, v):
        """Validate safety mode."""
        allowed_modes = ['permissive', 'standard', 'strict']
        if v not in allowed_modes:
            raise ValueError(f'safety_mode must be one of: {", ".join(allowed_modes)}')
        return v
    
    @field_validator('explanation_level')
    @classmethod
    def validate_explanation_level(cls, v):
        """Validate explanation level."""
        allowed_levels = ['minimal', 'standard', 'detailed', 'verbose']
        if v not in allowed_levels:
            raise ValueError(f'explanation_level must be one of: {", ".join(allowed_levels)}')
        return v
    
    @field_validator('personality')
    @classmethod
    def validate_personality(cls, v):
        """Validate personality type."""
        allowed_personalities = ['professional', 'friendly', 'helpful', 'concise', 'analytical']
        if v not in allowed_personalities:
            raise ValueError(f'personality must be one of: {", ".join(allowed_personalities)}')
        return v


class PromptsConfig(BaseModel):
    """Comprehensive prompt system configuration."""
    
    # Default settings
    default_template: str = Field(default="coding_agent", description="Default prompt template to use")
    max_context_tokens: int = Field(default=4000, ge=1000, le=32000, description="Default maximum context tokens")
    include_conversation_history: bool = Field(default=True, description="Include conversation history in prompts")
    max_history_entries: int = Field(default=10, ge=1, le=50, description="Maximum conversation history entries")
    
    # Context building settings
    auto_build_file_context: bool = Field(default=True, description="Automatically build file context when files are mentioned")
    auto_build_project_context: bool = Field(default=True, description="Automatically build project context")
    max_files_in_context: int = Field(default=5, ge=1, le=20, description="Maximum files to include in context")
    
    # Tool integration settings
    max_tools_in_description: int = Field(default=20, ge=5, le=100, description="Maximum tools to describe in prompts")
    include_tool_examples: bool = Field(default=True, description="Include tool usage examples")
    tool_call_format: str = Field(default="markdown_code", description="Format for tool calls: markdown_code, json, yaml")
    
    # Token management
    enable_token_estimation: bool = Field(default=True, description="Enable token count estimation")
    token_limit_behavior: str = Field(default="warn", description="Behavior when token limit exceeded: warn, truncate, error")
    context_compression_enabled: bool = Field(default=False, description="Enable context compression for large contexts")
    
    # Template-specific configurations
    templates: Dict[str, PromptTemplateConfig] = Field(
        default_factory=lambda: {
            "coding_agent": PromptTemplateConfig(
                personality="helpful",
                verbosity="detailed",
                max_context_tokens=3500
            ),
            "code_review": PromptTemplateConfig(
                personality="expert",
                verbosity="detailed",
                max_context_tokens=4000,
                focus_areas=["security", "performance", "maintainability"]
            ),
            "debugging": PromptTemplateConfig(
                personality="expert",
                verbosity="detailed",
                max_context_tokens=4000,
                focus_areas=["error_analysis", "root_cause", "solutions"]
            ),
            "file_analysis": PromptTemplateConfig(
                personality="detailed",
                verbosity="verbose",
                max_context_tokens=3500,
                focus_areas=["structure", "quality", "patterns"]
            ),
            "project_analysis": PromptTemplateConfig(
                personality="expert",
                verbosity="verbose", 
                max_context_tokens=4000,
                focus_areas=["architecture", "organization", "scalability"]
            )
        },
        description="Template-specific configurations"
    )
    
    @field_validator('default_template')
    @classmethod
    def validate_default_template(cls, v):
        """Validate default template name."""
        allowed_templates = ['coding_agent', 'code_review', 'debugging', 'file_analysis', 'project_analysis']
        if v not in allowed_templates:
            raise ValueError(f'default_template must be one of: {", ".join(allowed_templates)}')
        return v
    
    @field_validator('tool_call_format')
    @classmethod
    def validate_tool_call_format(cls, v):
        """Validate tool call format."""
        allowed_formats = ['markdown_code', 'json', 'yaml']
        if v not in allowed_formats:
            raise ValueError(f'tool_call_format must be one of: {", ".join(allowed_formats)}')
        return v
    
    @field_validator('token_limit_behavior')
    @classmethod
    def validate_token_limit_behavior(cls, v):
        """Validate token limit behavior."""
        allowed_behaviors = ['warn', 'truncate', 'error']
        if v not in allowed_behaviors:
            raise ValueError(f'token_limit_behavior must be one of: {", ".join(allowed_behaviors)}')
        return v


class LokiCodeConfig(BaseModel):
    """Main configuration model that combines all sections."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
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