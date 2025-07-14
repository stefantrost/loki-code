"""
Tool type definitions and schemas for Loki Code.

This module defines the core types, enums, and data structures used
throughout the tool system, designed for MCP compatibility from the start.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import json
import time

# Import config for default value extraction
try:
    from ..config import get_config
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False


def _get_config_value(path: str, default: Any) -> Any:
    """Helper to get configuration values with fallback."""
    if not _HAS_CONFIG:
        return default
    
    try:
        config = get_config()
        parts = path.split('.')
        value = config
        for part in parts:
            value = getattr(value, part, None)
            if value is None:
                return default
        return value
    except Exception:
        return default


class SecurityLevel(Enum):
    """Security levels for tool operations."""
    SAFE = "safe"               # Read-only, no side effects
    CAUTION = "caution"         # Limited writes, low risk
    MODERATE = "moderate"       # Writes files, limited scope
    DANGEROUS = "dangerous"     # System commands, broad changes
    CRITICAL = "critical"       # High-risk system operations


# Alias for compatibility with agent system
ToolSecurityLevel = SecurityLevel


class ToolCapability(Enum):
    """Capabilities that tools can declare."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_COMMAND = "execute_command"
    NETWORK_ACCESS = "network_access"
    SYSTEM_MODIFICATION = "system_modification"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    PROJECT_NAVIGATION = "project_navigation"
    VERSION_CONTROL = "version_control"
    PACKAGE_MANAGEMENT = "package_management"


class ToolStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    NEEDS_CONFIRMATION = "needs_confirmation"


class ConfirmationLevel(Enum):
    """Levels of confirmation required for tool operations."""
    NONE = "none"               # No confirmation needed
    PROMPT = "prompt"           # Simple yes/no prompt
    DETAILED = "detailed"       # Show detailed operation info
    EXPLICIT = "explicit"       # Require explicit command confirmation


@dataclass
class SafetySettings:
    """Safety settings for tool execution."""
    require_confirmation_for: List[SecurityLevel] = field(default_factory=lambda: [SecurityLevel.DANGEROUS])
    allowed_paths: List[str] = field(default_factory=lambda: _get_config_value("tools.allowed_paths", ["./"]))
    restricted_commands: List[str] = field(default_factory=lambda: _get_config_value("ui.dangerous_commands", []))
    max_file_size_mb: int = field(default_factory=lambda: _get_config_value("performance.max_file_size_mb", 100))
    max_output_length: int = field(default_factory=lambda: _get_config_value("performance.max_output_length", 10000))
    timeout_seconds: int = field(default_factory=lambda: _get_config_value("performance.tool_timeout_seconds", 30))
    dry_run_mode: bool = False


@dataclass
class ToolSchema:
    """Schema definition for a tool, MCP-compatible."""
    name: str
    description: str
    input_schema: Dict[str, Any]        # JSON Schema compatible
    output_schema: Dict[str, Any]       # JSON Schema compatible
    capabilities: List[ToolCapability]
    security_level: SecurityLevel
    confirmation_level: ConfirmationLevel = ConfirmationLevel.NONE
    mcp_compatible: bool = True
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for MCP compatibility."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "metadata": {
                "capabilities": [cap.value for cap in self.capabilities],
                "securityLevel": self.security_level.value,
                "confirmationLevel": self.confirmation_level.value,
                "mcpCompatible": self.mcp_compatible,
                "version": self.version,
                "tags": self.tags
            },
            "examples": self.examples
        }
    
    @classmethod
    def from_json_schema(cls, schema: Dict[str, Any]) -> 'ToolSchema':
        """Create ToolSchema from JSON schema (for MCP integration)."""
        metadata = schema.get("metadata", {})
        
        return cls(
            name=schema["name"],
            description=schema["description"],
            input_schema=schema.get("inputSchema", {}),
            output_schema=schema.get("outputSchema", {}),
            capabilities=[ToolCapability(cap) for cap in metadata.get("capabilities", [])],
            security_level=SecurityLevel(metadata.get("securityLevel", "safe")),
            confirmation_level=ConfirmationLevel(metadata.get("confirmationLevel", "none")),
            mcp_compatible=metadata.get("mcpCompatible", True),
            version=metadata.get("version", "1.0.0"),
            tags=metadata.get("tags", []),
            examples=schema.get("examples", [])
        )


@dataclass
class ToolContext:
    """Context information provided to tools during execution."""
    project_path: str
    user_id: Optional[str] = None
    session_id: str = ""
    safety_settings: SafetySettings = field(default_factory=SafetySettings)
    llm_provider: Optional[Any] = None  # BaseLLMProvider (avoid circular import)
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: str = ""
    dry_run: bool = False
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if not self.session_id:
            self.session_id = f"session_{int(time.time())}"
        if not self.working_directory:
            self.working_directory = self.project_path


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: Any
    message: str
    status: ToolStatus = ToolStatus.SUCCESS
    side_effects: List[str] = field(default_factory=list)
    needs_confirmation: bool = False
    suggested_next_actions: List[str] = field(default_factory=list)
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set status based on success if not explicitly provided."""
        if self.status == ToolStatus.SUCCESS and not self.success:
            self.status = ToolStatus.FAILURE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "message": self.message,
            "status": self.status.value,
            "side_effects": self.side_effects,
            "needs_confirmation": self.needs_confirmation,
            "suggested_next_actions": self.suggested_next_actions,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }
    
    @classmethod
    def success_result(
        cls, 
        output: Any, 
        message: str = "Operation completed successfully",
        **kwargs
    ) -> 'ToolResult':
        """Create a successful result."""
        return cls(
            success=True,
            output=output,
            message=message,
            status=ToolStatus.SUCCESS,
            **kwargs
        )
    
    @classmethod
    def failure_result(
        cls, 
        message: str, 
        output: Any = None,
        **kwargs
    ) -> 'ToolResult':
        """Create a failure result."""
        return cls(
            success=False,
            output=output,
            message=message,
            status=ToolStatus.FAILURE,
            **kwargs
        )
    
    @classmethod
    def confirmation_needed(
        cls, 
        message: str, 
        output: Any = None,
        **kwargs
    ) -> 'ToolResult':
        """Create a result requiring confirmation."""
        return cls(
            success=False,
            output=output,
            message=message,
            status=ToolStatus.NEEDS_CONFIRMATION,
            needs_confirmation=True,
            **kwargs
        )


@dataclass
class ToolCall:
    """Represents a call to a tool."""
    tool_name: str
    input_data: Any
    context: ToolContext
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Generate call ID if not provided."""
        if not self.call_id:
            self.call_id = f"call_{self.tool_name}_{int(self.timestamp)}"


@dataclass
class ToolExecution:
    """Complete record of a tool execution."""
    call: ToolCall
    result: ToolResult
    started_at: float
    completed_at: float
    
    @property
    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        return (self.completed_at - self.started_at) * 1000


class InputValidationSchema:
    """Helper class for creating JSON schemas for tool inputs."""
    
    @staticmethod
    def string_field(
        description: str, 
        required: bool = True,
        pattern: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a string field schema."""
        schema = {
            "type": "string",
            "description": description
        }
        if pattern:
            schema["pattern"] = pattern
        if min_length is not None:
            schema["minLength"] = min_length
        if max_length is not None:
            schema["maxLength"] = max_length
        return schema
    
    @staticmethod
    def file_path_field(description: str, required: bool = True) -> Dict[str, Any]:
        """Create a file path field schema."""
        return {
            "type": "string",
            "description": description,
            "pattern": r"^[^<>:\"|?*\x00-\x1f]*$"  # Basic path validation
        }
    
    @staticmethod
    def boolean_field(description: str, default: Optional[bool] = None) -> Dict[str, Any]:
        """Create a boolean field schema."""
        schema = {
            "type": "boolean",
            "description": description
        }
        if default is not None:
            schema["default"] = default
        return schema
    
    @staticmethod
    def integer_field(
        description: str,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create an integer field schema."""
        schema = {
            "type": "integer",
            "description": description
        }
        if minimum is not None:
            schema["minimum"] = minimum
        if maximum is not None:
            schema["maximum"] = maximum
        return schema
    
    @staticmethod
    def array_field(
        description: str,
        items_schema: Dict[str, Any],
        min_items: Optional[int] = None,
        max_items: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create an array field schema."""
        schema = {
            "type": "array",
            "description": description,
            "items": items_schema
        }
        if min_items is not None:
            schema["minItems"] = min_items
        if max_items is not None:
            schema["maxItems"] = max_items
        return schema
    
    @staticmethod
    def create_schema(
        properties: Dict[str, Dict[str, Any]],
        required: List[str] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a complete JSON schema."""
        schema = {
            "type": "object",
            "properties": properties
        }
        if description:
            schema["description"] = description
        if required:
            schema["required"] = required
        return schema


class OutputValidationSchema:
    """Helper class for creating JSON schemas for tool outputs."""
    
    @staticmethod
    def success_output_schema(
        data_schema: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Create schema for successful tool output."""
        properties = {
            "success": {"type": "boolean", "const": True},
            "output": data_schema or {"type": ["string", "object", "array", "null"]},
            "message": {"type": "string"},
            "status": {"type": "string", "enum": [s.value for s in ToolStatus]}
        }
        
        if include_metadata:
            properties.update({
                "side_effects": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggested_next_actions": {
                    "type": "array", 
                    "items": {"type": "string"}
                },
                "execution_time_ms": {"type": "number"},
                "metadata": {"type": "object"}
            })
        
        return {
            "type": "object",
            "properties": properties,
            "required": ["success", "output", "message", "status"]
        }


# Common schemas for reuse
COMMON_SCHEMAS = {
    "file_path": InputValidationSchema.file_path_field("Path to file"),
    "directory_path": InputValidationSchema.string_field("Path to directory"),
    "content": InputValidationSchema.string_field("Text content"),
    "recursive": InputValidationSchema.boolean_field("Recursive operation", False),
    "dry_run": InputValidationSchema.boolean_field("Preview changes without executing", False),
    "force": InputValidationSchema.boolean_field("Force operation without confirmation", False)
}