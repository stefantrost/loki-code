"""
Safety-first error recovery and validation system for Loki Code agent.

This module implements immutable safety rules and intelligent error recovery
to ensure the agent never causes harm and can gracefully handle failures.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from pathlib import Path

from .permission_manager import ToolAction
from ...utils.logging import get_logger


class SafetyLevel(Enum):
    """Safety levels for different types of validation."""
    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Strategies for error recovery."""
    STOP_AND_ASK = "stop_and_ask"              # Stop and ask user for guidance
    SUGGEST_ALTERNATIVES = "suggest_alternatives" # Suggest alternative approaches
    PARTIAL_COMPLETION = "partial_completion"   # Complete what's possible
    RETRY_WITH_CHANGES = "retry_with_changes"   # Retry with modified approach
    GRACEFUL_DEGRADATION = "graceful_degradation" # Provide reduced functionality


@dataclass
class SafetyViolation:
    """Represents a safety rule violation."""
    rule_name: str
    severity: SafetyLevel
    message: str
    suggested_action: str = ""
    auto_recoverable: bool = False


@dataclass
class SafetyResult:
    """Result of safety validation."""
    approved: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def approved_safe(cls) -> "SafetyResult":
        """Create an approved safety result."""
        return cls(approved=True)
    
    @classmethod
    def denied(cls, violation: SafetyViolation) -> "SafetyResult":
        """Create a denied safety result."""
        return cls(
            approved=False,
            violations=[violation]
        )
    
    @classmethod
    def denied_with_message(cls, rule_name: str, message: str, 
                           severity: SafetyLevel = SafetyLevel.DANGEROUS) -> "SafetyResult":
        """Create a denied result with a simple message."""
        violation = SafetyViolation(
            rule_name=rule_name,
            severity=severity,
            message=message
        )
        return cls.denied(violation)


@dataclass
class RecoveryPlan:
    """Plan for recovering from an error or failure."""
    strategy: RecoveryStrategy
    message: str
    alternatives: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    retry_possible: bool = False
    user_input_needed: bool = False


@dataclass
class TaskContext:
    """Context about the current task for safety evaluation."""
    project_path: Optional[str] = None
    current_file: Optional[str] = None
    target_files: List[str] = field(default_factory=list)
    operation_type: str = "unknown"
    user_intent: str = ""
    previous_errors: List[Exception] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Context about the project for boundary enforcement."""
    project_path: str
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.yaml', '.yml', '.json'
    })
    max_file_size_mb: int = 10
    max_files_per_operation: int = 100


@dataclass
class SafetyConfig:
    """Configuration for the safety system."""
    immutable_rules_enabled: bool = True
    project_boundary_enforcement: bool = True
    resource_limit_enforcement: bool = True
    path_traversal_protection: bool = True
    max_file_size_mb: int = 100
    max_files_per_operation: int = 50
    max_operation_time_seconds: int = 300
    
    # Paths that are never allowed
    forbidden_paths: Set[str] = field(default_factory=lambda: {
        '/etc', '/usr/bin', '/System', '/Windows', '/private',
        '/system32', '/boot', '/root'
    })
    
    # File patterns that are never allowed
    forbidden_patterns: Set[str] = field(default_factory=lambda: {
        '*.exe', '*.dll', '*.so', '*.dylib', '*.bin'
    })


class SafetyManager:
    """
    Enforces immutable safety rules and provides intelligent error recovery.
    
    This system ensures the agent never causes harm by:
    1. Enforcing immutable safety rules that cannot be overridden
    2. Validating all operations before execution
    3. Providing intelligent error recovery strategies
    4. Maintaining project boundary enforcement
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Immutable safety rules - these can never be disabled
        self.immutable_rules = self._initialize_immutable_rules()
        
        # Error recovery knowledge base
        self.recovery_patterns = self._initialize_recovery_patterns()
    
    def validate_action(self, action: ToolAction, context: Optional[TaskContext] = None) -> SafetyResult:
        """
        Validate an action against all safety rules.
        
        This is the main entry point for safety validation. It runs all
        safety checks and returns whether the action is safe to execute.
        """
        try:
            violations = []
            warnings = []
            
            # Run immutable safety rules first
            for rule_name, rule_func in self.immutable_rules.items():
                try:
                    violation = rule_func(action, context)
                    if violation:
                        violations.append(violation)
                        if violation.severity in [SafetyLevel.DANGEROUS, SafetyLevel.CRITICAL]:
                            # Critical violations stop evaluation immediately
                            return SafetyResult(approved=False, violations=violations)
                except Exception as e:
                    self.logger.error(f"Error in safety rule {rule_name}: {e}")
                    # If a safety rule fails, deny the action
                    violations.append(SafetyViolation(
                        rule_name=rule_name,
                        severity=SafetyLevel.CRITICAL,
                        message=f"Safety rule execution failed: {str(e)}"
                    ))
                    return SafetyResult(approved=False, violations=violations)
            
            # If we have any violations, deny
            if violations:
                return SafetyResult(approved=False, violations=violations, warnings=warnings)
            
            # All checks passed
            return SafetyResult.approved_safe()
            
        except Exception as e:
            self.logger.error(f"Safety validation error: {e}")
            # Fail safe - deny if validation itself fails
            return SafetyResult.denied_with_message(
                "validation_error",
                f"Safety validation failed: {str(e)}",
                SafetyLevel.CRITICAL
            )
    
    def _initialize_immutable_rules(self) -> Dict[str, callable]:
        """Initialize the immutable safety rules that can never be disabled."""
        return {
            "project_boundary": self._rule_project_boundary,
            "path_traversal": self._rule_path_traversal,
            "forbidden_paths": self._rule_forbidden_paths,
            "forbidden_patterns": self._rule_forbidden_patterns,
            "resource_limits": self._rule_resource_limits,
            "destructive_confirmation": self._rule_destructive_confirmation,
            "file_existence": self._rule_file_existence
        }
    
    def _rule_project_boundary(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Ensure operations stay within project boundaries."""
        if not self.config.project_boundary_enforcement:
            return None
        
        if not context or not context.project_path:
            # If no project context, be very restrictive
            if action.file_paths:
                return SafetyViolation(
                    rule_name="project_boundary",
                    severity=SafetyLevel.DANGEROUS,
                    message="No project context provided, cannot validate file operations",
                    suggested_action="Specify project path or working directory"
                )
            return None
        
        project_path = Path(context.project_path).resolve()
        
        for file_path in action.file_paths:
            try:
                abs_path = Path(file_path).resolve()
                # Check if file is within project directory
                abs_path.relative_to(project_path)
            except ValueError:
                return SafetyViolation(
                    rule_name="project_boundary",
                    severity=SafetyLevel.DANGEROUS,
                    message=f"File operation outside project directory: {file_path}",
                    suggested_action="Only operate on files within the project directory"
                )
        
        return None
    
    def _rule_path_traversal(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Prevent path traversal attacks."""
        if not self.config.path_traversal_protection:
            return None
        
        dangerous_patterns = ['../', '..\\\\', '%2e%2e%2f', '%2e%2e%5c']
        
        for file_path in action.file_paths:
            file_str = str(file_path).lower()
            for pattern in dangerous_patterns:
                if pattern in file_str:
                    return SafetyViolation(
                        rule_name="path_traversal",
                        severity=SafetyLevel.CRITICAL,
                        message=f"Path traversal attempt detected in: {file_path}",
                        suggested_action="Use absolute paths or validate file paths properly"
                    )
        
        return None
    
    def _rule_forbidden_paths(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Prevent access to forbidden system paths."""
        for file_path in action.file_paths:
            path_str = str(Path(file_path).resolve())
            
            for forbidden in self.config.forbidden_paths:
                if path_str.startswith(forbidden):
                    return SafetyViolation(
                        rule_name="forbidden_paths",
                        severity=SafetyLevel.CRITICAL,
                        message=f"Access to forbidden system path: {file_path}",
                        suggested_action="Only access files in allowed directories"
                    )
        
        return None
    
    def _rule_forbidden_patterns(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Prevent access to forbidden file patterns."""
        for file_path in action.file_paths:
            path = Path(file_path)
            
            for pattern in self.config.forbidden_patterns:
                if path.match(pattern):
                    return SafetyViolation(
                        rule_name="forbidden_patterns",
                        severity=SafetyLevel.DANGEROUS,
                        message=f"Access to forbidden file type: {file_path}",
                        suggested_action="Only access text and code files"
                    )
        
        return None
    
    def _rule_resource_limits(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Enforce resource limits."""
        if not self.config.resource_limit_enforcement:
            return None
        
        # Check file count limit
        if len(action.file_paths) > self.config.max_files_per_operation:
            return SafetyViolation(
                rule_name="resource_limits",
                severity=SafetyLevel.WARNING,
                message=f"Too many files in operation: {len(action.file_paths)} > {self.config.max_files_per_operation}",
                suggested_action="Process files in smaller batches"
            )
        
        # Check file sizes
        for file_path in action.file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    if size_mb > self.config.max_file_size_mb:
                        return SafetyViolation(
                            rule_name="resource_limits",
                            severity=SafetyLevel.WARNING,
                            message=f"File too large: {file_path} ({size_mb:.1f}MB > {self.config.max_file_size_mb}MB)",
                            suggested_action="Process smaller files or increase limits"
                        )
            except Exception:
                # If we can't check file size, allow it but log
                self.logger.warning(f"Could not check size of file: {file_path}")
        
        return None
    
    def _rule_destructive_confirmation(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Ensure destructive operations have proper confirmation."""
        if action.is_destructive and not action.has_confirmation:
            return SafetyViolation(
                rule_name="destructive_confirmation",
                severity=SafetyLevel.DANGEROUS,
                message="Destructive operation requires explicit confirmation",
                suggested_action="Confirm the operation before proceeding",
                auto_recoverable=True
            )
        
        return None
    
    def _rule_file_existence(self, action: ToolAction, context: Optional[TaskContext]) -> Optional[SafetyViolation]:
        """Validate file existence for read operations."""
        # Only check for read operations
        read_tools = {'file_reader', 'code_analyzer', 'file_inspector'}
        
        if action.tool_name in read_tools:
            for file_path in action.file_paths:
                if not Path(file_path).exists():
                    return SafetyViolation(
                        rule_name="file_existence",
                        severity=SafetyLevel.WARNING,
                        message=f"File does not exist: {file_path}",
                        suggested_action="Verify file path or create the file first",
                        auto_recoverable=True
                    )
        
        return None
    
    async def handle_error(self, error: Exception, context: TaskContext) -> RecoveryPlan:
        """
        Provide intelligent error recovery with safety priority.
        
        This method analyzes errors and provides recovery strategies that
        prioritize safety over functionality.
        """
        try:
            # Safety first - never make things worse
            if self._could_cause_harm(error, context):
                return RecoveryPlan(
                    strategy=RecoveryStrategy.STOP_AND_ASK,
                    message="I encountered an error that could be risky to auto-recover from. Please advise on how to proceed safely.",
                    user_input_needed=True
                )
            
            # Try pattern-based recovery
            for pattern, recovery_func in self.recovery_patterns.items():
                if self._error_matches_pattern(error, pattern):
                    return recovery_func(error, context)
            
            # Default graceful recovery
            return RecoveryPlan(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                message=f"I encountered an error ({type(error).__name__}: {str(error)}) but can provide partial assistance based on what I was able to analyze.",
                suggested_actions=["Review the error details", "Try a simpler approach", "Check file permissions"]
            )
            
        except Exception as recovery_error:
            self.logger.error(f"Error in error recovery: {recovery_error}")
            # Ultimate fallback
            return RecoveryPlan(
                strategy=RecoveryStrategy.STOP_AND_ASK,
                message="I encountered an error and my recovery system also failed. Please provide guidance.",
                user_input_needed=True
            )
    
    def _initialize_recovery_patterns(self) -> Dict[str, callable]:
        """Initialize error recovery patterns."""
        return {
            "file_not_found": self._recover_file_not_found,
            "permission_denied": self._recover_permission_denied,
            "invalid_syntax": self._recover_invalid_syntax,
            "encoding_error": self._recover_encoding_error,
            "resource_exhausted": self._recover_resource_exhausted
        }
    
    def _error_matches_pattern(self, error: Exception, pattern: str) -> bool:
        """Check if an error matches a recovery pattern."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        pattern_map = {
            "file_not_found": lambda: "filenotfound" in error_type or "no such file" in error_message,
            "permission_denied": lambda: "permission" in error_type or "permission denied" in error_message,
            "invalid_syntax": lambda: "syntax" in error_type or "invalid syntax" in error_message,
            "encoding_error": lambda: "encoding" in error_type or "decode" in error_message,
            "resource_exhausted": lambda: "memory" in error_message or "too large" in error_message
        }
        
        check_func = pattern_map.get(pattern)
        return check_func() if check_func else False
    
    def _recover_file_not_found(self, error: Exception, context: TaskContext) -> RecoveryPlan:
        """Recover from file not found errors."""
        error_message = str(error)
        
        # Try to extract filename from error
        filename = self._extract_filename_from_error(error_message)
        
        if filename and context.project_path:
            # Look for similar files
            similar_files = self._find_similar_files(filename, context.project_path)
            
            if similar_files:
                return RecoveryPlan(
                    strategy=RecoveryStrategy.SUGGEST_ALTERNATIVES,
                    message=f"File '{filename}' not found. Did you mean one of these?",
                    alternatives=similar_files[:5],
                    retry_possible=True
                )
        
        return RecoveryPlan(
            strategy=RecoveryStrategy.PARTIAL_COMPLETION,
            message="I couldn't find the specified file, but I can help you with other aspects of your request.",
            suggested_actions=[
                "Check the file path spelling",
                "Verify the file exists",
                "Use a file listing tool to find the correct path"
            ]
        )
    
    def _recover_permission_denied(self, error: Exception, context: TaskContext) -> RecoveryPlan:
        """Recover from permission errors."""
        return RecoveryPlan(
            strategy=RecoveryStrategy.SUGGEST_ALTERNATIVES,
            message="I don't have permission to access this file or directory.",
            suggested_actions=[
                "Check file permissions",
                "Try accessing a different file",
                "Use sudo if appropriate (be careful!)"
            ],
            user_input_needed=True
        )
    
    def _recover_invalid_syntax(self, error: Exception, context: TaskContext) -> RecoveryPlan:
        """Recover from syntax errors."""
        return RecoveryPlan(
            strategy=RecoveryStrategy.PARTIAL_COMPLETION,
            message="I found syntax errors in the code, but I can still help analyze the structure and provide suggestions.",
            suggested_actions=[
                "Fix syntax errors first",
                "Focus on specific functions or sections",
                "Use a linter to identify issues"
            ]
        )
    
    def _recover_encoding_error(self, error: Exception, context: TaskContext) -> RecoveryPlan:
        """Recover from encoding errors."""
        return RecoveryPlan(
            strategy=RecoveryStrategy.RETRY_WITH_CHANGES,
            message="I encountered a text encoding issue. I can try different encodings or focus on other files.",
            suggested_actions=[
                "Try opening the file with a different encoding",
                "Convert the file to UTF-8",
                "Skip binary or non-text files"
            ],
            retry_possible=True
        )
    
    def _recover_resource_exhausted(self, error: Exception, context: TaskContext) -> RecoveryPlan:
        """Recover from resource exhaustion."""
        return RecoveryPlan(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            message="The operation was too large for available resources. I can process smaller chunks or provide a summary.",
            suggested_actions=[
                "Process files in smaller batches",
                "Focus on specific files or sections",
                "Increase available resources if possible"
            ]
        )
    
    def _could_cause_harm(self, error: Exception, context: TaskContext) -> bool:
        """Determine if auto-recovery could cause harm."""
        # Conservative approach - if in doubt, ask user
        harmful_patterns = [
            "permission denied",
            "access is denied", 
            "operation not permitted",
            "file system",
            "disk space",
            "critical error"
        ]
        
        error_message = str(error).lower()
        return any(pattern in error_message for pattern in harmful_patterns)
    
    def _extract_filename_from_error(self, error_message: str) -> Optional[str]:
        """Extract filename from error message."""
        # Common patterns for file not found errors
        patterns = [
            r"'([^']+)'",  # 'filename'
            r'"([^"]+)"',  # "filename"
            r"file '([^']+)' not found",
            r"no such file or directory: '([^']+)'"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)
        
        return None
    
    def _find_similar_files(self, filename: str, project_path: str) -> List[str]:
        """Find files similar to the requested filename."""
        try:
            project_dir = Path(project_path)
            all_files = []
            
            # Get all files in project
            for file_path in project_dir.rglob("*"):
                if file_path.is_file():
                    all_files.append(str(file_path.relative_to(project_dir)))
            
            # Find files with similar names
            filename_lower = filename.lower()
            similar = []
            
            for file_path in all_files:
                file_name = Path(file_path).name.lower()
                
                # Exact match (case insensitive)
                if file_name == filename_lower:
                    similar.append(file_path)
                # Contains the filename
                elif filename_lower in file_name:
                    similar.append(file_path)
                # Filename contains part of the requested name
                elif any(part in file_name for part in filename_lower.split('.') if len(part) > 2):
                    similar.append(file_path)
            
            return similar[:10]  # Limit results
            
        except Exception as e:
            self.logger.warning(f"Error finding similar files: {e}")
            return []
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get a summary of safety system status."""
        return {
            "immutable_rules_count": len(self.immutable_rules),
            "project_boundary_enforcement": self.config.project_boundary_enforcement,
            "path_traversal_protection": self.config.path_traversal_protection,
            "resource_limit_enforcement": self.config.resource_limit_enforcement,
            "max_file_size_mb": self.config.max_file_size_mb,
            "max_files_per_operation": self.config.max_files_per_operation,
            "forbidden_paths_count": len(self.config.forbidden_paths),
            "recovery_patterns_count": len(self.recovery_patterns)
        }