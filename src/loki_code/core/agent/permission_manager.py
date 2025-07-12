"""
Permission-based autonomy system for Loki Code agent.

This module implements a sophisticated permission system that allows the agent
to operate autonomously for safe operations while requesting user permission
for potentially risky or ambiguous actions.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from pathlib import Path

from ...tools.types import ToolSecurityLevel
from ...utils.logging import get_logger


class PermissionLevel(Enum):
    """Permission levels for different types of operations."""
    AUTO_GRANT = "auto_grant"       # Safe operations - no permission needed
    ASK_ONCE = "ask_once"          # Need permission for this specific action
    ASK_SESSION = "ask_session"    # Ask once per session for this type
    ALWAYS_ASK = "always_ask"      # Never auto-grant, always ask
    NEVER_ALLOW = "never_allow"    # Permanently blocked operations


class PermissionScope(Enum):
    """Scope of permission grants."""
    SPECIFIC = "specific"          # This exact action only
    SIMILAR = "similar"           # Similar actions (same tool, similar params)
    TOOL_TYPE = "tool_type"       # All actions of this tool type
    GLOBAL = "global"             # All actions globally


@dataclass
class ToolAction:
    """Represents a tool action that needs permission evaluation."""
    tool_name: str
    description: str
    input_data: Dict[str, Any]
    file_paths: List[str] = field(default_factory=list)
    security_level: ToolSecurityLevel = ToolSecurityLevel.SAFE
    is_destructive: bool = False
    has_confirmation: bool = False
    estimated_impact: str = "low"
    
    def get_action_key(self) -> str:
        """Get a key that identifies this type of action."""
        # Create a key based on tool and general parameters
        key_parts = [self.tool_name]
        
        # Add security level
        key_parts.append(self.security_level.value)
        
        # Add destructive flag
        if self.is_destructive:
            key_parts.append("destructive")
        
        return ":".join(key_parts)
    
    def get_similarity_key(self) -> str:
        """Get a key for similar actions."""
        return f"{self.tool_name}:{self.security_level.value}"


@dataclass
class PermissionResult:
    """Result of a permission request."""
    granted: bool
    reason: str = ""
    scope: PermissionScope = PermissionScope.SPECIFIC
    remember_choice: bool = False
    expires_at: Optional[float] = None
    
    @classmethod
    def granted_with_scope(cls, scope: PermissionScope, remember: bool = False) -> "PermissionResult":
        """Create a granted permission with specified scope."""
        return cls(
            granted=True,
            reason="User granted permission",
            scope=scope,
            remember_choice=remember
        )
    
    @classmethod
    def denied(cls, reason: str) -> "PermissionResult":
        """Create a denied permission."""
        return cls(
            granted=False,
            reason=reason
        )


@dataclass
class PermissionConfig:
    """Configuration for the permission system."""
    default_mode: PermissionLevel = PermissionLevel.ASK_ONCE
    remember_session_choices: bool = True
    remember_permanent_choices: bool = True
    auto_grant_safe_operations: bool = True
    session_timeout_hours: float = 8.0
    permission_cache_file: str = "~/.loki-code/permissions.json"
    
    # Security level mappings
    security_level_permissions: Dict[str, PermissionLevel] = field(default_factory=lambda: {
        "safe": PermissionLevel.AUTO_GRANT,
        "caution": PermissionLevel.ASK_SESSION,
        "dangerous": PermissionLevel.ALWAYS_ASK,
        "critical": PermissionLevel.ALWAYS_ASK
    })


class PermissionManager:
    """
    Manages permission-based autonomy for the Loki Code agent.
    
    This system allows the agent to operate autonomously for safe operations
    while requesting user permission for potentially risky actions.
    """
    
    def __init__(self, config: PermissionConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # In-memory permission cache
        self.session_permissions: Dict[str, PermissionResult] = {}
        self.permanent_permissions: Dict[str, PermissionResult] = {}
        self.never_permissions: Set[str] = set()
        
        # Load persistent permissions
        self._load_persistent_permissions()
        
        # Session management
        self.session_start_time = time.time()
    
    async def request_permission(self, action: ToolAction, context: str = "") -> PermissionResult:
        """
        Request permission for a tool action.
        
        This is the main entry point for the permission system. It evaluates
        the action and either grants permission automatically, uses cached
        permissions, or asks the user.
        """
        try:
            # Step 1: Check if this action type is never allowed
            action_key = action.get_action_key()
            if action_key in self.never_permissions:
                return PermissionResult.denied("Action type permanently blocked by user")
            
            # Step 2: Auto-grant safe operations if configured
            if self.config.auto_grant_safe_operations and action.security_level == ToolSecurityLevel.SAFE:
                return PermissionResult.granted_with_scope(PermissionScope.TOOL_TYPE)
            
            # Step 3: Check cached permissions
            cached_permission = self._get_cached_permission(action)
            if cached_permission:
                return cached_permission
            
            # Step 4: Determine permission level needed
            permission_level = self._get_permission_level(action)
            
            # Step 5: Auto-grant if permission level allows
            if permission_level == PermissionLevel.AUTO_GRANT:
                return PermissionResult.granted_with_scope(PermissionScope.TOOL_TYPE)
            
            # Step 6: Ask user for permission
            return await self._ask_user_permission(action, context, permission_level)
            
        except Exception as e:
            self.logger.error(f"Error in permission request: {e}")
            # Fail safe - deny permission if there's an error
            return PermissionResult.denied(f"Permission system error: {str(e)}")
    
    def _get_permission_level(self, action: ToolAction) -> PermissionLevel:
        """Determine the permission level required for an action."""
        # Check security level mapping
        security_mapping = self.config.security_level_permissions.get(
            action.security_level.value.lower(),
            self.config.default_mode
        )
        
        # Escalate for destructive operations
        if action.is_destructive and security_mapping == PermissionLevel.AUTO_GRANT:
            return PermissionLevel.ASK_ONCE
        
        return security_mapping
    
    def _get_cached_permission(self, action: ToolAction) -> Optional[PermissionResult]:
        """Check if we have a cached permission for this action."""
        action_key = action.get_action_key()
        similarity_key = action.get_similarity_key()
        
        # Check exact match in session permissions
        if action_key in self.session_permissions:
            permission = self.session_permissions[action_key]
            if self._is_permission_valid(permission):
                return permission
        
        # Check similar actions in session
        if similarity_key in self.session_permissions:
            permission = self.session_permissions[similarity_key]
            if self._is_permission_valid(permission) and permission.scope != PermissionScope.SPECIFIC:
                return permission
        
        # Check permanent permissions
        if action_key in self.permanent_permissions:
            return self.permanent_permissions[action_key]
        
        if similarity_key in self.permanent_permissions:
            permission = self.permanent_permissions[similarity_key]
            if permission.scope != PermissionScope.SPECIFIC:
                return permission
        
        return None
    
    def _is_permission_valid(self, permission: PermissionResult) -> bool:
        """Check if a cached permission is still valid."""
        if permission.expires_at is None:
            return True
        
        return time.time() < permission.expires_at
    
    async def _ask_user_permission(self, action: ToolAction, context: str, 
                                  permission_level: PermissionLevel) -> PermissionResult:
        """Ask the user for permission with intelligent options."""
        
        # Build the permission prompt
        prompt = self._build_permission_prompt(action, context, permission_level)
        
        # Get user response (this would be replaced with actual UI interaction)
        response = await self._get_user_response(prompt)
        
        # Process the response
        return self._process_permission_response(action, response)
    
    def _build_permission_prompt(self, action: ToolAction, context: str, 
                                permission_level: PermissionLevel) -> str:
        """Build an intelligent permission prompt for the user."""
        
        # Risk indicators
        risk_emoji = {
            ToolSecurityLevel.SAFE: "✅",
            ToolSecurityLevel.CAUTION: "⚠️",
            ToolSecurityLevel.DANGEROUS: "🔶",
            ToolSecurityLevel.CRITICAL: "🚨"
        }
        
        files_text = ""
        if action.file_paths:
            if len(action.file_paths) == 1:
                files_text = f"**File**: {action.file_paths[0]}"
            else:
                files_text = f"**Files**: {len(action.file_paths)} files affected"
                if len(action.file_paths) <= 3:
                    files_text += f" ({', '.join(action.file_paths)})"
        
        destructive_warning = ""
        if action.is_destructive:
            destructive_warning = "⚠️ **Warning**: This action may modify or delete files.\\n"
        
        prompt = f"""
🤖 **Loki Code Permission Request**

{risk_emoji.get(action.security_level, "🔶")} **Action**: {action.description}
📝 **Tool**: {action.tool_name}
🛡️ **Security Level**: {action.security_level.value.title()}
{files_text}
📍 **Context**: {context}
{destructive_warning}

**Choose an option:**
1. **Yes, once** - Allow this specific action
2. **Yes, similar** - Allow similar {action.tool_name} operations this session
3. **Yes, always** - Remember this choice permanently for {action.tool_name}
4. **No** - Deny this action
5. **Never** - Never allow this type of operation again

Your choice (1-5): """
        
        return prompt
    
    async def _get_user_response(self, prompt: str) -> str:
        """Get user response (placeholder for actual UI integration)."""
        # This would be replaced with actual user interaction
        # For now, simulate interactive input
        print(prompt)
        try:
            response = input().strip()
            return response
        except (EOFError, KeyboardInterrupt):
            return "4"  # Default to deny
    
    def _process_permission_response(self, action: ToolAction, response: str) -> PermissionResult:
        """Process user's permission response and update caches."""
        
        response = response.strip().lower()
        
        # Parse response
        if response in ["1", "yes", "y", "once"]:
            # Allow once
            result = PermissionResult.granted_with_scope(PermissionScope.SPECIFIC)
            
        elif response in ["2", "similar", "session"]:
            # Allow similar actions this session
            result = PermissionResult.granted_with_scope(
                PermissionScope.SIMILAR, 
                remember=True
            )
            result.expires_at = time.time() + (self.config.session_timeout_hours * 3600)
            
            # Cache for session
            similarity_key = action.get_similarity_key()
            self.session_permissions[similarity_key] = result
            
        elif response in ["3", "always", "remember"]:
            # Remember permanently
            result = PermissionResult.granted_with_scope(
                PermissionScope.SIMILAR,
                remember=True
            )
            
            # Cache permanently
            similarity_key = action.get_similarity_key()
            self.permanent_permissions[similarity_key] = result
            self._save_persistent_permissions()
            
        elif response in ["5", "never", "block"]:
            # Never allow this type
            action_key = action.get_action_key()
            self.never_permissions.add(action_key)
            self._save_persistent_permissions()
            
            result = PermissionResult.denied("Action type permanently blocked by user")
            
        else:
            # Default to deny (including "4", "no", "n")
            result = PermissionResult.denied("Permission denied by user")
        
        return result
    
    def _load_persistent_permissions(self):
        """Load persistent permissions from disk."""
        try:
            permission_file = Path(self.config.permission_cache_file).expanduser()
            if permission_file.exists():
                with open(permission_file, 'r') as f:
                    data = json.load(f)
                
                # Load permanent permissions
                for key, perm_data in data.get("permanent", {}).items():
                    self.permanent_permissions[key] = PermissionResult(**perm_data)
                
                # Load never permissions
                self.never_permissions = set(data.get("never", []))
                
                self.logger.debug(f"Loaded {len(self.permanent_permissions)} permanent permissions")
                
        except Exception as e:
            self.logger.warning(f"Failed to load persistent permissions: {e}")
    
    def _save_persistent_permissions(self):
        """Save persistent permissions to disk."""
        try:
            permission_file = Path(self.config.permission_cache_file).expanduser()
            permission_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "permanent": {
                    key: {
                        "granted": perm.granted,
                        "reason": perm.reason,
                        "scope": perm.scope.value,
                        "remember_choice": perm.remember_choice,
                        "expires_at": perm.expires_at
                    }
                    for key, perm in self.permanent_permissions.items()
                },
                "never": list(self.never_permissions)
            }
            
            with open(permission_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug("Saved persistent permissions")
            
        except Exception as e:
            self.logger.error(f"Failed to save persistent permissions: {e}")
    
    def clear_session_permissions(self):
        """Clear all session permissions (called on new session)."""
        self.session_permissions.clear()
        self.session_start_time = time.time()
        self.logger.debug("Cleared session permissions")
    
    def revoke_permission(self, action_key: str) -> bool:
        """Revoke a previously granted permission."""
        revoked = False
        
        if action_key in self.session_permissions:
            del self.session_permissions[action_key]
            revoked = True
        
        if action_key in self.permanent_permissions:
            del self.permanent_permissions[action_key]
            self._save_persistent_permissions()
            revoked = True
        
        if revoked:
            self.logger.info(f"Revoked permission for: {action_key}")
        
        return revoked
    
    def get_permission_summary(self) -> Dict[str, Any]:
        """Get a summary of current permissions."""
        return {
            "session_permissions": len(self.session_permissions),
            "permanent_permissions": len(self.permanent_permissions),
            "never_permissions": len(self.never_permissions),
            "session_age_hours": (time.time() - self.session_start_time) / 3600,
            "auto_grant_safe": self.config.auto_grant_safe_operations
        }