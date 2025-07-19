"""
Confirmation management system for handling pending tool operations.

This module manages the state of operations that require user confirmation,
allowing users to confirm or deny pending operations through natural language.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .types import ConversationContext
from ..agent.permission_manager import PermissionResult, PermissionScope
from ...utils.logging import get_logger


class ConfirmationStatus(Enum):
    """Status of a confirmation request."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class PendingConfirmation:
    """Represents a pending confirmation request."""
    
    confirmation_id: str
    tool_name: str
    input_data: Dict[str, Any]
    context: Any  # ToolContext - avoiding circular import
    user_message: str
    created_at: float
    expires_at: float
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    prompt_shown: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if confirmation has expired."""
        return time.time() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if confirmation is still active."""
        return self.status == ConfirmationStatus.PENDING and not self.is_expired()


@dataclass
class ConfirmationResponse:
    """Response to a confirmation request."""
    
    confirmation_id: str
    response_type: str  # "confirm", "deny", "always", "never"
    user_input: str
    scope: PermissionScope = PermissionScope.SPECIFIC
    remember_choice: bool = False


class ConfirmationManager:
    """Manages pending confirmations and handles confirmation responses."""
    
    def __init__(self, default_timeout_seconds: float = 300.0):  # 5 minutes default
        self.logger = get_logger(__name__)
        self.default_timeout = default_timeout_seconds
        
        # Store pending confirmations
        self.pending_confirmations: Dict[str, PendingConfirmation] = {}
        
        # Store per-session state
        self.session_pending: Dict[str, str] = {}  # session_id -> confirmation_id
        
    def create_confirmation(
        self, 
        tool_name: str,
        input_data: Dict[str, Any], 
        context: Any,
        user_message: str,
        timeout_seconds: Optional[float] = None
    ) -> PendingConfirmation:
        """Create a new pending confirmation."""
        
        confirmation_id = str(uuid.uuid4())
        timeout = timeout_seconds or self.default_timeout
        
        # Clean up any existing confirmation for this session
        session_id = getattr(context, 'session_id', 'default')
        self._cleanup_session_confirmation(session_id)
        
        # Create confirmation
        confirmation = PendingConfirmation(
            confirmation_id=confirmation_id,
            tool_name=tool_name,
            input_data=input_data,
            context=context,
            user_message=user_message,
            created_at=time.time(),
            expires_at=time.time() + timeout,
            prompt_shown=self._generate_confirmation_prompt(tool_name, input_data, context)
        )
        
        # Store confirmation
        self.pending_confirmations[confirmation_id] = confirmation
        self.session_pending[session_id] = confirmation_id
        
        self.logger.info(f"Created confirmation {confirmation_id} for tool {tool_name}")
        return confirmation
    
    def get_pending_confirmation(self, session_id: str) -> Optional[PendingConfirmation]:
        """Get pending confirmation for a session."""
        confirmation_id = self.session_pending.get(session_id)
        if not confirmation_id:
            return None
            
        confirmation = self.pending_confirmations.get(confirmation_id)
        if not confirmation or not confirmation.is_active():
            # Clean up expired/invalid confirmations
            self._cleanup_session_confirmation(session_id)
            return None
            
        return confirmation
    
    def parse_confirmation_response(self, user_input: str) -> Optional[ConfirmationResponse]:
        """Parse user input to determine if it's a confirmation response."""
        
        user_input_lower = user_input.lower().strip()
        
        # Positive confirmations
        positive_patterns = [
            "yes", "y", "confirm", "ok", "okay", "sure", "proceed", "go ahead",
            "do it", "execute", "run", "continue", "accept", "agreed", "fine"
        ]
        
        # Negative confirmations  
        negative_patterns = [
            "no", "n", "deny", "cancel", "stop", "abort", "reject", "refuse",
            "don't", "do not", "skip", "nevermind", "never mind"
        ]
        
        # Always allow patterns
        always_patterns = [
            "yes always", "always yes", "always allow", "always", "remember yes",
            "yes and remember", "yes for all", "allow all", "always confirm"
        ]
        
        # Never allow patterns
        never_patterns = [
            "never", "no never", "never allow", "always no", "always deny",
            "block always", "remember no", "no for all"
        ]
        
        # Check patterns in order of specificity
        if any(pattern in user_input_lower for pattern in always_patterns):
            return ConfirmationResponse(
                confirmation_id="",  # Will be set by caller
                response_type="always",
                user_input=user_input,
                scope=PermissionScope.SIMILAR,
                remember_choice=True
            )
        
        elif any(pattern in user_input_lower for pattern in never_patterns):
            return ConfirmationResponse(
                confirmation_id="",
                response_type="never", 
                user_input=user_input,
                scope=PermissionScope.SIMILAR,
                remember_choice=True
            )
        
        elif any(pattern in user_input_lower for pattern in positive_patterns):
            return ConfirmationResponse(
                confirmation_id="",
                response_type="confirm",
                user_input=user_input,
                scope=PermissionScope.SPECIFIC
            )
        
        elif any(pattern in user_input_lower for pattern in negative_patterns):
            return ConfirmationResponse(
                confirmation_id="",
                response_type="deny",
                user_input=user_input,
                scope=PermissionScope.SPECIFIC
            )
        
        # Check for session-specific patterns
        session_patterns = [
            "yes for session", "session yes", "yes this session", "allow session",
            "yes similar", "similar yes", "allow similar"
        ]
        
        if any(pattern in user_input_lower for pattern in session_patterns):
            return ConfirmationResponse(
                confirmation_id="",
                response_type="session",
                user_input=user_input,
                scope=PermissionScope.SIMILAR,
                remember_choice=False  # Session only
            )
        
        return None
    
    def handle_confirmation_response(
        self, 
        session_id: str, 
        user_input: str
    ) -> Tuple[Optional[PendingConfirmation], Optional[ConfirmationResponse]]:
        """Handle a confirmation response and return the result."""
        
        # Get pending confirmation
        confirmation = self.get_pending_confirmation(session_id)
        if not confirmation:
            return None, None
        
        # Parse response
        response = self.parse_confirmation_response(user_input)
        if not response:
            return confirmation, None  # Not a confirmation response
        
        # Set confirmation ID
        response.confirmation_id = confirmation.confirmation_id
        
        # Update confirmation status
        if response.response_type in ["confirm", "always", "session"]:
            confirmation.status = ConfirmationStatus.CONFIRMED
        else:
            confirmation.status = ConfirmationStatus.DENIED
        
        # Clean up session state
        self._cleanup_session_confirmation(session_id)
        
        self.logger.info(
            f"Handled confirmation {confirmation.confirmation_id}: {response.response_type}"
        )
        
        return confirmation, response
    
    def _generate_confirmation_prompt(
        self, 
        tool_name: str, 
        input_data: Dict[str, Any], 
        context: Any
    ) -> str:
        """Generate a user-friendly confirmation prompt."""
        
        # Extract key information from input
        file_path = input_data.get("file_path", "")
        content_preview = ""
        
        if "content" in input_data:
            content = input_data["content"]
            if content:
                # Show first line or indicate content type
                first_line = content.split('\n')[0][:50]
                content_preview = f" with content: '{first_line}...'" if first_line else " with content"
            else:
                content_preview = " (empty file)"
        
        # Generate action description
        if tool_name == "file_writer":
            if file_path:
                action_desc = f"create file '{file_path}'{content_preview}"
            else:
                action_desc = f"create a file{content_preview}"
        
        elif tool_name == "file_reader":
            action_desc = f"read file '{file_path}'" if file_path else "read a file"
        
        else:
            action_desc = f"execute {tool_name}"
            if file_path:
                action_desc += f" on '{file_path}'"
        
        return f"""This will {action_desc}.

**Options:**
• Type 'yes' or 'confirm' to proceed
• Type 'no' or 'cancel' to cancel
• Type 'yes always' to always allow similar operations
• Type 'never' to block this type of operation

Your choice:"""
    
    def _cleanup_session_confirmation(self, session_id: str) -> None:
        """Clean up confirmation for a session."""
        confirmation_id = self.session_pending.get(session_id)
        if confirmation_id:
            # Remove from session mapping
            del self.session_pending[session_id]
            
            # Mark confirmation as expired/processed
            if confirmation_id in self.pending_confirmations:
                confirmation = self.pending_confirmations[confirmation_id]
                if confirmation.status == ConfirmationStatus.PENDING:
                    confirmation.status = ConfirmationStatus.EXPIRED
    
    def cleanup_expired_confirmations(self) -> int:
        """Clean up expired confirmations and return count removed."""
        current_time = time.time()
        expired_ids = []
        
        for conf_id, confirmation in self.pending_confirmations.items():
            if confirmation.expires_at < current_time:
                expired_ids.append(conf_id)
        
        # Remove expired confirmations
        for conf_id in expired_ids:
            del self.pending_confirmations[conf_id]
        
        # Clean up session mappings
        sessions_to_remove = []
        for session_id, conf_id in self.session_pending.items():
            if conf_id in expired_ids:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.session_pending[session_id]
        
        if expired_ids:
            self.logger.debug(f"Cleaned up {len(expired_ids)} expired confirmations")
        
        return len(expired_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get confirmation manager statistics."""
        total_pending = len(self.pending_confirmations)
        active_pending = sum(1 for c in self.pending_confirmations.values() if c.is_active())
        
        return {
            "total_confirmations": total_pending,
            "active_confirmations": active_pending,
            "session_mappings": len(self.session_pending),
        }