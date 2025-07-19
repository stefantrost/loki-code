"""
UI Module for Loki Code

This module provides the UI interface layer and implementations
for different user interface types with streaming support.
"""

from .interface import (
    UIInterface,
    UIMessage,
    UIResponse,
    UIEvent,
    UIResponseType,
    UIResponseFormatter,
    StreamingUIService
)

from .message_bus import (
    UIMessageBus,
    MessageType,
    BusMessage,
    get_global_message_bus,
    publish_user_input,
    publish_agent_response,
    publish_system_event,
    subscribe_to_user_input,
    subscribe_to_agent_responses
)

from .tui_adapter import TUIAdapter, create_tui_adapter

# Try to import TUI components (optional dependency)
try:
    from .textual_app import LokiApp, create_loki_app
    from .repl_app import LokiREPL, create_repl_app
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False
    LokiApp = None
    create_loki_app = None
    LokiREPL = None
    create_repl_app = None

__all__ = [
    # Core UI Interface
    "UIInterface",
    "UIMessage", 
    "UIResponse",
    "UIEvent",
    "UIResponseType",
    "UIResponseFormatter",
    "StreamingUIService",
    
    # Message Bus
    "UIMessageBus",
    "MessageType",
    "BusMessage",
    "get_global_message_bus",
    "publish_user_input",
    "publish_agent_response",
    "publish_system_event",
    "subscribe_to_user_input",
    "subscribe_to_agent_responses",
    
    # TUI Implementation
    "TUIAdapter",
    "create_tui_adapter",
    "LokiApp",
    "create_loki_app",
    "LokiREPL",
    "create_repl_app",
    
    # Constants
    "TUI_AVAILABLE",
]


def create_ui_interface(ui_type: str = "tui", config=None) -> UIInterface:
    """
    Factory function to create UI interface based on type.
    
    Args:
        ui_type: Type of UI interface ("tui", "cli", "web")
        config: Application configuration
        
    Returns:
        UIInterface: The requested UI interface
        
    Raises:
        ValueError: If UI type is not supported
        ImportError: If required dependencies are not available
    """
    if ui_type == "tui":
        if not TUI_AVAILABLE:
            raise ImportError("TUI not available. Install with: pip install textual")
        return create_tui_adapter(config)
    elif ui_type == "cli":
        # Future: CLI-only interface
        raise ValueError("CLI interface not yet implemented")
    elif ui_type == "web":
        # Future: Web interface
        raise ValueError("Web interface not yet implemented")
    else:
        raise ValueError(f"Unsupported UI type: {ui_type}")


def get_available_ui_types() -> list[str]:
    """
    Get list of available UI types.
    
    Returns:
        List of available UI type names
    """
    available = []
    
    if TUI_AVAILABLE:
        available.append("tui")
    
    # Future UI types will be added here
    
    return available