"""
Beautiful progress indicators and status displays for Loki Code.

This module provides stunning visual progress indicators, real-time feedback,
and professional status displays for long-running operations like code analysis,
LLM processing, and tool execution.

Key Components:
- ProgressIndicator: Core progress display components
- ProgressManagers: Task-specific progress management
- ProgressAnimations: Beautiful animations and effects
- ProgressIntegration: Integration with agent and tool systems

Usage:
    from loki_code.ui.progress import ProgressIntegration
    
    progress_mgr = ProgressIntegration(console)
    result = await progress_mgr.wrap_agent_call(agent.process_request, "analyze code")
"""

from .indicators import (
    ProgressIndicator,
    ProgressContext,
    StreamingProgress
)

from .managers import (
    LLMProgressManager,
    ToolProgressManager,
    ProjectProgressManager
)

from .animations import (
    ProgressAnimations,
    ProgressEffects
)

from .integration import (
    ProgressIntegration,
    CancellableProgress
)

__all__ = [
    # Core indicators
    "ProgressIndicator",
    "ProgressContext", 
    "StreamingProgress",
    
    # Progress managers
    "LLMProgressManager",
    "ToolProgressManager",
    "ProjectProgressManager",
    
    # Animations and effects
    "ProgressAnimations",
    "ProgressEffects",
    
    # Integration
    "ProgressIntegration",
    "CancellableProgress"
]

# Version info
__version__ = "0.1.0"

# Progress system information
PROGRESS_SYSTEM_INFO = {
    "version": __version__,
    "features": [
        "real_time_indicators",
        "task_specific_visualization", 
        "streaming_progress",
        "cancellable_operations",
        "agent_tool_integration",
        "beautiful_animations"
    ],
    "supported_operations": [
        "llm_processing",
        "tool_execution",
        "file_analysis",
        "project_scanning",
        "code_generation",
        "streaming_responses"
    ]
}


def get_progress_system_info() -> dict:
    """Get information about the progress display system.
    
    Returns:
        Dictionary with progress system information
    """
    return PROGRESS_SYSTEM_INFO.copy()