"""
Backward compatibility wrapper for LokiAgent.

This module provides the LokiAgent class for backward compatibility,
mapping to the new simplified LokiCodeAgent.
"""

from .agent_core import LokiCodeAgent
from .types import AgentConfig, AgentResponse, RequestContext, AgentState

# Backward compatibility alias
LokiAgent = LokiCodeAgent

__all__ = [
    "LokiAgent", 
    "LokiCodeAgent",
    "AgentConfig",
    "AgentResponse", 
    "RequestContext",
    "AgentState"
]