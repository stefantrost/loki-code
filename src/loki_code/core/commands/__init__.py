"""
Core command processing module for Loki Code.

This module contains UI-agnostic command processing logic including
parsing, routing, and execution orchestration using modern NLP techniques.
"""

from .types import *
from .router import CommandRouter

# Avoid circular imports by not importing processor or nlp_parser here
# Use direct imports where needed instead