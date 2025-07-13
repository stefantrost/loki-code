"""
CLI module for Loki Code.

This module contains the command-line interface logic extracted from main.py
to simplify the main entry point and provide a cleaner separation of concerns.
"""

from .commands import create_parser, parse_args
from .handlers import handle_cli_command

__all__ = ["create_parser", "parse_args", "handle_cli_command"]