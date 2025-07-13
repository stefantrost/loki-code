"""
Command-line argument parser for Loki Code.

This module defines all CLI commands and arguments, extracted from main.py
for better organization and maintainability.
"""

import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Loki Code - A local coding agent using LangChain and local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  loki-code                           # Start Textual TUI (default)
  loki-code --chat                    # Simple chat mode
  loki-code --test-llm               # Test LLM connection
  loki-code --list-tools             # List available tools
  loki-code --analyze-file path.py   # Analyze a code file
        """
    )
    
    # Basic options
    parser.add_argument(
        "--version",
        action="version",
        version="Loki Code 0.1.0"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    
    mode_group.add_argument(
        "--chat",
        action="store_true",
        help="Start in simple chat mode"
    )
    
    mode_group.add_argument(
        "--tui",
        action="store_true",
        default=True,
        help="Start Textual TUI (default mode)"
    )
    
    # LLM testing
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help="Test LLM connection and exit"
    )
    
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available LLM providers"
    )
    
    # Tool management
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools"
    )
    
    parser.add_argument(
        "--tool-info",
        type=str,
        metavar="NAME",
        help="Show information about a specific tool"
    )
    
    # File operations
    parser.add_argument(
        "--analyze-file",
        type=str,
        metavar="PATH",
        help="Analyze a specific code file"
    )
    
    parser.add_argument(
        "--read-file",
        type=str,
        metavar="PATH",
        help="Read a file using the file reader tool"
    )
    
    # Chat options
    parser.add_argument(
        "--prompt",
        type=str,
        metavar="TEXT",
        help="Single prompt for chat mode"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        metavar="NAME",
        help="Specify which model to use"
    )
    
    return parser


def parse_args(args=None):
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args(args)