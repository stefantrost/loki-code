#!/usr/bin/env python3
"""
Loki Code - A local coding agent using LangChain and local LLMs

Simplified main entry point focusing on core functionality.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from loki_code.cli import parse_args, handle_cli_command
from loki_code.config import load_config
from loki_code.utils.logging import setup_logging, get_logger
from loki_code.ui.textual_app import create_loki_app


def main() -> int:
    """Main entry point for Loki Code."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Handle CLI commands (test, list, etc.)
        if any([
            args.test_llm, args.list_providers, args.list_tools, 
            args.tool_info, args.analyze_file, args.read_file, args.chat
        ]):
            return handle_cli_command(args)
        
        # Default: Start Textual TUI
        config = load_config(args.config)
        setup_logging(config, verbose=args.verbose)
        
        app = create_loki_app(config)
        if app is None:
            print("❌ Textual UI not available. Install with: pip install textual")
            return 1
            
        app.run()
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())