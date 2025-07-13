"""
CLI command handlers for Loki Code.

This module contains the actual implementation of CLI commands,
separated from the argument parsing logic for better organization.
"""

import sys
from typing import Any, Dict, Optional
from pathlib import Path

from ..config import load_config, ConfigurationError
from ..utils import setup_logging, get_logger
from ..core.llm_test import test_ollama_connection, format_test_report
from ..core.providers import create_llm_provider, list_available_providers
from ..tools.registry import ToolRegistry
from ..tools.file_reader import FileReaderTool


def handle_cli_command(args) -> int:
    """
    Handle CLI commands based on parsed arguments.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Load configuration
        config = load_config(args.config)
        setup_logging(config, verbose=args.verbose)
        logger = get_logger(__name__)
        
        # Handle specific commands
        if args.test_llm:
            return _handle_test_llm(config, args)
        elif args.list_providers:
            return _handle_list_providers(config, args)
        elif args.list_tools:
            return _handle_list_tools(config, args)
        elif args.tool_info:
            return _handle_tool_info(config, args)
        elif args.analyze_file:
            return _handle_analyze_file(config, args)
        elif args.read_file:
            return _handle_read_file(config, args)
        elif args.chat:
            return _handle_chat_mode(config, args)
        else:
            # Default: Start Textual TUI
            return _handle_tui_mode(config, args)
            
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1


def _handle_test_llm(config, args) -> int:
    """Test LLM connection."""
    print("🔍 Testing LLM connection...")
    
    try:
        provider = create_llm_provider(config)
        test_result = test_ollama_connection(provider)
        report = format_test_report(test_result)
        print(report)
        return 0 if test_result.success else 1
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return 1


def _handle_list_providers(config, args) -> int:
    """List available LLM providers."""
    print("📋 Available LLM Providers:")
    
    try:
        providers = list_available_providers()
        for provider_type, provider_info in providers.items():
            status = "✅ Available" if provider_info.get("available", False) else "❌ Not available"
            print(f"  {provider_type}: {status}")
            if provider_info.get("models"):
                for model in provider_info["models"][:3]:  # Show first 3 models
                    print(f"    - {model}")
        return 0
    except Exception as e:
        print(f"❌ Error listing providers: {e}")
        return 1


def _handle_list_tools(config, args) -> int:
    """List available tools."""
    print("🔧 Available Tools:")
    
    try:
        registry = ToolRegistry()
        tools = registry.list_tools()
        
        if not tools:
            print("  No tools available")
            return 0
            
        for tool_name, tool_info in tools.items():
            print(f"  {tool_name}: {tool_info.get('description', 'No description')}")
        return 0
    except Exception as e:
        print(f"❌ Error listing tools: {e}")
        return 1


def _handle_tool_info(config, args) -> int:
    """Show information about a specific tool."""
    tool_name = args.tool_info
    print(f"🔍 Tool Information: {tool_name}")
    
    try:
        registry = ToolRegistry()
        tool_info = registry.get_tool_info(tool_name)
        
        if not tool_info:
            print(f"❌ Tool '{tool_name}' not found")
            return 1
            
        print(f"  Description: {tool_info.get('description', 'No description')}")
        print(f"  Capabilities: {', '.join(tool_info.get('capabilities', []))}")
        return 0
    except Exception as e:
        print(f"❌ Error getting tool info: {e}")
        return 1


def _handle_analyze_file(config, args) -> int:
    """Analyze a specific code file."""
    file_path = Path(args.analyze_file)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 1
        
    print(f"🔍 Analyzing file: {file_path}")
    
    try:
        from ..core.code_analysis import analyze_file_quick
        result = analyze_file_quick(str(file_path))
        print(result)
        return 0
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return 1


def _handle_read_file(config, args) -> int:
    """Read a file using the file reader tool."""
    file_path = Path(args.read_file)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 1
        
    print(f"📖 Reading file: {file_path}")
    
    try:
        tool = FileReaderTool()
        result = tool.execute({"file_path": str(file_path)})
        print(result.content)
        return 0
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 1


def _handle_chat_mode(config, args) -> int:
    """Handle simple chat mode."""
    print("💬 Starting simple chat mode...")
    
    if args.prompt:
        # Single prompt mode
        try:
            provider = create_llm_provider(config)
            from ..core.providers import GenerationRequest
            
            request = GenerationRequest(prompt=args.prompt)
            response = provider.generate_sync(request)
            print(f"\n🤖 {response.content}")
            return 0
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return 1
    else:
        # Interactive chat mode (simplified)
        print("🤖 Simple chat mode. Type 'exit' to quit.")
        
        try:
            provider = create_llm_provider(config)
            
            while True:
                try:
                    user_input = input("👤 You: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        break
                        
                    from ..core.providers import GenerationRequest
                    request = GenerationRequest(prompt=user_input)
                    response = provider.generate_sync(request)
                    print(f"🤖 Assistant: {response.content}\n")
                    
                except KeyboardInterrupt:
                    break
                    
            print("👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"❌ Error in chat mode: {e}")
            return 1


def _handle_tui_mode(config, args) -> int:
    """Handle Textual TUI mode (default)."""
    print("🚀 Starting Textual TUI...")
    
    try:
        from ..ui.textual_app import LokiApp
        app = LokiApp(config)
        app.run()
        return 0
    except ImportError:
        print("❌ Textual UI not available. Install with: pip install textual")
        return 1
    except Exception as e:
        print(f"❌ Error starting TUI: {e}")
        return 1