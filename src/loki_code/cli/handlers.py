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
from ..core.tool_registry import get_global_registry
from ..tools.file_reader import FileReaderTool


def handle_cli_command(args) -> int:
    """
    Handle CLI commands based on parsed arguments.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Load configuration (use built-in defaults if no config specified)
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
        # Test the connection using the config (not the provider object)
        test_result = test_ollama_connection(config, verbose=args.verbose)
        report = format_test_report(test_result, verbose=args.verbose)
        print(report)
        return 0 if test_result.overall_status.value == "success" else 1
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
        registry = get_global_registry()
        tools = registry.list_tools()
        
        if not tools:
            print("  No tools available")
            return 0
            
        for tool_schema in tools:
            print(f"  {tool_schema.name}: {tool_schema.description}")
        return 0
    except Exception as e:
        print(f"❌ Error listing tools: {e}")
        return 1


def _handle_tool_info(config, args) -> int:
    """Show information about a specific tool."""
    tool_name = args.tool_info
    print(f"🔍 Tool Information: {tool_name}")
    
    try:
        registry = get_global_registry()
        tool_schema = registry.get_tool_schema(tool_name)
        
        if not tool_schema:
            print(f"❌ Tool '{tool_name}' not found")
            return 1
            
        print(f"  Description: {tool_schema.description}")
        print(f"  Capabilities: {', '.join([cap.value for cap in tool_schema.capabilities])}")
        print(f"  Security Level: {tool_schema.security_level.value}")
        print(f"  Confirmation Level: {tool_schema.confirmation_level.value}")
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
    """Handle chat mode using the same agent system as TUI."""
    print("💬 Starting chat mode with agent support...")
    
    if args.prompt:
        # Single prompt mode using agent system
        try:
            result = _process_chat_input(config, args.prompt)
            print(f"\n🤖 {result}")
            return 0
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return 1
    else:
        # Interactive chat mode using agent system
        print("🤖 Chat mode with LLM classification and tool support. Type 'exit' to quit.")
        
        try:
            while True:
                try:
                    user_input = input("👤 You: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        break
                        
                    result = _process_chat_input(config, user_input)
                    print(f"🤖 Assistant: {result}\n")
                    
                except KeyboardInterrupt:
                    break
                    
            print("👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"❌ Error in chat mode: {e}")
            return 1


def _process_chat_input(config, user_input: str) -> str:
    """Process chat input using the unified agent service."""
    import asyncio
    
    async def process_async():
        # Use the unified agent service
        from ..core.services import get_agent_service
        
        try:
            # Initialize the agent service
            agent_service = await get_agent_service(config, "chat_session")
            
            # Process the message
            response = await agent_service.process_message(user_input)
            
            # Format the response for display
            formatted_response = response.content
            
            # Add tool usage information
            if response.tools_used:
                formatted_response += f"\n\n🔧 Tools used: {', '.join(response.tools_used)}"
            
            # Add reasoning steps in verbose mode (check if we have access to args)
            try:
                # Try to get verbose flag from config or default to False
                verbose = getattr(config, 'verbose', False)
                if verbose and response.metadata and 'reasoning_steps' in response.metadata:
                    reasoning_steps = response.metadata['reasoning_steps']
                    if reasoning_steps:
                        formatted_response += "\n\n💭 Reasoning steps:"
                        for step in reasoning_steps:
                            formatted_response += f"\n  - {step}"
            except:
                pass  # Ignore if we can't access verbose flag
            
            return formatted_response
            
        except Exception as e:
            return f"Error: {e}"
    
    # Run the async function
    return asyncio.run(process_async())


def _handle_tui_mode(config, args) -> int:
    """Handle Textual TUI mode (default)."""
    try:
        # Check if configured model is available
        if not _check_model_available(config):
            return 1
        
        # Use the transformed TUI without adapter for now
        from ..ui.textual_app import create_loki_app
        app = create_loki_app(config, use_adapter=False)
        if app:
            app.run()
            return 0
        else:
            print("❌ Textual UI not available. Install with: pip install textual")
            return 1
    except Exception as e:
        print(f"❌ Error starting TUI: {e}")
        return 1


def _check_model_available(config) -> bool:
    """
    Silently check if the configured model is available.
    Only shows error message if model is not available.
    
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        import requests
        from requests.exceptions import ConnectionError, Timeout, RequestException
        
        # Get model and base URL from config
        model = config.llm.model
        base_url = config.llm.base_url
        
        # Check if Ollama is running and model is available
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            if model in model_names:
                return True
            else:
                print(f"❌ Model '{model}' not available. Download with: ollama pull {model}")
                return False
        else:
            print(f"❌ Cannot connect to Ollama at {base_url}")
            return False
            
    except (ConnectionError, Timeout):
        print(f"❌ Ollama not running at {base_url}. Start with: ollama serve")
        return False
    except RequestException as e:
        print(f"❌ Error checking model availability: {e}")
        return False
    except Exception as e:
        # Silent fallback - don't block startup for unexpected errors
        return True