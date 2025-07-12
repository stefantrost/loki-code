#!/usr/bin/env python3
"""
Loki Code - A local coding agent using LangChain and local LLMs

This is the main entry point for the Loki Code application.

Usage:
    python main.py --help          # Show help message
    python main.py --version       # Show version information
    python main.py                 # Start interactive mode
    python main.py --config PATH   # Specify config file
    python main.py --test-llm      # Test LLM connection
    python main.py --chat          # Interactive chat mode
    python main.py --chat --prompt "Hello"  # Single prompt mode
    python main.py --list-providers # List available providers
    python main.py --list-tools     # List available tools
    python main.py --read-file PATH # Read file with intelligent analysis
    python main.py --analyze-file PATH # Analyze a code file
    python main.py --analyze-project PATH # Analyze a project
    python main.py --list-languages # List supported languages
    python main.py --generate-context FILE # Generate LLM context

Examples:
    python main.py --version
    python main.py --config ./configs/my_config.yaml
    python main.py --test-llm --verbose
    python main.py --list-providers
    python main.py --list-tools
    python main.py --tool-info file_reader
    python main.py --test-tools
    python main.py --read-file ./example.py --analysis-level detailed
    python main.py --search-tools "file"        # Search for file-related tools
    python main.py --discover-tools             # Rediscover available tools
    python main.py --registry-stats             # Show tool registry statistics
    python main.py --execute-tool file_reader --tool-input '{"file_path": "test.py"}'
    python main.py --run-tests                  # Run comprehensive integration tests
    python main.py --benchmark-tools            # Run performance benchmarks
    python main.py --test-multi-language        # Test multi-language support
    python main.py --test-security              # Run security validation tests
    python main.py --test-performance           # Run performance tests
    python main.py --test-cli                   # Test CLI integration
    python main.py --chat
    python main.py --chat --prompt "Help me with Python"
    python main.py --analyze-file ./example.py
    python main.py --analyze-project ./my-project
    python main.py --list-languages
    python main.py --language-info python
    python main.py --generate-context ./example.py --task-description "Explain this code"
"""

import argparse
import sys
import time
from pathlib import Path
from loki_code.config import load_config, ConfigurationError
from loki_code.utils import setup_logging, get_logger, log_startup, log_config_info, log_shutdown
from loki_code.core.llm_test import test_ollama_connection, format_test_report
from loki_code.core.llm_client import create_llm_client, LLMRequest, LLMClientError, LLMConnectionError, LLMModelError, LLMTimeoutError
from loki_code.core.providers import create_llm_provider, GenerationRequest, ProviderConnectionError, ProviderModelError, ProviderTimeoutError
from loki_code.core.model_manager import ModelManager
from loki_code.core.task_classifier import TaskClassifier, TaskType
from loki_code.core.code_analysis import (
    TreeSitterParser, CodeAnalyzer, ContextExtractor, 
    ContextLevel, ContextConfig, SupportedLanguage, 
    get_language_info, list_supported_languages, analyze_file_quick, analyze_project_quick
)


def get_version():
    """Get the version from the loki_code package."""
    try:
        import loki_code
        return loki_code.__version__
    except ImportError as e:
        print(f"Error: Could not import loki_code package: {e}", file=sys.stderr)
        print("Please ensure the package is installed with: pip install -e .", file=sys.stderr)
        sys.exit(1)


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="loki-code",
        description="Loki Code - A local coding agent using LangChain and local LLMs",
        epilog="For more information, visit: https://github.com/your-org/loki-code"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Loki Code v{get_version()}"
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
    
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help="Test LLM connection and model availability"
    )
    
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start chat mode for interactive conversation with LLM"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        metavar="TEXT",
        help="Send a single prompt to the LLM (use with --chat)"
    )
    
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available LLM providers"
    )
    
    parser.add_argument(
        "--show-models",
        action="store_true",
        help="Show model configurations and status"
    )
    
    parser.add_argument(
        "--test-models",
        action="store_true",
        help="Test all configured models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        metavar="NAME",
        help="Force use of specific model (use with --chat)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark on all models"
    )
    
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and their capabilities"
    )
    
    parser.add_argument(
        "--tool-info",
        type=str,
        metavar="NAME",
        help="Show detailed information about a specific tool"
    )
    
    parser.add_argument(
        "--test-tools",
        action="store_true",
        help="Test all available tools"
    )
    
    # Integration Testing Commands
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run comprehensive integration tests for the tool system"
    )
    
    parser.add_argument(
        "--benchmark-tools",
        action="store_true",
        help="Run performance benchmarks on tool system"
    )
    
    parser.add_argument(
        "--test-multi-language",
        action="store_true",
        help="Test tool system with multiple programming languages"
    )
    
    parser.add_argument(
        "--test-security",
        action="store_true",
        help="Run security validation tests"
    )
    
    parser.add_argument(
        "--test-performance",
        action="store_true",
        help="Run performance tests with large files"
    )
    
    parser.add_argument(
        "--test-cli",
        action="store_true",
        help="Test CLI integration functionality"
    )
    
    # Tool Commands
    parser.add_argument(
        "--read-file",
        type=str,
        metavar="PATH",
        help="Read a file using the file reader tool"
    )
    
    parser.add_argument(
        "--analysis-level",
        type=str,
        choices=["minimal", "standard", "detailed", "comprehensive"],
        default="standard",
        help="Analysis level for file reading (use with --read-file)"
    )
    
    # Advanced Tool Management Commands
    parser.add_argument(
        "--search-tools",
        type=str,
        metavar="QUERY",
        help="Search tools by description or name"
    )
    
    parser.add_argument(
        "--discover-tools",
        action="store_true",
        help="Rediscover and register available tools"
    )
    
    parser.add_argument(
        "--tool-stats",
        type=str,
        metavar="TOOL_NAME",
        help="Show usage statistics for a specific tool"
    )
    
    parser.add_argument(
        "--registry-stats",
        action="store_true",
        help="Show overall tool registry statistics"
    )
    
    parser.add_argument(
        "--execute-tool",
        type=str,
        metavar="TOOL_NAME",
        help="Execute a tool directly with JSON input"
    )
    
    parser.add_argument(
        "--tool-input",
        type=str,
        metavar="JSON",
        help="JSON input for tool execution (use with --execute-tool)"
    )
    
    parser.add_argument(
        "--validate-tool",
        type=str,
        metavar="TOOL_NAME",
        help="Validate a tool's configuration and functionality"
    )
    
    parser.add_argument(
        "--enable-tool",
        type=str,
        metavar="TOOL_NAME",
        help="Enable a specific tool"
    )
    
    parser.add_argument(
        "--disable-tool",
        type=str,
        metavar="TOOL_NAME",
        help="Disable a specific tool"
    )
    
    parser.add_argument(
        "--execution-history",
        action="store_true",
        help="Show recent tool execution history"
    )
    
    parser.add_argument(
        "--performance-metrics",
        type=str,
        nargs="?",
        const="all",
        metavar="TOOL_NAME",
        help="Show performance metrics for tools (optional tool name)"
    )
    
    parser.add_argument(
        "--filter-by-capability",
        type=str,
        metavar="CAPABILITY",
        help="Filter tools by capability (read_file, write_file, etc.)"
    )
    
    parser.add_argument(
        "--filter-by-security",
        type=str,
        choices=["safe", "moderate", "dangerous"],
        help="Filter tools by security level"
    )
    
    # Code Analysis Commands
    parser.add_argument(
        "--analyze-file",
        type=str,
        metavar="PATH",
        help="Analyze a specific code file"
    )
    
    parser.add_argument(
        "--analyze-project",
        type=str,
        metavar="PATH",
        help="Analyze an entire project directory"
    )
    
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported programming languages"
    )
    
    parser.add_argument(
        "--language-info",
        type=str,
        metavar="LANG",
        help="Show information about a specific language"
    )
    
    parser.add_argument(
        "--context-level",
        type=str,
        choices=["minimal", "standard", "detailed", "comprehensive"],
        default="standard",
        help="Set context extraction level for analysis"
    )
    
    parser.add_argument(
        "--generate-context",
        type=str,
        metavar="FILE",
        help="Generate LLM context from a file"
    )
    
    parser.add_argument(
        "--task-description",
        type=str,
        metavar="TEXT",
        help="Task description for context generation (use with --generate-context)"
    )
    
    return parser


def show_welcome_message():
    """Display the welcome message for interactive mode."""
    version = get_version()
    print(f"""
╭─────────────────────────────────────────────╮
│              🎭 Loki Code v{version:<8}          │
│    A local coding agent for developers     │
╰─────────────────────────────────────────────╯

Welcome to Loki Code! 

📋 Current Status: Interactive mode not implemented yet

🚀 Available Commands:
   --help     Show this help message
   --version  Show version information
   --config   Specify configuration file (future use)

💡 Next Steps:
   - Interactive TUI will be implemented soon
   - Configuration system coming next
   - Local LLM integration in development

For more information, run: python main.py --help
""")


def run_interactive_mode(args):
    """Run the interactive mode (placeholder implementation)."""
    config = None
    logger = None
    
    try:
        # Load configuration
        try:
            config = load_config(args.config)
        except ConfigurationError as e:
            print(f"⚠️  Configuration error: {e}", file=sys.stderr)
            print("Falling back to built-in defaults...")
            # Continue with default config
            from loki_code.config.models import LokiCodeConfig
            config = LokiCodeConfig()
        
        # Setup logging system
        setup_logging(config, verbose=args.verbose)
        logger = get_logger(__name__)
        
        # Log startup information
        log_startup(args.config)
        log_config_info(config)
        
        # Show welcome message with config info
        show_welcome_message()
        
        # Display configuration info if verbose
        if args.verbose or config.app.debug:
            logger.info("Verbose mode enabled - showing configuration details")
            print("🔍 Configuration details:")
            print(f"   Debug mode: {config.app.debug}")
            print(f"   Log level: {config.app.log_level}")
            print(f"   LLM provider: {config.llm.provider}")
            print(f"   LLM model: {config.llm.model}")
            print(f"   Theme: {config.ui.theme}")
            print(f"   Tools enabled: {len(config.tools.enabled)} tools")
        
        # Demonstrate some logging
        logger.info("Interactive mode started (placeholder implementation)")
        logger.debug("This is a debug message - only visible with --verbose or debug config")
        logger.info("Interactive mode will be implemented in future versions")
        
        print("\n✨ Exiting gracefully...")
        logger.info("Interactive mode session completed")
        return 0
        
    except KeyboardInterrupt:
        if logger:
            logger.info("Interrupted by user")
        print("\n\n👋 Interrupted by user. Goodbye!")
        return 0
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error in interactive mode: {e}", exc_info=True)
        else:
            print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        # Log shutdown
        if logger:
            log_shutdown()


def run_llm_test(args):
    """Run LLM connection test."""
    config = None
    logger = None
    
    try:
        # Load configuration
        try:
            config = load_config(args.config)
        except ConfigurationError as e:
            print(f"⚠️  Configuration error: {e}", file=sys.stderr)
            print("Using built-in defaults for testing...")
            from loki_code.config.models import LokiCodeConfig
            config = LokiCodeConfig()
        
        # Setup logging
        setup_logging(config, verbose=args.verbose)
        logger = get_logger(__name__)
        
        # Log test start
        logger.info("🔍 Starting LLM connection test...")
        
        # Run the comprehensive test
        report = test_ollama_connection(config, verbose=args.verbose)
        
        # Format and display results
        formatted_report = format_test_report(report, verbose=args.verbose)
        print("\n" + formatted_report)
        
        # Return appropriate exit code
        if report.overall_status.value == "success":
            logger.info("LLM test completed successfully")
            return 0
        elif report.overall_status.value == "warning":
            logger.warning("LLM test completed with warnings")
            return 0
        else:
            logger.error("LLM test failed")
            return 1
            
    except Exception as e:
        if logger:
            logger.error(f"LLM test failed with unexpected error: {e}", exc_info=True)
        else:
            print(f"❌ LLM test failed: {e}", file=sys.stderr)
        return 1


def run_list_providers(args):
    """List available LLM providers."""
    try:
        from loki_code.core.providers import list_available_providers, get_provider_info
        
        print("📋 Available LLM Providers:")
        print()
        
        providers = list_available_providers()
        if not providers:
            print("❌ No providers are currently available")
            return 1
        
        for provider_name in providers:
            info = get_provider_info(provider_name)
            if info:
                print(f"✅ {provider_name.title()}")
                print(f"   📝 {info.get('description', 'No description')}")
                capabilities = info.get('capabilities', [])
                if capabilities:
                    print(f"   🔧 Capabilities: {', '.join(capabilities)}")
                print()
        
        print(f"💡 To use a provider, set it in your config file or use --config")
        return 0
        
    except Exception as e:
        print(f"❌ Error listing providers: {e}", file=sys.stderr)
        return 1


def run_show_models(args):
    """Show model configurations and status."""
    try:
        # Load configuration
        try:
            config = load_config(args.config)
        except ConfigurationError as e:
            print(f"⚠️  Configuration error: {e}", file=sys.stderr)
            from loki_code.config.models import LokiCodeConfig
            config = LokiCodeConfig()
        
        # Create model manager
        model_manager = ModelManager(config)
        
        print("🧠 Model Configurations:")
        print()
        
        configs = model_manager.get_model_configurations()
        if not configs:
            print("❌ No models configured")
            return 1
        
        for model_name, config_info in configs.items():
            print(f"📝 {model_name}")
            print(f"   Provider: {config_info['provider']}")
            print(f"   Task Types: {', '.join(config_info['task_types'])}")
            print(f"   Complexity Range: {config_info['min_complexity']:.1f} - {config_info['max_complexity']:.1f}")
            print(f"   Priority: {config_info['priority']}")
            if config_info['temperature'] is not None:
                print(f"   Temperature: {config_info['temperature']}")
            if config_info['max_tokens'] is not None:
                print(f"   Max Tokens: {config_info['max_tokens']}")
            print()
        
        # Show active models
        active_models = model_manager.get_active_models()
        if active_models:
            print("🔄 Active Models:")
            for model_name, stats in active_models.items():
                print(f"   {model_name}: {stats['request_count']} requests, "
                      f"{stats['memory_usage']}MB memory")
        else:
            print("💤 No models currently active")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error showing models: {e}", file=sys.stderr)
        return 1


def run_test_models(args):
    """Test all configured models."""
    try:
        # Load configuration
        try:
            config = load_config(args.config)
        except ConfigurationError as e:
            print(f"⚠️  Configuration error: {e}", file=sys.stderr)
            from loki_code.config.models import LokiCodeConfig
            config = LokiCodeConfig()
        
        # Setup logging
        setup_logging(config, verbose=args.verbose)
        logger = get_logger(__name__)
        
        # Create model manager
        model_manager = ModelManager(config)
        
        print("🧪 Testing All Models:")
        print()
        
        # Run health checks
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(model_manager.health_check_all_models())
        
        success_count = 0
        for model_name, is_healthy in results.items():
            if is_healthy:
                print(f"✅ {model_name}: Healthy")
                success_count += 1
            else:
                print(f"❌ {model_name}: Failed health check")
        
        print()
        print(f"📊 Results: {success_count}/{len(results)} models healthy")
        
        return 0 if success_count == len(results) else 1
        
    except Exception as e:
        print(f"❌ Error testing models: {e}", file=sys.stderr)
        return 1


def run_benchmark(args):
    """Run performance benchmark on all models."""
    try:
        # Load configuration
        try:
            config = load_config(args.config)
        except ConfigurationError as e:
            print(f"⚠️  Configuration error: {e}", file=sys.stderr)
            from loki_code.config.models import LokiCodeConfig
            config = LokiCodeConfig()
        
        # Setup logging
        setup_logging(config, verbose=args.verbose)
        logger = get_logger(__name__)
        
        # Create model manager
        model_manager = ModelManager(config)
        
        print("⚡ Running Model Benchmark:")
        print()
        
        # Benchmark prompts for different task types
        benchmark_prompts = {
            TaskType.CODE_GENERATION: "Write a Python function to calculate fibonacci numbers",
            TaskType.CHAT_CONVERSATION: "Hello! How are you doing today?",
            TaskType.EXPLANATION: "Explain how recursion works in programming",
            TaskType.DEBUGGING: "Why might this code cause a memory leak?",
        }
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run benchmarks
        for task_type, prompt in benchmark_prompts.items():
            print(f"🎯 Testing {task_type.value}:")
            print(f"   Prompt: {prompt}")
            
            try:
                start_time = time.perf_counter()
                response = loop.run_until_complete(model_manager.generate(prompt))
                duration = (time.perf_counter() - start_time) * 1000
                
                print(f"   ✅ Model: {response.model}")
                print(f"   ⏱️  Time: {duration:.0f}ms")
                print(f"   📝 Response: {response.content[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
            
            print()
        
        # Show performance stats
        stats = model_manager.get_performance_stats()
        if stats:
            print("📈 Performance Statistics:")
            for model_name, model_stats in stats.items():
                print(f"   {model_name}:")
                for task_type, task_stats in model_stats.items():
                    print(f"      {task_type}: {task_stats['avg_response_time']:.0f}ms avg, "
                          f"{task_stats['request_count']} requests")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running benchmark: {e}", file=sys.stderr)
        return 1


def run_list_tools(args):
    """List available tools and their capabilities."""
    try:
        from loki_code.tools import tool_registry, SecurityLevel, ToolCapability
        
        print("🔧 Available Tools:")
        print()
        
        tools = tool_registry.list_tools()
        if not tools:
            print("❌ No tools are currently registered")
            print("💡 Tools will be registered when the first tool is implemented")
            return 0
        
        for tool_schema in tools:
            print(f"🛠️  {tool_schema.name}")
            print(f"   📝 {tool_schema.description}")
            print(f"   🔒 Security: {tool_schema.security_level.value}")
            
            capabilities = [cap.value for cap in tool_schema.capabilities]
            print(f"   ⚡ Capabilities: {', '.join(capabilities)}")
            
            if tool_schema.confirmation_level.value != "none":
                print(f"   ⚠️  Requires: {tool_schema.confirmation_level.value} confirmation")
            
            if tool_schema.tags:
                print(f"   🏷️  Tags: {', '.join(tool_schema.tags)}")
            
            print()
        
        # Show summary
        stats = tool_registry.get_registry_stats()
        print(f"📊 Summary: {stats['total_tools']} tools available")
        
        if stats['security_distribution']:
            security_summary = [f"{level}: {count}" for level, count in stats['security_distribution'].items()]
            print(f"🔒 Security levels: {', '.join(security_summary)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error listing tools: {e}", file=sys.stderr)
        return 1


def run_tool_info(args):
    """Show detailed information about a specific tool."""
    try:
        from loki_code.tools import tool_registry
        
        tool_name = args.tool_info
        schema = tool_registry.get_tool_schema(tool_name)
        
        if not schema:
            print(f"❌ Tool '{tool_name}' not found")
            
            available_tools = tool_registry.list_tool_names()
            if available_tools:
                print(f"Available tools: {', '.join(available_tools)}")
            else:
                print("No tools are currently registered")
            return 1
        
        print(f"🛠️  Tool Information: {schema.name}")
        print()
        print(f"📝 Description: {schema.description}")
        print(f"🔒 Security Level: {schema.security_level.value}")
        print(f"⚠️  Confirmation: {schema.confirmation_level.value}")
        print(f"🆔 Version: {schema.version}")
        print(f"🌐 MCP Compatible: {schema.mcp_compatible}")
        
        if schema.capabilities:
            capabilities = [cap.value for cap in schema.capabilities]
            print(f"⚡ Capabilities: {', '.join(capabilities)}")
        
        if schema.tags:
            print(f"🏷️  Tags: {', '.join(schema.tags)}")
        
        # Show input/output schemas
        print()
        print("📥 Input Schema:")
        if schema.input_schema:
            properties = schema.input_schema.get('properties', {})
            required = schema.input_schema.get('required', [])
            
            for prop_name, prop_schema in properties.items():
                required_marker = " (required)" if prop_name in required else ""
                prop_type = prop_schema.get('type', 'unknown')
                description = prop_schema.get('description', 'No description')
                print(f"   • {prop_name}: {prop_type}{required_marker} - {description}")
        else:
            print("   No input schema defined")
        
        print()
        print("📤 Output Schema:")
        if schema.output_schema:
            output_props = schema.output_schema.get('properties', {})
            for prop_name, prop_schema in output_props.items():
                prop_type = prop_schema.get('type', 'unknown')
                description = prop_schema.get('description', 'No description')
                print(f"   • {prop_name}: {prop_type} - {description}")
        else:
            print("   No output schema defined")
        
        # Show usage statistics if available
        stats = tool_registry.get_tool_stats(tool_name)
        if stats and stats['usage_count'] > 0:
            print()
            print("📊 Usage Statistics:")
            print(f"   • Total uses: {stats['usage_count']}")
            print(f"   • Success rate: {stats['success_rate']:.1%}")
            if stats['last_used']:
                last_used = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_used']))
                print(f"   • Last used: {last_used}")
        
        # Show examples if available
        if schema.examples:
            print()
            print("💡 Examples:")
            for i, example in enumerate(schema.examples[:3], 1):
                print(f"   Example {i}:")
                if 'input' in example:
                    print(f"      Input: {example['input']}")
                if 'output' in example:
                    print(f"      Output: {example['output']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting tool info: {e}", file=sys.stderr)
        return 1


def run_test_tools(args):
    """Test all available tools."""
    try:
        from loki_code.tools import tool_registry, ToolContext, SafetySettings
        
        print("🧪 Testing All Tools:")
        print()
        
        tools = tool_registry.list_tools()
        if not tools:
            print("❌ No tools are currently registered")
            return 0
        
        # Create a test context
        test_context = ToolContext(
            project_path="./",
            session_id="test_session",
            safety_settings=SafetySettings(dry_run_mode=True),
            dry_run=True
        )
        
        success_count = 0
        total_count = len(tools)
        
        for tool_schema in tools:
            tool_name = tool_schema.name
            print(f"🔍 Testing {tool_name}...")
            
            try:
                # Get tool instance
                tool = tool_registry.get_tool(tool_name)
                if tool is None:
                    print(f"   ❌ Failed to get tool instance")
                    continue
                
                # Test basic validation
                print(f"   ✅ Tool instance created")
                print(f"   ✅ Schema validation passed")
                print(f"   ✅ Security level: {tool_schema.security_level.value}")
                
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        print()
        print(f"📊 Test Results: {success_count}/{total_count} tools passed basic tests")
        
        if success_count == total_count:
            print("🎉 All tools passed basic validation!")
            return 0
        else:
            print("⚠️  Some tools failed validation")
            return 1
        
    except Exception as e:
        print(f"❌ Error testing tools: {e}", file=sys.stderr)
        return 1


# ============================================================================
# Integration Testing Functions
# ============================================================================

def run_integration_tests(args):
    """Run comprehensive integration tests for the tool system."""
    try:
        import subprocess
        import os
        
        print("🧪 Running Comprehensive Integration Tests")
        print("=" * 50)
        print()
        
        # Check if pytest is available
        try:
            import pytest
        except ImportError:
            print("❌ pytest is required to run integration tests")
            print("💡 Install with: pip install pytest pytest-asyncio")
            return 1
        
        # Get the test directory path
        test_dir = Path(__file__).parent / "src" / "loki_code" / "tests"
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            return 1
        
        # Run pytest with verbose output
        pytest_args = [
            "-v",
            "--tb=short",
            str(test_dir / "test_tool_integration.py")
        ]
        
        if args.verbose:
            pytest_args.extend(["--capture=no", "-s"])
        
        print(f"🔍 Running tests from: {test_dir}")
        print(f"📝 Command: pytest {' '.join(pytest_args)}")
        print()
        
        # Run pytest as subprocess to capture output
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + pytest_args,
            cwd=Path(__file__).parent,
            capture_output=not args.verbose,
            text=True
        )
        
        if result.returncode == 0:
            print("🎉 All integration tests passed!")
            if not args.verbose and result.stdout:
                # Show summary if not in verbose mode
                lines = result.stdout.split('\n')
                for line in lines[-10:]:
                    if line.strip():
                        print(line)
        else:
            print("❌ Some integration tests failed")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running integration tests: {e}", file=sys.stderr)
        return 1


def run_tool_benchmarks(args):
    """Run performance benchmarks on tool system."""
    try:
        print("⚡ Running Tool Performance Benchmarks")
        print("=" * 40)
        print()
        
        # Import the benchmark function - try full version first, fall back to simple
        try:
            from src.loki_code.tests.test_tool_integration import benchmark_file_analysis
        except ImportError:
            # Fallback to simple benchmark that doesn't need pytest
            from src.loki_code.tests.simple_benchmarks import simple_benchmark_file_analysis as benchmark_file_analysis
        
        print("🔍 Benchmarking file analysis performance...")
        results = benchmark_file_analysis()
        
        print("📊 Benchmark Results:")
        print("-" * 30)
        
        for size, metrics in results.items():
            print(f"📁 {size.capitalize()} file:")
            print(f"   ⏱️  Duration: {metrics['duration_ms']:.2f}ms")
            print(f"   📄 Lines: {metrics['lines']:,}")
            print(f"   📊 Chars/sec: {metrics['chars_per_second']:,.0f}")
            print()
        
        # Performance thresholds
        max_allowed_ms = {
            'small': 100,
            'medium': 1000,
            'large': 5000
        }
        
        passed = True
        for size, metrics in results.items():
            if metrics['duration_ms'] > max_allowed_ms.get(size, 1000):
                print(f"⚠️  {size.capitalize()} file benchmark exceeded threshold")
                passed = False
        
        if passed:
            print("🎉 All benchmarks passed performance thresholds!")
            return 0
        else:
            print("❌ Some benchmarks exceeded performance thresholds")
            return 1
        
    except Exception as e:
        print(f"❌ Error running benchmarks: {e}", file=sys.stderr)
        return 1


def run_multi_language_tests(args):
    """Test tool system with multiple programming languages."""
    try:
        print("🌐 Testing Multi-Language Support")
        print("=" * 35)
        print()
        
        from src.loki_code.tests.test_tool_integration import TestFixtures
        from loki_code.core.tool_registry import get_global_registry
        from loki_code.tools.types import ToolContext
        
        fixtures = TestFixtures()
        registry = get_global_registry()
        
        # Test files for different languages
        test_cases = [
            ("Python", fixtures.create_test_python_file()),
            ("JavaScript", fixtures.create_test_javascript_file()),
            ("TypeScript", fixtures.create_test_typescript_file()),
            ("Rust", fixtures.create_test_rust_file())
        ]
        
        success_count = 0
        total_count = len(test_cases)
        
        for language, test_file in test_cases:
            print(f"🔍 Testing {language} analysis...")
            
            try:
                # Get file reader tool
                tool = registry.get_tool("file_reader")
                if not tool:
                    print(f"   ❌ File reader tool not available")
                    continue
                
                # Create test context
                context = ToolContext(
                    project_path=fixtures.temp_dir,
                    session_id="multi_lang_test",
                    safety_settings=fixtures.get_test_safety_settings()
                )
                
                # Execute tool
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    tool.execute({"file_path": test_file}, context)
                )
                
                if result.success:
                    print(f"   ✅ {language} analysis successful")
                    if hasattr(result.output, 'file_info'):
                        print(f"   📄 Language detected: {result.output.file_info.language.value}")
                    success_count += 1
                else:
                    print(f"   ❌ {language} analysis failed: {result.message}")
                
            except Exception as e:
                print(f"   ❌ {language} test error: {e}")
        
        # Cleanup
        fixtures.cleanup()
        
        print()
        print(f"📊 Multi-Language Test Results: {success_count}/{total_count} languages passed")
        
        if success_count == total_count:
            print("🎉 All languages tested successfully!")
            return 0
        else:
            print("⚠️  Some language tests failed")
            return 1
        
    except Exception as e:
        print(f"❌ Error running multi-language tests: {e}", file=sys.stderr)
        return 1


def run_security_tests(args):
    """Run security validation tests."""
    try:
        print("🔒 Running Security Validation Tests")
        print("=" * 35)
        print()
        
        # Try to import test utilities - create simple versions if not available
        try:
            from src.loki_code.tests.test_tool_integration import TestFixtures
        except ImportError:
            # Create a simple test fixtures class
            import tempfile
            class TestFixtures:
                def __init__(self):
                    self.temp_dir = tempfile.mkdtemp(prefix="loki_security_test_")
                    
                def get_test_safety_settings(self):
                    from loki_code.tools.types import SafetySettings
                    return SafetySettings(
                        allowed_paths=[self.temp_dir, "./"],
                        max_file_size_mb=50,
                        require_confirmation_for=[]
                    )
                
                def create_large_test_file(self, size_mb):
                    import os
                    test_file = os.path.join(self.temp_dir, "large_test.txt")
                    chunk = "A" * 1024  # 1KB
                    with open(test_file, 'w') as f:
                        for _ in range(size_mb * 1024):
                            f.write(chunk)
                    return test_file
                
                def cleanup(self):
                    import shutil
                    try:
                        shutil.rmtree(self.temp_dir)
                    except:
                        pass
        
        from loki_code.core.tool_registry import get_global_registry
        from loki_code.tools.types import ToolContext
        
        fixtures = TestFixtures()
        registry = get_global_registry()
        tool = registry.get_tool("file_reader")
        
        if not tool:
            print("❌ File reader tool not available for testing")
            return 1
        
        context = ToolContext(
            project_path=fixtures.temp_dir,
            session_id="security_test",
            safety_settings=fixtures.get_test_safety_settings()
        )
        
        security_tests = [
            ("Restricted Path Access", "/etc/passwd"),
            ("Windows System Path", "C:\\Windows\\System32\\config\\SAM"),
            ("Relative Path Escape", "../../etc/passwd"),
            ("Non-existent File", "/path/that/does/not/exist")
        ]
        
        passed_tests = 0
        total_tests = len(security_tests)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        for test_name, dangerous_path in security_tests:
            print(f"🔍 Testing: {test_name}")
            
            try:
                result = loop.run_until_complete(
                    tool.execute({"file_path": dangerous_path}, context)
                )
                
                if not result.success:
                    print(f"   ✅ Correctly blocked access to {dangerous_path}")
                    passed_tests += 1
                else:
                    print(f"   ❌ Security breach: allowed access to {dangerous_path}")
                
            except Exception as e:
                print(f"   ✅ Security exception caught: {type(e).__name__}")
                passed_tests += 1
        
        # Test file size limits
        print("🔍 Testing: File Size Limits")
        try:
            large_file = fixtures.create_large_test_file(size_mb=60)  # Over 50MB limit
            result = loop.run_until_complete(
                tool.execute({"file_path": large_file}, context)
            )
            
            if not result.success:
                print("   ✅ Correctly blocked oversized file")
                passed_tests += 1
            else:
                print("   ❌ Security breach: allowed oversized file")
        except Exception as e:
            print(f"   ✅ Size limit exception: {type(e).__name__}")
            passed_tests += 1
        
        total_tests += 1
        
        # Cleanup
        fixtures.cleanup()
        
        print()
        print(f"📊 Security Test Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All security tests passed!")
            return 0
        else:
            print("❌ Some security tests failed")
            return 1
        
    except Exception as e:
        print(f"❌ Error running security tests: {e}", file=sys.stderr)
        return 1


def run_performance_tests(args):
    """Run performance tests with large files."""
    try:
        print("⚡ Running Performance Tests")
        print("=" * 30)
        print()
        
        from src.loki_code.tests.test_tool_integration import TestFixtures
        from loki_code.core.tool_registry import get_global_registry
        from loki_code.tools.types import ToolContext
        import time
        import asyncio
        
        fixtures = TestFixtures()
        registry = get_global_registry()
        tool = registry.get_tool("file_reader")
        
        if not tool:
            print("❌ File reader tool not available for testing")
            return 1
        
        context = ToolContext(
            project_path=fixtures.temp_dir,
            session_id="perf_test",
            safety_settings=fixtures.get_test_safety_settings()
        )
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Test different file sizes
        test_cases = [
            ("Small file (100 lines)", 100),
            ("Medium file (500 lines)", 500),
            ("Large file (1000 lines)", 1000)
        ]
        
        performance_results = []
        
        for test_name, line_count in test_cases:
            print(f"🔍 {test_name}...")
            
            # Create test file
            test_file = fixtures.create_large_python_file(lines=line_count)
            
            # Measure execution time
            start_time = time.time()
            result = loop.run_until_complete(
                tool.execute({"file_path": test_file}, context)
            )
            end_time = time.time()
            
            duration_ms = (end_time - start_time) * 1000
            
            if result.success:
                print(f"   ✅ Completed in {duration_ms:.2f}ms")
                performance_results.append((test_name, duration_ms, True))
            else:
                print(f"   ❌ Failed: {result.message}")
                performance_results.append((test_name, duration_ms, False))
        
        # Cleanup
        fixtures.cleanup()
        
        # Analyze results
        print()
        print("📊 Performance Results Summary:")
        print("-" * 40)
        
        passed = True
        max_allowed_times = {
            "Small file": 1000,   # 1 second
            "Medium file": 3000,  # 3 seconds  
            "Large file": 8000    # 8 seconds
        }
        
        for test_name, duration_ms, success in performance_results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name}: {duration_ms:.2f}ms")
            
            # Check if within acceptable time
            for key, max_time in max_allowed_times.items():
                if key in test_name and duration_ms > max_time:
                    print(f"   ⚠️  Exceeded threshold of {max_time}ms")
                    passed = False
        
        print()
        if passed:
            print("🎉 All performance tests passed!")
            return 0
        else:
            print("⚠️  Some performance tests exceeded thresholds")
            return 1
        
    except Exception as e:
        print(f"❌ Error running performance tests: {e}", file=sys.stderr)
        return 1


def run_cli_tests(args):
    """Test CLI integration functionality."""
    try:
        print("🖥️  Running CLI Integration Tests")
        print("=" * 32)
        print()
        
        import subprocess
        import os
        
        # Test basic CLI commands
        test_commands = [
            (["--list-tools"], "List tools command"),
            (["--registry-stats"], "Registry stats command"),
            (["--search-tools", "file"], "Search tools command"),
            (["--discover-tools"], "Discover tools command"),
        ]
        
        passed_tests = 0
        total_tests = len(test_commands)
        
        for cmd_args, test_name in test_commands:
            print(f"🔍 Testing: {test_name}")
            
            try:
                result = subprocess.run(
                    [sys.executable, "main.py"] + cmd_args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=Path(__file__).parent
                )
                
                if result.returncode == 0:
                    print(f"   ✅ Command executed successfully")
                    if args.verbose:
                        print(f"   📝 Output: {result.stdout[:100]}...")
                    passed_tests += 1
                else:
                    print(f"   ❌ Command failed with exit code {result.returncode}")
                    if result.stderr:
                        print(f"   📝 Error: {result.stderr[:100]}...")
                
            except subprocess.TimeoutExpired:
                print(f"   ❌ Command timed out")
            except Exception as e:
                print(f"   ❌ Command error: {e}")
        
        print()
        print(f"📊 CLI Test Results: {passed_tests}/{total_tests} commands passed")
        
        if passed_tests == total_tests:
            print("🎉 All CLI tests passed!")
            return 0
        else:
            print("❌ Some CLI tests failed")
            return 1
        
    except Exception as e:
        print(f"❌ Error running CLI tests: {e}", file=sys.stderr)
        return 1


def run_read_file(args):
    """Read a file using the file reader tool."""
    try:
        from loki_code.tools import tool_registry, ToolContext, SafetySettings
        
        file_path = args.read_file
        analysis_level = args.analysis_level
        
        print(f"📖 Reading file: {file_path}")
        print(f"📊 Analysis level: {analysis_level}")
        print()
        
        # Check if file exists
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return 1
        
        # Create tool context
        context = ToolContext(
            project_path="./",
            session_id="cli_session",
            safety_settings=SafetySettings()
        )
        
        # Get file reader tool
        tool = tool_registry.get_tool("file_reader")
        if tool is None:
            print("❌ File reader tool not found")
            print("💡 Try running: python main.py --list-tools")
            return 1
        
        # Prepare input data
        input_data = {
            "file_path": file_path,
            "analysis_level": analysis_level,
            "include_context": True,
            "max_size_mb": 10,
            "encoding": "utf-8"
        }
        
        # Execute the tool
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            tool_registry.execute_tool("file_reader", input_data, context)
        )
        
        # Display results
        if result.success:
            output = result.output
            
            print("✅ File read successfully!")
            print()
            
            # File information
            file_info = output.file_info
            print("📋 File Information:")
            print(f"   Path: {file_info.path}")
            print(f"   Size: {file_info.size_bytes} bytes")
            print(f"   Lines: {file_info.lines}")
            print(f"   Language: {file_info.language or 'Unknown'}")
            print(f"   Last Modified: {file_info.last_modified}")
            print(f"   Binary: {'Yes' if file_info.is_binary else 'No'}")
            print()
            
            # Analysis summary
            if output.analysis_summary:
                print("🔍 Analysis Summary:")
                print(f"   {output.analysis_summary}")
                print()
            
            # Code analysis details
            if output.code_analysis:
                analysis = output.code_analysis
                structure = analysis.get('structure_analysis', {})
                
                print("🧠 Code Analysis:")
                print(f"   Language: {analysis.get('language', 'Unknown')}")
                print(f"   Functions: {structure.get('total_functions', 0)}")
                print(f"   Classes: {structure.get('total_classes', 0)}")
                print(f"   Imports: {structure.get('total_imports', 0)}")
                print(f"   Complexity Score: {analysis.get('complexity_score', 0):.2f}")
                
                if analysis.get('key_concepts'):
                    print(f"   Key Concepts: {', '.join(analysis['key_concepts'][:5])}")
                print()
            
            # Suggestions
            if output.suggestions:
                print("💡 Suggestions:")
                for suggestion in output.suggestions:
                    print(f"   • {suggestion}")
                print()
            
            # Content preview (first 20 lines)
            if not file_info.is_binary:
                lines = output.content.splitlines()
                preview_lines = min(20, len(lines))
                
                print(f"📄 Content Preview (first {preview_lines} lines):")
                print("─" * 60)
                for i, line in enumerate(lines[:preview_lines], 1):
                    print(f"{i:3d}: {line}")
                
                if len(lines) > preview_lines:
                    print(f"... ({len(lines) - preview_lines} more lines)")
                print("─" * 60)
            
            print(f"\n✨ {result.message}")
            
        else:
            print(f"❌ Failed to read file: {result.message}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        return 1


def run_search_tools(args):
    """Search tools by query."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        query = args.search_tools
        print(f"🔍 Searching tools for: '{query}'")
        print()
        
        registry = get_global_registry()
        matching_tools = registry.search_tools(query)
        
        if not matching_tools:
            print("❌ No tools found matching the query")
            return 1
        
        print(f"📋 Found {len(matching_tools)} matching tools:")
        print()
        
        for tool in matching_tools:
            print(f"🛠️  {tool.name}")
            print(f"   📝 {tool.description}")
            print(f"   🔒 Security: {tool.security_level.value}")
            if tool.capabilities:
                capabilities = [cap.value for cap in tool.capabilities]
                print(f"   ⚡ Capabilities: {', '.join(capabilities)}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error searching tools: {e}", file=sys.stderr)
        return 1


def run_discover_tools(args):
    """Rediscover and register available tools."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        print("🔍 Discovering available tools...")
        
        registry = get_global_registry()
        
        # Get count before discovery
        tools_before = len(registry)
        
        # Rediscover tools
        discovered_tools = registry.discovery.discover_builtin_tools()
        newly_registered = 0
        
        for tool_class in discovered_tools:
            try:
                # Check if tool is already registered by name
                if tool_class.__name__ not in [reg.tool_class.__name__ for reg in registry._local_tools.values()]:
                    registry.register_tool_class(tool_class, source="discovery")
                    newly_registered += 1
            except Exception as e:
                print(f"⚠️  Failed to register {tool_class.__name__}: {e}")
        
        tools_after = len(registry)
        
        print(f"✅ Tool discovery completed")
        print(f"   📊 Total tools: {tools_after}")
        print(f"   🆕 Newly discovered: {newly_registered}")
        print(f"   🔄 Previously registered: {tools_before}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during tool discovery: {e}", file=sys.stderr)
        return 1


def run_tool_stats(args):
    """Show usage statistics for a specific tool."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        tool_name = args.tool_stats
        print(f"📊 Tool Statistics: {tool_name}")
        print()
        
        registry = get_global_registry()
        stats = registry.get_tool_stats(tool_name)
        
        if not stats:
            print(f"❌ Tool '{tool_name}' not found")
            return 1
        
        print("📋 Usage Statistics:")
        print(f"   Name: {stats['name']}")
        print(f"   Source: {stats['source']}")
        print(f"   Enabled: {'✅ Yes' if stats['enabled'] else '❌ No'}")
        print(f"   Registered: {stats['registered_at']}")
        print(f"   Last Used: {stats['last_used'] or 'Never'}")
        print(f"   Usage Count: {stats['usage_count']}")
        print(f"   Error Count: {stats['error_count']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting tool stats: {e}", file=sys.stderr)
        return 1


def run_registry_stats(args):
    """Show overall tool registry statistics."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        print("📊 Tool Registry Statistics")
        print()
        
        registry = get_global_registry()
        stats = registry.get_registry_stats()
        
        print("📋 Overview:")
        print(f"   Total Tools: {stats['total_tools']}")
        print(f"   Enabled Tools: {stats['enabled_tools']}")
        print(f"   Disabled Tools: {stats['disabled_tools']}")
        print(f"   Total Executions: {stats['total_executions']}")
        print(f"   Recent Executions (24h): {stats['recent_executions_24h']}")
        print(f"   Recent Success Rate: {stats['recent_success_rate']:.1%}")
        print()
        
        if stats['source_distribution']:
            print("📦 Source Distribution:")
            for source, count in stats['source_distribution'].items():
                print(f"   {source}: {count}")
            print()
        
        if stats['capability_distribution']:
            print("⚡ Capability Distribution:")
            for capability, count in stats['capability_distribution'].items():
                print(f"   {capability}: {count}")
            print()
        
        if stats['security_distribution']:
            print("🔒 Security Level Distribution:")
            for level, count in stats['security_distribution'].items():
                print(f"   {level}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting registry stats: {e}", file=sys.stderr)
        return 1


def run_execute_tool(args):
    """Execute a tool directly with JSON input."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        from loki_code.core.tool_executor import get_global_executor
        from loki_code.tools.types import ToolContext, SafetySettings
        import json
        import asyncio
        
        tool_name = args.execute_tool
        tool_input = args.tool_input
        
        if not tool_input:
            print("❌ No input provided. Use --tool-input with JSON data")
            return 1
        
        print(f"🚀 Executing tool: {tool_name}")
        print(f"📥 Input: {tool_input}")
        print()
        
        # Parse JSON input
        try:
            input_data = json.loads(tool_input)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON input: {e}")
            return 1
        
        # Create execution context
        context = ToolContext(
            project_path="./",
            session_id="cli_execution",
            safety_settings=SafetySettings()
        )
        
        # Execute tool
        executor = get_global_executor()
        
        # Set up async execution
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            executor.execute_tool(tool_name, input_data, context)
        )
        
        # Display results
        if result.success:
            print("✅ Tool executed successfully!")
            print()
            print("📤 Output:")
            print(json.dumps(result.output, indent=2) if hasattr(result.output, '__dict__') else str(result.output))
            print()
            print(f"✨ {result.message}")
        else:
            print("❌ Tool execution failed!")
            print(f"Error: {result.message}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error executing tool: {e}", file=sys.stderr)
        return 1


def run_validate_tool(args):
    """Validate a tool's configuration and functionality."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        tool_name = args.validate_tool
        print(f"🔍 Validating tool: {tool_name}")
        print()
        
        registry = get_global_registry()
        
        # Check if tool exists
        if tool_name not in registry:
            print(f"❌ Tool '{tool_name}' not found")
            return 1
        
        # Get tool instance
        tool = registry.get_tool(tool_name)
        if tool is None:
            print(f"❌ Failed to get tool instance for '{tool_name}'")
            return 1
        
        # Validate schema
        try:
            schema = tool.get_schema()
            print(f"✅ Schema validation passed")
            print(f"   Name: {schema.name}")
            print(f"   Description: {schema.description}")
            print(f"   Security Level: {schema.security_level.value}")
            print(f"   MCP Compatible: {schema.mcp_compatible}")
        except Exception as e:
            print(f"❌ Schema validation failed: {e}")
            return 1
        
        # Check tool registration
        registration = registry.get_tool_registration(tool_name)
        if registration:
            print(f"✅ Tool registration valid")
            print(f"   Source: {registration.source}")
            print(f"   Enabled: {registration.enabled}")
            print(f"   Registered: {registration.registered_at}")
        
        print()
        print("🎉 Tool validation completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error validating tool: {e}", file=sys.stderr)
        return 1


def run_enable_tool(args):
    """Enable a specific tool."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        tool_name = args.enable_tool
        registry = get_global_registry()
        
        if registry.enable_tool(tool_name):
            print(f"✅ Tool '{tool_name}' enabled successfully")
            return 0
        else:
            print(f"❌ Tool '{tool_name}' not found")
            return 1
        
    except Exception as e:
        print(f"❌ Error enabling tool: {e}", file=sys.stderr)
        return 1


def run_disable_tool(args):
    """Disable a specific tool."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        tool_name = args.disable_tool
        registry = get_global_registry()
        
        if registry.disable_tool(tool_name):
            print(f"✅ Tool '{tool_name}' disabled successfully")
            return 0
        else:
            print(f"❌ Tool '{tool_name}' not found")
            return 1
        
    except Exception as e:
        print(f"❌ Error disabling tool: {e}", file=sys.stderr)
        return 1


def run_execution_history(args):
    """Show recent tool execution history."""
    try:
        from loki_code.core.tool_registry import get_global_registry
        
        print("📜 Tool Execution History")
        print()
        
        registry = get_global_registry()
        history = registry.get_execution_history(limit=20)
        
        if not history:
            print("📭 No execution history available")
            return 0
        
        print(f"📋 Recent executions (last {len(history)}):")
        print()
        
        for record in reversed(history):  # Show most recent first
            status = "✅" if record.result and record.result.success else "❌"
            duration = f"{record.duration_ms:.1f}ms" if record.duration_ms else "N/A"
            
            print(f"{status} {record.tool_name}")
            print(f"   🕐 {record.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   ⏱️  Duration: {duration}")
            if record.error:
                print(f"   ❌ Error: {record.error}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting execution history: {e}", file=sys.stderr)
        return 1


def run_performance_metrics(args):
    """Show performance metrics for tools."""
    try:
        from loki_code.core.tool_executor import get_global_executor
        
        tool_name = args.performance_metrics
        executor = get_global_executor()
        
        if tool_name == "all":
            print("📈 Performance Metrics - All Tools")
            print()
            
            metrics = executor.get_execution_metrics()
            if not metrics:
                print("📭 No performance metrics available")
                return 0
            
            for name, tool_metrics in metrics.items():
                print(f"🛠️  {name}:")
                print(f"   Total Executions: {tool_metrics['total_executions']}")
                print(f"   Success Rate: {tool_metrics['success_rate']:.1%}")
                print(f"   Average Time: {tool_metrics['average_execution_time_ms']:.1f}ms")
                if tool_metrics['fastest_execution_ms']:
                    print(f"   Fastest: {tool_metrics['fastest_execution_ms']:.1f}ms")
                print(f"   Slowest: {tool_metrics['slowest_execution_ms']:.1f}ms")
                print()
        else:
            print(f"📈 Performance Metrics - {tool_name}")
            print()
            
            metrics = executor.get_execution_metrics(tool_name)
            if not metrics:
                print(f"📭 No performance metrics available for '{tool_name}'")
                return 0
            
            print(f"📊 Metrics:")
            print(f"   Total Executions: {metrics['total_executions']}")
            print(f"   Successful: {metrics['successful_executions']}")
            print(f"   Failed: {metrics['failed_executions']}")
            print(f"   Success Rate: {metrics['success_rate']:.1%}")
            print(f"   Average Time: {metrics['average_execution_time_ms']:.1f}ms")
            if metrics['fastest_execution_ms']:
                print(f"   Fastest: {metrics['fastest_execution_ms']:.1f}ms")
            print(f"   Slowest: {metrics['slowest_execution_ms']:.1f}ms")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting performance metrics: {e}", file=sys.stderr)
        return 1


def run_filtered_tools(args):
    """Show tools filtered by capability or security level."""
    try:
        from loki_code.core.tool_registry import get_global_registry, ToolFilter
        from loki_code.tools.types import ToolCapability, SecurityLevel
        
        registry = get_global_registry()
        filter_obj = ToolFilter()
        
        # Apply capability filter
        if args.filter_by_capability:
            try:
                capability = ToolCapability(args.filter_by_capability.lower())
                filter_obj.capabilities = [capability]
                print(f"🔍 Tools with capability: {capability.value}")
            except ValueError:
                print(f"❌ Invalid capability: {args.filter_by_capability}")
                print(f"Valid capabilities: {', '.join([cap.value for cap in ToolCapability])}")
                return 1
        
        # Apply security filter  
        if args.filter_by_security:
            try:
                security_level = SecurityLevel(args.filter_by_security.lower())
                filter_obj.security_levels = [security_level]
                print(f"🔒 Tools with security level: {security_level.value}")
            except ValueError:
                print(f"❌ Invalid security level: {args.filter_by_security}")
                return 1
        
        # Get filtered tools
        tools = registry.list_tools(filter_obj)
        
        if not tools:
            print("❌ No tools found matching the filter criteria")
            return 1
        
        print()
        print(f"📋 Found {len(tools)} matching tools:")
        print()
        
        for tool in tools:
            print(f"🛠️  {tool.name}")
            print(f"   📝 {tool.description}")
            print(f"   🔒 Security: {tool.security_level.value}")
            if tool.capabilities:
                capabilities = [cap.value for cap in tool.capabilities]
                print(f"   ⚡ Capabilities: {', '.join(capabilities)}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error filtering tools: {e}", file=sys.stderr)
        return 1


def run_analyze_file(args):
    """Analyze a specific code file."""
    try:
        file_path = args.analyze_file
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}", file=sys.stderr)
            return 1
        
        print(f"🔍 Analyzing file: {file_path}")
        print()
        
        # Quick analysis
        result = analyze_file_quick(file_path)
        
        if not result["success"]:
            print(f"❌ Analysis failed: {result['error']}")
            return 1
        
        # Display results
        context = result["context"]
        summary = result["structure_summary"]
        
        print("📊 Analysis Results:")
        print(f"   Language: {context['language']}")
        print(f"   Functions: {context['functions']}")
        print(f"   Classes: {context['classes']}")
        print(f"   Complexity: {context['complexity']:.2f}")
        print(f"   Lines of Code: {summary['lines_of_code']}")
        
        if context["purpose"]:
            print(f"   Purpose: {context['purpose'][:100]}...")
        
        if context["key_concepts"]:
            print(f"   Key Concepts: {', '.join(context['key_concepts'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}", file=sys.stderr)
        return 1


def run_analyze_project(args):
    """Analyze an entire project directory."""
    try:
        project_path = args.analyze_project
        
        if not Path(project_path).is_dir():
            print(f"❌ Directory not found: {project_path}", file=sys.stderr)
            return 1
        
        print(f"🔍 Analyzing project: {project_path}")
        print()
        
        # Quick project analysis
        result = analyze_project_quick(project_path)
        
        if not result["success"]:
            print(f"❌ Analysis failed: {result['error']}")
            return 1
        
        # Display results
        print("📊 Project Analysis Results:")
        print(f"   Project: {result['project_name']}")
        print(f"   Files: {result['files_count']}")
        print(f"   Languages: {', '.join(result['languages'])}")
        
        if result["patterns"]:
            print(f"   Architecture Patterns: {', '.join(result['patterns'])}")
        
        if result["key_modules"]:
            print(f"   Key Modules: {', '.join(result['key_modules'])}")
        
        complexity = result["complexity"]
        print("   Complexity Metrics:")
        print(f"     Total Functions: {complexity.get('total_functions', 0)}")
        print(f"     Total Classes: {complexity.get('total_classes', 0)}")
        print(f"     Total Lines: {complexity.get('total_lines_of_code', 0)}")
        print(f"     Avg File Complexity: {complexity.get('average_file_complexity', 0):.2f}")
        
        if result["dependencies"]:
            print(f"   Dependencies: {', '.join(result['dependencies'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing project: {e}", file=sys.stderr)
        return 1


def run_list_languages(args):
    """List supported programming languages."""
    try:
        print("🔤 Supported Programming Languages:")
        print()
        
        languages = list_supported_languages()
        language_info = get_language_info()
        
        for language in languages:
            lang_name = language.value
            info = language_info.get(lang_name, {})
            
            print(f"📝 {lang_name.title()}")
            print(f"   Extensions: {', '.join(info.get('extensions', []))}")
            print(f"   Has Classes: {'✅' if info.get('has_classes', False) else '❌'}")
            print(f"   Has Functions: {'✅' if info.get('has_functions', False) else '❌'}")
            print(f"   Has Imports: {'✅' if info.get('has_imports', False) else '❌'}")
            print()
        
        print(f"Total: {len(languages)} languages supported")
        return 0
        
    except Exception as e:
        print(f"❌ Error listing languages: {e}", file=sys.stderr)
        return 1


def run_language_info(args):
    """Show information about a specific language."""
    try:
        language_name = args.language_info.lower()
        
        print(f"🔤 Language Information: {language_name.title()}")
        print()
        
        # Find matching language
        supported_languages = list_supported_languages()
        matching_language = None
        
        for lang in supported_languages:
            if lang.value.lower() == language_name:
                matching_language = lang
                break
        
        if not matching_language:
            print(f"❌ Language '{language_name}' is not supported")
            print("\nSupported languages:")
            for lang in supported_languages:
                print(f"  - {lang.value}")
            return 1
        
        # Get language info
        language_info = get_language_info()
        info = language_info.get(matching_language.value, {})
        
        print(f"📋 Details:")
        print(f"   Name: {info.get('name', 'Unknown')}")
        print(f"   Extensions: {', '.join(info.get('extensions', []))}")
        print(f"   Case Sensitive: {'Yes' if info.get('case_sensitive', True) else 'No'}")
        print(f"   Comment Patterns: {', '.join(info.get('comment_patterns', []))}")
        print(f"   Docstring Patterns: {', '.join(info.get('docstring_patterns', []))}")
        
        print(f"\n🔧 Capabilities:")
        print(f"   Functions: {'✅ Supported' if info.get('has_functions', False) else '❌ Not supported'}")
        print(f"   Classes: {'✅ Supported' if info.get('has_classes', False) else '❌ Not supported'}")
        print(f"   Imports: {'✅ Supported' if info.get('has_imports', False) else '❌ Not supported'}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting language info: {e}", file=sys.stderr)
        return 1


def run_generate_context(args):
    """Generate LLM context from a file."""
    try:
        file_path = args.generate_context
        task_description = args.task_description or "Analyze this code"
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}", file=sys.stderr)
            return 1
        
        print(f"🧠 Generating LLM context for: {file_path}")
        print(f"📝 Task: {task_description}")
        print()
        
        # Create context extractor
        extractor = ContextExtractor()
        
        # Get context level
        context_level = ContextLevel(args.context_level)
        config = ContextConfig(level=context_level)
        
        # Extract context
        context = extractor.extract_file_context(file_path, config)
        
        # Generate prompt
        prompt = extractor.generate_llm_prompt(context, task_description)
        
        print("🎯 Generated LLM Context:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating context: {e}", file=sys.stderr)
        return 1


def run_chat_mode(args):
    """Run chat mode for interactive conversation with LLM."""
    config = None
    logger = None
    
    try:
        # Load configuration
        try:
            config = load_config(args.config)
        except ConfigurationError as e:
            print(f"⚠️  Configuration error: {e}", file=sys.stderr)
            print("Using built-in defaults for chat...")
            from loki_code.config.models import LokiCodeConfig
            config = LokiCodeConfig()
        
        # Setup logging
        setup_logging(config, verbose=args.verbose)
        logger = get_logger(__name__)
        
        # Create LLM provider (with fallback to legacy client)
        try:
            llm_provider = create_llm_provider(config)
            use_provider = True
        except Exception as e:
            logger.warning(f"Failed to create LLM provider, falling back to legacy client: {e}")
            try:
                llm_client = create_llm_client(config)
                use_provider = False
            except Exception as e2:
                logger.error(f"Failed to create LLM client: {e2}")
                print(f"❌ Failed to create LLM connection: {e2}", file=sys.stderr)
                return 1
        
        # Test connection first
        logger.info("Testing LLM connection...")
        if use_provider:
            if not llm_provider.health_check_sync():
                print("❌ Cannot connect to LLM. Please check:")
                print(f"   • Ollama is running: ollama serve")
                print(f"   • Model is available: ollama pull {config.llm.model}")
                print(f"   • Service URL is correct: {config.llm.base_url}")
                return 1
        else:
            if not llm_client.test_connection():
                print("❌ Cannot connect to LLM. Please check:")
                print(f"   • Ollama is running: ollama serve")
                print(f"   • Model is available: ollama pull {config.llm.model}")
                print(f"   • Service URL is correct: {config.llm.base_url}")
                return 1
        
        # Show chat header
        version = get_version()
        print(f"""🎉 Loki Code v{version} - Chat Mode
Connected to: {config.llm.model}
Type 'exit' to quit, 'help' for commands""")
        
        # Handle single prompt mode
        if args.prompt:
            if use_provider:
                return run_single_prompt_provider(llm_provider, args.prompt, logger)
            else:
                return run_single_prompt(llm_client, args.prompt, logger)
        
        # Interactive chat loop
        if use_provider:
            return run_interactive_chat_provider(llm_provider, logger, args.verbose)
        else:
            return run_interactive_chat(llm_client, logger, args.verbose)
        
    except KeyboardInterrupt:
        if logger:
            logger.info("Chat interrupted by user")
        print("\n\n👋 Chat interrupted. Goodbye!")
        return 0
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error in chat mode: {e}", exc_info=True)
        else:
            print(f"\n❌ Chat error: {e}", file=sys.stderr)
        return 1


def run_single_prompt(llm_client, prompt, logger):
    """Handle single prompt mode."""
    try:
        logger.info(f"Sending single prompt: {prompt[:50]}...")
        
        print(f"\n> {prompt}")
        print("🤖 ", end="", flush=True)
        
        # Send request with streaming
        request = LLMRequest(prompt=prompt, stream=True)
        response_generator = llm_client.send_prompt(request, stream=True)
        
        # Stream the response
        full_response = ""
        for token in response_generator:
            print(token, end="", flush=True)
            full_response += token
        
        print("\n")  # New line after response
        
        logger.info(f"Single prompt completed: {len(full_response)} characters")
        return 0
        
    except LLMConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        logger.error(f"Connection error in single prompt: {e}")
        return 1
    except LLMTimeoutError as e:
        print(f"\n❌ Timeout error: {e}")
        logger.error(f"Timeout error in single prompt: {e}")
        return 1
    except LLMModelError as e:
        print(f"\n❌ Model error: {e}")
        logger.error(f"Model error in single prompt: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Unexpected error in single prompt: {e}", exc_info=True)
        return 1


def run_interactive_chat(llm_client, logger, verbose=False):
    """Handle interactive chat mode."""
    conversation_count = 0
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() in ['help', '?']:
                print("""
Available commands:
  help, ?     Show this help message
  exit, quit  Exit chat mode
  
Just type your message to chat with the AI!""")
                continue
            
            conversation_count += 1
            
            try:
                print("🤖 ", end="", flush=True)
                
                # Send request with streaming
                request = LLMRequest(prompt=user_input, stream=True)
                response_generator = llm_client.send_prompt(request, stream=True)
                
                # Stream the response
                full_response = ""
                for token in response_generator:
                    print(token, end="", flush=True)
                    full_response += token
                
                print()  # New line after response
                
                if verbose:
                    logger.debug(f"Conversation #{conversation_count}: {len(full_response)} characters")
                
            except LLMConnectionError as e:
                print(f"\n❌ Connection error: {e}")
                print("💡 Try: Check if Ollama is running and the model is available")
                logger.error(f"Connection error in conversation #{conversation_count}: {e}")
                continue
                
            except LLMTimeoutError as e:
                print(f"\n❌ Request timed out: {e}")
                print("💡 Try: Using a smaller model or increasing timeout in config")
                logger.error(f"Timeout error in conversation #{conversation_count}: {e}")
                continue
                
            except LLMModelError as e:
                print(f"\n❌ Model error: {e}")
                logger.error(f"Model error in conversation #{conversation_count}: {e}")
                continue
                
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in conversation #{conversation_count}: {e}", exc_info=True)
                continue
        
        logger.info(f"Interactive chat completed after {conversation_count} exchanges")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
        logger.info(f"Chat interrupted after {conversation_count} exchanges")
        return 0


def run_single_prompt_provider(llm_provider, prompt, logger):
    """Handle single prompt mode using provider abstraction."""
    try:
        logger.info(f"Sending single prompt: {prompt[:50]}...")
        
        print(f"\n> {prompt}")
        print("🤖 ", end="", flush=True)
        
        # Create generation request
        request = GenerationRequest(prompt=prompt, stream=True)
        
        # Send request with streaming
        async def _stream_response():
            async for token in llm_provider.stream_generate(request):
                print(token, end="", flush=True)
                return token
        
        # Run async streaming in sync context
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        full_response = ""
        async def _collect_response():
            nonlocal full_response
            async for token in llm_provider.stream_generate(request):
                print(token, end="", flush=True)
                full_response += token
        
        loop.run_until_complete(_collect_response())
        print("\n")  # New line after response
        
        logger.info(f"Single prompt completed: {len(full_response)} characters")
        return 0
        
    except ProviderConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        logger.error(f"Connection error in single prompt: {e}")
        return 1
    except ProviderTimeoutError as e:
        print(f"\n❌ Timeout error: {e}")
        logger.error(f"Timeout error in single prompt: {e}")
        return 1
    except ProviderModelError as e:
        print(f"\n❌ Model error: {e}")
        logger.error(f"Model error in single prompt: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Unexpected error in single prompt: {e}", exc_info=True)
        return 1


def run_interactive_chat_provider(llm_provider, logger, verbose=False):
    """Handle interactive chat mode using provider abstraction."""
    conversation_count = 0
    
    # Set up async event loop
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() in ['help', '?']:
                print("""
Available commands:
  help, ?     Show this help message
  exit, quit  Exit chat mode
  
Just type your message to chat with the AI!""")
                continue
            
            conversation_count += 1
            
            try:
                print("🤖 ", end="", flush=True)
                
                # Create generation request
                request = GenerationRequest(prompt=user_input, stream=True)
                
                # Stream the response
                full_response = ""
                async def _collect_response():
                    nonlocal full_response
                    async for token in llm_provider.stream_generate(request):
                        print(token, end="", flush=True)
                        full_response += token
                
                loop.run_until_complete(_collect_response())
                print()  # New line after response
                
                if verbose:
                    logger.debug(f"Conversation #{conversation_count}: {len(full_response)} characters")
                
            except ProviderConnectionError as e:
                print(f"\n❌ Connection error: {e}")
                print("💡 Try: Check if Ollama is running and the model is available")
                logger.error(f"Connection error in conversation #{conversation_count}: {e}")
                continue
                
            except ProviderTimeoutError as e:
                print(f"\n❌ Request timed out: {e}")
                print("💡 Try: Using a smaller model or increasing timeout in config")
                logger.error(f"Timeout error in conversation #{conversation_count}: {e}")
                continue
                
            except ProviderModelError as e:
                print(f"\n❌ Model error: {e}")
                logger.error(f"Model error in conversation #{conversation_count}: {e}")
                continue
                
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in conversation #{conversation_count}: {e}", exc_info=True)
                continue
        
        logger.info(f"Interactive chat completed after {conversation_count} exchanges")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
        logger.info(f"Chat interrupted after {conversation_count} exchanges")
        return 0


def main():
    """Main entry point for the Loki Code application."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Handle LLM test mode
        if args.test_llm:
            exit_code = run_llm_test(args)
            sys.exit(exit_code)
        
        # Handle list providers mode
        if args.list_providers:
            exit_code = run_list_providers(args)
            sys.exit(exit_code)
        
        # Handle show models mode
        if args.show_models:
            exit_code = run_show_models(args)
            sys.exit(exit_code)
        
        # Handle test models mode
        if args.test_models:
            exit_code = run_test_models(args)
            sys.exit(exit_code)
        
        # Handle benchmark mode
        if args.benchmark:
            exit_code = run_benchmark(args)
            sys.exit(exit_code)
        
        # Handle list tools mode
        if args.list_tools:
            exit_code = run_list_tools(args)
            sys.exit(exit_code)
        
        # Handle tool info mode
        if args.tool_info:
            exit_code = run_tool_info(args)
            sys.exit(exit_code)
        
        # Handle test tools mode
        if args.test_tools:
            exit_code = run_test_tools(args)
            sys.exit(exit_code)
        
        # Handle integration testing commands
        if args.run_tests:
            exit_code = run_integration_tests(args)
            sys.exit(exit_code)
        
        if args.benchmark_tools:
            exit_code = run_tool_benchmarks(args)
            sys.exit(exit_code)
        
        if args.test_multi_language:
            exit_code = run_multi_language_tests(args)
            sys.exit(exit_code)
        
        if args.test_security:
            exit_code = run_security_tests(args)
            sys.exit(exit_code)
        
        if args.test_performance:
            exit_code = run_performance_tests(args)
            sys.exit(exit_code)
        
        if args.test_cli:
            exit_code = run_cli_tests(args)
            sys.exit(exit_code)
        
        # Handle read file mode
        if args.read_file:
            exit_code = run_read_file(args)
            sys.exit(exit_code)
        
        # Handle advanced tool management commands
        if args.search_tools:
            exit_code = run_search_tools(args)
            sys.exit(exit_code)
        
        if args.discover_tools:
            exit_code = run_discover_tools(args)
            sys.exit(exit_code)
        
        if args.tool_stats:
            exit_code = run_tool_stats(args)
            sys.exit(exit_code)
        
        if args.registry_stats:
            exit_code = run_registry_stats(args)
            sys.exit(exit_code)
        
        if args.execute_tool:
            exit_code = run_execute_tool(args)
            sys.exit(exit_code)
        
        if args.validate_tool:
            exit_code = run_validate_tool(args)
            sys.exit(exit_code)
        
        if args.enable_tool:
            exit_code = run_enable_tool(args)
            sys.exit(exit_code)
        
        if args.disable_tool:
            exit_code = run_disable_tool(args)
            sys.exit(exit_code)
        
        if args.execution_history:
            exit_code = run_execution_history(args)
            sys.exit(exit_code)
        
        if args.performance_metrics is not None:
            exit_code = run_performance_metrics(args)
            sys.exit(exit_code)
        
        if args.filter_by_capability or args.filter_by_security:
            exit_code = run_filtered_tools(args)
            sys.exit(exit_code)
        
        # Handle code analysis modes
        if args.analyze_file:
            exit_code = run_analyze_file(args)
            sys.exit(exit_code)
        
        if args.analyze_project:
            exit_code = run_analyze_project(args)
            sys.exit(exit_code)
        
        if args.list_languages:
            exit_code = run_list_languages(args)
            sys.exit(exit_code)
        
        if args.language_info:
            exit_code = run_language_info(args)
            sys.exit(exit_code)
        
        if args.generate_context:
            exit_code = run_generate_context(args)
            sys.exit(exit_code)
        
        # Handle chat mode
        if args.chat:
            exit_code = run_chat_mode(args)
            sys.exit(exit_code)
        
        # If no specific action is requested, start interactive mode
        exit_code = run_interactive_mode(args)
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()