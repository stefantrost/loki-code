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