"""
Test and validation utilities for the prompt system.

This module provides functions to test prompt templates and validate
the prompt system functionality.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from .templates import create_default_template_registry, PromptTemplate
from .builder import PromptBuilder, ToolCall
from .context import PromptContext, ContextBuilder, ConversationEntry
from ..tool_registry import get_global_registry
from ...utils.logging import get_logger


class PromptSystemValidator:
    """Validates the prompt system functionality."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def validate_complete_system(self) -> Dict[str, Any]:
        """Run complete validation of the prompt system."""
        results = {
            "template_validation": await self.validate_templates(),
            "context_building": await self.validate_context_building(),
            "prompt_building": await self.validate_prompt_building(),
            "tool_integration": await self.validate_tool_integration(),
            "tool_call_parsing": self.validate_tool_call_parsing()
        }
        
        # Overall success
        results["overall_success"] = all(
            result.get("success", False) for result in results.values()
        )
        
        return results
    
    async def validate_templates(self) -> Dict[str, Any]:
        """Validate all template definitions."""
        try:
            registry = create_default_template_registry()
            templates = registry.list_templates()
            
            validation_results = {}
            for template_name in templates:
                template = registry.get_template(template_name)
                validation_results[template_name] = self._validate_single_template(template)
            
            success = all(result["valid"] for result in validation_results.values())
            
            return {
                "success": success,
                "template_count": len(templates),
                "templates": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Template validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_single_template(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate a single template."""
        issues = []
        
        # Check required fields
        if not template.name:
            issues.append("Missing template name")
        if not template.system_prompt:
            issues.append("Missing system prompt")
        if not template.user_prompt_template:
            issues.append("Missing user prompt template")
        if not template.tool_calling_format:
            issues.append("Missing tool calling format")
        
        # Check system prompt structure
        if "{available_tools}" not in template.system_prompt:
            issues.append("System prompt should include {available_tools}")
        
        # Check user prompt structure
        if "{user_message}" not in template.user_prompt_template:
            issues.append("User prompt should include {user_message}")
        
        # Check tool calling format
        if "{tool_name}" not in template.tool_calling_format:
            issues.append("Tool calling format should include {tool_name}")
        if "{input}" not in template.tool_calling_format:
            issues.append("Tool calling format should include {input}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    async def validate_context_building(self) -> Dict[str, Any]:
        """Validate context building functionality."""
        try:
            context_builder = ContextBuilder()
            
            # Test basic context creation
            context = PromptContext(user_message="Test message")
            
            # Test conversation history
            context.add_conversation_entry("user", "Hello")
            context.add_conversation_entry("assistant", "Hi there!")
            
            # Test environment context
            env_context = context_builder.build_environment_context()
            
            # Test conversation context building
            conv_text = context_builder.build_conversation_context(context.conversation_history)
            
            return {
                "success": True,
                "conversation_entries": len(context.conversation_history),
                "environment_keys": len(env_context),
                "conversation_text_length": len(conv_text)
            }
            
        except Exception as e:
            self.logger.error(f"Context building validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_prompt_building(self) -> Dict[str, Any]:
        """Validate prompt building functionality."""
        try:
            # Create components
            template_registry = create_default_template_registry()
            tool_registry = get_global_registry()
            context_builder = ContextBuilder()
            
            builder = PromptBuilder(template_registry, tool_registry, context_builder)
            
            # Test simple prompt building
            context = PromptContext(
                user_message="Help me understand this code",
                current_task="Code analysis"
            )
            
            prompt = await builder.build_simple_prompt("coding_agent", "Test message")
            
            # Validate prompt structure
            validation = {
                "has_system_prompt": bool(prompt.system_prompt),
                "has_user_prompt": bool(prompt.user_prompt),
                "system_prompt_length": len(prompt.system_prompt),
                "user_prompt_length": len(prompt.user_prompt),
                "token_estimate": prompt.token_estimate,
                "template_name": prompt.template_name
            }
            
            # Check that essential elements are present
            essential_checks = {
                "has_loki_identity": "Loki Code" in prompt.system_prompt,
                "has_tools_section": "available tools" in prompt.system_prompt.lower(),
                "has_user_message": "Test message" in prompt.user_prompt,
                "has_tool_format": "```tool" in prompt.system_prompt
            }
            
            validation.update(essential_checks)
            validation["success"] = all(essential_checks.values())
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Prompt building validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_tool_integration(self) -> Dict[str, Any]:
        """Validate tool integration in prompts."""
        try:
            template_registry = create_default_template_registry()
            tool_registry = get_global_registry()
            
            builder = PromptBuilder(template_registry, tool_registry)
            tool_formatter = builder.get_tool_formatter()
            
            # Test tool description formatting
            tools_description = tool_formatter.format_available_tools(include_examples=True, max_tools=5)
            
            # Test tool call formatting
            template = template_registry.get_template("coding_agent")
            tool_call_text = tool_formatter.format_tool_call(
                "file_reader",
                {"file_path": "test.py", "analysis_level": "standard"},
                template
            )
            
            return {
                "success": True,
                "tools_description_length": len(tools_description),
                "has_tool_descriptions": "file_reader" in tools_description.lower(),
                "tool_call_formatted": "tool_name: file_reader" in tool_call_text,
                "tool_call_has_input": "file_path" in tool_call_text
            }
            
        except Exception as e:
            self.logger.error(f"Tool integration validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_tool_call_parsing(self) -> Dict[str, Any]:
        """Validate tool call parsing functionality."""
        try:
            template_registry = create_default_template_registry()
            tool_registry = get_global_registry()
            
            builder = PromptBuilder(template_registry, tool_registry)
            tool_formatter = builder.get_tool_formatter()
            
            # Test parsing valid tool calls
            valid_response = """
            I need to read the file to understand its structure.
            
            ```tool
            tool_name: file_reader
            input: {
              "file_path": "example.py",
              "analysis_level": "detailed"
            }
            ```
            
            This will help me analyze the code.
            """
            
            tool_calls = tool_formatter.parse_tool_calls(valid_response)
            
            # Test parsing invalid tool calls
            invalid_response = """
            ```tool
            tool_name: file_reader
            input: { invalid json }
            ```
            """
            
            invalid_calls = tool_formatter.parse_tool_calls(invalid_response)
            
            return {
                "success": True,
                "valid_calls_parsed": len(tool_calls),
                "invalid_calls_rejected": len(invalid_calls) == 0,
                "parsed_tool_name": tool_calls[0].tool_name if tool_calls else None,
                "parsed_input_valid": bool(tool_calls and tool_calls[0].input_data)
            }
            
        except Exception as e:
            self.logger.error(f"Tool call parsing validation failed: {e}")
            return {"success": False, "error": str(e)}


async def run_prompt_system_tests():
    """Run comprehensive tests of the prompt system."""
    print("🧪 Running Prompt System Tests")
    print("=" * 40)
    
    validator = PromptSystemValidator()
    results = await validator.validate_complete_system()
    
    # Print results
    for test_category, result in results.items():
        if test_category == "overall_success":
            continue
            
        status = "✅" if result.get("success", False) else "❌"
        print(f"{status} {test_category.replace('_', ' ').title()}")
        
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            print(f"   Error: {error}")
        else:
            # Print some key metrics
            if "template_count" in result:
                print(f"   Templates validated: {result['template_count']}")
            if "conversation_entries" in result:
                print(f"   Conversation entries: {result['conversation_entries']}")
            if "token_estimate" in result:
                print(f"   Token estimate: {result['token_estimate']}")
    
    print()
    overall_status = "✅" if results["overall_success"] else "❌"
    print(f"{overall_status} Overall System Status: {'PASSED' if results['overall_success'] else 'FAILED'}")
    
    return results


def test_template_usage_examples():
    """Test template usage with examples."""
    print("\n📝 Template Usage Examples")
    print("-" * 30)
    
    # Example 1: Simple coding agent
    print("Example 1: Simple Coding Agent")
    print("Template: coding_agent")
    print("Context: General help request")
    print("Expected: Helpful assistant with tool access")
    print()
    
    # Example 2: Code review specialist
    print("Example 2: Code Review Specialist")
    print("Template: code_review")
    print("Context: File analysis with security focus")
    print("Expected: Expert reviewer focused on quality")
    print()
    
    # Example 3: Debugging expert
    print("Example 3: Debugging Expert")
    print("Template: debugging")
    print("Context: Error analysis with stack trace")
    print("Expected: Systematic debugger with analysis")
    print()


if __name__ == "__main__":
    # Run tests
    results = asyncio.run(run_prompt_system_tests())
    
    # Show examples
    test_template_usage_examples()
    
    # Exit with appropriate code
    exit(0 if results["overall_success"] else 1)