"""
Dynamic prompt builder with tool integration.

This module provides the core prompt building functionality that combines
templates, context, and tool information to create rich, context-aware
prompts for LLM interactions.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .templates import PromptTemplate, PromptTemplateRegistry
from .context import PromptContext, ContextBuilder, ConversationEntry
from ..tool_registry import ToolRegistry
from ...tools.types import ToolSchema
from ...utils.logging import get_logger


class PromptBuildError(Exception):
    """Exception raised when prompt building fails."""
    pass


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM response."""
    tool_name: str
    input_data: Dict[str, Any]
    reasoning: Optional[str] = None
    call_id: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        """Generate call ID if not provided."""
        if not self.call_id:
            import time
            self.call_id = f"call_{self.tool_name}_{int(time.time() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "input_data": self.input_data,
            "reasoning": self.reasoning,
            "call_id": self.call_id,
            "confidence": self.confidence
        }


@dataclass
class BuiltPrompt:
    """A complete, built prompt ready for LLM consumption."""
    system_prompt: str
    user_prompt: str
    template_name: str
    context_summary: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_full_prompt(self) -> str:
        """Get the complete prompt as a single string."""
        return f"System: {self.system_prompt}\n\nUser: {self.user_prompt}"
    
    def get_messages_format(self) -> List[Dict[str, str]]:
        """Get prompt in OpenAI messages format."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt}
        ]


class ToolPromptFormatter:
    """Handles tool-related prompt formatting and parsing."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.logger = get_logger(__name__)
    
    def format_available_tools(self, include_examples: bool = True, max_tools: Optional[int] = None) -> str:
        """Generate a formatted description of available tools."""
        tools = self.tool_registry.list_tools()
        
        if max_tools:
            tools = tools[:max_tools]
        
        if not tools:
            return "No tools are currently available."
        
        descriptions = []
        descriptions.append("Available Tools:")
        descriptions.append("================")
        
        for tool_schema in tools:
            # Tool header
            tool_desc = f"\n**{tool_schema.name}**"
            tool_desc += f"\n- Description: {tool_schema.description}"
            
            # Capabilities
            if tool_schema.capabilities:
                caps = [cap.value for cap in tool_schema.capabilities]
                tool_desc += f"\n- Capabilities: {', '.join(caps)}"
            
            # Security level
            tool_desc += f"\n- Security Level: {tool_schema.security_level.value}"
            
            # Input schema (simplified)
            if tool_schema.input_schema:
                required_fields = tool_schema.input_schema.get("required", [])
                if required_fields:
                    tool_desc += f"\n- Required Input: {', '.join(required_fields)}"
            
            # Example usage
            if include_examples and tool_schema.examples:
                example = tool_schema.examples[0]
                tool_desc += f"\n- Example: {json.dumps(example, indent=2)}"
            
            descriptions.append(tool_desc)
        
        return "\n".join(descriptions)
    
    def format_tool_call(self, tool_name: str, input_data: Dict[str, Any], 
                        template: PromptTemplate) -> str:
        """Format a tool call according to template specification."""
        # Validate tool exists
        if not self.tool_registry.get_tool(tool_name):
            raise PromptBuildError(f"Tool '{tool_name}' not found in registry")
        
        # Format the input data
        input_json = json.dumps(input_data, indent=2)
        
        # Apply template formatting
        return template.tool_calling_format.format(
            tool_name=tool_name,
            input=input_json
        )
    
    def parse_tool_calls(self, llm_response: str) -> List[ToolCall]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        
        # Look for ```tool blocks
        pattern = r'```tool\s*\ntool_name:\s*([^\n]+)\s*\ninput:\s*({.*?})\s*\n```'
        matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        for tool_name, input_json in matches:
            try:
                tool_name = tool_name.strip()
                input_data = json.loads(input_json)
                
                # Extract reasoning if present (look for text before the tool call)
                reasoning = self._extract_reasoning(llm_response, tool_name, input_json)
                
                tool_call = ToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    reasoning=reasoning
                )
                tool_calls.append(tool_call)
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse tool call input JSON: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Failed to parse tool call: {e}")
                continue
        
        return tool_calls
    
    def validate_tool_call(self, tool_call: ToolCall) -> List[str]:
        """Validate a tool call against the tool's schema."""
        errors = []
        
        # Check if tool exists
        tool_schema = self.tool_registry.get_tool_schema(tool_call.tool_name)
        if not tool_schema:
            errors.append(f"Tool '{tool_call.tool_name}' not found")
            return errors
        
        # Validate input against schema
        try:
            required_fields = tool_schema.input_schema.get("required", [])
            for field in required_fields:
                if field not in tool_call.input_data:
                    errors.append(f"Missing required field: {field}")
            
            # Check field types (basic validation)
            properties = tool_schema.input_schema.get("properties", {})
            for field, value in tool_call.input_data.items():
                if field in properties:
                    expected_type = properties[field].get("type")
                    if expected_type and not self._validate_type(value, expected_type):
                        errors.append(f"Field '{field}' has invalid type (expected {expected_type})")
        
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")
        
        return errors
    
    def _extract_reasoning(self, response: str, tool_name: str, input_json: str) -> Optional[str]:
        """Extract reasoning text that appears before a tool call."""
        try:
            # Find the tool call in the response
            tool_block = f"```tool\ntool_name: {tool_name}\ninput: {input_json}\n```"
            tool_index = response.find(tool_block)
            
            if tool_index == -1:
                return None
            
            # Get text before the tool call
            before_tool = response[:tool_index].strip()
            
            # Extract the last paragraph as reasoning
            paragraphs = before_tool.split('\n\n')
            if paragraphs:
                reasoning = paragraphs[-1].strip()
                # Only return if it's substantial and not just formatting
                if len(reasoning) > 10 and not reasoning.startswith('```'):
                    return reasoning
            
            return None
            
        except Exception:
            return None
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Basic type validation for JSON schema types."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it


class PromptBuilder:
    """
    Dynamic prompt builder that combines templates, context, and tools.
    
    This is the main class for creating rich, context-aware prompts
    that enable effective LLM and tool interactions.
    """
    
    def __init__(self, 
                 template_registry: PromptTemplateRegistry,
                 tool_registry: ToolRegistry,
                 context_builder: Optional[ContextBuilder] = None):
        self.template_registry = template_registry
        self.tool_registry = tool_registry
        self.context_builder = context_builder or ContextBuilder()
        self.tool_formatter = ToolPromptFormatter(tool_registry)
        self.logger = get_logger(__name__)
    
    async def build_prompt(self, template_name: str, context: PromptContext) -> BuiltPrompt:
        """Build a complete prompt from template and context."""
        try:
            # Get template
            template = self.template_registry.get_template(template_name)
            if not template:
                raise PromptBuildError(f"Template '{template_name}' not found")
            
            # Build context variables
            context_vars = await self._build_context_variables(template, context)
            
            # Validate context variables
            missing_vars = template.validate_context_vars(context_vars)
            if missing_vars:
                self.logger.warning(f"Missing context variables: {missing_vars}")
                # Fill missing variables with defaults
                for var in missing_vars:
                    context_vars[var] = f"[{var} not available]"
            
            # Format system prompt
            system_prompt = template.system_prompt.format(**context_vars)
            
            # Format user prompt
            user_prompt = template.user_prompt_template.format(**context_vars)
            
            # Estimate token count (rough approximation)
            token_estimate = self._estimate_tokens(system_prompt + user_prompt)
            
            # Check token limits
            if token_estimate > template.max_context_tokens:
                self.logger.warning(f"Prompt exceeds token limit: {token_estimate} > {template.max_context_tokens}")
                # Could implement truncation logic here
            
            # Build context summary for metadata
            context_summary = self._build_context_summary(context)
            
            return BuiltPrompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                template_name=template_name,
                context_summary=context_summary,
                token_estimate=token_estimate,
                metadata={
                    "template_type": template.template_type.value,
                    "personality": template.personality.value,
                    "tools_available": len(self.tool_registry.list_tools()),
                    "context_sections": template.context_sections
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build prompt: {e}")
            raise PromptBuildError(f"Prompt building failed: {str(e)}") from e
    
    async def _build_context_variables(self, template: PromptTemplate, context: PromptContext) -> Dict[str, Any]:
        """Build all context variables needed for the template."""
        context_vars = {}
        
        # Always include user message
        context_vars["user_message"] = context.user_message
        
        # Build available tools description
        context_vars["available_tools"] = self.tool_formatter.format_available_tools(
            include_examples=True,
            max_tools=20  # Limit to prevent prompt bloat
        )
        
        # Build each context section as needed
        for section in template.context_sections:
            if section == "project_context":
                context_vars["project_context"] = await self._build_project_context_text(context)
            elif section == "file_context":
                context_vars["file_context"] = await self._build_file_context_text(context)
            elif section == "conversation_history":
                context_vars["conversation_history"] = self._build_conversation_context_text(context)
            elif section == "current_task":
                context_vars["current_task"] = context.current_task or "General assistance"
            elif section == "error_context":
                context_vars["error_context"] = context.error_context or "No error information"
            elif section == "stack_trace":
                context_vars["stack_trace"] = context.stack_trace or "No stack trace available"
            elif section == "analysis_scope":
                context_vars["analysis_scope"] = context.analysis_scope or "General analysis"
            elif section == "focus_areas":
                context_vars["focus_areas"] = ", ".join(context.focus_areas) if context.focus_areas else "All areas"
            elif section == "environment_context":
                context_vars["environment_context"] = self._format_environment_context(context.environment_context)
            elif section == "target_file":
                context_vars["target_file"] = context.current_file or "No specific file"
            elif section == "current_file":
                context_vars["current_file"] = context.current_file or "No file selected"
            elif section == "related_files":
                context_vars["related_files"] = ", ".join(context.target_files) if context.target_files else "No related files"
            elif section == "file_analysis":
                context_vars["file_analysis"] = await self._build_file_analysis_text(context)
        
        return context_vars
    
    async def _build_project_context_text(self, context: PromptContext) -> str:
        """Build project context text."""
        if context.project_context:
            return context.project_context.to_string()
        elif context.project_path:
            # Build project context on demand
            project_context = self.context_builder.build_project_context(context.project_path)
            context.project_context = project_context
            return project_context.to_string()
        else:
            return "No project context available"
    
    async def _build_file_context_text(self, context: PromptContext) -> str:
        """Build file context text."""
        if not context.current_file and not context.target_files:
            return "No files specified"
        
        contexts = []
        
        # Current file context
        if context.current_file:
            file_context = context.get_file_context(context.current_file)
            if not file_context:
                file_context = await self.context_builder.build_file_context(context.current_file)
                context.set_file_context(context.current_file, file_context)
            contexts.append(f"Current file:\n{file_context.to_string()}")
        
        # Target files context
        for file_path in context.target_files[:5]:  # Limit to 5 files
            file_context = context.get_file_context(file_path)
            if not file_context:
                file_context = await self.context_builder.build_file_context(file_path)
                context.set_file_context(file_path, file_context)
            contexts.append(f"Target file:\n{file_context.to_string()}")
        
        return "\n\n".join(contexts) if contexts else "No file context available"
    
    def _build_conversation_context_text(self, context: PromptContext) -> str:
        """Build conversation context text."""
        return self.context_builder.build_conversation_context(context.conversation_history, max_entries=10)
    
    async def _build_file_analysis_text(self, context: PromptContext) -> str:
        """Build detailed file analysis text."""
        if context.current_file:
            file_context = context.get_file_context(context.current_file)
            if not file_context:
                file_context = await self.context_builder.build_file_context(context.current_file)
                context.set_file_context(context.current_file, file_context)
            return file_context.to_string()
        return "No file selected for analysis"
    
    def _format_environment_context(self, env_context: Dict[str, Any]) -> str:
        """Format environment context as readable text."""
        if not env_context:
            return "No environment information"
        
        formatted = []
        for key, value in env_context.items():
            formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)
    
    def _build_context_summary(self, context: PromptContext) -> Dict[str, Any]:
        """Build a summary of the context for metadata."""
        return {
            "has_project": bool(context.project_path),
            "has_current_file": bool(context.current_file),
            "target_files_count": len(context.target_files),
            "conversation_entries": len(context.conversation_history),
            "has_error_context": bool(context.error_context),
            "focus_areas_count": len(context.focus_areas)
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters ≈ 1 token)."""
        return len(text) // 4
    
    def get_tool_formatter(self) -> ToolPromptFormatter:
        """Get the tool formatter for external use."""
        return self.tool_formatter
    
    def create_context_from_message(self, user_message: str, **kwargs) -> PromptContext:
        """Create a basic PromptContext from a user message."""
        return PromptContext(
            user_message=user_message,
            **kwargs
        )
    
    async def build_simple_prompt(self, template_name: str, user_message: str, **context_kwargs) -> BuiltPrompt:
        """Build a prompt with minimal setup - convenience method."""
        context = self.create_context_from_message(user_message, **context_kwargs)
        return await self.build_prompt(template_name, context)