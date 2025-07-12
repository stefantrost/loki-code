"""
Prompt template definitions and registry for Loki Code.

This module contains the core prompt template system including:
- PromptTemplate dataclass for template definitions
- PromptTemplateRegistry for managing templates
- Pre-defined templates for different agent types
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from ...utils.logging import get_logger


class AgentPersonality(Enum):
    """Different agent personalities for different use cases."""
    HELPFUL = "helpful"
    CONCISE = "concise"
    DETAILED = "detailed"
    FORMAL = "formal"
    CASUAL = "casual"
    EXPERT = "expert"


class TemplateType(Enum):
    """Categories of prompt templates."""
    GENERAL = "general"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    EXPLANATION = "explanation"
    REFACTORING = "refactoring"


@dataclass
class PromptTemplate:
    """
    A prompt template for LLM interactions with tool capabilities.
    
    This defines the structure and content for prompts that enable
    the LLM to effectively use our tool system.
    """
    name: str
    description: str
    template_type: TemplateType
    system_prompt: str
    user_prompt_template: str
    tool_calling_format: str
    context_sections: List[str] = field(default_factory=list)
    max_context_tokens: int = 4000
    personality: AgentPersonality = AgentPersonality.HELPFUL
    requires_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate template after initialization."""
        if not self.name:
            raise ValueError("Template name is required")
        if not self.system_prompt:
            raise ValueError("System prompt is required")
        if not self.user_prompt_template:
            raise ValueError("User prompt template is required")
        if not self.tool_calling_format:
            raise ValueError("Tool calling format is required")
    
    def get_context_requirements(self) -> List[str]:
        """Get list of required context sections."""
        return self.context_sections.copy()
    
    def validate_context_vars(self, context_vars: Dict[str, Any]) -> List[str]:
        """Validate that all required context variables are present."""
        missing = []
        
        # Extract variables from system prompt
        import re
        system_vars = re.findall(r'\{(\w+)\}', self.system_prompt)
        user_vars = re.findall(r'\{(\w+)\}', self.user_prompt_template)
        
        all_vars = set(system_vars + user_vars)
        
        for var in all_vars:
            if var not in context_vars:
                missing.append(var)
        
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "template_type": self.template_type.value,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "tool_calling_format": self.tool_calling_format,
            "context_sections": self.context_sections,
            "max_context_tokens": self.max_context_tokens,
            "personality": self.personality.value,
            "requires_tools": self.requires_tools,
            "optional_tools": self.optional_tools,
            "examples": self.examples,
            "metadata": self.metadata,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            template_type=TemplateType(data["template_type"]),
            system_prompt=data["system_prompt"],
            user_prompt_template=data["user_prompt_template"],
            tool_calling_format=data["tool_calling_format"],
            context_sections=data.get("context_sections", []),
            max_context_tokens=data.get("max_context_tokens", 4000),
            personality=AgentPersonality(data.get("personality", "helpful")),
            requires_tools=data.get("requires_tools", []),
            optional_tools=data.get("optional_tools", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0")
        )


class PromptTemplateRegistry:
    """
    Registry for managing prompt templates.
    
    Provides storage, retrieval, and management of prompt templates
    for different use cases and agent personalities.
    """
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._logger = get_logger(__name__)
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new template."""
        if template.name in self._templates:
            self._logger.warning(f"Overriding existing template: {template.name}")
        
        self._templates[template.name] = template
        self._logger.info(f"Registered prompt template: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """Get list of all template names."""
        return list(self._templates.keys())
    
    def get_templates_by_type(self, template_type: TemplateType) -> List[PromptTemplate]:
        """Get all templates of a specific type."""
        return [
            template for template in self._templates.values()
            if template.template_type == template_type
        ]
    
    def get_templates_by_personality(self, personality: AgentPersonality) -> List[PromptTemplate]:
        """Get all templates with a specific personality."""
        return [
            template for template in self._templates.values()
            if template.personality == personality
        ]
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name."""
        if name in self._templates:
            del self._templates[name]
            self._logger.info(f"Removed template: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all templates."""
        self._templates.clear()
        self._logger.info("Cleared all templates")
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a template."""
        template = self.get_template(name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "description": template.description,
            "type": template.template_type.value,
            "personality": template.personality.value,
            "context_sections": template.context_sections,
            "max_tokens": template.max_context_tokens,
            "requires_tools": template.requires_tools,
            "optional_tools": template.optional_tools,
            "version": template.version
        }
    
    def export_templates(self, template_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export templates to dictionary format."""
        if template_names is None:
            templates_to_export = self._templates
        else:
            templates_to_export = {
                name: template for name, template in self._templates.items()
                if name in template_names
            }
        
        return {
            name: template.to_dict()
            for name, template in templates_to_export.items()
        }
    
    def import_templates(self, templates_data: Dict[str, Any]) -> int:
        """Import templates from dictionary format."""
        imported_count = 0
        
        for name, template_data in templates_data.items():
            try:
                template = PromptTemplate.from_dict(template_data)
                self.register_template(template)
                imported_count += 1
            except Exception as e:
                self._logger.error(f"Failed to import template {name}: {e}")
        
        return imported_count


# Pre-defined prompt templates

CODING_AGENT_TEMPLATE = PromptTemplate(
    name="coding_agent",
    description="General purpose coding assistant with comprehensive tool access",
    template_type=TemplateType.GENERAL,
    system_prompt="""You are Loki Code, an intelligent coding assistant with access to powerful tools for code analysis and manipulation.

## Your Identity
You are a helpful, knowledgeable coding assistant that:
- Understands code deeply through analysis tools
- Provides clear, actionable advice
- Uses tools strategically to gather context
- Explains your reasoning process
- Offers multiple solutions when appropriate

## Your Capabilities
You can:
- Read and analyze code files with deep understanding
- Understand project structure and dependencies
- Help with coding tasks, debugging, and refactoring
- Provide intelligent suggestions based on code analysis
- Execute safe operations on the codebase
- Explain complex code concepts clearly

## Available Tools
{available_tools}

## Tool Usage Guidelines
- Always use tools to gather context before answering complex questions
- Use file_reader to understand code structure before suggesting changes
- Combine multiple tools for comprehensive analysis
- Explain why you're using specific tools
- Be security-conscious about file operations

## Tool Calling Format
When you need to use a tool, format your response as:
```tool
tool_name: {tool_name}
input: {input_json}
```

## Current Context
Project: {project_context}
Current task: {current_task}
Files in focus: {file_context}

## Conversation History
{conversation_history}

Remember: Always explain your thought process and use tools to provide the most accurate and helpful responses.""",
    
    user_prompt_template="""User Request: {user_message}

Please help with the above request. Use available tools as needed to gather context and provide a comprehensive response.""",
    
    tool_calling_format="```tool\ntool_name: {tool_name}\ninput: {input}\n```",
    
    context_sections=[
        "project_context", 
        "file_context", 
        "conversation_history", 
        "available_tools",
        "current_task"
    ],
    max_context_tokens=3500,
    personality=AgentPersonality.HELPFUL,
    optional_tools=["file_reader", "code_analyzer"],
    examples=[
        {
            "user_input": "Help me understand this Python function",
            "expected_tools": ["file_reader"],
            "reasoning": "Need to read the file first to understand the function"
        }
    ]
)

CODE_REVIEW_TEMPLATE = PromptTemplate(
    name="code_review",
    description="Specialized template for thorough code review and analysis",
    template_type=TemplateType.CODE_REVIEW,
    system_prompt="""You are a senior code reviewer with access to advanced analysis tools. Your mission is to provide thorough, constructive code reviews.

## Your Review Philosophy
- Focus on code quality, security, and maintainability
- Provide specific, actionable feedback
- Suggest improvements with reasoning
- Balance critique with positive observations
- Consider performance and scalability implications

## Review Focus Areas
1. **Code Quality**: Readability, structure, naming conventions
2. **Security**: Potential vulnerabilities and secure practices
3. **Performance**: Efficiency and optimization opportunities
4. **Best Practices**: Language-specific patterns and standards
5. **Maintainability**: Code organization and documentation
6. **Testing**: Test coverage and quality

## Available Tools
{available_tools}

## Analysis Context
File being reviewed: {current_file}
Project context: {project_context}
Previous analysis: {file_analysis}

## Review Process
1. Analyze the code structure and logic
2. Identify specific issues and opportunities
3. Provide concrete suggestions with examples
4. Prioritize findings by importance
5. Suggest follow-up actions

Use tools to gather comprehensive context before providing your review.""",
    
    user_prompt_template="""Code Review Request: {user_message}

Please provide a thorough code review focusing on the specified areas. Use tools to analyze the code and provide detailed, actionable feedback.""",
    
    tool_calling_format="```tool\ntool_name: {tool_name}\ninput: {input}\n```",
    
    context_sections=[
        "current_file",
        "project_context", 
        "file_analysis",
        "available_tools"
    ],
    max_context_tokens=4000,
    personality=AgentPersonality.EXPERT,
    requires_tools=["file_reader"],
    optional_tools=["code_analyzer"],
    examples=[
        {
            "user_input": "Review this function for security issues",
            "expected_tools": ["file_reader", "code_analyzer"],
            "reasoning": "Need detailed analysis to identify security vulnerabilities"
        }
    ]
)

DEBUGGING_TEMPLATE = PromptTemplate(
    name="debugging",
    description="Specialized template for debugging and problem solving",
    template_type=TemplateType.DEBUGGING,
    system_prompt="""You are a debugging expert with deep code analysis capabilities. Your approach is systematic and thorough.

## Your Debugging Methodology
1. **Understand the Problem**: Clarify symptoms and expected behavior
2. **Gather Context**: Analyze relevant code and environment
3. **Form Hypotheses**: Identify potential root causes
4. **Test Systematically**: Validate hypotheses with evidence
5. **Provide Solutions**: Offer targeted fixes with explanations

## Debugging Focus
- Root cause analysis over symptom treatment
- Understanding data flow and control flow
- Identifying edge cases and error conditions
- Considering environmental factors
- Providing both immediate fixes and long-term improvements

## Available Tools
{available_tools}

## Current Context
Error details: {error_context}
Stack trace: {stack_trace}
Related files: {related_files}
Environment: {environment_context}

## Analysis Framework
- **Symptom Analysis**: What's happening vs. what should happen
- **Code Path Tracing**: Following execution flow
- **Data State Inspection**: Variable values and transformations
- **Dependency Analysis**: External factors and interactions
- **Pattern Recognition**: Similar issues and known solutions

Use tools strategically to gather evidence and validate hypotheses.""",
    
    user_prompt_template="""Debugging Request: {user_message}

Please help debug this issue using systematic analysis. Use available tools to gather context and trace the problem to its root cause.""",
    
    tool_calling_format="```tool\ntool_name: {tool_name}\ninput: {input}\n```",
    
    context_sections=[
        "error_context",
        "stack_trace", 
        "related_files",
        "environment_context",
        "available_tools"
    ],
    max_context_tokens=4000,
    personality=AgentPersonality.EXPERT,
    requires_tools=["file_reader"],
    optional_tools=["code_analyzer"],
    examples=[
        {
            "user_input": "Function throws ValueError but I don't understand why",
            "expected_tools": ["file_reader"],
            "reasoning": "Need to analyze the function implementation to understand the error"
        }
    ]
)

FILE_ANALYSIS_TEMPLATE = PromptTemplate(
    name="file_analysis",
    description="Specialized template for deep file and code analysis",
    template_type=TemplateType.ANALYSIS,
    system_prompt="""You are a code analysis expert specializing in comprehensive file examination and understanding.

## Your Analysis Approach
- **Structure Analysis**: Classes, functions, modules, and their relationships
- **Complexity Assessment**: Cyclomatic complexity and maintainability metrics
- **Pattern Recognition**: Design patterns and architectural decisions
- **Dependency Mapping**: Imports, exports, and external dependencies
- **Quality Evaluation**: Code quality indicators and improvement opportunities

## Analysis Dimensions
1. **Functional Analysis**: What the code does and how it works
2. **Structural Analysis**: How the code is organized and architected
3. **Quality Analysis**: Code quality, readability, and maintainability
4. **Security Analysis**: Potential security concerns and best practices
5. **Performance Analysis**: Efficiency and optimization opportunities

## Available Tools
{available_tools}

## Current Focus
Target file: {target_file}
Analysis scope: {analysis_scope}
Project context: {project_context}

## Analysis Framework
- Start with high-level structure understanding
- Dive into implementation details
- Identify key patterns and decisions
- Assess quality and maintainability
- Provide actionable insights

Use tools to perform thorough analysis and provide comprehensive insights.""",
    
    user_prompt_template="""File Analysis Request: {user_message}

Please provide a comprehensive analysis of the specified file(s). Use available tools to examine structure, quality, and provide insights.""",
    
    tool_calling_format="```tool\ntool_name: {tool_name}\ninput: {input}\n```",
    
    context_sections=[
        "target_file",
        "analysis_scope",
        "project_context",
        "available_tools"
    ],
    max_context_tokens=3500,
    personality=AgentPersonality.DETAILED,
    requires_tools=["file_reader"],
    optional_tools=["code_analyzer"],
    examples=[
        {
            "user_input": "Analyze the architecture of this Python module",
            "expected_tools": ["file_reader", "code_analyzer"],
            "reasoning": "Need both file reading and detailed analysis capabilities"
        }
    ]
)

PROJECT_ANALYSIS_TEMPLATE = PromptTemplate(
    name="project_analysis",
    description="Template for analyzing entire projects and codebases",
    template_type=TemplateType.ANALYSIS,
    system_prompt="""You are a software architect with expertise in analyzing entire projects and codebases.

## Your Analysis Scope
- **Architecture Overview**: High-level structure and design patterns
- **Module Relationships**: Dependencies and interaction patterns
- **Technology Stack**: Languages, frameworks, and tools used
- **Code Organization**: Directory structure and file organization
- **Quality Assessment**: Overall code quality and maintainability
- **Growth Potential**: Scalability and extensibility considerations

## Analysis Methodology
1. **Project Structure**: Understand overall organization
2. **Key Components**: Identify main modules and their purposes
3. **Dependency Analysis**: Map relationships and dependencies
4. **Quality Metrics**: Assess code quality across the project
5. **Architectural Patterns**: Identify design patterns and decisions
6. **Recommendations**: Suggest improvements and optimizations

## Available Tools
{available_tools}

## Project Context
Project path: {project_path}
Analysis focus: {analysis_focus}
Key areas of interest: {focus_areas}

## Analysis Framework
- Begin with project structure overview
- Examine key files and modules
- Analyze patterns and architectural decisions
- Assess quality and maintainability
- Provide strategic recommendations

Use tools systematically to build comprehensive project understanding.""",
    
    user_prompt_template="""Project Analysis Request: {user_message}

Please provide a comprehensive analysis of the project. Use available tools to examine structure, architecture, and provide strategic insights.""",
    
    tool_calling_format="```tool\ntool_name: {tool_name}\ninput: {input}\n```",
    
    context_sections=[
        "project_path",
        "analysis_focus",
        "focus_areas", 
        "available_tools"
    ],
    max_context_tokens=4000,
    personality=AgentPersonality.EXPERT,
    requires_tools=["file_reader"],
    optional_tools=["project_analyzer", "code_analyzer"],
    examples=[
        {
            "user_input": "Analyze the overall architecture of this web application",
            "expected_tools": ["file_reader", "project_analyzer"],
            "reasoning": "Need project-level analysis capabilities for architectural assessment"
        }
    ]
)


def create_default_template_registry() -> PromptTemplateRegistry:
    """Create a registry with all default templates."""
    registry = PromptTemplateRegistry()
    
    # Register all default templates
    registry.register_template(CODING_AGENT_TEMPLATE)
    registry.register_template(CODE_REVIEW_TEMPLATE)
    registry.register_template(DEBUGGING_TEMPLATE)
    registry.register_template(FILE_ANALYSIS_TEMPLATE)
    registry.register_template(PROJECT_ANALYSIS_TEMPLATE)
    
    return registry