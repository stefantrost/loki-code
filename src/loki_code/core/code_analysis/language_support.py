"""
Language support configuration for Tree-sitter integration.

This module defines supported languages and their Tree-sitter configurations,
including queries for extracting functions, classes, imports, and other
code structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import os


class SupportedLanguage(Enum):
    """Supported programming languages for code analysis."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"


@dataclass
class LanguageConfig:
    """Configuration for a specific programming language."""
    tree_sitter_language: str
    file_extensions: List[str]
    function_query: str
    class_query: str
    import_query: str
    variable_query: str
    comment_patterns: List[str]
    docstring_patterns: List[str] = field(default_factory=list)
    # Additional language-specific settings
    has_classes: bool = True
    has_functions: bool = True
    has_imports: bool = True
    case_sensitive: bool = True


# Tree-sitter queries for different code constructs
LANGUAGE_CONFIGS = {
    SupportedLanguage.PYTHON: LanguageConfig(
        tree_sitter_language="python",
        file_extensions=[".py", ".pyw", ".pyi"],
        function_query="""
        (function_definition
          name: (identifier) @function.name
          parameters: (parameters) @function.params
          body: (block) @function.body) @function.def
        
        (async_function_definition
          name: (identifier) @function.name
          parameters: (parameters) @function.params
          body: (block) @function.body) @function.def
        """,
        class_query="""
        (class_definition
          name: (identifier) @class.name
          superclasses: (argument_list)? @class.bases
          body: (block) @class.body) @class.def
        """,
        import_query="""
        (import_statement
          name: (dotted_name) @import.name) @import.stmt
        
        (import_from_statement
          module_name: (dotted_name) @import.module
          name: (dotted_name) @import.name) @import.stmt
        
        (import_from_statement
          module_name: (dotted_name) @import.module
          name: (import_list) @import.names) @import.stmt
        """,
        variable_query="""
        (assignment
          left: (identifier) @variable.name
          right: (_) @variable.value) @variable.assignment
        """,
        comment_patterns=["#"],
        docstring_patterns=['"""', "'''"],
        has_classes=True,
        has_functions=True,
        has_imports=True,
        case_sensitive=True
    ),
    
    SupportedLanguage.JAVASCRIPT: LanguageConfig(
        tree_sitter_language="javascript",
        file_extensions=[".js", ".jsx", ".mjs"],
        function_query="""
        (function_declaration
          name: (identifier) @function.name
          parameters: (formal_parameters) @function.params
          body: (statement_block) @function.body) @function.def
        
        (arrow_function
          parameter: (identifier) @function.params
          body: (_) @function.body) @function.def
        
        (arrow_function
          parameters: (formal_parameters) @function.params
          body: (_) @function.body) @function.def
        
        (method_definition
          name: (property_identifier) @function.name
          parameters: (formal_parameters) @function.params
          body: (statement_block) @function.body) @function.def
        """,
        class_query="""
        (class_declaration
          name: (identifier) @class.name
          superclass: (class_heritage)? @class.bases
          body: (class_body) @class.body) @class.def
        """,
        import_query="""
        (import_statement
          source: (string) @import.source) @import.stmt
        
        (import_statement
          import_clause: (import_clause) @import.clause
          source: (string) @import.source) @import.stmt
        """,
        variable_query="""
        (variable_declaration
          (variable_declarator
            name: (identifier) @variable.name
            value: (_)? @variable.value)) @variable.declaration
        
        (lexical_declaration
          (variable_declarator
            name: (identifier) @variable.name
            value: (_)? @variable.value)) @variable.declaration
        """,
        comment_patterns=["//", "/*"],
        docstring_patterns=["/**"],
        has_classes=True,
        has_functions=True,
        has_imports=True,
        case_sensitive=True
    ),
    
    SupportedLanguage.TYPESCRIPT: LanguageConfig(
        tree_sitter_language="typescript",
        file_extensions=[".ts", ".tsx"],
        function_query="""
        (function_declaration
          name: (identifier) @function.name
          parameters: (formal_parameters) @function.params
          return_type: (type_annotation)? @function.return_type
          body: (statement_block) @function.body) @function.def
        
        (method_definition
          name: (property_identifier) @function.name
          parameters: (formal_parameters) @function.params
          return_type: (type_annotation)? @function.return_type
          body: (statement_block) @function.body) @function.def
        """,
        class_query="""
        (class_declaration
          name: (type_identifier) @class.name
          type_parameters: (type_parameters)? @class.generics
          class_heritage: (class_heritage)? @class.bases
          body: (class_body) @class.body) @class.def
        
        (interface_declaration
          name: (type_identifier) @class.name
          type_parameters: (type_parameters)? @class.generics
          body: (object_type) @class.body) @class.def
        """,
        import_query="""
        (import_statement
          import_clause: (import_clause) @import.clause
          source: (string) @import.source) @import.stmt
        """,
        variable_query="""
        (variable_declaration
          (variable_declarator
            name: (identifier) @variable.name
            type: (type_annotation)? @variable.type
            value: (_)? @variable.value)) @variable.declaration
        """,
        comment_patterns=["//", "/*"],
        docstring_patterns=["/**"],
        has_classes=True,
        has_functions=True,
        has_imports=True,
        case_sensitive=True
    ),
    
    SupportedLanguage.RUST: LanguageConfig(
        tree_sitter_language="rust",
        file_extensions=[".rs"],
        function_query="""
        (function_item
          name: (identifier) @function.name
          parameters: (parameters) @function.params
          return_type: (type_identifier)? @function.return_type
          body: (block) @function.body) @function.def
        """,
        class_query="""
        (struct_item
          name: (type_identifier) @class.name
          body: (field_declaration_list) @class.body) @class.def
        
        (impl_item
          type: (type_identifier) @class.name
          body: (declaration_list) @class.body) @class.def
        """,
        import_query="""
        (use_declaration
          argument: (use_clause) @import.clause) @import.stmt
        """,
        variable_query="""
        (let_declaration
          pattern: (identifier) @variable.name
          type: (type_identifier)? @variable.type
          value: (_)? @variable.value) @variable.declaration
        """,
        comment_patterns=["//", "/*"],
        docstring_patterns=["///", "/**"],
        has_classes=True,
        has_functions=True,
        has_imports=True,
        case_sensitive=True
    ),
    
    SupportedLanguage.GO: LanguageConfig(
        tree_sitter_language="go",
        file_extensions=[".go"],
        function_query="""
        (function_declaration
          name: (identifier) @function.name
          parameters: (parameter_list) @function.params
          result: (parameter_list)? @function.return_type
          body: (block) @function.body) @function.def
        
        (method_declaration
          receiver: (parameter_list) @function.receiver
          name: (field_identifier) @function.name
          parameters: (parameter_list) @function.params
          result: (parameter_list)? @function.return_type
          body: (block) @function.body) @function.def
        """,
        class_query="""
        (type_declaration
          (type_spec
            name: (type_identifier) @class.name
            type: (struct_type) @class.body)) @class.def
        
        (type_declaration
          (type_spec
            name: (type_identifier) @class.name
            type: (interface_type) @class.body)) @class.def
        """,
        import_query="""
        (import_declaration
          (import_spec
            path: (interpreted_string_literal) @import.path)) @import.stmt
        
        (import_declaration
          (import_spec
            name: (package_identifier) @import.alias
            path: (interpreted_string_literal) @import.path)) @import.stmt
        """,
        variable_query="""
        (var_declaration
          (var_spec
            name: (identifier) @variable.name
            type: (type_identifier)? @variable.type
            value: (expression_list)? @variable.value)) @variable.declaration
        """,
        comment_patterns=["//", "/*"],
        docstring_patterns=[],
        has_classes=True,
        has_functions=True,
        has_imports=True,
        case_sensitive=True
    ),
    
    SupportedLanguage.JAVA: LanguageConfig(
        tree_sitter_language="java",
        file_extensions=[".java"],
        function_query="""
        (method_declaration
          modifiers: (modifiers)? @function.modifiers
          type: (_) @function.return_type
          name: (identifier) @function.name
          parameters: (formal_parameters) @function.params
          body: (block) @function.body) @function.def
        """,
        class_query="""
        (class_declaration
          modifiers: (modifiers)? @class.modifiers
          name: (identifier) @class.name
          superclass: (superclass)? @class.bases
          interfaces: (super_interfaces)? @class.interfaces
          body: (class_body) @class.body) @class.def
        
        (interface_declaration
          modifiers: (modifiers)? @class.modifiers
          name: (identifier) @class.name
          extends: (extends_interfaces)? @class.bases
          body: (interface_body) @class.body) @class.def
        """,
        import_query="""
        (import_declaration
          (scoped_identifier) @import.name) @import.stmt
        
        (import_declaration
          (asterisk) @import.wildcard) @import.stmt
        """,
        variable_query="""
        (field_declaration
          modifiers: (modifiers)? @variable.modifiers
          type: (_) @variable.type
          declarator: (variable_declarator
            name: (identifier) @variable.name
            value: (_)? @variable.value)) @variable.declaration
        """,
        comment_patterns=["//", "/*"],
        docstring_patterns=["/**"],
        has_classes=True,
        has_functions=True,
        has_imports=True,
        case_sensitive=True
    )
}


def get_language_from_extension(file_path: str) -> Optional[SupportedLanguage]:
    """Get the programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SupportedLanguage if recognized, None otherwise
    """
    if not file_path:
        return None
    
    # Get the file extension
    _, ext = os.path.splitext(file_path.lower())
    
    # Search through language configurations
    for language, config in LANGUAGE_CONFIGS.items():
        if ext in config.file_extensions:
            return language
    
    return None


def get_language_config(language: SupportedLanguage) -> Optional[LanguageConfig]:
    """Get configuration for a specific language.
    
    Args:
        language: The programming language
        
    Returns:
        LanguageConfig if supported, None otherwise
    """
    return LANGUAGE_CONFIGS.get(language)


def is_language_supported(language: SupportedLanguage) -> bool:
    """Check if a language is supported.
    
    Args:
        language: The programming language to check
        
    Returns:
        True if supported, False otherwise
    """
    return language in LANGUAGE_CONFIGS


def list_supported_languages() -> List[SupportedLanguage]:
    """Get list of all supported languages.
    
    Returns:
        List of supported programming languages
    """
    return list(LANGUAGE_CONFIGS.keys())


def get_supported_extensions() -> List[str]:
    """Get list of all supported file extensions.
    
    Returns:
        List of file extensions (including the dot)
    """
    extensions = []
    for config in LANGUAGE_CONFIGS.values():
        extensions.extend(config.file_extensions)
    return sorted(set(extensions))


def get_language_info() -> Dict[str, Dict[str, any]]:
    """Get information about all supported languages.
    
    Returns:
        Dictionary with language information
    """
    info = {}
    for language, config in LANGUAGE_CONFIGS.items():
        info[language.value] = {
            "name": language.value,
            "extensions": config.file_extensions,
            "has_classes": config.has_classes,
            "has_functions": config.has_functions,
            "has_imports": config.has_imports,
            "case_sensitive": config.case_sensitive,
            "comment_patterns": config.comment_patterns,
            "docstring_patterns": config.docstring_patterns
        }
    return info