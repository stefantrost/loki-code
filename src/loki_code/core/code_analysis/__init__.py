"""
Code analysis package for Loki Code.

This package provides comprehensive code analysis capabilities using Tree-sitter
for multi-language parsing, structure extraction, and LLM context generation.

Core Components:
- TreeSitterParser: Multi-language code parsing using Tree-sitter
- CodeAnalyzer: Structure extraction from parsed code
- ContextExtractor: LLM-ready context generation

Usage:
    from loki_code.core.code_analysis import TreeSitterParser, CodeAnalyzer, ContextExtractor
    
    # Parse and analyze code
    parser = TreeSitterParser()
    analyzer = CodeAnalyzer(parser)
    extractor = ContextExtractor(parser)
    
    # Analyze a file
    structure = analyzer.analyze_file("example.py")
    context = extractor.extract_file_context("example.py")
    
    # Generate LLM prompt
    prompt = extractor.generate_llm_prompt(context, "Explain this code")
"""

from .tree_sitter_parser import (
    TreeSitterParser,
    ParseResult,
    ParserStats
)

from .code_analyzer import (
    CodeAnalyzer,
    CodeStructure,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    VariableInfo
)

from .context_extractor import (
    ContextExtractor,
    ContextLevel,
    ContextConfig,
    FunctionContext,
    ClassContext,
    FileContext,
    ProjectContext
)

from .language_support import (
    SupportedLanguage,
    LanguageConfig,
    get_language_from_extension,
    get_language_config,
    is_language_supported,
    list_supported_languages,
    get_supported_extensions,
    get_language_info
)

# Version info
__version__ = "0.1.0"

# Export all public APIs
__all__ = [
    # Parser
    "TreeSitterParser",
    "ParseResult", 
    "ParserStats",
    
    # Analyzer
    "CodeAnalyzer",
    "CodeStructure",
    "FunctionInfo",
    "ClassInfo", 
    "ImportInfo",
    "VariableInfo",
    
    # Context Extractor
    "ContextExtractor",
    "ContextLevel",
    "ContextConfig",
    "FunctionContext",
    "ClassContext",
    "FileContext", 
    "ProjectContext",
    
    # Language Support
    "SupportedLanguage",
    "LanguageConfig",
    "get_language_from_extension",
    "get_language_config",
    "is_language_supported",
    "list_supported_languages",
    "get_supported_extensions",
    "get_language_info"
]

# System information for discovery
CODE_ANALYSIS_INFO = {
    "version": __version__,
    "supported_languages": [lang.value for lang in SupportedLanguage],
    "context_levels": [level.value for level in ContextLevel],
    "capabilities": [
        "multi_language_parsing",
        "code_structure_extraction", 
        "llm_context_generation",
        "project_analysis",
        "complexity_metrics"
    ]
}


def get_code_analysis_info() -> dict:
    """Get information about the code analysis system.
    
    Returns:
        Dictionary with system information
    """
    return CODE_ANALYSIS_INFO.copy()


def create_analyzer_pipeline(cache_enabled: bool = True) -> tuple:
    """Create a complete code analysis pipeline.
    
    Args:
        cache_enabled: Whether to enable caching
        
    Returns:
        Tuple of (parser, analyzer, extractor) ready to use
    """
    parser = TreeSitterParser(cache_enabled=cache_enabled)
    analyzer = CodeAnalyzer(parser)
    extractor = ContextExtractor(parser)
    
    return parser, analyzer, extractor


def analyze_file_quick(file_path: str) -> dict:
    """Quick file analysis with minimal setup.
    
    Args:
        file_path: Path to file to analyze
        
    Returns:
        Dictionary with analysis results
    """
    try:
        parser, analyzer, extractor = create_analyzer_pipeline()
        
        # Analyze the file
        structure = analyzer.analyze_file(file_path)
        context = extractor.extract_file_context(file_path)
        
        return {
            "success": True,
            "structure_summary": structure.get_summary(),
            "context": {
                "language": context.language,
                "purpose": context.purpose,
                "functions": len(context.functions),
                "classes": len(context.classes),
                "complexity": context.complexity_score,
                "key_concepts": context.key_concepts[:5]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def analyze_project_quick(project_path: str) -> dict:
    """Quick project analysis with minimal setup.
    
    Args:
        project_path: Path to project directory
        
    Returns:
        Dictionary with analysis results
    """
    try:
        parser, analyzer, extractor = create_analyzer_pipeline()
        
        # Analyze the project
        project_context = extractor.extract_project_context(project_path)
        
        return {
            "success": True,
            "project_name": project_context.name,
            "languages": project_context.main_languages,
            "files_count": len(project_context.files),
            "patterns": project_context.architecture_patterns,
            "complexity": project_context.complexity_metrics,
            "key_modules": project_context.key_modules,
            "dependencies": project_context.dependencies[:10]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }