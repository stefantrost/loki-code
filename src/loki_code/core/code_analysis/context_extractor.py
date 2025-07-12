"""
Context extractor for generating LLM-ready context from code analysis.

This module provides the ContextExtractor class that takes analyzed code
structures and converts them into rich, contextual information that can
be used by LLMs for code understanding, generation, and assistance.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .code_analyzer import CodeStructure, FunctionInfo, ClassInfo, ImportInfo, VariableInfo
from .tree_sitter_parser import TreeSitterParser
from .language_support import SupportedLanguage
from ...utils.logging import get_logger


class ContextLevel(Enum):
    """Different levels of context detail for LLM consumption."""
    MINIMAL = "minimal"          # Just basic structure
    STANDARD = "standard"        # Standard detail level
    DETAILED = "detailed"        # Full detail with code snippets
    COMPREHENSIVE = "comprehensive"  # Everything including dependencies


@dataclass
class ContextConfig:
    """Configuration for context extraction."""
    level: ContextLevel = ContextLevel.STANDARD
    include_code_snippets: bool = True
    include_comments: bool = True
    include_docstrings: bool = True
    include_imports: bool = True
    include_variables: bool = False
    max_snippet_length: int = 200
    max_functions_per_class: int = 10
    include_complexity_metrics: bool = True
    include_relationships: bool = True


@dataclass
class FunctionContext:
    """LLM context for a function."""
    name: str
    signature: str
    purpose: Optional[str] = None
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_info: Optional[str] = None
    complexity: Optional[float] = None
    snippet: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)


@dataclass
class ClassContext:
    """LLM context for a class."""
    name: str
    purpose: Optional[str] = None
    inheritance: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    methods: List[FunctionContext] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    complexity: Optional[float] = None
    relationships: List[str] = field(default_factory=list)


@dataclass
class FileContext:
    """LLM context for a file."""
    path: str
    language: str
    purpose: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    exported_items: List[str] = field(default_factory=list)
    functions: List[FunctionContext] = field(default_factory=list)
    classes: List[ClassContext] = field(default_factory=list)
    global_variables: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    key_concepts: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """LLM context for an entire project."""
    name: str
    structure: Dict[str, Any] = field(default_factory=dict)
    files: List[FileContext] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    main_languages: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    key_modules: List[str] = field(default_factory=list)


class ContextExtractor:
    """
    Extractor for generating LLM-ready context from code analysis.
    
    Converts analyzed code structures into rich, contextual information
    that LLMs can use for understanding and generating code.
    """
    
    def __init__(self, parser: Optional[TreeSitterParser] = None):
        """Initialize the context extractor.
        
        Args:
            parser: Optional TreeSitterParser instance
        """
        self.logger = get_logger(__name__)
        self.parser = parser or TreeSitterParser()
        
        # Default configuration
        self.default_config = ContextConfig()
        
        # Cache for extracted contexts
        self._context_cache: Dict[str, FileContext] = {}
    
    def extract_file_context(self, file_path: str, config: Optional[ContextConfig] = None) -> FileContext:
        """Extract LLM context from a single file.
        
        Args:
            file_path: Path to the file to analyze
            config: Optional configuration for extraction
            
        Returns:
            FileContext with LLM-ready information
        """
        config = config or self.default_config
        
        # Check cache
        cache_key = f"{file_path}:{config.level.value}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        try:
            # Parse and analyze the file
            from .code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer(self.parser)
            structure = analyzer.analyze_file(file_path)
            
            # Convert to context
            context = self._structure_to_file_context(structure, config)
            
            # Cache the result
            self._context_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error extracting context from {file_path}: {e}")
            return FileContext(
                path=file_path,
                language="unknown",
                purpose="Error during analysis"
            )
    
    def extract_code_context(self, code: str, language: SupportedLanguage, 
                           config: Optional[ContextConfig] = None) -> FileContext:
        """Extract LLM context from code string.
        
        Args:
            code: Source code to analyze
            language: Programming language of the code
            config: Optional configuration for extraction
            
        Returns:
            FileContext with LLM-ready information
        """
        config = config or self.default_config
        
        try:
            # Analyze the code
            from .code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer(self.parser)
            structure = analyzer.analyze_code(code, language)
            
            # Convert to context
            return self._structure_to_file_context(structure, config)
            
        except Exception as e:
            self.logger.error(f"Error extracting context from code: {e}")
            return FileContext(
                path="<inline>",
                language=language.value,
                purpose="Error during analysis"
            )
    
    def extract_project_context(self, project_path: str, config: Optional[ContextConfig] = None) -> ProjectContext:
        """Extract LLM context from an entire project.
        
        Args:
            project_path: Path to the project directory
            config: Optional configuration for extraction
            
        Returns:
            ProjectContext with project-wide information
        """
        config = config or self.default_config
        
        try:
            project_dir = Path(project_path)
            project_name = project_dir.name
            
            # Parse all files in the project
            from .code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer(self.parser)
            parse_results = self.parser.parse_directory(
                str(project_dir), 
                recursive=True,
                exclude_patterns=["*.pyc", "__pycache__", ".git", "node_modules", "venv", ".env"]
            )
            
            # Analyze each file
            file_contexts = []
            all_structures = []
            
            for parse_result in parse_results:
                if parse_result.success:
                    structure = analyzer.analyze_parse_result(parse_result)
                    all_structures.append(structure)
                    
                    file_context = self._structure_to_file_context(structure, config)
                    file_contexts.append(file_context)
            
            # Build project context
            project_context = ProjectContext(name=project_name)
            project_context.files = file_contexts
            project_context.structure = self._analyze_project_structure(project_dir)
            project_context.dependencies = self._extract_project_dependencies(all_structures)
            project_context.main_languages = self._get_main_languages(all_structures)
            project_context.architecture_patterns = self._detect_architecture_patterns(all_structures)
            project_context.complexity_metrics = self._calculate_project_complexity(all_structures)
            project_context.key_modules = self._identify_key_modules(file_contexts)
            
            return project_context
            
        except Exception as e:
            self.logger.error(f"Error extracting project context from {project_path}: {e}")
            return ProjectContext(name=Path(project_path).name)
    
    def generate_llm_prompt(self, context: Union[FileContext, ProjectContext], 
                          task_description: str) -> str:
        """Generate an LLM prompt with context and task description.
        
        Args:
            context: FileContext or ProjectContext
            task_description: Description of the task for the LLM
            
        Returns:
            Formatted prompt string for the LLM
        """
        if isinstance(context, FileContext):
            return self._generate_file_prompt(context, task_description)
        elif isinstance(context, ProjectContext):
            return self._generate_project_prompt(context, task_description)
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
    
    def extract_relevant_context(self, query: str, contexts: List[FileContext], 
                                max_contexts: int = 5) -> List[FileContext]:
        """Extract most relevant contexts for a query.
        
        Args:
            query: User query or task description
            contexts: List of available file contexts
            max_contexts: Maximum number of contexts to return
            
        Returns:
            List of most relevant FileContext objects
        """
        # Simple relevance scoring based on keyword matching
        scored_contexts = []
        
        query_words = set(query.lower().split())
        
        for context in contexts:
            score = 0.0
            
            # Score based on file path
            if any(word in context.path.lower() for word in query_words):
                score += 3.0
            
            # Score based on function names
            for func in context.functions:
                if any(word in func.name.lower() for word in query_words):
                    score += 2.0
            
            # Score based on class names
            for cls in context.classes:
                if any(word in cls.name.lower() for word in query_words):
                    score += 2.0
            
            # Score based on imports
            for imp in context.imports:
                if any(word in imp.lower() for word in query_words):
                    score += 1.0
            
            # Score based on key concepts
            for concept in context.key_concepts:
                if any(word in concept.lower() for word in query_words):
                    score += 1.5
            
            scored_contexts.append((score, context))
        
        # Sort by score and return top contexts
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        return [context for _, context in scored_contexts[:max_contexts]]
    
    def _structure_to_file_context(self, structure: CodeStructure, config: ContextConfig) -> FileContext:
        """Convert CodeStructure to FileContext.
        
        Args:
            structure: CodeStructure from analyzer
            config: Configuration for extraction
            
        Returns:
            FileContext with LLM-ready information
        """
        context = FileContext(
            path=structure.file_path or "<unknown>",
            language=structure.language.value,
            summary=structure.get_summary(),
            complexity_score=structure.complexity_score
        )
        
        # Extract purpose from docstrings or comments
        if structure.docstrings:
            context.purpose = structure.docstrings[0][:200] + "..." if len(structure.docstrings[0]) > 200 else structure.docstrings[0]
        elif structure.comments:
            context.purpose = structure.comments[0][:100] + "..." if len(structure.comments[0]) > 100 else structure.comments[0]
        
        # Extract imports
        if config.include_imports:
            context.imports = [f"{imp.module}" + (f".{'.'.join(imp.names)}" if imp.names else "") 
                             for imp in structure.imports]
        
        # Extract functions
        for func in structure.functions:
            func_context = self._function_to_context(func, config)
            context.functions.append(func_context)
        
        # Extract classes
        for cls in structure.classes:
            cls_context = self._class_to_context(cls, config)
            context.classes.append(cls_context)
        
        # Extract global variables
        context.global_variables = [var.name for var in structure.variables 
                                  if var.scope == "global"]
        
        # Extract key concepts
        context.key_concepts = self._extract_key_concepts(structure)
        
        # Identify exported items (simplified)
        context.exported_items = [func.name for func in structure.functions if not func.name.startswith('_')]
        context.exported_items.extend([cls.name for cls in structure.classes if not cls.name.startswith('_')])
        
        return context
    
    def _function_to_context(self, func: FunctionInfo, config: ContextConfig) -> FunctionContext:
        """Convert FunctionInfo to FunctionContext.
        
        Args:
            func: FunctionInfo from analyzer
            config: Configuration for extraction
            
        Returns:
            FunctionContext with LLM-ready information
        """
        # Build signature
        params_str = ", ".join(func.parameters)
        signature = f"{func.name}({params_str})"
        if func.return_type:
            signature += f" -> {func.return_type}"
        
        # Extract purpose from docstring
        purpose = None
        if func.docstring:
            purpose = func.docstring[:150] + "..." if len(func.docstring) > 150 else func.docstring
        
        # Build parameter info
        parameters = []
        for param in func.parameters:
            param_info = {"name": param}
            # Add type info if available (simplified)
            if ":" in param:
                name, type_hint = param.split(":", 1)
                param_info["name"] = name.strip()
                param_info["type"] = type_hint.strip()
            parameters.append(param_info)
        
        # Get code snippet
        snippet = None
        if config.include_code_snippets and func.body:
            snippet = func.body[:config.max_snippet_length]
            if len(func.body) > config.max_snippet_length:
                snippet += "..."
        
        return FunctionContext(
            name=func.name,
            signature=signature,
            purpose=purpose,
            parameters=parameters,
            return_info=func.return_type,
            complexity=func.complexity_score,
            snippet=snippet
        )
    
    def _class_to_context(self, cls: ClassInfo, config: ContextConfig) -> ClassContext:
        """Convert ClassInfo to ClassContext.
        
        Args:
            cls: ClassInfo from analyzer
            config: Configuration for extraction
            
        Returns:
            ClassContext with LLM-ready information
        """
        # Extract purpose from docstring
        purpose = None
        if cls.docstring:
            purpose = cls.docstring[:150] + "..." if len(cls.docstring) > 150 else cls.docstring
        
        # Convert methods (limit based on config)
        methods = []
        for method in cls.methods[:config.max_functions_per_class]:
            method_context = self._function_to_context(method, config)
            method_context.name = method.name  # Mark as method
            methods.append(method_context)
        
        return ClassContext(
            name=cls.name,
            purpose=purpose,
            inheritance=cls.base_classes,
            interfaces=cls.interfaces,
            methods=methods,
            properties=cls.properties
        )
    
    def _extract_key_concepts(self, structure: CodeStructure) -> List[str]:
        """Extract key concepts from code structure.
        
        Args:
            structure: CodeStructure to analyze
            
        Returns:
            List of key concept strings
        """
        concepts = set()
        
        # Extract from function names
        for func in structure.functions:
            # Split camelCase and snake_case
            words = func.name.replace("_", " ").split()
            concepts.update(word.lower() for word in words if len(word) > 2)
        
        # Extract from class names
        for cls in structure.classes:
            words = cls.name.replace("_", " ").split()
            concepts.update(word.lower() for word in words if len(word) > 2)
        
        # Extract from imports
        for imp in structure.imports:
            module_parts = imp.module.split(".")
            concepts.update(part.lower() for part in module_parts if len(part) > 2)
        
        # Filter common programming terms
        programming_terms = {"get", "set", "create", "init", "main", "test", "util", "lib"}
        concepts -= programming_terms
        
        return sorted(list(concepts))
    
    def _analyze_project_structure(self, project_dir: Path) -> Dict[str, Any]:
        """Analyze project directory structure.
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            "directories": [],
            "files_by_type": {},
            "total_files": 0
        }
        
        try:
            for item in project_dir.rglob("*"):
                if item.is_file():
                    structure["total_files"] += 1
                    ext = item.suffix.lower()
                    if ext not in structure["files_by_type"]:
                        structure["files_by_type"][ext] = 0
                    structure["files_by_type"][ext] += 1
                elif item.is_dir():
                    rel_path = str(item.relative_to(project_dir))
                    structure["directories"].append(rel_path)
        
        except Exception as e:
            self.logger.error(f"Error analyzing project structure: {e}")
        
        return structure
    
    def _extract_project_dependencies(self, structures: List[CodeStructure]) -> List[str]:
        """Extract project-wide dependencies.
        
        Args:
            structures: List of CodeStructure objects
            
        Returns:
            List of dependency names
        """
        dependencies = set()
        
        for structure in structures:
            for imp in structure.imports:
                if imp.module:
                    # Extract top-level module name
                    top_module = imp.module.split(".")[0]
                    dependencies.add(top_module)
        
        return sorted(list(dependencies))
    
    def _get_main_languages(self, structures: List[CodeStructure]) -> List[str]:
        """Get main programming languages used in the project.
        
        Args:
            structures: List of CodeStructure objects
            
        Returns:
            List of language names sorted by usage
        """
        language_counts = {}
        
        for structure in structures:
            lang = structure.language.value
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Sort by usage
        sorted_langs = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        return [lang for lang, _ in sorted_langs]
    
    def _detect_architecture_patterns(self, structures: List[CodeStructure]) -> List[str]:
        """Detect architectural patterns in the project.
        
        Args:
            structures: List of CodeStructure objects
            
        Returns:
            List of detected pattern names
        """
        patterns = set()
        
        # Simple pattern detection based on file/class names
        all_names = []
        for structure in structures:
            all_names.extend([func.name.lower() for func in structure.functions])
            all_names.extend([cls.name.lower() for cls in structure.classes])
            if structure.file_path:
                all_names.append(Path(structure.file_path).stem.lower())
        
        # Detect common patterns
        if any("controller" in name for name in all_names):
            patterns.add("MVC")
        if any("service" in name for name in all_names):
            patterns.add("Service Layer")
        if any("repository" in name for name in all_names):
            patterns.add("Repository Pattern")
        if any("factory" in name for name in all_names):
            patterns.add("Factory Pattern")
        if any("observer" in name for name in all_names):
            patterns.add("Observer Pattern")
        if any("adapter" in name for name in all_names):
            patterns.add("Adapter Pattern")
        
        return sorted(list(patterns))
    
    def _calculate_project_complexity(self, structures: List[CodeStructure]) -> Dict[str, float]:
        """Calculate project-wide complexity metrics.
        
        Args:
            structures: List of CodeStructure objects
            
        Returns:
            Dictionary with complexity metrics
        """
        total_complexity = sum(s.complexity_score for s in structures)
        total_files = len(structures)
        total_functions = sum(len(s.functions) for s in structures)
        total_classes = sum(len(s.classes) for s in structures)
        total_lines = sum(s.lines_of_code for s in structures)
        
        return {
            "total_complexity": round(total_complexity, 2),
            "average_file_complexity": round(total_complexity / max(total_files, 1), 2),
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_lines_of_code": total_lines,
            "functions_per_file": round(total_functions / max(total_files, 1), 2),
            "classes_per_file": round(total_classes / max(total_files, 1), 2)
        }
    
    def _identify_key_modules(self, contexts: List[FileContext]) -> List[str]:
        """Identify key modules in the project.
        
        Args:
            contexts: List of FileContext objects
            
        Returns:
            List of key module names
        """
        # Score modules based on various factors
        module_scores = {}
        
        for context in contexts:
            score = 0.0
            
            # Score based on complexity
            score += context.complexity_score * 0.3
            
            # Score based on number of functions/classes
            score += len(context.functions) * 0.5
            score += len(context.classes) * 1.0
            
            # Score based on imports (likely to be central)
            score += len(context.imports) * 0.2
            
            # Score based on file name (main, core, etc.)
            path_lower = context.path.lower()
            if any(keyword in path_lower for keyword in ["main", "core", "app", "index"]):
                score += 5.0
            
            module_scores[context.path] = score
        
        # Return top modules
        sorted_modules = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)
        return [Path(module).stem for module, _ in sorted_modules[:5]]
    
    def _generate_file_prompt(self, context: FileContext, task_description: str) -> str:
        """Generate LLM prompt for file context.
        
        Args:
            context: FileContext to include
            task_description: Task description
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"# Code Analysis Context",
            f"",
            f"## File: {context.path}",
            f"Language: {context.language}",
            f"Complexity Score: {context.complexity_score}",
        ]
        
        if context.purpose:
            prompt_parts.extend([
                f"",
                f"## Purpose",
                f"{context.purpose}"
            ])
        
        if context.imports:
            prompt_parts.extend([
                f"",
                f"## Imports",
                f"```",
                *context.imports,
                f"```"
            ])
        
        if context.functions:
            prompt_parts.extend([
                f"",
                f"## Functions ({len(context.functions)})"
            ])
            for func in context.functions[:5]:  # Limit to first 5
                prompt_parts.append(f"- `{func.signature}`")
                if func.purpose:
                    prompt_parts.append(f"  {func.purpose}")
        
        if context.classes:
            prompt_parts.extend([
                f"",
                f"## Classes ({len(context.classes)})"
            ])
            for cls in context.classes[:3]:  # Limit to first 3
                prompt_parts.append(f"- `{cls.name}`")
                if cls.purpose:
                    prompt_parts.append(f"  {cls.purpose}")
                if cls.methods:
                    prompt_parts.append(f"  Methods: {', '.join(m.name for m in cls.methods[:3])}")
        
        if context.key_concepts:
            prompt_parts.extend([
                f"",
                f"## Key Concepts",
                f"{', '.join(context.key_concepts[:10])}"
            ])
        
        prompt_parts.extend([
            f"",
            f"## Task",
            f"{task_description}",
            f"",
            f"Please provide assistance based on the above code context."
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_project_prompt(self, context: ProjectContext, task_description: str) -> str:
        """Generate LLM prompt for project context.
        
        Args:
            context: ProjectContext to include
            task_description: Task description
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"# Project Analysis Context",
            f"",
            f"## Project: {context.name}",
            f"Languages: {', '.join(context.main_languages)}",
            f"Files: {len(context.files)}",
        ]
        
        if context.architecture_patterns:
            prompt_parts.extend([
                f"Patterns: {', '.join(context.architecture_patterns)}"
            ])
        
        if context.dependencies:
            prompt_parts.extend([
                f"",
                f"## Dependencies",
                f"{', '.join(context.dependencies[:10])}"
            ])
        
        if context.key_modules:
            prompt_parts.extend([
                f"",
                f"## Key Modules",
                f"{', '.join(context.key_modules)}"
            ])
        
        if context.complexity_metrics:
            prompt_parts.extend([
                f"",
                f"## Complexity Metrics",
                f"Total Functions: {context.complexity_metrics.get('total_functions', 0)}",
                f"Total Classes: {context.complexity_metrics.get('total_classes', 0)}",
                f"Average File Complexity: {context.complexity_metrics.get('average_file_complexity', 0)}"
            ])
        
        # Include summary of key files
        if context.files:
            prompt_parts.extend([
                f"",
                f"## Key Files"
            ])
            for file_ctx in sorted(context.files, key=lambda f: f.complexity_score, reverse=True)[:5]:
                prompt_parts.append(f"- `{file_ctx.path}` ({file_ctx.language})")
                if file_ctx.purpose:
                    prompt_parts.append(f"  {file_ctx.purpose[:100]}...")
        
        prompt_parts.extend([
            f"",
            f"## Task",
            f"{task_description}",
            f"",
            f"Please provide assistance based on the above project context."
        ])
        
        return "\n".join(prompt_parts)