"""
Code structure analyzer for extracting meaningful information from parsed code.

This module provides the CodeAnalyzer class that works with Tree-sitter parse
results to extract functions, classes, imports, and other code structures,
making them available for LLM context generation and code understanding.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from tree_sitter import Node, Tree
import logging

from .tree_sitter_parser import ParseResult, TreeSitterParser
from .language_support import (
    SupportedLanguage, LanguageConfig, get_language_config
)
from ...utils.logging import get_logger


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    start_byte: int = 0
    end_byte: int = 0
    body: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None
    complexity_score: float = 0.0


@dataclass
class ClassInfo:
    """Information about a class or interface."""
    name: str
    base_classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    start_byte: int = 0
    end_byte: int = 0
    modifiers: List[str] = field(default_factory=list)
    is_interface: bool = False
    generics: List[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_wildcard: bool = False
    start_line: int = 0
    end_line: int = 0
    import_type: str = "import"  # import, from, use, etc.


@dataclass
class VariableInfo:
    """Information about a variable declaration."""
    name: str
    type_annotation: Optional[str] = None
    value: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    scope: str = "local"  # local, global, class, parameter
    modifiers: List[str] = field(default_factory=list)


@dataclass
class CodeStructure:
    """Complete structure information for a code file."""
    language: SupportedLanguage
    file_path: Optional[str] = None
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    docstrings: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    complexity_score: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the code structure."""
        return {
            "language": self.language.value,
            "file_path": self.file_path,
            "function_count": len(self.functions),
            "class_count": len(self.classes),
            "import_count": len(self.imports),
            "variable_count": len(self.variables),
            "lines_of_code": self.lines_of_code,
            "complexity_score": self.complexity_score,
            "top_level_functions": [f.name for f in self.functions if not f.is_method],
            "class_names": [c.name for c in self.classes],
            "imported_modules": list(set(i.module for i in self.imports if i.module))
        }


class CodeAnalyzer:
    """
    Analyzer for extracting structured information from parsed code.
    
    Uses Tree-sitter parse results to extract functions, classes, imports,
    and other code structures with language-specific handling.
    """
    
    def __init__(self, parser: Optional[TreeSitterParser] = None):
        """Initialize the code analyzer.
        
        Args:
            parser: Optional TreeSitterParser instance (creates one if not provided)
        """
        self.logger = get_logger(__name__)
        self.parser = parser or TreeSitterParser()
        
        # Cache for extracted structures
        self._structure_cache: Dict[str, CodeStructure] = {}
    
    def analyze_code(self, code: str, language: SupportedLanguage, 
                    file_path: Optional[str] = None) -> CodeStructure:
        """Analyze code structure from source code.
        
        Args:
            code: Source code to analyze
            language: Programming language of the code
            file_path: Optional file path for context
            
        Returns:
            CodeStructure with extracted information
        """
        # Parse the code
        parse_result = self.parser.parse_code(code, language, file_path)
        
        if not parse_result.success:
            self.logger.error(f"Failed to parse code: {parse_result.error_message}")
            return CodeStructure(language=language, file_path=file_path)
        
        return self.analyze_parse_result(parse_result)
    
    def analyze_file(self, file_path: str, language: Optional[SupportedLanguage] = None) -> CodeStructure:
        """Analyze code structure from a file.
        
        Args:
            file_path: Path to the file to analyze
            language: Optional language override
            
        Returns:
            CodeStructure with extracted information
        """
        # Parse the file
        parse_result = self.parser.parse_file(file_path, language)
        
        if not parse_result.success:
            self.logger.error(f"Failed to parse file {file_path}: {parse_result.error_message}")
            return CodeStructure(language=parse_result.language, file_path=file_path)
        
        return self.analyze_parse_result(parse_result)
    
    def analyze_parse_result(self, parse_result: ParseResult) -> CodeStructure:
        """Analyze a Tree-sitter parse result.
        
        Args:
            parse_result: ParseResult from TreeSitterParser
            
        Returns:
            CodeStructure with extracted information
        """
        if not parse_result.success or parse_result.tree is None:
            return CodeStructure(language=parse_result.language, file_path=parse_result.file_path)
        
        # Get language configuration
        config = get_language_config(parse_result.language)
        if config is None:
            self.logger.error(f"No configuration found for language: {parse_result.language}")
            return CodeStructure(language=parse_result.language, file_path=parse_result.file_path)
        
        # Create structure object
        structure = CodeStructure(
            language=parse_result.language,
            file_path=parse_result.file_path,
            lines_of_code=len(parse_result.source_code.splitlines())
        )
        
        try:
            # Extract different code structures
            if config.has_functions:
                structure.functions = self._extract_functions(parse_result, config)
            
            if config.has_classes:
                structure.classes = self._extract_classes(parse_result, config)
            
            if config.has_imports:
                structure.imports = self._extract_imports(parse_result, config)
            
            structure.variables = self._extract_variables(parse_result, config)
            structure.comments = self._extract_comments(parse_result, config)
            structure.docstrings = self._extract_docstrings(parse_result, config)
            
            # Calculate complexity score
            structure.complexity_score = self._calculate_complexity(structure)
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing parse result: {e}")
            return structure
    
    def _extract_functions(self, parse_result: ParseResult, config: LanguageConfig) -> List[FunctionInfo]:
        """Extract function information from the parse tree.
        
        Args:
            parse_result: ParseResult with tree to analyze
            config: Language configuration
            
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        
        if not config.function_query.strip():
            return functions
        
        try:
            # Create query object
            query = parse_result.tree.language.query(config.function_query)
            
            # Execute query
            captures = query.captures(parse_result.tree.root_node)
            
            # Group captures by function
            function_captures = {}
            for node, capture_name in captures:
                if capture_name.startswith('function.'):
                    # Extract function index (assuming one function per capture group)
                    func_key = (node.start_byte, node.end_byte)
                    if func_key not in function_captures:
                        function_captures[func_key] = {}
                    function_captures[func_key][capture_name] = node
            
            # Process each function
            for captures_dict in function_captures.values():
                func_info = self._build_function_info(captures_dict, parse_result)
                if func_info:
                    functions.append(func_info)
            
        except Exception as e:
            self.logger.error(f"Error extracting functions: {e}")
        
        return functions
    
    def _extract_classes(self, parse_result: ParseResult, config: LanguageConfig) -> List[ClassInfo]:
        """Extract class information from the parse tree.
        
        Args:
            parse_result: ParseResult with tree to analyze
            config: Language configuration
            
        Returns:
            List of ClassInfo objects
        """
        classes = []
        
        if not config.class_query.strip():
            return classes
        
        try:
            # Create query object
            query = parse_result.tree.language.query(config.class_query)
            
            # Execute query
            captures = query.captures(parse_result.tree.root_node)
            
            # Group captures by class
            class_captures = {}
            for node, capture_name in captures:
                if capture_name.startswith('class.'):
                    # Extract class index
                    class_key = (node.start_byte, node.end_byte)
                    if class_key not in class_captures:
                        class_captures[class_key] = {}
                    class_captures[class_key][capture_name] = node
            
            # Process each class
            for captures_dict in class_captures.values():
                class_info = self._build_class_info(captures_dict, parse_result)
                if class_info:
                    classes.append(class_info)
            
        except Exception as e:
            self.logger.error(f"Error extracting classes: {e}")
        
        return classes
    
    def _extract_imports(self, parse_result: ParseResult, config: LanguageConfig) -> List[ImportInfo]:
        """Extract import information from the parse tree.
        
        Args:
            parse_result: ParseResult with tree to analyze
            config: Language configuration
            
        Returns:
            List of ImportInfo objects
        """
        imports = []
        
        if not config.import_query.strip():
            return imports
        
        try:
            # Create query object
            query = parse_result.tree.language.query(config.import_query)
            
            # Execute query
            captures = query.captures(parse_result.tree.root_node)
            
            # Group captures by import
            import_captures = {}
            for node, capture_name in captures:
                if capture_name.startswith('import.'):
                    # Extract import index
                    import_key = (node.start_byte, node.end_byte)
                    if import_key not in import_captures:
                        import_captures[import_key] = {}
                    import_captures[import_key][capture_name] = node
            
            # Process each import
            for captures_dict in import_captures.values():
                import_info = self._build_import_info(captures_dict, parse_result)
                if import_info:
                    imports.append(import_info)
            
        except Exception as e:
            self.logger.error(f"Error extracting imports: {e}")
        
        return imports
    
    def _extract_variables(self, parse_result: ParseResult, config: LanguageConfig) -> List[VariableInfo]:
        """Extract variable information from the parse tree.
        
        Args:
            parse_result: ParseResult with tree to analyze
            config: Language configuration
            
        Returns:
            List of VariableInfo objects
        """
        variables = []
        
        if not config.variable_query.strip():
            return variables
        
        try:
            # Create query object
            query = parse_result.tree.language.query(config.variable_query)
            
            # Execute query
            captures = query.captures(parse_result.tree.root_node)
            
            # Group captures by variable
            variable_captures = {}
            for node, capture_name in captures:
                if capture_name.startswith('variable.'):
                    # Extract variable index
                    var_key = (node.start_byte, node.end_byte)
                    if var_key not in variable_captures:
                        variable_captures[var_key] = {}
                    variable_captures[var_key][capture_name] = node
            
            # Process each variable
            for captures_dict in variable_captures.values():
                var_info = self._build_variable_info(captures_dict, parse_result)
                if var_info:
                    variables.append(var_info)
            
        except Exception as e:
            self.logger.error(f"Error extracting variables: {e}")
        
        return variables
    
    def _extract_comments(self, parse_result: ParseResult, config: LanguageConfig) -> List[str]:
        """Extract comments from the parse tree.
        
        Args:
            parse_result: ParseResult with tree to analyze
            config: Language configuration
            
        Returns:
            List of comment strings
        """
        comments = []
        
        try:
            # Simple comment extraction based on comment patterns
            lines = parse_result.source_code.splitlines()
            
            for line in lines:
                stripped = line.strip()
                for pattern in config.comment_patterns:
                    if stripped.startswith(pattern):
                        # Extract comment text
                        comment_text = stripped[len(pattern):].strip()
                        if comment_text:
                            comments.append(comment_text)
                        break
        
        except Exception as e:
            self.logger.error(f"Error extracting comments: {e}")
        
        return comments
    
    def _extract_docstrings(self, parse_result: ParseResult, config: LanguageConfig) -> List[str]:
        """Extract docstrings from the parse tree.
        
        Args:
            parse_result: ParseResult with tree to analyze
            config: Language configuration
            
        Returns:
            List of docstring texts
        """
        docstrings = []
        
        try:
            # Simple docstring extraction based on patterns
            content = parse_result.source_code
            
            for pattern in config.docstring_patterns:
                start = 0
                while True:
                    # Find opening pattern
                    start_pos = content.find(pattern, start)
                    if start_pos == -1:
                        break
                    
                    # Find closing pattern
                    end_pos = content.find(pattern, start_pos + len(pattern))
                    if end_pos == -1:
                        break
                    
                    # Extract docstring content
                    docstring_content = content[start_pos + len(pattern):end_pos].strip()
                    if docstring_content:
                        docstrings.append(docstring_content)
                    
                    start = end_pos + len(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting docstrings: {e}")
        
        return docstrings
    
    def _build_function_info(self, captures: Dict[str, Node], parse_result: ParseResult) -> Optional[FunctionInfo]:
        """Build FunctionInfo from captured nodes.
        
        Args:
            captures: Dictionary of captured nodes
            parse_result: ParseResult for context
            
        Returns:
            FunctionInfo if successful, None otherwise
        """
        try:
            # Get function name
            name_node = captures.get('function.name')
            if not name_node:
                return None
            
            name = self._get_node_text(name_node, parse_result.source_code)
            
            # Get parameters
            params_node = captures.get('function.params')
            parameters = []
            if params_node:
                # Extract parameter names (simplified)
                param_text = self._get_node_text(params_node, parse_result.source_code)
                if param_text and param_text != "()":
                    # Simple parameter parsing
                    param_text = param_text.strip("()")
                    if param_text:
                        parameters = [p.strip().split(":")[0].strip() for p in param_text.split(",")]
            
            # Get function definition node for position info
            def_node = captures.get('function.def') or name_node
            start_line = def_node.start_point[0] + 1
            end_line = def_node.end_point[0] + 1
            
            # Get return type if available
            return_type = None
            return_node = captures.get('function.return_type')
            if return_node:
                return_type = self._get_node_text(return_node, parse_result.source_code)
            
            # Get body if available
            body = None
            body_node = captures.get('function.body')
            if body_node:
                body = self._get_node_text(body_node, parse_result.source_code)
            
            return FunctionInfo(
                name=name,
                parameters=parameters,
                return_type=return_type,
                start_line=start_line,
                end_line=end_line,
                start_byte=def_node.start_byte,
                end_byte=def_node.end_byte,
                body=body
            )
            
        except Exception as e:
            self.logger.error(f"Error building function info: {e}")
            return None
    
    def _build_class_info(self, captures: Dict[str, Node], parse_result: ParseResult) -> Optional[ClassInfo]:
        """Build ClassInfo from captured nodes.
        
        Args:
            captures: Dictionary of captured nodes
            parse_result: ParseResult for context
            
        Returns:
            ClassInfo if successful, None otherwise
        """
        try:
            # Get class name
            name_node = captures.get('class.name')
            if not name_node:
                return None
            
            name = self._get_node_text(name_node, parse_result.source_code)
            
            # Get base classes
            base_classes = []
            bases_node = captures.get('class.bases')
            if bases_node:
                bases_text = self._get_node_text(bases_node, parse_result.source_code)
                if bases_text:
                    # Simple base class parsing
                    base_classes = [b.strip() for b in bases_text.split(",") if b.strip()]
            
            # Get class definition node for position info
            def_node = captures.get('class.def') or name_node
            start_line = def_node.start_point[0] + 1
            end_line = def_node.end_point[0] + 1
            
            return ClassInfo(
                name=name,
                base_classes=base_classes,
                start_line=start_line,
                end_line=end_line,
                start_byte=def_node.start_byte,
                end_byte=def_node.end_byte
            )
            
        except Exception as e:
            self.logger.error(f"Error building class info: {e}")
            return None
    
    def _build_import_info(self, captures: Dict[str, Node], parse_result: ParseResult) -> Optional[ImportInfo]:
        """Build ImportInfo from captured nodes.
        
        Args:
            captures: Dictionary of captured nodes
            parse_result: ParseResult for context
            
        Returns:
            ImportInfo if successful, None otherwise
        """
        try:
            # Get import statement node for position info
            stmt_node = captures.get('import.stmt')
            if not stmt_node:
                return None
            
            start_line = stmt_node.start_point[0] + 1
            end_line = stmt_node.end_point[0] + 1
            
            # Get module name
            module = ""
            module_node = captures.get('import.module') or captures.get('import.source')
            if module_node:
                module = self._get_node_text(module_node, parse_result.source_code).strip('"\'')
            
            # Get imported names
            names = []
            name_node = captures.get('import.name') or captures.get('import.names')
            if name_node:
                name_text = self._get_node_text(name_node, parse_result.source_code)
                if name_text:
                    names = [n.strip() for n in name_text.split(",") if n.strip()]
            
            # Check for wildcard
            is_wildcard = captures.get('import.wildcard') is not None
            
            return ImportInfo(
                module=module,
                names=names,
                is_wildcard=is_wildcard,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            self.logger.error(f"Error building import info: {e}")
            return None
    
    def _build_variable_info(self, captures: Dict[str, Node], parse_result: ParseResult) -> Optional[VariableInfo]:
        """Build VariableInfo from captured nodes.
        
        Args:
            captures: Dictionary of captured nodes
            parse_result: ParseResult for context
            
        Returns:
            VariableInfo if successful, None otherwise
        """
        try:
            # Get variable name
            name_node = captures.get('variable.name')
            if not name_node:
                return None
            
            name = self._get_node_text(name_node, parse_result.source_code)
            
            # Get declaration node for position info
            decl_node = captures.get('variable.declaration') or captures.get('variable.assignment') or name_node
            start_line = decl_node.start_point[0] + 1
            end_line = decl_node.end_point[0] + 1
            
            # Get type annotation
            type_annotation = None
            type_node = captures.get('variable.type')
            if type_node:
                type_annotation = self._get_node_text(type_node, parse_result.source_code)
            
            # Get value
            value = None
            value_node = captures.get('variable.value')
            if value_node:
                value = self._get_node_text(value_node, parse_result.source_code)
            
            return VariableInfo(
                name=name,
                type_annotation=type_annotation,
                value=value,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            self.logger.error(f"Error building variable info: {e}")
            return None
    
    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Get text content of a node.
        
        Args:
            node: Tree-sitter node
            source_code: Source code string
            
        Returns:
            Text content of the node
        """
        return source_code[node.start_byte:node.end_byte]
    
    def _calculate_complexity(self, structure: CodeStructure) -> float:
        """Calculate complexity score for the code structure.
        
        Args:
            structure: CodeStructure to analyze
            
        Returns:
            Complexity score (higher = more complex)
        """
        score = 0.0
        
        # Base complexity from line count
        score += structure.lines_of_code * 0.1
        
        # Function complexity
        for func in structure.functions:
            score += 2.0  # Base function complexity
            score += len(func.parameters) * 0.5  # Parameter complexity
            if func.body:
                # Simple complexity based on body length
                score += len(func.body.splitlines()) * 0.2
        
        # Class complexity
        for cls in structure.classes:
            score += 3.0  # Base class complexity
            score += len(cls.methods) * 1.0  # Method complexity
            score += len(cls.base_classes) * 0.5  # Inheritance complexity
        
        # Import complexity
        score += len(structure.imports) * 0.3
        
        return round(score, 2)