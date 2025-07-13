"""
Simplified context extractor for code analysis.

Focuses on core extraction functionality with cleaner separation of concerns.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from .context_types import (
    ContextLevel, ContextConfig, FunctionContext, 
    ClassContext, FileContext, ProjectContext
)
from .tree_sitter_parser import TreeSitterParser
from .language_support import SupportedLanguage, get_language_from_extension
from ...utils.logging import get_logger


class ContextExtractor:
    """
    Simplified context extractor for generating LLM-ready context from code analysis.
    
    Much simpler than the original 783-line version, focusing on core functionality.
    """
    
    def __init__(self, parser: Optional[TreeSitterParser] = None):
        """Initialize the context extractor."""
        self.parser = parser or TreeSitterParser()
        self.logger = get_logger(__name__)
    
    def extract_file_context(
        self, 
        file_path: str, 
        level: str = "standard", 
        config: Optional[ContextConfig] = None
    ) -> str:
        """
        Extract context from a file and return as formatted string.
        
        This is the main method that replaces the complex original implementation.
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return f"File not found: {file_path}"
            
            # Get language
            language = get_language_from_extension(file_path)
            if not language:
                return self._extract_text_file_context(file_path)
            
            # Parse file
            content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
            tree = self.parser.parse_content(content, language)
            
            if not tree:
                return self._extract_text_file_context(file_path)
            
            # Extract structured context
            file_context = self._extract_structured_context(file_path, content, tree, language)
            
            # Format for LLM
            return self._format_context_for_llm(file_context, level)
            
        except Exception as e:
            self.logger.error(f"Error extracting context from {file_path}: {e}")
            return f"Error extracting context: {e}"
    
    def _extract_text_file_context(self, file_path: str) -> str:
        """Extract context from non-code text files."""
        try:
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Basic text file info
            return f"""File: {file_path}
Type: Text file
Lines: {len(lines)}
Size: {len(content)} characters

Content preview:
{content[:500]}{'...' if len(content) > 500 else ''}"""
            
        except Exception as e:
            return f"Error reading text file {file_path}: {e}"
    
    def _extract_structured_context(
        self, 
        file_path: str, 
        content: str, 
        tree: Any, 
        language: SupportedLanguage
    ) -> FileContext:
        """Extract structured context from parsed tree."""
        # Extract basic file info
        lines = content.split('\n')
        
        # Extract functions
        functions = self.parser.extract_functions(tree, language)
        function_contexts = [
            FunctionContext(
                name=func.name,
                signature=func.signature,
                line_range=(func.start_line, func.end_line)
            )
            for func in functions
        ]
        
        # Extract classes
        classes = self.parser.extract_classes(tree, language)
        class_contexts = [
            ClassContext(
                name=cls.name,
                line_range=(cls.start_line, cls.end_line)
            )
            for cls in classes
        ]
        
        # Create file context
        file_context = FileContext(
            file_path=file_path,
            language=language.value,
            functions=function_contexts,
            classes=class_contexts,
            line_count=len(lines)
        )
        
        return file_context
    
    def _format_context_for_llm(self, file_context: FileContext, level: str) -> str:
        """Format the context for LLM consumption."""
        output = []
        
        # File header
        output.append(f"File: {file_context.file_path}")
        output.append(f"Language: {file_context.language}")
        output.append(f"Lines: {file_context.line_count}")
        
        # Functions
        if file_context.functions:
            output.append(f"\nFunctions ({len(file_context.functions)}):")
            for func in file_context.functions:
                if level in ["minimal"]:
                    output.append(f"  - {func.name}")
                else:
                    output.append(f"  - {func.signature} (lines {func.line_range[0]}-{func.line_range[1]})")
        
        # Classes
        if file_context.classes:
            output.append(f"\nClasses ({len(file_context.classes)}):")
            for cls in file_context.classes:
                if level in ["minimal"]:
                    output.append(f"  - {cls.name}")
                else:
                    output.append(f"  - {cls.name} (lines {cls.line_range[0]}-{cls.line_range[1]})")
                    if level in ["detailed", "comprehensive"] and cls.methods:
                        for method in cls.methods:
                            output.append(f"    - {method.name}()")
        
        # Additional details for higher levels
        if level in ["detailed", "comprehensive"]:
            if file_context.imports:
                output.append(f"\nImports: {', '.join(file_context.imports[:10])}")
        
        return '\n'.join(output)
    
    def extract_project_context(self, project_path: str, level: str = "standard") -> str:
        """Extract context from an entire project."""
        try:
            project_path_obj = Path(project_path)
            if not project_path_obj.exists():
                return f"Project path not found: {project_path}"
            
            # Find code files
            code_files = []
            for pattern in ["*.py", "*.js", "*.ts", "*.rs", "*.java", "*.cpp", "*.c"]:
                code_files.extend(project_path_obj.rglob(pattern))
            
            # Limit number of files for performance
            if len(code_files) > 50:
                code_files = code_files[:50]
            
            output = []
            output.append(f"Project: {project_path}")
            output.append(f"Code files found: {len(code_files)}")
            
            # Extract context from each file
            for file_path in code_files[:10]:  # Limit for performance
                try:
                    relative_path = file_path.relative_to(project_path_obj)
                    file_context = self.extract_file_context(str(file_path), level="minimal")
                    output.append(f"\n--- {relative_path} ---")
                    output.append(file_context)
                except Exception as e:
                    output.append(f"\n--- {file_path} ---")
                    output.append(f"Error: {e}")
            
            return '\n'.join(output)
            
        except Exception as e:
            self.logger.error(f"Error extracting project context: {e}")
            return f"Error extracting project context: {e}"
    
    def extract_function_context(self, file_path: str, function_name: str) -> str:
        """Extract context for a specific function."""
        try:
            file_context = self.extract_file_context(file_path, level="detailed")
            
            # Find the specific function
            # This is a simplified search - could be enhanced
            if function_name in file_context:
                return f"Function '{function_name}' found in:\n{file_context}"
            else:
                return f"Function '{function_name}' not found in {file_path}"
                
        except Exception as e:
            return f"Error extracting function context: {e}"
    
    def generate_summary(self, context: str, max_length: int = 500) -> str:
        """Generate a summary of the context."""
        if len(context) <= max_length:
            return context
        
        lines = context.split('\n')
        summary_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > max_length:
                break
            summary_lines.append(line)
            current_length += len(line)
        
        summary = '\n'.join(summary_lines)
        if len(summary) < len(context):
            summary += "\n... (truncated)"
        
        return summary