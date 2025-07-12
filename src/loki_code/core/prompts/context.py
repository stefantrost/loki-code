"""
Context management for prompt templates.

This module provides context building and management capabilities
for generating rich, context-aware prompts.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..code_analysis import CodeAnalyzer, TreeSitterParser, SupportedLanguage
from ...utils.logging import get_logger


class ContextType(Enum):
    """Types of context that can be included in prompts."""
    PROJECT = "project"
    FILE = "file"
    CONVERSATION = "conversation"
    ERROR = "error"
    ENVIRONMENT = "environment"
    TASK = "task"


class ContextBuildError(Exception):
    """Exception raised when context building fails."""
    pass


@dataclass
class ConversationEntry:
    """Represents a single conversation entry."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert to a string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.role}: {self.content[:100]}..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class CodeContext:
    """Rich context information about code files."""
    file_path: str
    language: Optional[SupportedLanguage] = None
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    line_count: int = 0
    summary: str = ""
    errors: List[str] = field(default_factory=list)
    
    def to_string(self) -> str:
        """Convert to a readable string representation."""
        if self.errors:
            return f"File: {self.file_path} (analysis failed: {', '.join(self.errors)})"
        
        parts = [f"File: {self.file_path}"]
        
        if self.language:
            parts.append(f"Language: {self.language.value}")
        
        if self.line_count > 0:
            parts.append(f"Lines: {self.line_count}")
        
        if self.functions:
            parts.append(f"Functions: {len(self.functions)} ({', '.join(self.functions[:3])}{'...' if len(self.functions) > 3 else ''})")
        
        if self.classes:
            parts.append(f"Classes: {len(self.classes)} ({', '.join(self.classes[:3])}{'...' if len(self.classes) > 3 else ''})")
        
        if self.complexity_score > 0:
            parts.append(f"Complexity: {self.complexity_score:.2f}")
        
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        
        return "\n".join(f"  {part}" for part in parts)


@dataclass
class ProjectContext:
    """Context information about the entire project."""
    project_path: str
    name: str = ""
    description: str = ""
    languages: List[SupportedLanguage] = field(default_factory=list)
    file_count: int = 0
    key_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    structure: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert to a readable string representation."""
        parts = [f"Project: {self.name or Path(self.project_path).name}"]
        
        if self.description:
            parts.append(f"Description: {self.description}")
        
        parts.append(f"Path: {self.project_path}")
        
        if self.file_count > 0:
            parts.append(f"Files: {self.file_count}")
        
        if self.languages:
            parts.append(f"Languages: {', '.join(lang.value for lang in self.languages)}")
        
        if self.key_files:
            parts.append(f"Key files: {', '.join(self.key_files[:5])}{'...' if len(self.key_files) > 5 else ''}")
        
        if self.dependencies:
            parts.append(f"Dependencies: {', '.join(self.dependencies[:5])}{'...' if len(self.dependencies) > 5 else ''}")
        
        return "\n".join(f"  {part}" for part in parts)


@dataclass
class PromptContext:
    """
    Complete context for prompt generation.
    
    This contains all the information needed to build
    rich, context-aware prompts for LLM interactions.
    """
    user_message: str
    current_task: Optional[str] = None
    project_path: Optional[str] = None
    current_file: Optional[str] = None
    target_files: List[str] = field(default_factory=list)
    conversation_history: List[ConversationEntry] = field(default_factory=list)
    file_contexts: Dict[str, CodeContext] = field(default_factory=dict)
    project_context: Optional[ProjectContext] = None
    error_context: Optional[str] = None
    stack_trace: Optional[str] = None
    environment_context: Dict[str, Any] = field(default_factory=dict)
    analysis_scope: Optional[str] = None
    focus_areas: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_conversation_entry(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new conversation entry."""
        entry = ConversationEntry(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.conversation_history.append(entry)
    
    def get_file_context(self, file_path: str) -> Optional[CodeContext]:
        """Get context for a specific file."""
        return self.file_contexts.get(file_path)
    
    def set_file_context(self, file_path: str, context: CodeContext) -> None:
        """Set context for a specific file."""
        self.file_contexts[file_path] = context
    
    def get_recent_conversation(self, max_entries: int = 10) -> List[ConversationEntry]:
        """Get recent conversation entries."""
        return self.conversation_history[-max_entries:] if self.conversation_history else []


class ContextBuilder:
    """
    Builder for creating rich context objects.
    
    This class provides methods to analyze files, projects,
    and other context sources to build comprehensive
    PromptContext objects.
    """
    
    def __init__(self, 
                 code_analyzer: Optional[CodeAnalyzer] = None,
                 tree_sitter_parser: Optional[TreeSitterParser] = None):
        self.code_analyzer = code_analyzer
        self.tree_sitter_parser = tree_sitter_parser
        self.logger = get_logger(__name__)
    
    async def build_file_context(self, file_path: str) -> CodeContext:
        """Build rich context about a specific file."""
        try:
            abs_path = Path(file_path).resolve()
            
            if not abs_path.exists():
                return CodeContext(
                    file_path=file_path,
                    errors=[f"File does not exist: {file_path}"]
                )
            
            # Get basic file info
            stat = abs_path.stat()
            
            # Read file content
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return CodeContext(
                    file_path=file_path,
                    errors=["File is not text or uses unsupported encoding"]
                )
            
            # Count lines
            line_count = len(content.split('\n'))
            
            # Detect language
            language = self._detect_language(abs_path)
            
            # Initialize context
            context = CodeContext(
                file_path=file_path,
                language=language,
                line_count=line_count
            )
            
            # Try to get detailed analysis if analyzer is available
            if self.code_analyzer and language:
                try:
                    analysis_result = await self._analyze_with_tree_sitter(abs_path, content, language)
                    if analysis_result:
                        context.functions = analysis_result.get("functions", [])
                        context.classes = analysis_result.get("classes", [])
                        context.imports = analysis_result.get("imports", [])
                        context.complexity_score = analysis_result.get("complexity", 0.0)
                        context.summary = analysis_result.get("summary", "")
                except Exception as e:
                    context.errors.append(f"Analysis failed: {str(e)}")
                    self.logger.warning(f"File analysis failed for {file_path}: {e}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to build file context for {file_path}: {e}")
            return CodeContext(
                file_path=file_path,
                errors=[f"Context building failed: {str(e)}"]
            )
    
    def build_project_context(self, project_path: str) -> ProjectContext:
        """Build context about the overall project."""
        try:
            project_dir = Path(project_path).resolve()
            
            if not project_dir.exists() or not project_dir.is_dir():
                return ProjectContext(
                    project_path=project_path,
                    name="Unknown Project"
                )
            
            # Get project name
            name = project_dir.name
            
            # Scan for files
            code_files = []
            languages = set()
            
            for file_path in project_dir.rglob("*"):
                if file_path.is_file() and self._is_code_file(file_path):
                    code_files.append(str(file_path.relative_to(project_dir)))
                    
                    # Detect language
                    lang = self._detect_language(file_path)
                    if lang:
                        languages.add(lang)
            
            # Identify key files
            key_files = self._identify_key_files(code_files)
            
            # Look for dependencies
            dependencies = self._find_dependencies(project_dir)
            
            # Try to get project description
            description = self._get_project_description(project_dir)
            
            return ProjectContext(
                project_path=project_path,
                name=name,
                description=description,
                languages=list(languages),
                file_count=len(code_files),
                key_files=key_files,
                dependencies=dependencies
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build project context for {project_path}: {e}")
            return ProjectContext(
                project_path=project_path,
                name="Error Loading Project"
            )
    
    def build_conversation_context(self, history: List[ConversationEntry], max_entries: int = 10) -> str:
        """Build context from conversation history."""
        if not history:
            return "No previous conversation."
        
        recent_history = history[-max_entries:]
        
        if len(recent_history) == 1:
            return f"Previous message: {recent_history[0].to_string()}"
        
        lines = ["Recent conversation:"]
        for entry in recent_history:
            lines.append(f"  {entry.to_string()}")
        
        return "\n".join(lines)
    
    def build_environment_context(self) -> Dict[str, Any]:
        """Build context about the current environment."""
        return {
            "os": os.name,
            "platform": os.sys.platform,
            "python_version": os.sys.version.split()[0],
            "working_directory": str(Path.cwd()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_with_tree_sitter(self, file_path: Path, content: str, language: SupportedLanguage) -> Optional[Dict[str, Any]]:
        """Analyze file with Tree-sitter if available."""
        if not self.code_analyzer:
            return None
        
        try:
            # This would use the actual code analyzer
            # For now, provide a basic implementation
            functions = []
            classes = []
            imports = []
            
            # Simple parsing for Python files
            if language == SupportedLanguage.PYTHON:
                lines = content.split('\n')
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('def '):
                        func_name = stripped.split('(')[0].replace('def ', '')
                        functions.append(func_name)
                    elif stripped.startswith('class '):
                        class_name = stripped.split('(')[0].split(':')[0].replace('class ', '')
                        classes.append(class_name)
                    elif stripped.startswith('import ') or stripped.startswith('from '):
                        imports.append(stripped)
            
            # Calculate complexity (simple line-based approximation)
            complexity = len([line for line in content.split('\n') if line.strip().startswith(('if ', 'for ', 'while ', 'elif ', 'except ', 'try:'))])
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports[:10],  # Limit to avoid noise
                "complexity": float(complexity),
                "summary": f"Contains {len(functions)} functions, {len(classes)} classes"
            }
            
        except Exception as e:
            self.logger.warning(f"Tree-sitter analysis failed: {e}")
            return None
    
    def _detect_language(self, file_path: Path) -> Optional[SupportedLanguage]:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        
        language_map = {
            '.py': SupportedLanguage.PYTHON,
            '.js': SupportedLanguage.JAVASCRIPT,
            '.jsx': SupportedLanguage.JAVASCRIPT,
            '.ts': SupportedLanguage.TYPESCRIPT,
            '.tsx': SupportedLanguage.TYPESCRIPT,
            '.rs': SupportedLanguage.RUST,
            '.go': SupportedLanguage.GO,
            '.java': SupportedLanguage.JAVA
        }
        
        return language_map.get(suffix)
    
    def _is_code_file(self, file_path: Path) -> bool:
        """Check if a file is a code file."""
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.rs', '.go', '.java', '.cpp', '.c', '.h'}
        return file_path.suffix.lower() in code_extensions
    
    def _identify_key_files(self, code_files: List[str]) -> List[str]:
        """Identify key files in the project."""
        key_patterns = [
            'main.py', 'app.py', 'index.js', 'index.ts', 'main.rs', 'main.go',
            'setup.py', 'package.json', 'Cargo.toml', 'go.mod',
            'README.md', 'README.txt', 'requirements.txt'
        ]
        
        key_files = []
        for pattern in key_patterns:
            matching_files = [f for f in code_files if f.endswith(pattern)]
            key_files.extend(matching_files[:1])  # Take first match
        
        return key_files[:10]  # Limit to 10 key files
    
    def _find_dependencies(self, project_dir: Path) -> List[str]:
        """Find project dependencies."""
        dependencies = []
        
        # Python dependencies
        requirements_files = ['requirements.txt', 'setup.py', 'pyproject.toml']
        for req_file in requirements_files:
            req_path = project_dir / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text()
                    # Simple parsing - could be enhanced
                    for line in content.split('\n')[:10]:  # Limit to 10
                        if line.strip() and not line.startswith('#'):
                            dep = line.split('=')[0].split('>')[0].split('<')[0].strip()
                            if dep:
                                dependencies.append(dep)
                except Exception:
                    pass
        
        # JavaScript dependencies
        package_json = project_dir / 'package.json'
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text())
                deps = data.get('dependencies', {})
                dependencies.extend(list(deps.keys())[:10])
            except Exception:
                pass
        
        return dependencies[:15]  # Limit total dependencies
    
    def _get_project_description(self, project_dir: Path) -> str:
        """Try to get project description from README or package files."""
        # Check README files
        readme_files = ['README.md', 'README.txt', 'README.rst']
        for readme in readme_files:
            readme_path = project_dir / readme
            if readme_path.exists():
                try:
                    content = readme_path.read_text()
                    # Get first paragraph as description
                    paragraphs = content.split('\n\n')
                    for para in paragraphs:
                        cleaned = para.strip().replace('\n', ' ')
                        if len(cleaned) > 20 and not cleaned.startswith('#'):
                            return cleaned[:200] + ('...' if len(cleaned) > 200 else '')
                except Exception:
                    pass
        
        # Check package.json description
        package_json = project_dir / 'package.json'
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text())
                description = data.get('description', '')
                if description:
                    return description
            except Exception:
                pass
        
        return ""