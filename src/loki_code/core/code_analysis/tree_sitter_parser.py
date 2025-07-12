"""
Tree-sitter parser for multi-language code analysis in Loki Code.

This module provides the main TreeSitterParser class that handles parsing
of various programming languages using Tree-sitter, enabling intelligent
code analysis and context extraction for LLM operations.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
import time
import tree_sitter
from tree_sitter import Language, Parser, Node, Tree

from .language_support import (
    SupportedLanguage, LanguageConfig, 
    get_language_from_extension, get_language_config,
    is_language_supported, list_supported_languages
)
from ...utils.logging import get_logger


@dataclass
class ParseResult:
    """Result of parsing a code file."""
    language: SupportedLanguage
    tree: Tree
    root_node: Node
    source_code: str
    file_path: Optional[str] = None
    parse_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def get_functions(self) -> List[Dict[str, Any]]:
        """Extract function information from the parse tree."""
        return []  # Implemented by CodeAnalyzer
    
    def get_classes(self) -> List[Dict[str, Any]]:
        """Extract class information from the parse tree."""
        return []  # Implemented by CodeAnalyzer
    
    def get_imports(self) -> List[Dict[str, Any]]:
        """Extract import information from the parse tree."""
        return []  # Implemented by CodeAnalyzer


@dataclass
class ParserStats:
    """Statistics for the Tree-sitter parser."""
    total_parses: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    total_parse_time_ms: float = 0.0
    language_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_parses == 0:
            return 0.0
        return (self.successful_parses / self.total_parses) * 100.0
    
    @property
    def average_parse_time_ms(self) -> float:
        """Calculate average parse time in milliseconds."""
        if self.successful_parses == 0:
            return 0.0
        return self.total_parse_time_ms / self.successful_parses


class TreeSitterParser:
    """
    Multi-language code parser using Tree-sitter.
    
    Provides parsing capabilities for various programming languages,
    enabling intelligent code analysis and context extraction.
    """
    
    def __init__(self, cache_enabled: bool = True, cache_size: int = 100):
        """Initialize the Tree-sitter parser.
        
        Args:
            cache_enabled: Whether to enable parse result caching
            cache_size: Maximum number of cached parse results
        """
        self.logger = get_logger(__name__)
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        
        # Initialize parsers for each supported language
        self._parsers: Dict[SupportedLanguage, Parser] = {}
        self._languages: Dict[SupportedLanguage, Language] = {}
        
        # Parse cache for performance
        self._parse_cache: Dict[str, ParseResult] = {}
        self._cache_access_times: Dict[str, float] = {}
        
        # Statistics
        self.stats = ParserStats()
        
        # Initialize supported languages
        self._initialize_languages()
    
    def _initialize_languages(self) -> None:
        """Initialize Tree-sitter languages and parsers."""
        for language in list_supported_languages():
            try:
                config = get_language_config(language)
                if config is None:
                    continue
                
                # Import the language module
                ts_language = self._import_tree_sitter_language(config.tree_sitter_language)
                
                if ts_language is not None:
                    # Create parser
                    parser = Parser()
                    parser.set_language(ts_language)
                    
                    self._languages[language] = ts_language
                    self._parsers[language] = parser
                    
                    self.logger.info(f"Initialized Tree-sitter parser for {language.value}")
                else:
                    self.logger.warning(f"Could not load Tree-sitter language for {language.value}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize {language.value} parser: {e}")
    
    def _import_tree_sitter_language(self, language_name: str) -> Optional[Language]:
        """Import a Tree-sitter language module.
        
        Args:
            language_name: Name of the language module to import
            
        Returns:
            Language object if successful, None otherwise
        """
        try:
            if language_name == "python":
                import tree_sitter_python as ts_python
                return Language(ts_python.language(), "python")
            elif language_name == "javascript":
                import tree_sitter_javascript as ts_javascript
                return Language(ts_javascript.language(), "javascript")
            elif language_name == "typescript":
                import tree_sitter_typescript as ts_typescript
                return Language(ts_typescript.language(), "typescript")
            else:
                # For other languages, try dynamic import
                module_name = f"tree_sitter_{language_name}"
                try:
                    module = __import__(module_name)
                    return Language(module.language(), language_name)
                except ImportError:
                    return None
                    
        except ImportError as e:
            self.logger.debug(f"Tree-sitter language not available: {language_name} - {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading Tree-sitter language {language_name}: {e}")
            return None
    
    def get_language_from_extension(self, file_path: str) -> Optional[SupportedLanguage]:
        """Get programming language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SupportedLanguage if recognized, None otherwise
        """
        return get_language_from_extension(file_path)
    
    def get_language_from_content(self, content: str, file_path: Optional[str] = None) -> Optional[SupportedLanguage]:
        """Attempt to detect language from file content.
        
        Args:
            content: File content to analyze
            file_path: Optional file path for extension-based detection
            
        Returns:
            SupportedLanguage if detected, None otherwise
        """
        # First try extension-based detection if file path is provided
        if file_path:
            lang = self.get_language_from_extension(file_path)
            if lang:
                return lang
        
        # Fallback to content-based detection using simple heuristics
        content_lower = content.lower().strip()
        
        # Python detection
        if any(keyword in content for keyword in ['def ', 'import ', 'from ', 'class ']):
            if 'function' not in content_lower or 'var ' not in content_lower:
                return SupportedLanguage.PYTHON
        
        # JavaScript/TypeScript detection
        if any(keyword in content for keyword in ['function ', 'const ', 'let ', 'var ']):
            if ': ' in content and 'interface ' in content:
                return SupportedLanguage.TYPESCRIPT
            return SupportedLanguage.JAVASCRIPT
        
        # Rust detection
        if any(keyword in content for keyword in ['fn ', 'struct ', 'impl ', 'use ']):
            return SupportedLanguage.RUST
        
        # Go detection
        if any(keyword in content for keyword in ['func ', 'package ', 'import "', 'type ']):
            return SupportedLanguage.GO
        
        # Java detection
        if any(keyword in content for keyword in ['public class ', 'private ', 'import java']):
            return SupportedLanguage.JAVA
        
        return None
    
    def is_language_supported(self, language: SupportedLanguage) -> bool:
        """Check if a language is supported and available.
        
        Args:
            language: Language to check
            
        Returns:
            True if language is supported and parser is available
        """
        return language in self._parsers
    
    def list_available_languages(self) -> List[SupportedLanguage]:
        """Get list of available languages with initialized parsers.
        
        Returns:
            List of available SupportedLanguage values
        """
        return list(self._parsers.keys())
    
    def parse_code(self, code: str, language: SupportedLanguage, file_path: Optional[str] = None) -> ParseResult:
        """Parse code content using Tree-sitter.
        
        Args:
            code: Source code to parse
            language: Programming language of the code
            file_path: Optional file path for context
            
        Returns:
            ParseResult with parse tree and metadata
        """
        start_time = time.perf_counter()
        
        # Update statistics
        self.stats.total_parses += 1
        self.stats.language_counts[language.value] = self.stats.language_counts.get(language.value, 0) + 1
        
        # Check if language is supported
        if not self.is_language_supported(language):
            error_msg = f"Language {language.value} is not supported or available"
            self.logger.error(error_msg)
            self.stats.failed_parses += 1
            return ParseResult(
                language=language,
                tree=None,
                root_node=None,
                source_code=code,
                file_path=file_path,
                success=False,
                error_message=error_msg
            )
        
        # Check cache if enabled
        if self.cache_enabled and file_path:
            cache_key = self._get_cache_key(file_path, code)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        try:
            # Get parser for the language
            parser = self._parsers[language]
            
            # Parse the code
            tree = parser.parse(bytes(code, 'utf-8'))
            root_node = tree.root_node
            
            # Calculate parse time
            end_time = time.perf_counter()
            parse_time_ms = (end_time - start_time) * 1000.0
            
            # Create result
            result = ParseResult(
                language=language,
                tree=tree,
                root_node=root_node,
                source_code=code,
                file_path=file_path,
                parse_time_ms=parse_time_ms,
                success=True
            )
            
            # Update statistics
            self.stats.successful_parses += 1
            self.stats.total_parse_time_ms += parse_time_ms
            
            # Cache the result if enabled
            if self.cache_enabled and file_path:
                self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to parse code: {str(e)}"
            self.logger.error(error_msg)
            self.stats.failed_parses += 1
            
            return ParseResult(
                language=language,
                tree=None,
                root_node=None,
                source_code=code,
                file_path=file_path,
                success=False,
                error_message=error_msg
            )
    
    def parse_file(self, file_path: str, language: Optional[SupportedLanguage] = None) -> ParseResult:
        """Parse a source code file.
        
        Args:
            file_path: Path to the file to parse
            language: Optional language override (detected from extension if not provided)
            
        Returns:
            ParseResult with parse tree and metadata
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect language if not provided
            if language is None:
                language = self.get_language_from_extension(file_path)
                if language is None:
                    language = self.get_language_from_content(content, file_path)
            
            if language is None:
                error_msg = f"Could not detect language for file: {file_path}"
                self.logger.error(error_msg)
                return ParseResult(
                    language=SupportedLanguage.PYTHON,  # Default fallback
                    tree=None,
                    root_node=None,
                    source_code=content,
                    file_path=file_path,
                    success=False,
                    error_message=error_msg
                )
            
            # Parse the content
            return self.parse_code(content, language, file_path)
            
        except Exception as e:
            error_msg = f"Failed to read or parse file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return ParseResult(
                language=SupportedLanguage.PYTHON,  # Default fallback
                tree=None,
                root_node=None,
                source_code="",
                file_path=file_path,
                success=False,
                error_message=error_msg
            )
    
    async def parse_file_async(self, file_path: str, language: Optional[SupportedLanguage] = None) -> ParseResult:
        """Parse a file asynchronously.
        
        Args:
            file_path: Path to the file to parse
            language: Optional language override
            
        Returns:
            ParseResult with parse tree and metadata
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse_file, file_path, language)
    
    def parse_directory(self, directory_path: str, recursive: bool = True, 
                       include_patterns: Optional[List[str]] = None,
                       exclude_patterns: Optional[List[str]] = None) -> List[ParseResult]:
        """Parse all supported files in a directory.
        
        Args:
            directory_path: Path to the directory to parse
            recursive: Whether to parse subdirectories recursively
            include_patterns: Optional list of file patterns to include
            exclude_patterns: Optional list of file patterns to exclude
            
        Returns:
            List of ParseResult objects for all parsed files
        """
        results = []
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return results
        
        # Get all files to parse
        if recursive:
            files = directory.rglob("*")
        else:
            files = directory.iterdir()
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Check if file should be included
            if not self._should_include_file(str(file_path), include_patterns, exclude_patterns):
                continue
            
            # Check if language is supported
            language = self.get_language_from_extension(str(file_path))
            if language is None or not self.is_language_supported(language):
                continue
            
            # Parse the file
            result = self.parse_file(str(file_path), language)
            results.append(result)
        
        return results
    
    def get_parser_stats(self) -> ParserStats:
        """Get parser statistics.
        
        Returns:
            ParserStats object with usage statistics
        """
        return self.stats
    
    def clear_cache(self) -> None:
        """Clear the parse result cache."""
        self._parse_cache.clear()
        self._cache_access_times.clear()
        self.logger.info("Cleared Tree-sitter parse cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self.cache_enabled,
            "size": len(self._parse_cache),
            "max_size": self.cache_size,
            "hit_count": getattr(self, '_cache_hits', 0),
            "miss_count": getattr(self, '_cache_misses', 0)
        }
    
    def _get_cache_key(self, file_path: str, content: str) -> str:
        """Generate cache key for a file and content.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Cache key string
        """
        import hashlib
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{file_path}:{content_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[ParseResult]:
        """Get result from cache.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached ParseResult if found, None otherwise
        """
        if cache_key in self._parse_cache:
            self._cache_access_times[cache_key] = time.time()
            self._cache_hits = getattr(self, '_cache_hits', 0) + 1
            return self._parse_cache[cache_key]
        
        self._cache_misses = getattr(self, '_cache_misses', 0) + 1
        return None
    
    def _add_to_cache(self, cache_key: str, result: ParseResult) -> None:
        """Add result to cache.
        
        Args:
            cache_key: Cache key
            result: ParseResult to cache
        """
        # Check cache size limit
        if len(self._parse_cache) >= self.cache_size:
            self._evict_oldest_cache_entry()
        
        self._parse_cache[cache_key] = result
        self._cache_access_times[cache_key] = time.time()
    
    def _evict_oldest_cache_entry(self) -> None:
        """Remove the oldest cache entry."""
        if not self._cache_access_times:
            return
        
        # Find oldest entry
        oldest_key = min(self._cache_access_times.keys(), 
                        key=lambda k: self._cache_access_times[k])
        
        # Remove from cache
        del self._parse_cache[oldest_key]
        del self._cache_access_times[oldest_key]
    
    def _should_include_file(self, file_path: str, include_patterns: Optional[List[str]], 
                           exclude_patterns: Optional[List[str]]) -> bool:
        """Check if a file should be included based on patterns.
        
        Args:
            file_path: Path to the file
            include_patterns: Optional list of include patterns
            exclude_patterns: Optional list of exclude patterns
            
        Returns:
            True if file should be included
        """
        import fnmatch
        
        file_name = os.path.basename(file_path)
        
        # Check exclude patterns first
        if exclude_patterns:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path, pattern):
                    return False
        
        # Check include patterns
        if include_patterns:
            for pattern in include_patterns:
                if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path, pattern):
                    return True
            return False  # No include pattern matched
        
        return True  # No patterns specified, include by default