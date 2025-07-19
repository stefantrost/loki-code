"""
Standalone tests for entity extraction functionality to avoid circular import issues.
"""

import pytest
import spacy
import re
from typing import Dict, List, Optional, Set
from unittest.mock import Mock


class StandaloneEntityExtractor:
    """Standalone version of EntityExtractor for testing."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.programming_languages = {
            "python", "javascript", "typescript", "java", "cpp", "c++", "c", 
            "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "perl",
            "shell", "bash", "sql", "html", "css", "json", "yaml", "xml"
        }
        self.naming_indicators = {
            "called", "named", "name", "call", "title", "save", "with"
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy's linguistic analysis."""
        try:
            doc = self.nlp(text)
            
            entities = {
                "files": [],
                "functions": [],
                "classes": [],
                "languages": [],
                "create_targets": [],
                "directories": [],
            }
            
            # Extract different entity types
            entities["files"] = self._extract_filenames(doc, text)
            entities["languages"] = self._extract_programming_languages(doc)
            entities["functions"] = self._extract_functions(doc)
            entities["classes"] = self._extract_classes(doc)
            
            # Remove duplicates and clean up
            for key in entities:
                entities[key] = list(set([item for item in entities[key] if item]))
            
            return entities
            
        except Exception as e:
            return self._get_empty_entities()
    
    def _extract_filenames(self, doc, text: str) -> List[str]:
        """Extract filenames using multiple approaches."""
        filenames = []
        
        # Method 1: Quoted strings
        quoted_filenames = self._extract_quoted_strings(text)
        filenames.extend(quoted_filenames)
        
        # Method 2: Dependency parsing
        dependency_filenames = self._extract_filenames_from_dependencies(doc)
        filenames.extend(dependency_filenames)
        
        # Method 3: Files with extensions
        extension_filenames = self._extract_files_with_extensions(text)
        filenames.extend(extension_filenames)
        
        return filenames
    
    def _extract_quoted_strings(self, text: str) -> List[str]:
        """Extract strings in quotes."""
        filenames = []
        
        quoted_patterns = [
            r"['\"]([^'\"]+)['\"]",
            r"'([^']+)'",
            r'"([^"]+)"'
        ]
        
        for pattern in quoted_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                quoted_text = match.group(1).strip()
                if self._looks_like_filename(quoted_text):
                    filenames.append(quoted_text)
        
        return filenames
    
    def _extract_filenames_from_dependencies(self, doc) -> List[str]:
        """Extract filenames using dependency parsing."""
        filenames = []
        
        for token in doc:
            if token.lemma_ in self.naming_indicators:
                filename = self._find_naming_target(token)
                if filename and self._looks_like_filename(filename):
                    filenames.append(filename)
        
        return filenames
    
    def _find_naming_target(self, token) -> Optional[str]:
        """Find the target of naming verbs."""
        for child in token.children:
            if child.dep_ in ["dobj", "pobj", "attr", "nsubj"]:
                name = child.text.strip("'\"")
                if name and len(name) < 100:
                    return name
        
        for descendant in token.subtree:
            if descendant.dep_ in ["pobj", "dobj"]:
                name = descendant.text.strip("'\"")
                if name and len(name) < 100:
                    return name
        
        return None
    
    def _extract_files_with_extensions(self, text: str) -> List[str]:
        """Extract files with extensions."""
        filenames = []
        
        file_pattern = r'\b([\w\-_./]+\.(py|js|ts|java|cpp|c|rs|go|rb|php|json|yaml|yml|md|txt|csv|xml|html|css|sh|bat))\b'
        
        matches = re.finditer(file_pattern, text, re.IGNORECASE)
        for match in matches:
            filename = match.group(1)
            if self._looks_like_filename(filename):
                filenames.append(filename)
        
        return filenames
    
    def _looks_like_filename(self, text: str) -> bool:
        """Determine if text looks like a filename."""
        if not text or len(text) > 100:
            return False
        
        common_words = {
            'file', 'name', 'it', 'this', 'that', 'the', 'a', 'an', 'with',
            'called', 'named', 'save', 'open', 'create', 'make', 'write'
        }
        
        if text.lower() in common_words:
            return False
        
        if ' ' in text and '.' not in text:
            return False
        
        if text.isupper() and len(text) > 10:
            return False
        
        return True
    
    def _extract_programming_languages(self, doc) -> List[str]:
        """Extract programming languages."""
        languages = []
        
        for token in doc:
            token_text = token.text.lower()
            token_lemma = token.lemma_.lower()
            
            if token_text in self.programming_languages:
                languages.append(token_text)
            elif token_lemma in self.programming_languages:
                languages.append(token_lemma)
            elif token_text == "js":
                languages.append("javascript")
            elif token_text == "ts":
                languages.append("typescript")
            elif token_text in ["c++", "cpp"]:
                languages.append("cpp")
        
        return languages
    
    def _extract_functions(self, doc) -> List[str]:
        """Extract function names."""
        functions = []
        
        for token in doc:
            if token.lemma_ in ["function", "method", "def"]:
                for child in token.children:
                    if child.dep_ in ["dobj", "attr"] and child.pos_ in ["NOUN", "PROPN"]:
                        functions.append(child.text)
            elif token.text.endswith("()"):
                func_name = token.text[:-2]
                if func_name.isidentifier():
                    functions.append(func_name)
        
        return functions
    
    def _extract_classes(self, doc) -> List[str]:
        """Extract class names."""
        classes = []
        
        for token in doc:
            if token.lemma_ == "class":
                for child in token.children:
                    if child.dep_ in ["dobj", "attr"] and child.pos_ in ["NOUN", "PROPN"]:
                        if child.text[0].isupper():
                            classes.append(child.text)
            elif (token.pos_ in ["NOUN", "PROPN"] and 
                  len(token.text) > 2 and 
                  token.text[0].isupper() and 
                  any(c.isupper() for c in token.text[1:])):
                classes.append(token.text)
        
        return classes
    
    def _get_empty_entities(self) -> Dict[str, List[str]]:
        """Return empty entities structure."""
        return {
            "files": [],
            "functions": [],
            "classes": [],
            "languages": [],
            "create_targets": [],
            "directories": [],
        }


class TestStandaloneEntityExtraction:
    """Test entity extraction without circular import issues."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return StandaloneEntityExtractor()

    def test_extract_quoted_filenames(self, extractor):
        """Test extraction of quoted filenames."""
        test_cases = [
            ("create a file with the name 'great'", ["great"]),
            ("make a python file called 'awesome'", ["awesome"]),
            ("write a new file named \"test\"", ["test"]),
            ("save it as 'config.json'", ["config.json"]),
        ]
        
        for text, expected in test_cases:
            entities = extractor.extract_entities(text)
            for expected_file in expected:
                assert expected_file in entities["files"], f"Failed to find '{expected_file}' in: {text}"

    def test_extract_programming_languages(self, extractor):
        """Test extraction of programming languages."""
        test_cases = [
            ("create a python file", ["python"]),
            ("write some go code", ["go"]),
            ("make a JavaScript function", ["javascript"]),
            ("create a cpp program", ["cpp"]),
        ]
        
        for text, expected in test_cases:
            entities = extractor.extract_entities(text)
            for expected_lang in expected:
                assert expected_lang in entities["languages"], f"Failed to find '{expected_lang}' in: {text}"

    def test_dependency_parsing_for_naming(self, extractor):
        """Test dependency parsing for naming patterns."""
        test_cases = [
            "create a file with the name 'great'",
            "make a file called 'awesome'", 
            "write a new file named 'test'",
        ]
        
        for text in test_cases:
            entities = extractor.extract_entities(text)
            assert len(entities["files"]) > 0, f"Failed to extract filename from: {text}"

    def test_looks_like_filename_heuristics(self, extractor):
        """Test filename detection heuristics."""
        valid_filenames = ["great", "test", "config", "main", "helper.py"]
        invalid_filenames = ["file", "name", "it", "this", "that"]
        
        for filename in valid_filenames:
            assert extractor._looks_like_filename(filename), f"Should accept: {filename}"
        
        for filename in invalid_filenames:
            assert not extractor._looks_like_filename(filename), f"Should reject: {filename}"

    def test_complex_extraction(self, extractor):
        """Test complex scenarios."""
        text = "create a python file called 'user_manager' with a class UserManager"
        entities = extractor.extract_entities(text)
        
        assert "user_manager" in entities["files"]
        assert "python" in entities["languages"]
        assert "UserManager" in entities["classes"]

    @pytest.mark.parametrize("input_text,expected_files", [
        ("create an empty go file with the name 'great'", ["great"]),
        ("make a python file called 'awesome'", ["awesome"]),
        ("write a new java file named 'test'", ["test"]),
        ("create a file with the name 'config' in go", ["config"]),
    ])
    def test_user_reported_scenarios(self, extractor, input_text, expected_files):
        """Test specific scenarios reported by users."""
        entities = extractor.extract_entities(input_text)
        
        for expected_file in expected_files:
            assert expected_file in entities["files"], f"Failed to extract '{expected_file}' from: {input_text}"