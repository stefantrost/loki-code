"""
Tree-sitter code analysis integration tests.

Tests the integration between tools and Tree-sitter analysis:
- Multi-language code parsing
- Context extraction and analysis
- Function and class detection
- Code structure analysis
"""

import pytest
from pathlib import Path

from ...core.code_analysis import (
    SupportedLanguage, CodeAnalyzer, ContextExtractor,
    analyze_file_quick, get_language_info, TreeSitterParser
)
from ...tools.file_reader import FileReaderTool
from ..fixtures.tool_test_fixtures import (
    test_config, temp_workspace, ToolTestHelpers,
    get_sample_code_path
)


class TestTreeSitterIntegration:
    """Test Tree-sitter integration with tools."""
    
    def test_python_analysis(self):
        """Test Python code analysis."""
        sample_file = get_sample_code_path("python")
        
        # Test quick analysis
        result = analyze_file_quick(str(sample_file))
        assert result is not None
        assert "fibonacci" in result
        assert "Calculator" in result
        assert "class" in result.lower()
        assert "function" in result.lower()
    
    def test_javascript_analysis(self):
        """Test JavaScript code analysis."""
        sample_file = get_sample_code_path("javascript")
        
        result = analyze_file_quick(str(sample_file))
        assert result is not None
        assert "Calculator" in result
        assert "fibonacci" in result
        assert "class" in result or "function" in result
    
    def test_typescript_analysis(self):
        """Test TypeScript code analysis."""
        sample_file = get_sample_code_path("typescript")
        
        result = analyze_file_quick(str(sample_file))
        assert result is not None
        assert "Calculator" in result
        assert "interface" in result.lower()
        assert "type" in result.lower()
    
    def test_rust_analysis(self):
        """Test Rust code analysis."""
        sample_file = get_sample_code_path("rust")
        
        result = analyze_file_quick(str(sample_file))
        assert result is not None
        assert "Calculator" in result
        assert "struct" in result.lower() or "impl" in result.lower()
    
    def test_language_detection(self):
        """Test automatic language detection."""
        test_cases = [
            ("test.py", SupportedLanguage.PYTHON),
            ("test.js", SupportedLanguage.JAVASCRIPT),
            ("test.ts", SupportedLanguage.TYPESCRIPT),
            ("test.rs", SupportedLanguage.RUST),
        ]
        
        for filename, expected_lang in test_cases:
            info = get_language_info(filename)
            assert info is not None
            assert info.name == expected_lang.value
    
    def test_code_analyzer_integration(self):
        """Test CodeAnalyzer with file reader tool."""
        sample_file = get_sample_code_path("python")
        
        analyzer = CodeAnalyzer()
        file_context = analyzer.analyze_file(str(sample_file))
        
        assert file_context is not None
        assert file_context.language == SupportedLanguage.PYTHON
        assert len(file_context.functions) > 0
        assert len(file_context.classes) > 0
        
        # Check specific functions/classes
        function_names = [f.name for f in file_context.functions]
        class_names = [c.name for c in file_context.classes]
        
        assert "fibonacci" in function_names
        assert "Calculator" in class_names
    
    def test_context_extractor_integration(self):
        """Test ContextExtractor with different analysis levels."""
        sample_file = get_sample_code_path("python")
        
        extractor = ContextExtractor()
        
        # Test minimal context
        minimal_context = extractor.extract_file_context(
            str(sample_file), level="minimal"
        )
        assert minimal_context is not None
        assert len(minimal_context) > 0
        
        # Test detailed context
        detailed_context = extractor.extract_file_context(
            str(sample_file), level="detailed"
        )
        assert detailed_context is not None
        assert len(detailed_context) > len(minimal_context)
        assert "fibonacci" in detailed_context
        assert "Calculator" in detailed_context
    
    def test_file_reader_with_analysis_levels(self):
        """Test file reader tool with different analysis levels."""
        sample_file = get_sample_code_path("python")
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Test each analysis level
        levels = ["minimal", "standard", "detailed", "comprehensive"]
        
        for level in levels:
            result = tool.execute({
                "file_path": str(sample_file),
                "analysis_level": level,
                "include_context": True
            }, context)
            
            ToolTestHelpers.assert_tool_result_success(result)
            
            # More detailed levels should have more content
            if level in ["detailed", "comprehensive"]:
                assert "fibonacci" in result.content
                assert "Calculator" in result.content
    
    def test_multi_language_project_analysis(self, temp_workspace):
        """Test analysis of a multi-language project."""
        # Create files in different languages
        files = {
            "main.py": "def main(): pass",
            "utils.js": "function util() { return true; }",
            "types.ts": "interface Config { name: string; }",
            "lib.rs": "pub fn hello() -> String { String::from(\"hello\") }"
        }
        
        for filename, content in files.items():
            file_path = temp_workspace / filename
            ToolTestHelpers.create_test_file(file_path, content)
        
        # Analyze each file
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        results = {}
        for filename in files:
            file_path = temp_workspace / filename
            result = tool.execute({
                "file_path": str(file_path),
                "analysis_level": "standard",
                "include_context": True
            }, context)
            
            ToolTestHelpers.assert_tool_result_success(result)
            results[filename] = result
        
        # Verify language-specific analysis
        assert "function" in results["main.py"].content.lower()
        assert "function" in results["utils.js"].content.lower()
        assert "interface" in results["types.ts"].content.lower()
        assert "function" in results["lib.rs"].content.lower()
    
    def test_parser_error_handling(self, temp_workspace):
        """Test parser error handling with invalid syntax."""
        # Create file with syntax errors
        invalid_py = temp_workspace / "invalid.py"
        ToolTestHelpers.create_test_file(invalid_py, "def broken_syntax(\n  # Missing closing parenthesis")
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        # Should still work but may have limited analysis
        result = tool.execute({
            "file_path": str(invalid_py),
            "analysis_level": "standard"
        }, context)
        
        # Should not crash, but may have warnings or limited analysis
        assert result.success or "syntax" in result.error.lower()
    
    def test_large_file_analysis(self, temp_workspace):
        """Test analysis of larger code files."""
        # Generate a larger Python file
        large_content = '''"""Large test file for performance testing."""

'''
        
        # Add many functions and classes
        for i in range(50):
            large_content += f'''
def function_{i}(param1: int, param2: str = "default") -> dict:
    """Function {i} for testing."""
    return {{"id": {i}, "param1": param1, "param2": param2}}

class Class_{i}:
    """Class {i} for testing."""
    
    def __init__(self):
        self.id = {i}
    
    def method_{i}(self):
        return f"Method from class {{self.id}}"
'''
        
        large_file = temp_workspace / "large_file.py"
        ToolTestHelpers.create_test_file(large_file, large_content)
        
        tool = FileReaderTool()
        context = ToolTestHelpers.create_mock_tool_context()
        
        result = tool.execute({
            "file_path": str(large_file),
            "analysis_level": "comprehensive"
        }, context)
        
        ToolTestHelpers.assert_tool_result_success(result)
        
        # Should detect multiple functions and classes
        assert "function_25" in result.content
        assert "Class_25" in result.content
        assert "Large test file" in result.content
    
    def test_tree_sitter_parser_direct(self):
        """Test Tree-sitter parser directly."""
        sample_file = get_sample_code_path("python")
        
        parser = TreeSitterParser()
        
        # Test parsing
        tree = parser.parse_file(str(sample_file), SupportedLanguage.PYTHON)
        assert tree is not None
        
        # Test node extraction
        functions = parser.extract_functions(tree, SupportedLanguage.PYTHON)
        classes = parser.extract_classes(tree, SupportedLanguage.PYTHON)
        
        assert len(functions) > 0
        assert len(classes) > 0
        
        # Check specific items
        function_names = [f.name for f in functions]
        class_names = [c.name for c in classes]
        
        assert "fibonacci" in function_names
        assert "Calculator" in class_names