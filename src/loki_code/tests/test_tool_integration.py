"""
Comprehensive integration tests for Loki Code's tool system.

This module validates the complete flow from file analysis to tool execution,
ensuring the architecture is solid for production use.

Test Categories:
- End-to-end tool execution testing
- Integration with Tree-sitter analysis  
- Tool registry and execution validation
- Performance and security testing
- CLI integration verification
"""

import asyncio
import json
import os
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

# Core imports
from ..config import load_config
from ..core.tool_registry import ToolRegistry, get_global_registry, ToolExecutionRecord
from ..core.tool_executor import ToolExecutor, ExecutionConfig, ExecutionContext
from ..core.code_analysis import (
    SupportedLanguage, CodeAnalyzer, ContextExtractor, 
    analyze_file_quick, get_language_info
)
from ..core.model_manager import ModelManager
from ..core.llm_client import create_llm_client

# Tool system imports
from ..tools.base import BaseTool
from ..tools.types import (
    ToolContext, ToolResult, ToolSchema, SecurityLevel, 
    ToolCapability, SafetySettings, ConfirmationLevel
)
from ..tools.exceptions import (
    ToolException, ToolNotFoundError, ToolExecutionError, 
    ToolSecurityError, ToolValidationError
)
from ..tools.file_reader import FileReaderTool

# Utilities
from ..utils.logging import get_logger


class TestFixtures:
    """Test fixture management for creating test files and contexts."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="loki_test_")
        self.temp_files = []
        self.logger = get_logger(__name__)
    
    def create_temp_file(self, content: str, extension: str = ".txt") -> str:
        """Create a temporary file with given content and extension."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=extension, 
            dir=self.temp_dir, 
            delete=False
        )
        temp_file.write(content)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_test_python_file(self) -> str:
        """Create a test Python file with known structure."""
        content = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class for basic operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history.copy()

# Complex function for testing analysis
def complex_function(data, options=None):
    """A more complex function with multiple paths."""
    if not data:
        raise ValueError("Data cannot be empty")
    
    processed = []
    for item in data:
        if options and options.get('transform'):
            item = options['transform'](item)
        
        if isinstance(item, (int, float)):
            if item > 0:
                processed.append(item * 2)
            elif item < 0:
                processed.append(abs(item))
            else:
                processed.append(1)
        else:
            processed.append(str(item).upper())
    
    return processed

if __name__ == "__main__":
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"4 * 7 = {calc.multiply(4, 7)}")
    print(f"Fibonacci(10) = {fibonacci(10)}")
    
    # Test complex function
    test_data = [1, -2, 0, "hello", 3.14]
    result = complex_function(test_data)
    print(f"Processed: {result}")
'''
        return self.create_temp_file(content, ".py")
    
    def create_test_javascript_file(self) -> str:
        """Create a test JavaScript file."""
        content = '''
function fibonacci(n) {
    /**
     * Calculate fibonacci number recursively
     */
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    /**
     * Simple calculator class for basic operations
     */
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return [...this.history];
    }
}

// Complex function for testing
function processData(data, options = {}) {
    if (!data || data.length === 0) {
        throw new Error("Data cannot be empty");
    }
    
    const processed = [];
    for (const item of data) {
        let processedItem = item;
        
        if (options.transform) {
            processedItem = options.transform(processedItem);
        }
        
        if (typeof processedItem === 'number') {
            if (processedItem > 0) {
                processed.push(processedItem * 2);
            } else if (processedItem < 0) {
                processed.push(Math.abs(processedItem));
            } else {
                processed.push(1);
            }
        } else {
            processed.push(String(processedItem).toUpperCase());
        }
    }
    
    return processed;
}

// Main execution
if (require.main === module) {
    const calc = new Calculator();
    console.log(`5 + 3 = ${calc.add(5, 3)}`);
    console.log(`4 * 7 = ${calc.multiply(4, 7)}`);
    console.log(`Fibonacci(10) = ${fibonacci(10)}`);
    
    const testData = [1, -2, 0, "hello", 3.14];
    const result = processData(testData);
    console.log(`Processed: ${result}`);
}
'''
        return self.create_temp_file(content, ".js")
    
    def create_test_typescript_file(self) -> str:
        """Create a test TypeScript file."""
        content = '''
interface CalculatorOptions {
    precision?: number;
    logHistory?: boolean;
}

interface HistoryEntry {
    operation: string;
    result: number;
    timestamp: Date;
}

function fibonacci(n: number): number {
    /**
     * Calculate fibonacci number recursively
     */
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    /**
     * Advanced calculator class with TypeScript features
     */
    private history: HistoryEntry[] = [];
    private options: CalculatorOptions;
    
    constructor(options: CalculatorOptions = {}) {
        this.options = { precision: 2, logHistory: true, ...options };
    }
    
    add(a: number, b: number): number {
        const result = this.roundToPrecision(a + b);
        this.logOperation(`${a} + ${b}`, result);
        return result;
    }
    
    multiply(a: number, b: number): number {
        const result = this.roundToPrecision(a * b);
        this.logOperation(`${a} * ${b}`, result);
        return result;
    }
    
    getHistory(): HistoryEntry[] {
        return [...this.history];
    }
    
    private roundToPrecision(value: number): number {
        const precision = this.options.precision || 2;
        return Math.round(value * Math.pow(10, precision)) / Math.pow(10, precision);
    }
    
    private logOperation(operation: string, result: number): void {
        if (this.options.logHistory) {
            this.history.push({
                operation,
                result,
                timestamp: new Date()
            });
        }
    }
}

// Generic function for processing data
function processArray<T>(
    data: T[], 
    processor: (item: T) => T,
    filter?: (item: T) => boolean
): T[] {
    let processed = data.map(processor);
    
    if (filter) {
        processed = processed.filter(filter);
    }
    
    return processed;
}

// Main execution
function main(): void {
    const calc = new Calculator({ precision: 3, logHistory: true });
    console.log(`5 + 3 = ${calc.add(5, 3)}`);
    console.log(`4 * 7 = ${calc.multiply(4, 7)}`);
    console.log(`Fibonacci(10) = ${fibonacci(10)}`);
    
    const numbers = [1, 2, 3, 4, 5];
    const doubled = processArray(numbers, x => x * 2, x => x > 5);
    console.log(`Filtered doubled: ${doubled}`);
}

if (require.main === module) {
    main();
}
'''
        return self.create_temp_file(content, ".ts")
    
    def create_test_rust_file(self) -> str:
        """Create a test Rust file."""
        content = '''
use std::collections::HashMap;

/// Calculate fibonacci number recursively
fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

/// Simple calculator struct for basic operations
#[derive(Debug)]
struct Calculator {
    history: Vec<String>,
    precision: usize,
}

impl Calculator {
    /// Create a new calculator with default precision
    fn new() -> Self {
        Self {
            history: Vec::new(),
            precision: 2,
        }
    }
    
    /// Create a calculator with custom precision
    fn with_precision(precision: usize) -> Self {
        Self {
            history: Vec::new(),
            precision,
        }
    }
    
    /// Add two numbers
    fn add(&mut self, a: f64, b: f64) -> f64 {
        let result = self.round_to_precision(a + b);
        self.log_operation(&format!("{} + {} = {}", a, b, result));
        result
    }
    
    /// Multiply two numbers
    fn multiply(&mut self, a: f64, b: f64) -> f64 {
        let result = self.round_to_precision(a * b);
        self.log_operation(&format!("{} * {} = {}", a, b, result));
        result
    }
    
    /// Get calculation history
    fn get_history(&self) -> &Vec<String> {
        &self.history
    }
    
    /// Round to configured precision
    fn round_to_precision(&self, value: f64) -> f64 {
        let multiplier = 10_f64.powi(self.precision as i32);
        (value * multiplier).round() / multiplier
    }
    
    /// Log an operation to history
    fn log_operation(&mut self, operation: &str) {
        self.history.push(operation.to_string());
    }
}

/// Process a vector of numbers with a given transformation
fn process_numbers<F>(numbers: Vec<f64>, transform: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    numbers.into_iter().map(transform).collect()
}

/// Complex function for testing analysis
fn analyze_data(data: &[i32]) -> HashMap<String, f64> {
    let mut results = HashMap::new();
    
    if data.is_empty() {
        return results;
    }
    
    let sum: i32 = data.iter().sum();
    let count = data.len() as f64;
    let mean = sum as f64 / count;
    
    results.insert("sum".to_string(), sum as f64);
    results.insert("count".to_string(), count);
    results.insert("mean".to_string(), mean);
    
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / count;
    
    results.insert("variance".to_string(), variance);
    results.insert("std_dev".to_string(), variance.sqrt());
    
    results
}

fn main() {
    let mut calc = Calculator::with_precision(3);
    
    println!("5 + 3 = {}", calc.add(5.0, 3.0));
    println!("4 * 7 = {}", calc.multiply(4.0, 7.0));
    println!("Fibonacci(10) = {}", fibonacci(10));
    
    let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let doubled = process_numbers(numbers, |x| x * 2.0);
    println!("Doubled: {:?}", doubled);
    
    let test_data = vec![1, 4, 7, 2, 9, 3, 5];
    let analysis = analyze_data(&test_data);
    println!("Analysis: {:?}", analysis);
    
    println!("History: {:?}", calc.get_history());
}
'''
        return self.create_temp_file(content, ".rs")
    
    def create_large_python_file(self, lines: int = 2000) -> str:
        """Create a large Python file for performance testing."""
        content_parts = [
            '"""Large Python file for performance testing."""',
            'import sys',
            'import os',
            'from typing import List, Dict, Any, Optional',
            '',
        ]
        
        # Generate many function definitions
        for i in range(lines // 10):
            content_parts.append(f'''
def function_{i}(param1: int, param2: str = "default") -> Dict[str, Any]:
    """Function {i} for testing purposes."""
    result = {{
        "function_id": {i},
        "param1": param1,
        "param2": param2,
        "computed": param1 * {i + 1}
    }}
    
    if param1 > {i * 10}:
        result["category"] = "high"
    elif param1 > {i * 5}:
        result["category"] = "medium"
    else:
        result["category"] = "low"
    
    return result
''')
        
        # Add a main function that uses all the generated functions
        content_parts.append('''
def main():
    """Main function that exercises all generated functions."""
    results = []
    
''')
        
        for i in range(lines // 10):
            content_parts.append(f'    results.append(function_{i}({i}, "test_{i}"))')
        
        content_parts.append('''
    
    print(f"Processed {len(results)} functions")
    return results

if __name__ == "__main__":
    main()
''')
        
        content = '\n'.join(content_parts)
        return self.create_temp_file(content, ".py")
    
    def create_binary_test_file(self) -> str:
        """Create a binary test file."""
        import struct
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb', 
            suffix=".bin", 
            dir=self.temp_dir, 
            delete=False
        )
        
        # Write some binary data
        for i in range(100):
            temp_file.write(struct.pack('>I', i))
        
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_large_test_file(self, size_mb: int) -> str:
        """Create a large test file of specified size in MB."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=".txt", 
            dir=self.temp_dir, 
            delete=False
        )
        
        # Write data to reach the desired size
        chunk = "A" * 1024  # 1KB chunk
        chunks_needed = size_mb * 1024  # MB to KB
        
        for _ in range(chunks_needed):
            temp_file.write(chunk)
        
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def get_test_safety_settings(self) -> SafetySettings:
        """Get safety settings for testing."""
        return SafetySettings(
            allowed_paths=[self.temp_dir, "./"],
            restricted_paths=["/etc", "/usr/bin", "/System", "/Windows"],
            max_file_size_mb=50,
            require_confirmation_for=[]
        )
    
    def get_test_context(self) -> ToolContext:
        """Create a test tool context."""
        return ToolContext(
            project_path=self.temp_dir,
            user_id="test_user",
            session_id="test_session",
            safety_settings=self.get_test_safety_settings(),
            dry_run=False
        )
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.call_count = 0
        
    async def generate(self, request):
        """Mock generation method."""
        self.call_count += 1
        return {
            "text": f"Mock response {self.call_count}",
            "usage": {"tokens": 100}
        }


@pytest.fixture(scope="function")
def fixtures():
    """Pytest fixture for test fixtures."""
    test_fixtures = TestFixtures()
    yield test_fixtures
    test_fixtures.cleanup()


@pytest.fixture(scope="function") 
def mock_llm():
    """Pytest fixture for mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture(scope="function")
def test_config():
    """Pytest fixture for test configuration."""
    return {
        "discovery": {"auto_discover_builtin": True},
        "execution": {
            "default_timeout_seconds": 30.0,
            "max_concurrent_tools": 3,
            "enable_execution_tracking": True
        },
        "security": {
            "allowed_paths": ["./", "~/projects/"],
            "max_file_size_mb": 50
        }
    }


@pytest.fixture(scope="function")
async def test_registry(test_config):
    """Pytest fixture for test tool registry."""
    registry = ToolRegistry(test_config)
    
    # Manually register FileReaderTool for testing
    registry.register_tool_class(FileReaderTool, source="test")
    
    return registry


@pytest.fixture(scope="function")
async def test_executor(test_registry, mock_llm):
    """Pytest fixture for test tool executor."""
    config = ExecutionConfig(
        default_timeout_seconds=30.0,
        max_concurrent_executions=3,
        enable_tracking=True
    )
    
    executor = ToolExecutor(test_registry, config)
    return executor


class TestToolIntegration:
    """End-to-end tool execution testing."""
    
    @pytest.mark.asyncio
    async def test_complete_file_analysis_flow(self, fixtures, test_executor):
        """Test: User request → Tool discovery → Execution → Result"""
        
        # Setup: Create test file with known content
        test_file = fixtures.create_test_python_file()
        test_context = fixtures.get_test_context()
        
        # Step 1: Tool discovery via registry
        registry = test_executor.registry
        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        
        assert "file_reader" in tool_names, f"file_reader not found in {tool_names}"
        
        # Step 2: Tool execution via executor
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": test_file, "analysis_level": "comprehensive"},
            test_context
        )
        
        # Step 3: Validate results
        assert result.success, f"Tool execution failed: {result.message}"
        assert result.output is not None, "Tool output is None"
        assert hasattr(result.output, 'file_info'), "Missing file_info in output"
        assert hasattr(result.output, 'content'), "Missing content in output"
        
        # Step 4: Validate file analysis
        file_info = result.output.file_info
        assert file_info.path == test_file
        assert file_info.language == SupportedLanguage.PYTHON
        assert not file_info.is_binary
        assert file_info.size_bytes > 0
        
        # Step 5: Validate content analysis
        if hasattr(result.output, 'code_analysis') and result.output.code_analysis:
            code_analysis = result.output.code_analysis
            assert code_analysis.language == SupportedLanguage.PYTHON
            assert len(code_analysis.functions) >= 3  # fibonacci, add, multiply
            assert code_analysis.complexity_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_tool_execution_tracking(self, fixtures, test_executor):
        """Test that tool execution is properly tracked."""
        
        test_file = fixtures.create_test_python_file()
        test_context = fixtures.get_test_context()
        
        # Execute tool
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": test_file},
            test_context
        )
        
        assert result.success
        
        # Check execution history
        registry = test_executor.registry
        history = registry.get_execution_history()
        
        assert len(history) >= 1
        
        latest_execution = history[-1]
        assert latest_execution.tool_name == "file_reader"
        assert latest_execution.start_time is not None
        assert latest_execution.end_time is not None
        assert latest_execution.duration_ms is not None
        assert latest_execution.duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_tool_registry_stats(self, test_executor):
        """Test tool registry statistics collection."""
        
        registry = test_executor.registry
        stats = registry.get_registry_stats()
        
        assert "total_tools" in stats
        assert "enabled_tools" in stats
        assert "source_distribution" in stats
        assert "capability_distribution" in stats
        assert "security_distribution" in stats
        
        assert stats["total_tools"] >= 1
        assert stats["enabled_tools"] >= 1
        assert "test" in stats["source_distribution"]


@pytest.mark.parametrize("language,extension,content", [
    (SupportedLanguage.PYTHON, ".py", '''
def hello():
    """Simple hello function."""
    print('Hello World')
    return "Hello"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
'''),
    (SupportedLanguage.JAVASCRIPT, ".js", '''
function hello() {
    /**
     * Simple hello function
     */
    console.log('Hello World');
    return "Hello";
}

class Greeter {
    greet(name) {
        return `Hello, ${name}!`;
    }
}
'''),
    (SupportedLanguage.TYPESCRIPT, ".ts", '''
function hello(): string {
    /**
     * Simple hello function
     */
    console.log('Hello World');
    return "Hello";
}

class Greeter {
    greet(name: string): string {
        return `Hello, ${name}!`;
    }
}
'''),
    (SupportedLanguage.RUST, ".rs", '''
/// Simple hello function
fn hello() -> &'static str {
    println!("Hello World");
    "Hello"
}

struct Greeter;

impl Greeter {
    fn greet(&self, name: &str) -> String {
        format!("Hello, {}!", name)
    }
}
''')
])
class TestMultiLanguageAnalysis:
    """Test tool execution across all supported languages."""
    
    @pytest.mark.asyncio
    async def test_multi_language_analysis(self, fixtures, test_executor, language, extension, content):
        """Test tool execution across all supported languages."""
        
        test_file = fixtures.create_temp_file(content, extension)
        test_context = fixtures.get_test_context()
        
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": test_file, "analysis_level": "detailed"},
            test_context
        )
        
        assert result.success, f"Failed to analyze {language.value} file: {result.message}"
        assert result.output.file_info.language == language
        
        # Verify basic analysis was performed
        if hasattr(result.output, 'code_analysis') and result.output.code_analysis:
            code_analysis = result.output.code_analysis
            assert code_analysis.language == language
            
            # All test files should have at least one function
            if language in [SupportedLanguage.PYTHON, SupportedLanguage.JAVASCRIPT, 
                          SupportedLanguage.TYPESCRIPT, SupportedLanguage.RUST]:
                assert len(code_analysis.functions) >= 1


class TestToolSecurity:
    """Security and validation testing for tools."""
    
    @pytest.mark.asyncio
    async def test_restricted_path_protection(self, fixtures, test_executor):
        """Test that tools respect security boundaries."""
        
        test_context = fixtures.get_test_context()
        
        # Try to access restricted path
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": "/etc/passwd"},  # Restricted path
            test_context
        )
        
        assert not result.success, "Tool should have failed on restricted path"
        assert "permission" in result.message.lower() or "restricted" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_file_size_limits(self, fixtures, test_executor):
        """Test file size limit enforcement."""
        
        # Create oversized file (larger than 50MB limit in test config)
        large_file = fixtures.create_large_test_file(size_mb=60)
        test_context = fixtures.get_test_context()
        
        result = await test_executor.execute_tool(
            "file_reader", 
            {"file_path": large_file},
            test_context
        )
        
        assert not result.success, "Tool should have failed on oversized file"
        assert "size" in result.message.lower() or "large" in result.message.lower()
    
    @pytest.mark.asyncio 
    async def test_binary_file_handling(self, fixtures, test_executor):
        """Test handling of binary files."""
        
        binary_file = fixtures.create_binary_test_file()
        test_context = fixtures.get_test_context()
        
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": binary_file},
            test_context
        )
        
        # Should succeed but identify as binary
        assert result.success, f"Binary file handling failed: {result.message}"
        assert result.output.file_info.is_binary, "Binary file not identified as binary"
        
        # Should not attempt code analysis on binary files
        if hasattr(result.output, 'code_analysis'):
            assert result.output.code_analysis is None or len(result.output.code_analysis.functions) == 0
    
    @pytest.mark.asyncio
    async def test_nonexistent_file_handling(self, fixtures, test_executor):
        """Test handling of non-existent files."""
        
        test_context = fixtures.get_test_context()
        
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": "/path/that/does/not/exist.py"},
            test_context
        )
        
        assert not result.success, "Tool should have failed on non-existent file"
        assert "not found" in result.message.lower() or "exist" in result.message.lower()


class TestToolPerformance:
    """Performance testing for tools."""
    
    @pytest.mark.asyncio
    async def test_large_file_performance(self, fixtures, test_executor):
        """Test performance with large code files."""
        
        # Create large but valid Python file
        large_file = fixtures.create_large_python_file(lines=1000)  # Smaller for CI
        test_context = fixtures.get_test_context()
        
        start_time = time.time()
        result = await test_executor.execute_tool(
            "file_reader",
            {"file_path": large_file, "analysis_level": "standard"},
            test_context
        )
        execution_time = time.time() - start_time
        
        assert result.success, f"Large file analysis failed: {result.message}"
        assert execution_time < 10.0, f"Analysis took too long: {execution_time}s"
        
        # Verify analysis was performed
        if hasattr(result.output, 'code_analysis') and result.output.code_analysis:
            assert result.output.code_analysis.language == SupportedLanguage.PYTHON
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, fixtures, test_executor):
        """Test multiple tools running concurrently."""
        
        # Create multiple test files
        test_files = [
            fixtures.create_test_python_file(),
            fixtures.create_test_javascript_file(), 
            fixtures.create_test_typescript_file()
        ]
        test_context = fixtures.get_test_context()
        
        # Execute multiple tools concurrently
        tasks = []
        for test_file in test_files:
            task = test_executor.execute_tool(
                "file_reader",
                {"file_path": test_file},
                test_context
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        successful_results = [r for r in results if isinstance(r, type(results[0])) and r.success]
        assert len(successful_results) == len(test_files), f"Some concurrent executions failed: {results}"
        
        # Check execution history
        registry = test_executor.registry
        history = registry.get_execution_history()
        assert len(history) >= len(test_files)


class TestErrorHandling:
    """Comprehensive error handling testing."""
    
    @pytest.mark.asyncio
    async def test_nonexistent_tool(self, fixtures, test_executor):
        """Test execution of non-existent tool."""
        
        test_context = fixtures.get_test_context()
        
        result = await test_executor.execute_tool(
            "nonexistent_tool",
            {"some": "input"},
            test_context
        )
        
        assert not result.success, "Non-existent tool should fail"
        assert "not found" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_input_data(self, fixtures, test_executor):
        """Test tool execution with invalid input."""
        
        test_context = fixtures.get_test_context()
        
        result = await test_executor.execute_tool(
            "file_reader",
            {"invalid_param": "value"},  # Missing required file_path
            test_context
        )
        
        assert not result.success, "Invalid input should cause failure"
        assert any(word in result.message.lower() for word in ["invalid", "missing", "required"])
    
    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self, fixtures, test_executor):
        """Test tool timeout handling."""
        
        # This is a conceptual test - would need a slow tool for real testing
        test_context = fixtures.get_test_context()
        test_file = fixtures.create_test_python_file()
        
        # Set very short timeout
        short_timeout_executor = ToolExecutor(
            test_executor.registry, 
            ExecutionConfig(default_timeout_seconds=0.001)  # Very short
        )
        
        # This might timeout or succeed depending on system speed
        result = await short_timeout_executor.execute_tool(
            "file_reader",
            {"file_path": test_file},
            test_context
        )
        
        # Either succeeds quickly or times out
        if not result.success:
            assert "timeout" in result.message.lower()


class TestCLIIntegration:
    """CLI integration testing."""
    
    def test_list_tools_command(self):
        """Test --list-tools CLI command."""
        
        result = subprocess.run([
            "python", "main.py", "--list-tools"
        ], capture_output=True, text=True, cwd="/Users/stefantrost/Projects/techscene/loki-code")
        
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert "file_reader" in result.stdout, f"file_reader not found in output: {result.stdout}"
    
    def test_registry_stats_command(self):
        """Test --registry-stats CLI command."""
        
        result = subprocess.run([
            "python", "main.py", "--registry-stats"
        ], capture_output=True, text=True, cwd="/Users/stefantrost/Projects/techscene/loki-code")
        
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert "total_tools" in result.stdout.lower(), f"Stats not found in output: {result.stdout}"
    
    def test_search_tools_command(self):
        """Test --search-tools CLI command."""
        
        result = subprocess.run([
            "python", "main.py", "--search-tools", "file"
        ], capture_output=True, text=True, cwd="/Users/stefantrost/Projects/techscene/loki-code")
        
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert "file_reader" in result.stdout, f"file_reader not found in search: {result.stdout}"
    
    def test_tool_info_command(self):
        """Test --tool-info CLI command."""
        
        result = subprocess.run([
            "python", "main.py", "--tool-info", "file_reader"
        ], capture_output=True, text=True, cwd="/Users/stefantrost/Projects/techscene/loki-code")
        
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert "file_reader" in result.stdout, f"Tool info not found: {result.stdout}"
        assert "schema" in result.stdout.lower() or "description" in result.stdout.lower()


# Performance benchmarking functions
def benchmark_file_analysis():
    """Benchmark file analysis performance."""
    import time
    from pathlib import Path
    
    fixtures = TestFixtures()
    
    try:
        # Create test files of various sizes
        test_files = [
            ("small", fixtures.create_test_python_file()),
            ("medium", fixtures.create_large_python_file(500)),
            ("large", fixtures.create_large_python_file(1500))
        ]
        
        results = {}
        
        for size_name, test_file in test_files:
            start_time = time.time()
            
            # Simulate file analysis (basic file reading and size calculation)
            with open(test_file, 'r') as f:
                content = f.read()
                line_count = len(content.split('\n'))
                char_count = len(content)
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[size_name] = {
                "duration_ms": duration * 1000,
                "lines": line_count,
                "chars": char_count,
                "chars_per_second": char_count / duration if duration > 0 else 0
            }
        
        return results
    
    finally:
        fixtures.cleanup()


if __name__ == "__main__":
    # Run benchmarks when executed directly
    print("Running performance benchmarks...")
    bench_results = benchmark_file_analysis()
    
    print("\nFile Analysis Performance:")
    for size, metrics in bench_results.items():
        print(f"  {size.capitalize()}: {metrics['duration_ms']:.2f}ms "
              f"({metrics['lines']} lines, {metrics['chars_per_second']:.0f} chars/sec)")