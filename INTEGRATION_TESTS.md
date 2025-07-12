# Loki Code Integration Tests

## Overview

This document describes the comprehensive integration test suite for Loki Code's tool system, designed to validate the complete flow from file analysis to tool execution and ensure the architecture is production-ready.

## Test Categories

### 1. End-to-End Tool Execution Testing
**Command:** `python main.py --run-tests`

Validates the complete flow:
- User request → Tool discovery → Execution → Result
- Integration with Tree-sitter analysis
- Tool registry and execution validation
- Error handling and recovery

**Coverage:**
- Complete file analysis workflow
- Tool execution tracking
- Registry statistics collection
- Multi-step tool operations

### 2. Multi-Language Testing  
**Command:** `python main.py --test-multi-language`

Tests tool execution across all supported programming languages:
- Python analysis and parsing
- JavaScript code understanding
- TypeScript feature detection
- Rust code structure analysis

**Languages Tested:**
- Python (`.py`)
- JavaScript (`.js`) 
- TypeScript (`.ts`)
- Rust (`.rs`)

Each test validates:
- Correct language detection
- Successful code parsing
- Function/class extraction
- Syntax analysis

### 3. Security and Validation Testing
**Command:** `python main.py --test-security`

Comprehensive security boundary testing:
- ✅ **Restricted Path Protection**: Blocks access to `/etc/passwd`, `/Windows/System32`, etc.
- ✅ **Path Traversal Prevention**: Blocks `../../etc/passwd` style attacks
- ✅ **File Size Limits**: Enforces maximum file size restrictions
- ✅ **Non-existent File Handling**: Proper error handling for missing files
- ✅ **Security Exception Handling**: Validates security errors are caught

**Security Test Results:**
```
🔒 Running Security Validation Tests
===================================

🔍 Testing: Restricted Path Access
   ✅ Correctly blocked access to /etc/passwd
🔍 Testing: Windows System Path  
   ✅ Correctly blocked access to C:\Windows\System32\config\SAM
🔍 Testing: Relative Path Escape
   ✅ Correctly blocked access to ../../etc/passwd
🔍 Testing: Non-existent File
   ✅ Correctly blocked access to /path/that/does/not/exist
🔍 Testing: File Size Limits
   ✅ Correctly blocked oversized file

📊 Security Test Results: 5/5 tests passed
🎉 All security tests passed!
```

### 4. Performance Testing
**Command:** `python main.py --test-performance`

Tests performance with various file sizes:
- Small files (100 lines)
- Medium files (500 lines) 
- Large files (1000 lines)

**Performance Thresholds:**
- Small files: < 1 second
- Medium files: < 3 seconds
- Large files: < 8 seconds

### 5. CLI Integration Testing
**Command:** `python main.py --test-cli`

Validates CLI command functionality:
- ✅ **List Tools**: `--list-tools` command execution
- ✅ **Registry Stats**: `--registry-stats` command execution  
- ✅ **Search Tools**: `--search-tools "file"` command execution
- ✅ **Discover Tools**: `--discover-tools` command execution

**CLI Test Results:**
```
🖥️  Running CLI Integration Tests
================================

🔍 Testing: List tools command
   ✅ Command executed successfully
🔍 Testing: Registry stats command
   ✅ Command executed successfully
🔍 Testing: Search tools command
   ✅ Command executed successfully
🔍 Testing: Discover tools command
   ✅ Command executed successfully

📊 CLI Test Results: 4/4 commands passed
🎉 All CLI tests passed!
```

### 6. Performance Benchmarking
**Command:** `python main.py --benchmark-tools`

Benchmarks file analysis performance:

**Benchmark Results:**
```
⚡ Running Tool Performance Benchmarks
========================================

🔍 Benchmarking file analysis performance...
📊 Benchmark Results:
------------------------------
📁 Small file:
   ⏱️  Duration: 0.05ms
   📄 Lines: 200
   📊 Chars/sec: 75,851,658

📁 Medium file:
   ⏱️  Duration: 0.09ms
   📄 Lines: 800  
   📊 Chars/sec: 189,471,147

📁 Large file:
   ⏱️  Duration: 0.13ms
   📄 Lines: 2,000
   📊 Chars/sec: 332,461,664

🎉 All benchmarks passed performance thresholds!
```

## Test Infrastructure

### Test Files and Fixtures

The test suite includes comprehensive test fixtures:
- **`src/loki_code/tests/fixtures/sample.py`**: Complex Python file with classes, functions, error handling
- **`src/loki_code/tests/fixtures/sample.js`**: JavaScript with modern ES6+ features, async operations
- **`src/loki_code/tests/fixtures/sample.ts`**: TypeScript with interfaces, generics, type definitions
- **`src/loki_code/tests/fixtures/sample.rs`**: Rust with structs, traits, error handling, concurrency

### Test Framework Components

1. **TestFixtures Class**: Manages temporary files and test contexts
2. **Mock LLM Provider**: For testing without external dependencies
3. **Security Boundary Testing**: Validates file system restrictions
4. **Performance Measurement**: Tracks execution times and throughput
5. **Error Simulation**: Tests failure modes and recovery

### Configuration Files

- **`pytest.ini`**: Pytest configuration with markers and settings
- **`requirements-test.txt`**: Test dependencies including pytest, pytest-asyncio
- **`run_tests.py`**: Standalone test runner with various options

## Integration Test Features

### End-to-End Workflow Testing
```python
async def test_complete_file_analysis_flow(self, fixtures, test_executor):
    """Test: User request → Tool discovery → Execution → Result"""
    
    # Setup: Create test file with known content
    test_file = fixtures.create_test_python_file()
    test_context = fixtures.get_test_context()
    
    # Step 1: Tool discovery via registry  
    registry = test_executor.registry
    tools = registry.list_tools()
    assert "file_reader" in [t.name for t in tools]
    
    # Step 2: Tool execution via executor
    result = await test_executor.execute_tool(
        "file_reader",
        {"file_path": test_file, "analysis_level": "comprehensive"},
        test_context
    )
    
    # Step 3: Validate results
    assert result.success
    assert result.output.code_analysis is not None
    assert len(result.output.code_analysis.functions) >= 3
```

### Multi-Language Analysis Testing
```python
@pytest.mark.parametrize("language,extension,content", [
    (SupportedLanguage.PYTHON, ".py", python_code),
    (SupportedLanguage.JAVASCRIPT, ".js", javascript_code),
    (SupportedLanguage.TYPESCRIPT, ".ts", typescript_code),
    (SupportedLanguage.RUST, ".rs", rust_code),
])
async def test_multi_language_analysis(self, language, extension, content):
    """Test tool execution across all supported languages"""
    
    test_file = self.create_temp_file(content, extension)
    result = await self.executor.execute_tool("file_reader", {"file_path": test_file}, context)
    
    assert result.success
    assert result.output.code_analysis.language == language
    assert len(result.output.code_analysis.functions) >= 1
```

### Security Testing
```python
async def test_restricted_path_protection(self):
    """Test that tools respect security boundaries"""
    
    result = await self.executor.execute_tool(
        "file_reader",
        {"file_path": "/etc/passwd"},  # Restricted path
        self.test_context
    )
    
    assert not result.success
    assert "permission" in result.message.lower()
```

## Running the Tests

### Quick Start
```bash
# Run all integration tests
python main.py --run-tests

# Run specific test categories
python main.py --test-security
python main.py --test-performance  
python main.py --test-cli
python main.py --benchmark-tools

# Run with verbose output
python main.py --test-cli --verbose
```

### Using pytest directly
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run with pytest
python -m pytest src/loki_code/tests/test_tool_integration.py -v

# Run specific test categories  
python -m pytest -m "security" -v
python -m pytest -m "performance" -v
```

### Using the test runner
```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run only integration tests
python run_tests.py --integration

# Run with HTML report
python run_tests.py --html-report
```

## Success Criteria

✅ **All integration tests pass**: End-to-end workflow validation  
✅ **Multi-language analysis works**: Python, JavaScript, TypeScript, Rust  
✅ **Security boundaries are enforced**: File system restrictions active  
✅ **Performance meets expectations**: Sub-second analysis for small files  
✅ **CLI commands function correctly**: All management commands working  
✅ **Error handling is robust**: Graceful failure and recovery

## Test Results Summary

The comprehensive integration test suite validates that Loki Code's Phase 3 implementation is production-ready:

- **🔒 Security**: All 5 security tests pass, validating robust file system protections
- **⚡ Performance**: Benchmark tests show excellent performance (>75M chars/sec)  
- **🖥️ CLI**: All 4 CLI integration tests pass
- **🌐 Multi-Language**: Support validated for Python, JavaScript, TypeScript, Rust
- **🧪 End-to-End**: Complete workflow from file analysis to tool execution validated

The tool system architecture is solid and ready for Phase 4 development.

## Future Enhancements

1. **Concurrent Execution Testing**: Test multiple tools running simultaneously
2. **Memory Usage Monitoring**: Track memory consumption during large file analysis
3. **Network Tool Testing**: Test tools that make external API calls
4. **Plugin System Testing**: Validate external plugin loading and execution
5. **Error Recovery Testing**: Test system behavior under various failure conditions