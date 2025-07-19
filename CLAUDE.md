# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Development installation with all dependencies
pip install -e ".[dev]"

# Production installation only
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests

# Run single test file
pytest tests/test_agent_basic.py

# Run with coverage
pytest --cov=loki_code --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Lint check
flake8 src/
```

### Application Entry Points
```bash
# Main CLI (comprehensive)
python main.py --help

# Package entry point
loki-code --help

# Key development commands (HTTP mode is now default)
python main.py --test-llm                           # Test LLM connection (HTTP mode)
python main.py --tui                               # Full TUI mode (HTTP mode)
python main.py --chat                              # Simple chat mode (HTTP mode)
python main.py --list-tools                        # Available tools

# Direct Mode (communicates directly with local LLM providers)
python main.py --config configs/direct-mode.yaml --tui  # TUI with direct LLM (Ollama)
python main.py --config configs/direct-mode.yaml --chat # Chat with direct LLM (Ollama)
```

## Architecture Overview

### Core Design Principles
- **Service-oriented architecture**: HTTP-based LLM server for scalability and performance (default)
- **Local-first execution**: Designed for enterprise environments with local LLMs
- **MCP-compatible**: Tool system built for Model Context Protocol integration
- **Modular architecture**: Clear separation between core logic, UI, and tools
- **Security-first**: Permission-based autonomy and safe execution boundaries

### High-Level Architecture

```
┌─────────────────┬──────────────────┬─────────────────┐
│   UI Layer      │   Core Layer     │   Tool Layer    │
├─────────────────┼──────────────────┼─────────────────┤
│ • Rich Console  │ • LLM Providers  │ • Tool Registry │
│ • Interactive   │ • Agent System   │ • File Tools    │
│ • Real-time     │ • Model Manager  │ • Code Analysis │
│   Input         │ • Task Routing   │ • MCP Support   │
└─────────────────┴──────────────────┴─────────────────┘
```

### Key Architectural Components

#### 1. Multi-Provider LLM System (`src/loki_code/core/providers/`)
- **Task-aware routing**: Different models for chat, code analysis, etc.
- **Provider abstraction**: Unified interface for Ollama, OpenAI, future providers
- **Resource optimization**: Automatic model selection based on task complexity

#### 2. Intelligent Agent System (`src/loki_code/core/agent/`)
- **Permission-based autonomy**: "yes once", "yes always", "deny" patterns
- **Safety boundaries**: Project-scoped execution with rollback capabilities
- **LangChain integration**: ReAct reasoning with self-correction

#### 3. Tree-sitter Code Analysis (`src/loki_code/core/code_analysis/`)
- **Multi-language parsing**: Python, JavaScript, TypeScript, Rust support
- **Intelligent context**: Generates LLM-optimized code summaries
- **Structure analysis**: Function/class extraction for precise assistance

#### 4. MCP-Compatible Tool System (`src/loki_code/tools/`)
- **Auto-discovery**: Registry scans for available tools
- **Security validation**: Input sanitization and execution boundaries  
- **Extension ready**: Built for MCP protocol integration

#### 5. Rich TUI System (`src/loki_code/ui/`)
- **Real-time input**: Character-by-character preview with optimized refresh
- **Live conversations**: Flicker-free message streaming
- **Session persistence**: Conversation history and state management

### Critical Integration Points

#### Agent ↔ Tool Integration
The agent system uses a standardized tool execution pipeline:
1. **Tool Discovery**: Registry provides available tools to agent
2. **Permission Check**: Safety manager validates tool requests
3. **Execution**: Tool executor runs with security boundaries
4. **Result Processing**: Response formatter converts tool output for LLM

#### LLM ↔ Provider Integration  
Task-aware model routing via `ModelManager`:
- **Conversation**: General chat models (llama3.1, claude-haiku)
- **Code Analysis**: Specialized code models (codellama, deepseek-coder)
- **Tool Planning**: High-reasoning models (claude-sonnet, gpt-4)

#### UI ↔ Agent Integration
Real-time interaction flow:
1. **Input Capture**: Real-time character preview with 30fps refresh
2. **Agent Processing**: Streaming responses with thinking indicators
3. **Tool Execution**: Live progress display with permission prompts
4. **Result Display**: Rich formatted output with syntax highlighting

### Configuration System (`src/loki_code/config/`)
- **Layered config**: defaults.yaml → env vars → CLI args
- **Pydantic validation**: Type-safe configuration with validation
- **Provider configs**: Model-specific settings and connection parameters

### Development Patterns

#### Testing Strategy
- **Unit tests**: Individual component testing (`tests/test_*.py`)
- **Integration tests**: End-to-end workflows (`tests/test_*_integration.py`)
- **Agent tests**: Conversation scenarios (`tests/fixtures/agent_test_scenarios.py`)

#### Code Organization
- **Feature modules**: Each major feature in its own package
- **Shared utilities**: Common code in `utils/` with logging, validation
- **Type safety**: Comprehensive type hints with mypy strict mode

#### Error Handling
- **Graceful degradation**: Fallback mechanisms for provider/tool failures
- **User guidance**: Clear error messages with suggested actions
- **Recovery patterns**: Automatic retry with exponential backoff

### Key Files for Extension

#### Adding New Tools
1. **Inherit from `ToolBase`** (`src/loki_code/tools/base.py`)
2. **Register with discovery** (`src/loki_code/tools/registry.py`)
3. **Add to test scenarios** (`tests/fixtures/agent_test_scenarios.py`)

#### Adding New Providers
1. **Implement `LLMProvider`** (`src/loki_code/core/providers/base.py`)
2. **Register in factory** (`src/loki_code/core/providers/factory.py`)
3. **Add provider config** (`src/loki_code/config/models.py`)

#### Extending UI Components
1. **Rich components** (`src/loki_code/ui/console/components.py`)
2. **Layout management** (`src/loki_code/ui/console/layouts.py`)
3. **Theme system** (`src/loki_code/ui/console/themes.py`)

### HTTP Mode Architecture (New)

**Loki Code** now supports two operational modes:

#### 1. **Direct Mode** (Traditional)
```
┌─────────────────┬──────────────────┬─────────────────┐
│   UI Layer      │   Core Layer     │   Tool Layer    │
├─────────────────┼──────────────────┼─────────────────┤
│ • TUI Interface │ • Agent Service  │ • Tool Registry │
│ • Chat Mode     │ • LLM Providers  │ • File Tools    │
│ • Commands      │ • Model Loading  │ • Code Analysis │
│                 │ • Direct LLM     │ • MCP Support   │
└─────────────────┴──────────────────┴─────────────────┘
```

#### 2. **HTTP Mode** (New Service-Oriented)
```
┌─────────────────┬──────────────────┬─────────────────┐
│ Loki Code App   │   HTTP Client    │ LLM Server      │
├─────────────────┼──────────────────┼─────────────────┤
│ • TUI Interface │ • HTTP Agent     │ • FastAPI       │
│ • Chat Mode     │ • Request/Reply  │ • Model Manager │
│ • Commands      │ • Error Handling │ • Transformers  │
│ • Tool System   │ • Streaming      │ • Ollama/OpenAI │
└─────────────────┴──────────────────┴─────────────────┘
```

**HTTP Mode Benefits**:
- ✅ **Fast startup**: No model loading in main process
- ✅ **No blocking**: Non-blocking TUI and chat interfaces
- ✅ **Scalability**: Server can handle multiple clients
- ✅ **Resource isolation**: LLM memory usage isolated to server
- ✅ **Independent deployment**: Server and client can be on different machines
- ✅ **Better debugging**: Separate logs and processes

**Configuration**:
```yaml
# configs/http-mode.yaml
llm:
  use_llm_server: true
  llm_server_url: "http://localhost:8765"
  llm_server_timeout: 30.0
  llm_server_retries: 3
```

### Current Development Phase
**Phase 5 Complete**: Interactive TUI with real-time input, flicker-free display, and full agent integration
**Phase 6 Complete**: HTTP-based LLM server architecture with independent deployment (now default)
**Phase 7 Complete**: TUI decoupling with streaming support and clean architecture

**Next Phases**:
- **Phase 8**: Enhanced TUI (panels, file browser, dependency analysis)
- **Phase 9**: Advanced planning system with multi-step task decomposition
- **Phase 10**: Additional UI implementations (web, mobile, API-only)
- **Phase 11**: Distributed deployment and scaling features

### Important Notes
- **HTTP-first architecture**: Default mode uses independent LLM server for better scalability
- **Decoupled UI architecture**: TUI properly separated from core with streaming support
- **Fallback to direct mode**: Use `--config configs/direct-mode.yaml` for direct LLM integration
- **Enterprise ready**: Security boundaries and permission systems built-in
- **MCP evolution**: Current tools are MCP-compatible, native MCP in Phase 8
- **Real-time performance**: TUI optimized for 30fps with intelligent refresh throttling and streaming