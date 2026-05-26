# Loki Code - AI Coding Agent

A Go-based AI coding agent that connects to Ollama, OpenAI, and OpenAI-compatible APIs for interactive chat with function calling capabilities.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)
- [Context Management](#context-management)
- [Plan Mode](#plan-mode)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Testing](#testing)
- [License](#license)

## Prerequisites

- Go 1.24+ installed
- Ollama running locally on port 11434 (for Ollama API)
- qwen3:32b model pulled in Ollama (or your preferred model)

## Setup

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Pull the qwen3:32b model (if not already available):
   ```bash
   ollama pull qwen3:32b
   ```

3. Create configuration:
   ```bash
   ./loki-code --create-config
   ```

## Usage

1. Build the application:
   ```bash
   go build -o loki-code .
   ```

2. Run the application:
   ```bash
   ./loki-code
   ```

3. Start chatting with the AI! Type your questions and see streaming responses.

4. Exit by typing `exit`, `quit`, or pressing Ctrl+C.

## Features

- Real-time streaming responses from the AI model
- **Multi-API support**: Ollama, OpenAI, and OpenAI-compatible APIs
- **Function calling with file manipulation tools** (create, read, update, delete, find, grep, exec, HTTP requests, static analysis)
- **Intelligent context management with token limits**
- **System prompt engineering** for coding assistant behavior
- **Plan mode** for read-only analysis
- **Concise/verbose response modes**
- **Task tracking** with auto-detection
- Project-aware tooling (auto-detects Go/Python/Node.js and surfaces relevant static analyzers)
- Simple CLI interface with special commands
- Graceful shutdown handling
- Security: command whitelisting, path traversal prevention, internal network blocking

### Example Usage

Ask the AI to perform file operations:

```
> Create a Python script that prints "Hello World"
> Read the contents of main.go
> List all files in the current directory
> Update README.md to add a new section
```

### Special Commands

- `/clear`: Clear conversation context
- `/compact`: Compress context using AI summarization
- `/plan`: Enter plan mode (read-only, creates execution plans)
- `/execute`: Exit plan mode (enable all tools)
- `/stats`: Show context statistics (tokens used, message count, current mode)
- `/concise`: Switch to concise responses
- `/verbose`: Switch to verbose responses
- `/task <description>`: Set an active task
- `/complete`: Mark current task as complete
- `exit` or `quit`: Exit the application

## Context Management

- **Automatic Management**: Stays within token limits automatically
- **Smart Trimming**: Preserves system prompt and recent conversation history
- **Tool Call Preservation**: Maintains tool call sequences during trimming
- **Context Compacting**: AI-powered summarization to compress long conversations
- **Real-time Statistics**: Shows token usage and message count
- **Dynamic Context Detection**: Automatically detects model context window size

### Dynamic Context Detection

The application automatically detects your model's context window at startup:

```bash
Loki Code - AI Coding Agent
Connecting to Ollama (qwen3:32b)...
✓ Detected context window: 32,768 tokens
✓ Set context limit: 24,576 tokens (75% utilization)
```

**Benefits:**
- **Model Agnostic**: Works with any Ollama or OpenAI-compatible model
- **Optimal Utilization**: Uses 75% of available context (25% reserved for responses)
- **Safe Fallback**: Uses 4,000 tokens if detection fails
- **Future Proof**: Adapts to new models without code changes

## Plan Mode

Plan Mode enables safe analysis and planning without executing changes. Perfect for exploring codebases and creating detailed execution plans.

### Features
- **Read-Only Operations**: `read_file`, `list_files`, `find_files`, `grep_content`, `get_pwd`, `tree_view`, `analyze_code`, `http_request` are allowed
- **Planning Focus**: AI creates structured, multi-step execution plans
- **Safe Exploration**: Analyze code without risk of changes
- **Visual Indicators**: `[PLAN] >` prompt shows current mode

### Usage Example
```bash
> /plan
🎯 Plan Mode Activated!

[PLAN] > Refactor the authentication system to use JWT tokens

📋 EXECUTION PLAN

## Overview
Refactor current session-based authentication to JWT token system

## Steps
1. **Install JWT Library**
   - Add github.com/golang-jwt/jwt to go.mod
   
2. **Create JWT Utils**
   - File: utils/jwt.go
   - Functions: GenerateToken, ValidateToken
   
3. **Update Auth Handlers**
   - File: handlers/auth.go
   - Modify login to return JWT instead of session

[PLAN] > /execute
⚡ Execute Mode Activated!

> # Now execute the plan steps
```

## Configuration

Configuration is loaded from multiple sources (highest priority first):
1. Command line flags
2. Environment variables (`LOKI_API_TYPE`, `LOKI_BASE_URL`, `LOKI_MODEL`, `LOKI_BEARER_TOKEN`, `LOKI_DEBUG`)
3. Config file (`llm.env`, `.env`, or `~/.loki-code/config.env`)

### Example Configuration (`llm.env`)

```env
# For Ollama (default)
API_TYPE=ollama
BASE_URL=http://localhost:11434
MODEL_NAME=qwen3:32b

# For OpenAI
# API_TYPE=openai
# BASE_URL=https://api.openai.com/v1
# MODEL_NAME=gpt-4
# BEARER_TOKEN=your_openai_api_key_here

# For OpenAI-compatible APIs
# API_TYPE=openai-compatible
# BASE_URL=https://your-api.example.com/v1
# MODEL_NAME=your-model
# BEARER_TOKEN=your_api_key
```

## Architecture

- `main.go`: CLI interface and user interaction loop
- `clients/`: Multi-provider client implementations
  - `interface.go`: Shared types and interfaces
  - `factory.go`: Client factory with API type detection
  - `ollama.go`: Ollama API client
  - `openai.go`: OpenAI-compatible API client
- `tools.go`: Tool definitions, execution logic, and project detection
- `context_manager.go`: Conversation context and token management
- `config.go`: Configuration loading from files and environment variables
- `ui.go`: Colored diff display and user confirmation

## Testing

Run tests:

```bash
go test ./...
```

## License

MIT License

For details, see [LICENSE.md](LICENSE.md)
