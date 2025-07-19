# Loki Code - AI Coding Agent

A simple Go-based coding agent that connects to local Ollama models for interactive chat with function calling capabilities.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)
- [Context Management](#context-management)
- [Architecture](#architecture)
- [License](#license)

## Prerequisites

- Go 1.19+ installed
- Ollama running locally on port 11434
- qwen3:32b model pulled in Ollama

## Setup

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Pull the qwen3:32b model (if not already available):
   ```bash
   ollama pull qwen3:32b
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
- **Function calling with file manipulation tools**
- **Intelligent context management with token limits**
- **System prompt engineering for coding assistant behavior**
- Simple CLI interface with special commands
- Graceful shutdown handling
- Error handling for network and API issues

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
- **Model Agnostic**: Works with any Ollama model automatically
- **Optimal Utilization**: Uses 75% of available context (25% reserved for responses)
- **Safe Fallback**: Uses 4,000 tokens if detection fails
- **Future Proof**: Adapts to new models without code changes

## Plan Mode

Plan Mode enables safe analysis and planning without executing changes. Perfect for exploring codebases and creating detailed execution plans.

### Features
- **Read-Only Operations**: Only `read_file` and `list_files` are allowed
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

## Architecture

- `main.go`: CLI interface and user interaction loop
- `orchestrator.go`: Core Ollama API client with streaming and function calling support
- `tools.go`: File manipulation tools and execution logic
- `context_manager.go`: Intelligent conversation context and token management

## License

MIT License

For details, see [LICENSE.md](LICENSE.md)
