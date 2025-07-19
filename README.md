# Loki Code - AI Coding Agent

A simple Go-based coding agent that connects to local Ollama models for interactive chat with function calling capabilities.

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

## Available Tools

The AI can execute the following file operations:

- **create_file**: Create new files with specified content
- **read_file**: Read the contents of existing files
- **update_file**: Update existing files with new content
- **delete_file**: Delete files
- **list_files**: List files in directories

## Example Usage

Ask the AI to perform file operations:

```
> Create a Python script that prints "Hello World"
> Read the contents of main.go
> List all files in the current directory
> Update README.md to add a new section
```

## Special Commands

- `/clear` - Clear conversation context
- `/compact` - Compress context using AI summarization
- `/stats` - Show context statistics (tokens used, message count)
- `exit` or `quit` - Exit the application

## Context Management

- **Automatic Management**: Stays within token limits automatically
- **Smart Trimming**: Preserves system prompt and recent conversation history
- **Tool Call Preservation**: Maintains tool call sequences during trimming
- **Context Compacting**: AI-powered summarization to compress long conversations
- **Real-time Statistics**: Shows token usage and message count

### Context Compacting

When conversations get long, use `/compact` to:
- Summarize older messages using the AI
- Preserve recent conversation and tool calls
- Reduce token usage by 60-80% typically
- Maintain conversation continuity

Example:
```
> /compact
Compacting conversation context...
🔄 Generating conversation summary...
✓ Context compacted: 1200 → 450 tokens (saved 750 tokens)
```

## Architecture

- `main.go`: CLI interface and user interaction loop
- `orchestrator.go`: Core Ollama API client with streaming and function calling support
- `tools.go`: File manipulation tools and execution logic
- `context_manager.go`: Intelligent conversation context and token management