# Loki Code - AI Coding Agent

A simple Go-based coding agent that connects to local Ollama models for interactive chat.

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
- Simple CLI interface
- Graceful shutdown handling
- Error handling for network and API issues

## Architecture

- `main.go`: CLI interface and user interaction loop
- `orchestrator.go`: Core Ollama API client with streaming support