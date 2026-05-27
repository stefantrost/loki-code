# Loki Code - AI Coding Agent

A Go-based AI coding agent that connects to LM Studio, OpenAI, Ollama, and any OpenAI-compatible API for interactive chat with function calling capabilities.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)
- [Context Management](#context-management)
- [Plan Mode](#plan-mode)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Testing & Quality](#testing--quality)
- [License](#license)

## Prerequisites

- Go 1.26.3+ installed
- [LM Studio](https://lmstudio.ai) running locally (or any OpenAI-compatible endpoint / Ollama)

## Setup

### LM Studio (recommended)

1. Download and start [LM Studio](https://lmstudio.ai), load a model, and enable the local server (default: `http://localhost:1234`).

2. Create a config file:
   ```bash
   cp llm.env.example llm.env
   # edit llm.env — set API_TYPE, BASE_URL, MODEL_NAME
   ```
   Or generate a fresh example:
   ```bash
   ./loki-code --create-config
   ```

### Ollama

1. Start Ollama and pull a model:
   ```bash
   ollama serve
   ollama pull qwen3:32b
   ```

## Usage

1. Build the application:
   ```bash
   go build -o loki-code .
   # or
   make build
   ```

2. Run the application:
   ```bash
   ./loki-code                                                        # reads llm.env / .env
   ./loki-code --model qwen/qwen3.6-35b-a3b --url http://localhost:1234/v1   # LM Studio
   ./loki-code --model qwen3:32b --url http://localhost:11434         # Ollama
   ```

3. Start chatting with the AI. Type your questions and see streaming responses.

4. Exit by typing `exit`, `quit`, or pressing Ctrl+C.

### CLI Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | `qwen3:32b` | Model name (overridden by config file in practice) |
| `--url` | | `http://localhost:11434` | API base URL (overridden by config file in practice) |
| `--api-type` | | *(auto-detected)* | `ollama`, `openai`, or `openai-compatible` |
| `--token` | | | Bearer token for OpenAI / compatible APIs |
| `--config` | | *(searches default paths)* | Path to config file |
| `--debug` | | | Enable debug logging |
| `--concise` | `-c` | | Start in concise response mode |
| `--list-models` | | | List available models (Ollama only) and exit |
| `--create-config` | | | Write an example `llm.env` and exit |

## Features

- Real-time streaming responses from the AI model
- **Multi-API support**: LM Studio, OpenAI, Ollama, and any OpenAI-compatible API
- **Function calling with tools**: create, read, update, delete, find, grep, exec, HTTP requests, static analysis
- **Intelligent context management** with automatic AI-powered compaction
- **System prompt engineering** for coding assistant behaviour
- **Plan mode** for read-only analysis before making changes
- **Concise/verbose response modes**
- **Task tracking** with auto-detection
- **Project-aware tooling**: auto-detects Go/Python/Node.js and surfaces relevant static analysers
- Simple REPL interface with special commands
- Graceful shutdown and interrupt handling
- **Security**: command whitelisting, path traversal prevention, internal network blocking
- **Zero external dependencies** — pure Go standard library

### Example Prompts

```
> Create a Python script that prints "Hello World"
> Read the contents of main.go
> List all files in the current directory
> Update README.md to add a new section
```

### Special Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation context |
| `/compact` | Manually compress context via AI summarisation |
| `/stats` | Show token usage, message count, and current mode |
| `/mode` | Show current active mode |
| `/plan` | Enter plan mode (read-only tools only) |
| `/execute` | Exit plan mode (re-enable all tools) |
| `/concise` | Switch to concise responses |
| `/verbose` | Switch to verbose responses |
| `/task <description>` | Set an active task |
| `/task` | Show the current active task |
| `/complete` | Mark the current task as complete |
| `exit` / `quit` | Exit the application |

## Context Management

- **Automatic compaction**: History accumulates freely; AI-powered summarisation fires automatically when token usage reaches 75% of the model's context window
- **Real token counts**: Exact token usage is read from each API response (`include_usage` for OpenAI-compatible/LM Studio; `prompt_eval_count` for Ollama), with a character-ratio heuristic as a cold-start fallback
- **Tool call preservation**: Compaction preserves tool-call sequences to maintain conversation integrity
- **Real-time statistics**: `/stats` shows exact token usage, message count, max tokens, and current mode
- **Dynamic context detection**: Automatically queries the model's actual context window at startup

### Dynamic Context Detection

The application queries the model's context window at startup and sets the compaction threshold accordingly:

```
Loki Code - AI Coding Agent
Connecting to LM Studio (qwen/qwen3.6-35b-a3b)...
✓ Detected context window: 32,768 tokens
✓ Set context limit: 24,576 tokens (75% utilisation)
```

**Benefits:**
- **Model agnostic**: Works with any LM Studio, Ollama, or OpenAI-compatible model
- **Optimal utilisation**: Uses 75% of available context (25% reserved for responses)
- **Safe fallback**: Uses 4,000 tokens if detection fails
- **Future proof**: Adapts to new models without code changes

## Plan Mode

Plan Mode enables safe analysis and planning without executing any changes. Perfect for exploring codebases and creating detailed execution plans before committing to edits.

### Allowed tools in plan mode
`read_file`, `list_files`, `find_files`, `grep_content`, `get_pwd`, `tree_view`, `analyze_code`, `http_request`

Mutating tools (`create_file`, `update_file`, `delete_file`, `exec_command`) are blocked and return a clear warning.

### Usage Example
```bash
> /plan
🎯 Plan Mode Activated!

[PLAN] > Refactor the authentication system to use JWT tokens

📋 EXECUTION PLAN

## Overview
Refactor current session-based authentication to JWT token system

## Steps
1. **Create JWT Utils**
   - File: utils/jwt.go
   - Functions: GenerateToken, ValidateToken

2. **Update Auth Handlers**
   - File: handlers/auth.go
   - Modify login to return JWT instead of session

[PLAN] > /execute
⚡ Execute Mode Activated!

> # Now execute the plan steps
```

## Configuration

Configuration is loaded from multiple sources (highest priority first):

1. Command-line flags
2. Environment variables (`LOKI_API_TYPE`, `LOKI_BASE_URL`, `LOKI_MODEL`, `LOKI_BEARER_TOKEN`, `LOKI_DEBUG`)
3. Config file — searched in order: `llm.env`, `.env`, `~/.loki-code/config.env`

### Example Configuration (`llm.env`)

```env
# LM Studio (local server — no token required)
API_TYPE=openai-compatible
BASE_URL=http://localhost:1234/v1
MODEL_NAME=qwen/qwen3.6-35b-a3b

# OpenAI
# API_TYPE=openai
# BASE_URL=https://api.openai.com/v1
# MODEL_NAME=gpt-4o
# BEARER_TOKEN=your_openai_api_key

# Ollama
# API_TYPE=ollama
# BASE_URL=http://localhost:11434
# MODEL_NAME=qwen3:32b
```

## Architecture

Two-package layout with a clean one-way dependency: `main` (root) and `clients/` know nothing about `internal/`; `internal/` is injected via interfaces and callbacks.

```
main.go                              — REPL loop, flag parsing, signal handling
clients/                             — LLM transport (protocol layer)
  interface.go                       — shared types and LLMClient interface
  factory.go                         — CreateClient(), API-type auto-detection
  base.go                            — shared client helpers
  ollama.go                          — Ollama streaming + context-window detection
  openai.go                          — OpenAI-compatible SSE streaming + token counting
  streaming.go                       — tool-call delta merging
internal/config/config.go            — multi-source config loading and validation
internal/session/context_manager.go  — message history, token accounting, compaction, modes
internal/session/task_tracker.go     — active task and task history
internal/tools/                      — tool schemas, dispatch, and all tool implementations
internal/ui/ui.go                    — coloured diff display and confirmation prompts
```

**Design principles:**
- **Zero external dependencies** — every import is from the Go standard library; enforced by an architecture test
- **Interface-driven integration** — `ToolExecutor`, `ToolSchemaProvider`, and `CompactFunc` callbacks are injected at client construction; `clients/` never imports concrete tool or session code
- **Security by default** — path traversal validation, command whitelisting, and internal-network blocking are the only guards between the LLM and the host

## Testing & Quality

Three quality tiers, increasing in thoroughness:

```bash
# Fast pre-commit check (vet + build + test, ~seconds)
make quick

# Full CI gate (quick + golangci-lint + govulncheck + coupling snapshot)
make check

# Audit (check + per-file statement-coverage floor enforcement)
make audit

# Rewrite coverage floors after intentional coverage changes
make cover-update

# Run tests directly
go test ./...
go test -run TestName ./...    # single test (package selector required)
go test ./clients/ -v
```

**Linters enabled** (`.golangci.yml` v2): `staticcheck`, `errcheck`, `govet`, `ineffassign`, `unused`, `gosec`, `gocyclo`, `gocognit`, `misspell`.

**Coverage floors** (`coverage-floors.txt`): per-file statement-coverage baselines anchored at current numbers with 2 pp slack; enforced by `scripts/coverage-floor.sh`.

**Architecture tests** (`architecture_test.go`): structural fitness checks — dependency direction, file-size budget (800 LOC max), registry completeness, no duplicate mode state, no terminal coupling outside `ui.go`/`main.go`, no tool-name literals in `clients/`, no third-party imports without `// allow:` marker, struct cohesion.

### Ollama lifecycle targets

```bash
make start-ollama   # Start Ollama in background
make stop-ollama    # Stop Ollama
make check-ollama   # Verify Ollama is reachable
make pull-model     # Pull the default model
make status         # Show system status
```

## License

MIT License

For details, see [LICENSE.md](LICENSE.md)
