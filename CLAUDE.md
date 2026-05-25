# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Next architecture re-review: 2026-08-25

Re-run the evaluation captured in `~/.claude/plans/evaluate-the-architecture-and-elegant-lantern.md` quarterly. Diff against the previous run, convert findings into issues. Update this date when the review completes.

## Project Overview

Loki Code is a Go CLI AI coding agent. It connects to Ollama, OpenAI, or any OpenAI-compatible API, exposes file/exec/HTTP/analysis tools via function calling, and streams responses to a REPL. Go 1.24+, single module `loki-code` (see `go.mod`).

## Common Commands

```bash
# Build / run
go build -o loki-code .
./loki-code                                  # uses llm.env / .env / ~/.loki-code/config.env
./loki-code --model qwen3:32b --url http://localhost:11434
./loki-code --create-config                  # write example llm.env

# Make targets (also manage Ollama lifecycle)
make build | make run | make dev             # dev = setup + run
make start-ollama | make stop-ollama | make check-ollama | make pull-model

# Tests
go test ./...
go test -run TestName ./...                  # single test (package selector required)
go test ./clients/ -v
```

Config precedence (highest first): CLI flags → env (`LOKI_API_TYPE`, `LOKI_BASE_URL`, `LOKI_MODEL`, `LOKI_BEARER_TOKEN`, `LOKI_DEBUG`) → config file. See `llm.env.example`.

## Architecture

Two-package layout: `main` (root) holds CLI/tooling/context; `clients/` holds LLM transport.

**Dependency direction is one-way:** `main` depends on `clients`; `clients` knows nothing about the concrete tool implementations or context manager. Integration happens via interfaces and callbacks defined in `clients/interface.go`:

- `LLMClient` — surface the REPL talks to (`StreamChat`, mode flags, stats, `DetectContextWindow`, `Interrupt`).
- `ContextManager` — conversation state, token accounting, plan/concise mode, active task.
- `ToolExecutor` / `ToolSchemaProvider` / `CompactFunc` — callbacks injected at client construction.

`main.go` wires it together: builds `ContextManager` (root pkg) → passes it plus `ExecuteToolWithPlanMode` and `GetAvailableTools` (from `tools.go`) into `clients.CreateClient`. The factory picks `OllamaClient` or `OpenAIClient` based on `APIType`. After construction, `client.DetectContextWindow()` is called and the context manager is sized to 75% of the model's window (4000-token fallback).

### Key files
- `main.go` — flag parsing, signal handling, REPL loop, slash-command dispatch.
- `config.go` — multi-source config loading, validation, example file generation.
- `context_manager.go` — message history, token estimation, smart trimming that preserves tool-call sequences, AI compaction, plan/concise/task state, system-prompt assembly.
- `tools.go` — tool schema list, dispatch (`ExecuteToolWithPlanMode`), plan-mode allow-list (`isToolAllowedInPlanMode`), per-tool implementations, project detection (Go/Python/Node) that conditionally exposes an `analyze_code` tool with the right analyzer.
- `ui.go` — colored diff / confirmation prompts used by mutating tools.
- `clients/factory.go` — `CreateClient`, `DetectAPIType` (URL heuristic), `ValidateConfig`.
- `clients/ollama.go`, `clients/openai.go` — streaming chat, tool-call assembly, context-window detection per provider.

### Modes & state
Plan mode and concise mode live in the context manager and are surfaced through `LLMClient`. Plan mode is enforced inside `ExecuteToolWithPlanMode` — only read tools (`read_file`, `list_files`, `find_files`, `grep_content`, `get_pwd`, `tree_view`, `analyze_code`, `http_request`) run; mutating calls are rejected before reaching their executors. The system prompt is reassembled on each request to reflect current mode and any active `UserTask`.

### Security model (in `tools.go`)
- `exec_command` is whitelisted (find/grep/pwd/tree/wc/sort/uniq/whoami/date/which).
- `validatePath` blocks path traversal for all file ops.
- `http_request` blocks internal/private networks.
When changing tool behavior, keep these invariants — they are the only guard between the LLM and the host.

## Adding a tool

1. Append a `clients.Tool` schema in `GetAvailableTools` (`tools.go`).
2. Add a case in `ExecuteToolWithPlanMode` and an `executeX` implementation.
3. If read-only, add it to `isToolAllowedInPlanMode`.
4. For mutating ops, route through `ui.go` for the diff/confirmation flow.

## Adding an LLM provider

Implement `clients.LLMClient` in a new file under `clients/`, then add a case to `CreateClient` (and optionally a URL heuristic to `DetectAPIType`). Do not import the root package — interact through the interfaces and callbacks already on `ClientConfig`/constructor args.
