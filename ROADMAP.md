# Loki Code — Roadmap

High-level feature roadmap. Items are grouped by theme and roughly ordered by priority within each group. Details get fleshed out in `.claude/plans/` before implementation starts.

---

## 🔥 Near-term

### Session Persistence & Recall
Save each session to disk (timestamp, project path, compacted summary, extracted key facts). A `recall` tool lets the agent search previous sessions — "what did we decide about the auth flow last week?" Memory architecture: **dual-store** — vector DB for semantic similarity search + knowledge graph for relational facts (architectural decisions, symbol dependencies, project conventions). Pure embedding search misses relational structure; pure graph misses semantic similarity. Keyword search as a zero-dep fallback.

### Git Integration
The agent currently has no awareness of git. Add read tools: `git_diff`, `git_log`, `git_status`, `git_blame`. Eventually `git_commit` with the same confirmation flow as file writes. The agent should auto-detect what changed since the last session and use that as implicit context.

### Sub-Agent System
Main REPL spawns child agents with scoped context and a specific goal, waits for a result, incorporates it. Child agents get restricted tool sets (e.g. read-only for research tasks, a dedicated test-runner agent). Key design question: context isolation — children receive only the relevant framing, not the full parent history. Enables parallelism and specialisation.

**Wire protocol: ACP (Agent Communication Protocol).** ACP (IBM/BeeAI) defines a REST/streaming interface for agent-to-agent calls — agents expose themselves as HTTP endpoints and an orchestrator calls them like services. Using ACP as the sub-agent wire protocol means child agents can be external processes or remote services, not just internal goroutines, and makes Loki composable with other ACP-compatible agents.

**Parallel tool dispatch.** Top SWE-bench agents issue up to 8 tool calls in parallel per turn — this alone adds 2+ benchmark points regardless of model. Loki's tool executor is currently single-threaded. Independent reads (`read_file`, `grep_content`, `list_files`) should fire concurrently within a single turn.

---

## 🟡 Medium-term

### Hierarchical Context Management
Current compaction is blunt (AI summarises at 75%). Context overflow is the dominant failure mode even for the strongest models — bigger context windows don't solve it, smarter assembly does. Planned improvements:
- **Importance scoring** — messages that were acted on are weighted higher than dead-ends
- **Pinned facts** — explicit items that survive compaction (`/pin`)
- **Hierarchical memory** — working memory (current task) + episodic (this session) + project facts (persistent); inspired by the OS RAM/disk model (Letta/MemGPT)
- **Selective / dynamic context building** — *(needs more design)* dynamically assemble the context window from the most relevant past turns and pinned facts per request, rather than a linear FIFO queue. Key insight from context engineering research: most relevant content belongs at the *beginning and end* of the window, never the middle ("lost in the middle" problem). Prereq for effective sub-agents.

### Speculative Tool Execution
Pre-execute the most likely next tool call *while the LLM is still generating the current response*, rather than waiting. The [PASTE framework](https://arxiv.org/abs/2510.04371) demonstrated 48.5% faster task completion and 1.8× tool throughput on multi-step tasks. Even a conservative version — pre-reading a file the model is about to request — reduces latency noticeably.

### Reflection & Self-Improvement Loop
Two connected ideas:
- **Reflection** — at session end (or on `/reflect`), agent distills raw session history into durable lessons stored in long-term memory: "the project uses X pattern", "previous attempt at Y failed because of Z". Produces curated knowledge, not just searchable history.
- **Self-improvement** — agent evaluates its own tool strategies and output quality over time, updating its internal heuristics and system prompt fragments. Connects per-session reflection with durable behavioural change across sessions. Could surface as a `/improve` command or run automatically after N sessions.

### Code Graph / Symbol Dependency Layer
Top SWE-bench agents (e.g. Augment Code) use **hybrid BM25 + embedding + code graph** for codebase retrieval — the graph tracks which symbol uses which across files. Qualitatively different from grep: understands that `Foo` in file A *calls* `Bar` in file B without reading every file. Would significantly improve `grep_content`, `find_files`, and `analyze_code`. Builds naturally on top of the session memory dual-store.

### Test Loop
`/test` mode: agent runs tests, reads failures, proposes and applies fixes, re-runs — without manual intervention. Currently requires the user to paste failure output. The loop is mechanical enough to automate; exit conditions are all-green or a configurable retry limit.

### Configurable Shell Tools
Let users define custom tools in `.loki-code/tools.yaml`: tool name, description, and a shell command template. The existing registry pattern is a natural extension point. Instant extensibility without writing Go.

### Exec Sandboxing
The current security model for `exec_command` is an allowlist of safe commands. This prevents the most obvious abuse but does not stop a whitelisted command from accessing files outside the project (e.g. `find / -name "*.pem"` is allowed and leaks credential paths).

The right solution is kernel-level sandboxing that wraps the **entire agent process** — not just `exec_command`. This makes all tools (file reads, file writes, exec) subject to the same policy automatically, with no changes to the tool implementations.

Preferred approach per platform:

- **macOS**: `sandbox-exec` with a Seatbelt profile — syscall interception via the XNU MAC framework, near-zero overhead, same filesystem and toolchain as the host. Technically deprecated by Apple (no documented replacement for CLI tools) but used in production by Anthropic for Claude Code.
- **Linux**: Landlock (Linux 5.13+) for filesystem access control + `seccomp` for syscall filtering — same "one policy, all tools" coherence as Seatbelt, no VM overhead.

Both are enabled via a `--sandbox` flag; the agent falls back gracefully if the platform does not support it.

**Why not Docker?**

| Approach | Coherent isolation | Complexity | Portability |
|---|---|---|---|
| Allowlist only (current) | No — exec limited, files open | Low | ✅ All platforms |
| Docker (exec only) | No — split env | Medium | ✅ All platforms |
| Docker (all tools proxied) | Yes | High | ✅ All platforms |
| macOS sandbox profiles | Yes | Medium (DSL learning curve) | ❌ macOS only |
| Landlock + seccomp | Yes | High | ❌ Linux only |

Docker on macOS adds a Linux VM layer, so `exec_command` runs in a Linux container while `read_file`/`write_file` remain on the host — a split execution environment with mismatched filesystem views and toolchain. Coherent Docker isolation requires proxying all tool calls through the container, which is a significant refactor for weaker per-call performance than native syscall filtering. Kernel-level sandboxing is strictly better for a local single-user tool.

### Observability
The current foundation (`--debug` flag, structured `slog` output) is good for local development but gives no visibility into what the agent actually did across a session — which tool calls fired, how long each took, how much context was consumed, and what it cost.

Three layers worth adding:

**Span-level tracing.** Each agent turn is a tree of spans: one root span for the model call, child spans for each tool invocation, with timing and input/output recorded at every node. [OpenTelemetry](https://opentelemetry.io) is the standard wire format; traces can be exported to any compatible backend (Jaeger, Honeycomb, Grafana Tempo) or to a local file for offline review. The existing `slog` calls become the span events.

**Cost tracking.** Token counts are already captured per turn (`lastKnownTokens`). Pairing them with a configurable price-per-token table gives per-session cost totals surfaced in `/stats` and written to the session log. Useful when running against paid APIs (OpenAI, Anthropic).

**Session log.** A structured JSONL file written alongside each session (e.g. `~/.loki-code/sessions/YYYY-MM-DD-HH-MM.jsonl`) capturing every turn: timestamp, model, prompt tokens, tool calls with inputs/outputs, latency, cost. Queryable after the fact without a backend. Natural input for the Reflection & Self-Improvement loop.

Platforms worth evaluating as optional exporters: [Langfuse](https://langfuse.com) (open-source, self-hostable), [Weights & Biases Weave](https://wandb.ai/site/weave). Both speak OpenTelemetry so they come for free once tracing is wired up.

### MCP Server Mode
Expose Loki Code's individual tools as an MCP server so they can be used by Claude Code or other MCP-compatible clients. Relatively low effort given the existing tool registry; high interoperability value. Distinct from ACP: MCP exposes *tools*, ACP exposes *Loki as a whole agent*.

### Multi-Model Routing
A `--fast-model` flag selects a cheap/fast model for inner-loop work (tool calls, lookups, simple edits) while the main model handles reasoning and synthesis. Useful when running against paid APIs. Natural fit with speculative tool execution (fast model for speculative pre-fetch, main model for verification).

### Conversation Branching
`/branch` forks the current context so you can explore a risky approach. `/merge` brings the result back; `/abandon` discards it. Useful for speculative refactors without polluting the main thread.

### Watch Mode
`loki-code --watch` monitors a set of files and re-runs a standing prompt whenever they change. Natural companion to the test loop — "keep the tests green as I edit."

---

## Deferred / Needs Design

- **Selective / dynamic context building** — see Hierarchical Context Management above
- **Dual-store session memory** — vector DB + knowledge graph; design the schema and query interface before picking storage backends
- **Multi-agent coordination** — beyond parent/child: peer agents that share a scratchpad and divide work. Needs the sub-agent + ACP foundation first.
- **Policy-learned context management** — RL-trained context assembly (frontier research); only worth considering after the heuristic hierarchical approach is proven
