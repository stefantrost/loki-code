package clients

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
)

// baseClient holds state and behavior that every streaming LLM client shares.
// Provider-specific clients embed it and supply only the protocol details
// (request building, stream parsing, context-window detection).
type baseClient struct {
	modelName          string
	toolExecutor       ToolExecutor
	toolSchemaProvider ToolSchemaProvider
	truncator          TruncationPolicy
	debug              bool
	responseActive     atomic.Bool
	interruptChan      chan struct{}
	contextManager     ContextManager
	httpClient         *http.Client
	outputWriter       io.Writer // nil → io.Discard; content tokens
	systemWriter       io.Writer // nil → io.Discard; tool-status blocks
	onStreamStart      func()    // nil → no-op; called at top of StreamChatWithHistory
}

// streamer is the protocol-specific surface that handleToolCalls needs back
// from the concrete client to issue a follow-up request after executing tools.
type streamer interface {
	StreamChatWithHistory(messages []ChatMessage) error
	planMode() bool
}

func (c *baseClient) debugLog(format string, args ...interface{}) {
	if c.debug {
		slog.Debug(fmt.Sprintf(format, args...))
	}
}

func (c *baseClient) ClearContext() {
	c.contextManager.Clear()
}

func (c *baseClient) GetStats() (int, int, int) {
	return c.contextManager.GetStats()
}

func (c *baseClient) CanCompact() bool {
	return c.contextManager.CanCompact()
}

func (c *baseClient) EnablePlanMode()  { c.contextManager.SetPlanMode(true) }
func (c *baseClient) DisablePlanMode() { c.contextManager.SetPlanMode(false) }
func (c *baseClient) IsInPlanMode() bool {
	return c.contextManager.IsInPlanMode()
}

func (c *baseClient) EnableConciseMode()  { c.contextManager.SetConciseMode(true) }
func (c *baseClient) DisableConciseMode() { c.contextManager.SetConciseMode(false) }
func (c *baseClient) IsInConciseMode() bool {
	return c.contextManager.IsInConciseMode()
}

func (c *baseClient) SetActiveTask(task string) {
	c.contextManager.SetActiveTask(task)
}

func (c *baseClient) GetActiveTask() string {
	if task := c.contextManager.GetActiveTask(); task != nil {
		return task.Goal
	}
	return ""
}

func (c *baseClient) CompleteCurrentTask() {
	c.contextManager.CompleteCurrentTask("")
}

func (c *baseClient) Interrupt() {
	select {
	case c.interruptChan <- struct{}{}:
	default:
	}
}

func (c *baseClient) IsResponseActive() bool {
	return c.responseActive.Load()
}

func (c *baseClient) SetDebug(enabled bool) {
	c.debug = enabled
}

// SetTruncator registers a tool-aware truncation policy. Pass nil to fall
// back to the default byte-truncation.
func (c *baseClient) SetTruncator(p TruncationPolicy) {
	c.truncator = p
}

// SetOutputWriter redirects streamed content tokens to w. Pass nil to restore
// the default (io.Discard). The agent sets this before each StreamChat.
func (c *baseClient) SetOutputWriter(w io.Writer) { c.outputWriter = w }

// getOutputWriter returns the active content writer, falling back to io.Discard.
// The agent always sets a writer before calling StreamChat; Discard is only
// reached in tests or edge cases.
func (c *baseClient) getOutputWriter() io.Writer {
	if c.outputWriter != nil {
		return c.outputWriter
	}
	return io.Discard
}

// SetSystemWriter redirects tool-status output to w. Pass nil to restore the
// default (io.Discard). The agent routes this to view.WriteSystem so tool
// blocks never mix with the glamour-rendered response region.
func (c *baseClient) SetSystemWriter(w io.Writer) { c.systemWriter = w }

// getSystemWriter returns the active system writer, falling back to io.Discard.
func (c *baseClient) getSystemWriter() io.Writer {
	if c.systemWriter != nil {
		return c.systemWriter
	}
	return io.Discard
}

// SetStreamStartCallback registers fn to be called at the top of each
// StreamChatWithHistory invocation. Pass nil to clear it.
func (c *baseClient) SetStreamStartCallback(fn func()) { c.onStreamStart = fn }

// notifyStreamStart calls onStreamStart if one is registered. Called at the
// top of each provider's StreamChatWithHistory, before any tokens are written.
func (c *baseClient) notifyStreamStart() {
	if c.onStreamStart != nil {
		c.onStreamStart()
	}
}

// toolBoxWidth is the fixed inner width of the tool-call box drawn in the chat
// pane. Wide enough to fit long file paths on an 80-col terminal with sidebar.
const toolBoxWidth = 64

// handleToolCalls executes every tool call in the assistant message, appends
// the results to context, and issues a follow-up streaming request via the
// concrete client.
func (c *baseClient) handleToolCalls(self streamer, assistantMessage ChatMessage) error {
	planMode := self.planMode()
	slog.Debug("handleToolCalls started",
		"assistant_message_content", truncateString(assistantMessage.Content, 500),
		"tool_calls_count", len(assistantMessage.ToolCalls), "plan_mode", planMode)

	sw := c.getSystemWriter()

	for i, toolCall := range assistantMessage.ToolCalls {
		name := toolCall.Function.Name
		slog.Debug("Processing tool call", "index", i, "tool_id", toolCall.ID,
			"tool_name", name, "tool_arguments", toolCall.Function.Arguments)

		// ── Top border ──────────────────────────────────────────────────
		//   ┌─ 🔧 tool_name ──────────────────────────────────────────┐
		label := fmt.Sprintf("─ 🔧 %s ", name)
		labelRunes := 5 + len([]rune(name)) // "─ 🔧 <name> "
		fill := toolBoxWidth - labelRunes - 1
		if fill < 1 {
			fill = 1
		}
		fmt.Fprintf(sw, "\n┌%s%s┐\n", label, strings.Repeat("─", fill))

		// ── Argument lines ────────────────────────────────────────────
		//   │ key: value                                               │
		for k, v := range toolCall.Function.Arguments {
			line := fmt.Sprintf("%s: %v", k, v)
			runes := []rune(line)
			if len(runes) > 400 {
				runes = append(runes[:399], '…')
				line = string(runes)
			}
			pad := toolBoxWidth - len([]rune(line))
			if pad < 0 {
				pad = 0
			}
			fmt.Fprintf(sw, "│ %s%s │\n", line, strings.Repeat(" ", pad))
		}

		// ── Bottom border ─────────────────────────────────────────────
		fmt.Fprintf(sw, "└%s┘\n", strings.Repeat("─", toolBoxWidth+2))

		// ── Execute ───────────────────────────────────────────────────
		result, err := c.toolExecutor(toolCall, planMode)
		if err != nil {
			slog.Error("Tool execution returned error", "tool_name", name, "error", err)
			result = fmt.Sprintf("Error: %v", err)
		}
		result = c.truncateToolResult(result, name)
		c.debugLog("Tool %s result length: %d characters", name, len(result))

		fmt.Fprintf(sw, "✓ %s  (%d chars)\n", name, len(result))

		c.contextManager.AddMessage(ChatMessage{
			Role:       "tool",
			Content:    result,
			ToolCallID: toolCall.ID,
		})
	}

	messages := c.contextManager.GetMessages()
	slog.Debug("Retrieved messages for follow-up request", "message_count", len(messages))
	return self.StreamChatWithHistory(messages)
}

// MaxToolResultChars caps tool output before it is fed back to the model.
// Exported so the application layer can shape its own truncation policy
// (line-aware vs byte-aware) on top of the same bound.
const MaxToolResultChars = 5000

// TruncationPolicy lets the application supply tool-name-aware truncation.
// The default in the clients package is byte-truncation; the loki-code
// application registers a smarter line-aware variant for list-style tools.
type TruncationPolicy func(result, toolName string) string

// defaultTruncate is the byte-only fallback. It knows nothing about specific
// tool names; that's a deliberate architectural rule (see
// TestArchitecture_NoToolNamesInClients).
func defaultTruncate(result, toolName string) string {
	if len(result) <= MaxToolResultChars {
		return result
	}
	return result[:MaxToolResultChars-50] + fmt.Sprintf("... (truncated at %d characters)", MaxToolResultChars-50)
}

// truncateToolResult applies the active TruncationPolicy. baseClient holds
// the policy as a field so per-client overrides are possible.
func (c *baseClient) truncateToolResult(result, toolName string) string {
	if c.truncator != nil {
		return c.truncator(result, toolName)
	}
	return defaultTruncate(result, toolName)
}

func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
