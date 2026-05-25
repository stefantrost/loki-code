package clients

import (
	"fmt"
	"log/slog"
	"net/http"
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

// handleToolCalls executes every tool call in the assistant message, appends
// the results to context, and issues a follow-up streaming request via the
// concrete client.
func (c *baseClient) handleToolCalls(self streamer, assistantMessage ChatMessage) error {
	planMode := self.planMode()
	slog.Debug("handleToolCalls started",
		"assistant_message_content", truncateString(assistantMessage.Content, 500),
		"tool_calls_count", len(assistantMessage.ToolCalls), "plan_mode", planMode)

	for i, toolCall := range assistantMessage.ToolCalls {
		slog.Debug("Processing tool call", "index", i, "tool_id", toolCall.ID,
			"tool_name", toolCall.Function.Name, "tool_arguments", toolCall.Function.Arguments)

		fmt.Printf("🔧 Executing tools...\n")
		fmt.Printf("Calling %s...\n", toolCall.Function.Name)

		result, err := c.toolExecutor(toolCall, planMode)
		if err != nil {
			slog.Error("Tool execution returned error", "tool_name", toolCall.Function.Name, "error", err)
			result = fmt.Sprintf("Error: %v", err)
		}
		result = c.truncateToolResult(result, toolCall.Function.Name)
		c.debugLog("Tool %s result length: %d characters", toolCall.Function.Name, len(result))

		fmt.Printf("✓ %s completed\n", toolCall.Function.Name)

		c.contextManager.AddMessage(ChatMessage{
			Role:    "tool",
			Content: result,
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
