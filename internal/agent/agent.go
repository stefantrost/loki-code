// Package agent contains the REPL loop that drives the LLM client via the
// View interface. It has no knowledge of any concrete view implementation
// (TUI, CLI, Web) or of any specific LLM provider — both are injected.
package agent

import (
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"loki-code/clients"
	"loki-code/internal/view"
)

// ── Writers ───────────────────────────────────────────────────────────────────

// tokenWriter implements io.Writer.  Each Write call forwards the raw bytes as
// streaming tokens to the view and appends them to an internal buffer so that
// CommitMessage can receive the full response text once streaming completes.
type tokenWriter struct {
	v   view.View
	buf strings.Builder
}

func (w *tokenWriter) Write(p []byte) (int, error) {
	s := string(p)
	w.v.WriteToken(s)
	w.buf.WriteString(s)
	return len(p), nil
}

// systemWriter implements io.Writer with line buffering.  handleToolCalls
// writes box-drawing blocks via fmt.Fprintf which embeds '\n' characters.
// We buffer partial lines and emit one WriteSystem call per complete line
// (stripped of its trailing '\n') so that CLIView (fmt.Println adds '\n')
// and TUIView (renderContent adds '\n') both display cleanly without
// double-spacing.
type systemWriter struct {
	v   view.View
	buf strings.Builder
}

func (w *systemWriter) Write(p []byte) (int, error) {
	w.buf.WriteString(string(p))
	s := w.buf.String()
	for {
		idx := strings.IndexByte(s, '\n')
		if idx < 0 {
			break
		}
		w.v.WriteSystem(s[:idx])
		s = s[idx+1:]
	}
	w.buf.Reset()
	w.buf.WriteString(s)
	return len(p), nil
}

// flush emits any buffered partial line (no trailing '\n' was received).
func (w *systemWriter) flush() {
	if w.buf.Len() > 0 {
		w.v.WriteSystem(w.buf.String())
		w.buf.Reset()
	}
}

// ── Public entry point ────────────────────────────────────────────────────────

// Run starts the agent REPL inside v.  v.Run blocks until the session ends.
func Run(client clients.LLMClient, v view.View) error {
	return v.Run(func() { runLoop(client, v) })
}

// ── REPL loop ─────────────────────────────────────────────────────────────────

func runLoop(client clients.LLMClient, v view.View) {
	// SIGINT / SIGTERM handling:
	//   • during an active response → interrupt the stream and continue
	//   • at the idle prompt → stop the view and exit cleanly
	// Note: in TUI (alt-screen + raw mode) Ctrl+C arrives as a tea.KeyMsg,
	// not a SIGINT, so this handler fires only for CLI mode or external signals.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)
	go func() {
		for range sigChan {
			if client.IsResponseActive() {
				client.Interrupt()
			} else {
				v.Stop()
				return
			}
		}
	}()

	updateStatus(client, v)

	for {
		input, err := v.ReadInput("")
		if err != nil {
			// io.EOF or "session ended" (TUIView closed inputCh) — clean exit.
			return
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}

		slog.Debug("User input", "len", len(input))

		if input == "exit" || input == "quit" {
			v.WriteSystem("Goodbye!")
			v.Stop()
			return
		}

		if handleSlashCommand(client, v, input) {
			updateStatus(client, v)
			continue
		}

		// Echo the user prompt as a separator in the chat pane.
		v.WriteSystem(fmt.Sprintf("─── You: %s ───", input))

		streamResponse(client, v, input)
	}
}

// ── Streaming ─────────────────────────────────────────────────────────────────

// streamResponse wires up the token/system writers, streams the LLM response,
// commits the glamour-rendered result, and auto-compacts if needed.
func streamResponse(client clients.LLMClient, v view.View, input string) {
	tw := &tokenWriter{v: v}
	sw := &systemWriter{v: v}

	// Anchor the start of this stream segment in the TUI viewport so that
	// CommitMessage only replaces the tokens written since this call.
	v.BeginStream()

	client.SetOutputWriter(tw)
	client.SetSystemWriter(sw)
	// Each StreamChatWithHistory call (follow-up after tool execution) fires
	// this callback, re-anchoring the viewport past the tool-block lines and
	// resetting the token buffer so CommitMessage receives only the latest
	// streaming segment.
	client.SetStreamStartCallback(func() {
		sw.flush()
		v.BeginStream()
		tw.buf.Reset()
	})

	if err := client.StreamChat(input); err != nil && !isInterruptErr(err) {
		slog.Error("StreamChat error", "error", err)
		v.WriteSystem(fmt.Sprintf("⚠️  Error: %v", err))
	}

	// Flush any buffered partial system line (shouldn't normally happen, but
	// safe to handle).
	sw.flush()

	// Glamour-render the accumulated response and replace the raw-token region.
	if tw.buf.Len() > 0 {
		v.CommitMessage("assistant", tw.buf.String())
	}

	// Reset writers so stale references can't leak writes after this turn.
	client.SetOutputWriter(nil)
	client.SetSystemWriter(nil)
	client.SetStreamStartCallback(nil)

	// Auto-compact when context reaches the 75% threshold.
	if client.CanCompact() {
		v.WriteSystem("⚡ Context at 75% — auto-compacting…")
		if err := client.CompactContext(); err != nil {
			slog.Warn("Auto-compact failed", "error", err)
		}
	}

	updateStatus(client, v)
}

// isInterruptErr returns true when the error represents a deliberate user
// interruption rather than a transport failure.
func isInterruptErr(err error) bool {
	return err != nil && strings.Contains(err.Error(), "interrupted")
}

// ── Slash commands ────────────────────────────────────────────────────────────

// handleSlashCommand processes slash commands and returns true if the input was
// consumed, false if it should be forwarded to the LLM.
func handleSlashCommand(client clients.LLMClient, v view.View, input string) bool { //nolint:cyclop
	switch {
	case input == "/clear":
		client.ClearContext()
		v.WriteSystem("✓ Context cleared")
		return true

	case input == "/stats":
		tokens, messages, maxTokens := client.GetStats()
		mode := "execute"
		if client.IsInPlanMode() {
			mode = "plan"
		}
		resp := "verbose"
		if client.IsInConciseMode() {
			resp = "concise"
		}
		task := client.GetActiveTask()
		if task == "" {
			task = "(none)"
		}
		v.WriteSystem(fmt.Sprintf(
			"📊 Context: %d/%d tokens · %d messages | Mode: %s | Response: %s | Task: %s",
			tokens, maxTokens, messages, mode, resp, task,
		))
		return true

	case input == "/concise":
		if client.IsInConciseMode() {
			v.WriteSystem("Already in concise mode")
			return true
		}
		client.EnableConciseMode()
		v.WriteSystem("📝 Concise mode — responses will be brief and to-the-point")
		return true

	case input == "/verbose":
		if !client.IsInConciseMode() {
			v.WriteSystem("Already in verbose mode")
			return true
		}
		client.DisableConciseMode()
		v.WriteSystem("📝 Verbose mode — detailed explanations enabled")
		return true

	case input == "/mode":
		mode := "verbose"
		if client.IsInConciseMode() {
			mode = "concise"
		}
		if client.IsInPlanMode() {
			mode += " (plan mode active)"
		}
		v.WriteSystem(fmt.Sprintf("Current mode: %s", mode))
		return true

	case input == "/plan":
		if client.IsInPlanMode() {
			v.WriteSystem("Already in plan mode")
			return true
		}
		client.EnablePlanMode()
		v.WriteSystem("🎯 Plan mode — only read operations allowed · /execute to exit")
		return true

	case input == "/execute":
		if !client.IsInPlanMode() {
			v.WriteSystem("Not in plan mode")
			return true
		}
		client.DisablePlanMode()
		v.WriteSystem("⚡ Execute mode — all tools available")
		return true

	case input == "/task":
		task := client.GetActiveTask()
		if task != "" {
			v.WriteSystem(fmt.Sprintf("🎯 Current task: %s", task))
		} else {
			v.WriteSystem("No active task")
		}
		return true

	case strings.HasPrefix(input, "/task "):
		desc := strings.TrimSpace(strings.TrimPrefix(input, "/task "))
		if desc == "" {
			v.WriteSystem("Usage: /task <description>")
			return true
		}
		client.SetActiveTask(desc)
		v.WriteSystem(fmt.Sprintf("🎯 Task set: %s", desc))
		return true

	case input == "/complete":
		if client.GetActiveTask() == "" {
			v.WriteSystem("No active task to complete")
			return true
		}
		client.CompleteCurrentTask()
		v.WriteSystem("✅ Task marked as complete")
		return true

	case input == "/compact":
		if !client.CanCompact() {
			v.WriteSystem("Context not ready for compacting (need 60%+ token usage)")
			return true
		}
		v.WriteSystem("Compacting conversation context…")
		if err := client.CompactContext(); err != nil {
			slog.Error("Compact failed", "error", err)
			v.WriteSystem(fmt.Sprintf("⚠️  Compact failed: %v", err))
			return true
		}
		v.WriteSystem("✓ Context compacted")
		return true
	}

	return false
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// updateStatus pushes the current client state into the view's sidebar /
// status line.
func updateStatus(client clients.LLMClient, v view.View) {
	tokens, _, maxTokens := client.GetStats()
	mode := "execute"
	if client.IsInPlanMode() {
		mode = "plan"
	}
	resp := "verbose"
	if client.IsInConciseMode() {
		resp = "concise"
	}
	v.UpdateStatus(view.Status{
		Tokens:    tokens,
		MaxTokens: maxTokens,
		Mode:      mode,
		Response:  resp,
		Task:      client.GetActiveTask(),
	})
}
