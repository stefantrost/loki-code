package agent

import (
	"context"
	"fmt"
	"io"
	"strings"
	"testing"

	"loki-code/clients"
	"loki-code/internal/view"
)

// ── mockView ──────────────────────────────────────────────────────────────────

// mockView is a synchronous View implementation for tests.
// Run calls fn() immediately; ReadInput drains the pre-loaded inputs channel.
type mockView struct {
	inputs    chan string
	tokens    []string
	system    []string
	committed []string
	statuses  []view.Status
	stopped   bool
}

func newMockView(inputs ...string) *mockView {
	ch := make(chan string, len(inputs))
	for _, s := range inputs {
		ch <- s
	}
	close(ch)
	return &mockView{inputs: ch}
}

func (v *mockView) Run(fn func()) error { fn(); return nil }

func (v *mockView) ReadInput(_ string) (string, error) {
	s, ok := <-v.inputs
	if !ok {
		return "", io.EOF
	}
	return s, nil
}

func (v *mockView) WriteToken(t string)           { v.tokens = append(v.tokens, t) }
func (v *mockView) WriteThinking(_ string)        {}
func (v *mockView) CommitMessage(_, c string)     { v.committed = append(v.committed, c) }
func (v *mockView) WriteSystem(m string)          { v.system = append(v.system, m) }
func (v *mockView) BeginStream()                  {}
func (v *mockView) UpdateStatus(s view.Status)    { v.statuses = append(v.statuses, s) }
func (v *mockView) ShowDiffAndConfirm(_, _, _ string) (bool, error) { return true, nil }
func (v *mockView) Confirm(_ string) (bool, error) { return true, nil }
func (v *mockView) Stop()                          { v.stopped = true }
func (v *mockView) IsCLI() bool                    { return false }

// containsSystem returns true if any element of v.system contains substr.
func (v *mockView) containsSystem(substr string) bool {
	for _, s := range v.system {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}

// ── mockClient ────────────────────────────────────────────────────────────────

// mockClient implements clients.LLMClient for testing.
// StreamChat writes a fixed token ("response") via the registered outputWriter.
type mockClient struct {
	outputWriter  io.Writer
	systemWriter  io.Writer
	onStart       func()
	planMode      bool
	conciseMode   bool
	task          string
	compactCalled bool
	streamErr     error // if non-nil, StreamChat returns this
}

// ── clients.LLMClient methods ─────────────────────────────────────────────────

func (c *mockClient) StreamChat(input string) error {
	if c.onStart != nil {
		c.onStart()
	}
	if c.outputWriter != nil {
		fmt.Fprint(c.outputWriter, "response")
	}
	return c.streamErr
}

func (c *mockClient) StreamChatWithHistory(_ []clients.ChatMessage) error { return nil }

func (c *mockClient) SetOutputWriter(w io.Writer)       { c.outputWriter = w }
func (c *mockClient) SetThinkingWriter(_ io.Writer)     {}
func (c *mockClient) SetSystemWriter(w io.Writer)       { c.systemWriter = w }
func (c *mockClient) SetStreamStartCallback(fn func())  { c.onStart = fn }

func (c *mockClient) ClearContext()              {}
func (c *mockClient) GetStats() (int, int, int)  { return 0, 0, 0 }
func (c *mockClient) CanCompact() bool           { return false }
func (c *mockClient) CompactContext() error       { c.compactCalled = true; return nil }

func (c *mockClient) IsInPlanMode() bool    { return c.planMode }
func (c *mockClient) EnablePlanMode()       { c.planMode = true }
func (c *mockClient) DisablePlanMode()      { c.planMode = false }
func (c *mockClient) IsInConciseMode() bool { return c.conciseMode }
func (c *mockClient) EnableConciseMode()    { c.conciseMode = true }
func (c *mockClient) DisableConciseMode()   { c.conciseMode = false }

func (c *mockClient) SetActiveTask(t string)   { c.task = t }
func (c *mockClient) GetActiveTask() string    { return c.task }
func (c *mockClient) CompleteCurrentTask()     { c.task = "" }

func (c *mockClient) Interrupt()              {}
func (c *mockClient) IsResponseActive() bool  { return false }
func (c *mockClient) DetectContextWindow() (int, error) { return 4096, nil }
func (c *mockClient) SetDebug(_ bool)         {}
func (c *mockClient) SetTruncator(_ clients.TruncationPolicy) {}

// ── tests ─────────────────────────────────────────────────────────────────────

func TestAgent_ExitCommand(t *testing.T) {
	mv := newMockView("exit")
	err := Run(context.Background(), &mockClient{}, mv)
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if !mv.stopped {
		t.Error("view.Stop() was not called on exit")
	}
}

func TestAgent_QuitCommand(t *testing.T) {
	mv := newMockView("quit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.stopped {
		t.Error("view.Stop() was not called on quit")
	}
}

func TestAgent_ReadInputEOF(t *testing.T) {
	// Channel already closed — ReadInput returns EOF immediately.
	mv := newMockView() // no inputs; channel closed
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run returned unexpected error on EOF: %v", err)
	}
}

func TestAgent_EmptyInputSkipped(t *testing.T) {
	mv := newMockView("", "   ", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	// No user-prompt separator should have been written for blank inputs.
	for _, s := range mv.system {
		if strings.Contains(s, "─── You:") && strings.TrimSpace(strings.TrimPrefix(s, "─── You:")) == "" {
			t.Errorf("unexpected system message for empty input: %q", s)
		}
	}
}

func TestAgent_SlashClear(t *testing.T) {
	mv := newMockView("/clear", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("✓ Context cleared") {
		t.Errorf("system messages %v — expected '✓ Context cleared'", mv.system)
	}
}

func TestAgent_SlashStats(t *testing.T) {
	mv := newMockView("/stats", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("📊 Context") {
		t.Errorf("system messages %v — expected stats output", mv.system)
	}
}

func TestAgent_SlashPlanAndExecute(t *testing.T) {
	mc := &mockClient{}
	mv := newMockView("/plan", "/execute", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if mc.planMode {
		t.Error("plan mode should be off after /execute")
	}
	if !mv.containsSystem("🎯 Plan mode") {
		t.Errorf("system messages %v — expected plan mode activation message", mv.system)
	}
	if !mv.containsSystem("⚡ Execute mode") {
		t.Errorf("system messages %v — expected execute mode message", mv.system)
	}
}

func TestAgent_SlashPlanAlreadyActive(t *testing.T) {
	mc := &mockClient{planMode: true}
	mv := newMockView("/plan", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("Already in plan mode") {
		t.Errorf("system messages %v — expected 'Already in plan mode'", mv.system)
	}
}

func TestAgent_SlashExecuteNotInPlanMode(t *testing.T) {
	mv := newMockView("/execute", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("Not in plan mode") {
		t.Errorf("system messages %v — expected 'Not in plan mode'", mv.system)
	}
}

func TestAgent_SlashConciseVerbose(t *testing.T) {
	mc := &mockClient{}
	mv := newMockView("/concise", "/verbose", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if mc.conciseMode {
		t.Error("concise mode should be off after /verbose")
	}
	if !mv.containsSystem("📝 Concise mode") {
		t.Errorf("system messages %v — expected concise mode message", mv.system)
	}
}

func TestAgent_SlashMode(t *testing.T) {
	mv := newMockView("/mode", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("Current mode") {
		t.Errorf("system messages %v — expected mode message", mv.system)
	}
}

func TestAgent_SlashTask(t *testing.T) {
	mc := &mockClient{}
	mv := newMockView("/task refactor auth", "/task", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if mc.task != "refactor auth" {
		t.Errorf("task = %q, want %q", mc.task, "refactor auth")
	}
	if !mv.containsSystem("🎯 Task set: refactor auth") {
		t.Errorf("system messages %v — expected task set message", mv.system)
	}
	if !mv.containsSystem("🎯 Current task: refactor auth") {
		t.Errorf("system messages %v — expected task show message", mv.system)
	}
}

func TestAgent_SlashTaskNoArg(t *testing.T) {
	mv := newMockView("/task", "exit") // no active task
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("No active task") {
		t.Errorf("system messages %v — expected 'No active task'", mv.system)
	}
}

func TestAgent_SlashComplete(t *testing.T) {
	mc := &mockClient{task: "some task"}
	mv := newMockView("/complete", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if mc.task != "" {
		t.Errorf("task = %q after /complete, want empty", mc.task)
	}
	if !mv.containsSystem("✅ Task marked as complete") {
		t.Errorf("system messages %v — expected complete message", mv.system)
	}
}

func TestAgent_SlashCompleteNoTask(t *testing.T) {
	mv := newMockView("/complete", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("No active task to complete") {
		t.Errorf("system messages %v — expected 'No active task to complete'", mv.system)
	}
}

func TestAgent_SlashCompactNotReady(t *testing.T) {
	mv := newMockView("/compact", "exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("not ready for compacting") {
		t.Errorf("system messages %v — expected compaction not-ready message", mv.system)
	}
}

func TestAgent_StreamResponse(t *testing.T) {
	mc := &mockClient{}
	mv := newMockView("hello", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	// CommitMessage should have been called with the accumulated "response" token.
	if len(mv.committed) == 0 {
		t.Fatal("CommitMessage was never called")
	}
	if mv.committed[0] != "response" {
		t.Errorf("CommitMessage content = %q, want %q", mv.committed[0], "response")
	}
	// Token should also have been forwarded live.
	if len(mv.tokens) == 0 || mv.tokens[0] != "response" {
		t.Errorf("tokens = %v, want [response]", mv.tokens)
	}
	// User prompt separator should appear in system output.
	if !mv.containsSystem("─── You: hello ───") {
		t.Errorf("system messages %v — expected user prompt separator", mv.system)
	}
}

func TestAgent_UpdateStatusCalled(t *testing.T) {
	mv := newMockView("exit")
	if err := Run(context.Background(), &mockClient{}, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if len(mv.statuses) == 0 {
		t.Error("UpdateStatus was never called")
	}
}

func TestAgent_StreamError(t *testing.T) {
	mc := &mockClient{streamErr: fmt.Errorf("transport failure")}
	mv := newMockView("hello", "exit")
	if err := Run(context.Background(), mc, mv); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !mv.containsSystem("⚠️  Error: transport failure") {
		t.Errorf("system messages %v — expected error message", mv.system)
	}
}
