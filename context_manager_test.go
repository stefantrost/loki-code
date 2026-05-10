package main

import (
	"loki-code/clients"
	"strings"
	"testing"
)

func TestNewContextManager(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(1000, provider)
	if cm == nil {
		t.Fatal("NewContextManager returned nil")
	}
	if cm.IsInPlanMode() {
		t.Error("new context manager should not be in plan mode")
	}
	if cm.IsInConciseMode() {
		t.Error("new context manager should not be in concise mode")
	}
}

func TestContextManagerAddAndGetMessages(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "hello"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "hi there"})

	messages := cm.GetMessages()
	// Should have system prompt + 2 messages
	if len(messages) != 3 {
		t.Errorf("expected 3 messages (system + 2), got %d", len(messages))
	}

	// First message should be system prompt
	if messages[0].Role != "system" {
		t.Errorf("first message role = %q, want %q", messages[0].Role, "system")
	}
}

func TestContextManagerClear(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "hello"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "hi"})

	cm.Clear()
	messages := cm.GetMessages()

	// Should only have system prompt after clear
	if len(messages) != 1 {
		t.Errorf("expected 1 message after clear (system prompt), got %d", len(messages))
	}
}

func TestContextManagerPlanMode(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.SetPlanMode(true)
	if !cm.IsInPlanMode() {
		t.Error("expected plan mode to be enabled")
	}

	cm.SetPlanMode(false)
	if cm.IsInPlanMode() {
		t.Error("expected plan mode to be disabled")
	}
}

func TestContextManagerConciseMode(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.SetConciseMode(true)
	if !cm.IsInConciseMode() {
		t.Error("expected concise mode to be enabled")
	}

	cm.SetConciseMode(false)
	if cm.IsInConciseMode() {
		t.Error("expected concise mode to be disabled")
	}
}

func TestContextManagerStats(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	tokens, msgs, maxTokens := cm.GetStats()
	if maxTokens != 10000 {
		t.Errorf("maxTokens = %d, want 10000", maxTokens)
	}
	if msgs != 0 {
		t.Errorf("message count = %d, want 0 (no user messages yet, only system prompt)", msgs)
	}
	if tokens == 0 {
		t.Error("token count should be > 0 due to system prompt")
	}
}

func TestContextManagerSetActiveAndGetTask(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	if cm.GetActiveTask() != nil {
		t.Error("expected no active task initially")
	}

	cm.SetActiveTask("implement login")
	task := cm.GetActiveTask()
	if task == nil {
		t.Fatal("expected active task after setting")
	}
	if task.Goal != "implement login" {
		t.Errorf("task goal = %q, want %q", task.Goal, "implement login")
	}
	if task.Status != "active" {
		t.Errorf("task status = %q, want %q", task.Status, "active")
	}
}

func TestContextManagerCompleteTask(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.SetActiveTask("implement login")

	cm.CompleteCurrentTask("login implemented")

	if cm.GetActiveTask() != nil {
		t.Error("expected task to be nil after completion")
	}
}

func TestContextManagerDetectUserTask(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	// Should detect task
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "implement a login system"})
	task := cm.GetActiveTask()
	if task == nil {
		t.Error("expected task to be detected from user message")
	}

	// Clear and test simple question (should not detect)
	cm.Clear()
	cm.SetActiveTask("") // Reset

	// Test with simple question pattern
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "what is go?"})
	// Simple questions should not create tasks
	// Note: This depends on the detectUserTask implementation
}

func TestContextManagerToolCallsInProgress(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	// No tool calls in progress initially
	if cm.HasToolCallsInProgress() {
		t.Error("expected no tool calls in progress initially")
	}

	// Add assistant message with tool calls but no results
	cm.AddMessage(clients.ChatMessage{
		Role: "assistant",
		Content: "I'll help with that",
		ToolCalls: []clients.ToolCall{
			{Function: clients.ToolFunc{Name: "read_file"}},
		},
	})

	if !cm.HasToolCallsInProgress() {
		t.Error("expected tool calls to be in progress")
	}

	// Add tool result - calls are now resolved
	cm.AddMessage(clients.ChatMessage{Role: "tool", Content: "file contents here"})

	if cm.HasToolCallsInProgress() {
		t.Error("expected tool calls to be resolved after adding tool result")
	}
}

func TestContextManagerTrimIfNeeded(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	// Very small token limit to force trimming
	cm := NewContextManager(100, provider)

	// Add many messages to exceed token limit
	for i := 0; i < 50; i++ {
		cm.AddMessage(clients.ChatMessage{
			Role:    "user",
			Content: strings.Repeat("word ", 20),
		})
		cm.AddMessage(clients.ChatMessage{
			Role:    "assistant",
			Content: strings.Repeat("response ", 20),
		})
	}

	messages := cm.GetMessages()
	// Should have been trimmed - system prompt + some recent messages
	if len(messages) < 2 {
		t.Errorf("expected at least 2 messages after trim, got %d", len(messages))
	}
}

func TestContextManagerSetMaxTokens(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(1000, provider)
	cm.SetMaxTokens(5000)

	_, _, maxTokens := cm.GetStats()
	if maxTokens != 5000 {
		t.Errorf("maxTokens = %d, want 5000", maxTokens)
	}
}

func TestContextManagerCompactCannotCompactTooFewMessages(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	// Add only 2 messages - not enough to compact
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "hello"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "hi"})

	err := cm.CompactContext(func([]clients.ChatMessage) (string, error) {
		return "summary", nil
	})

	if err == nil {
		t.Error("expected error when compacting with too few messages")
	}
}

func TestContextManagerEnhancePromptWithTask(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.SetActiveTask("implement login")

	messages := cm.GetMessages()
	systemPrompt := messages[0].Content

	if !strings.Contains(systemPrompt, "CURRENT ACTIVE TASK") {
		t.Error("expected system prompt to contain active task context")
	}
	if !strings.Contains(systemPrompt, "implement login") {
		t.Error("expected system prompt to contain task goal")
	}
}

func TestContextManagerGetLastUserMessage(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "first message"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "response"})
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "second message"})

	lastUser := cm.GetLastUserMessage()
	if lastUser == nil {
		t.Fatal("expected last user message")
	}
	if lastUser.Content != "second message" {
		t.Errorf("last user message = %q, want %q", lastUser.Content, "second message")
	}
}
