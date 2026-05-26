package session

import (
	"loki-code/clients"
	"strings"
	"sync"
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
	if len(messages) != 3 {
		t.Errorf("expected 3 messages (system + 2), got %d", len(messages))
	}

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

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "implement a login system"})
	task := cm.GetActiveTask()
	if task == nil {
		t.Error("expected task to be detected from user message")
	}

	cm.Clear()
	cm.SetActiveTask("")

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "what is go?"})
}

func TestContextManagerToolCallsInProgress(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	if cm.HasToolCallsInProgress() {
		t.Error("expected no tool calls in progress initially")
	}

	cm.AddMessage(clients.ChatMessage{
		Role:    "assistant",
		Content: "I'll help with that",
		ToolCalls: []clients.ToolCall{
			{Function: clients.ToolFunc{Name: "read_file"}},
		},
	})

	if !cm.HasToolCallsInProgress() {
		t.Error("expected tool calls to be in progress")
	}

	cm.AddMessage(clients.ChatMessage{Role: "tool", Content: "file contents here"})

	if cm.HasToolCallsInProgress() {
		t.Error("expected tool calls to be resolved after adding tool result")
	}
}

func TestContextManagerMessagesAccumulate(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(100, provider)

	for i := 0; i < 10; i++ {
		cm.AddMessage(clients.ChatMessage{Role: "user", Content: "word"})
		cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "ok"})
	}

	messages := cm.GetMessages()
	if len(messages) < 2 {
		t.Errorf("expected at least 2 messages, got %d", len(messages))
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

func TestContextManagerCompactSuccess(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	for i := 0; i < 20; i++ {
		cm.AddMessage(clients.ChatMessage{Role: "user", Content: strings.Repeat("word ", 100)})
		cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: strings.Repeat("response ", 100)})
	}

	err := cm.CompactContext(func([]clients.ChatMessage) (string, error) {
		return "This is a summary of the conversation.", nil
	})

	if err != nil {
		t.Fatalf("CompactContext error: %v", err)
	}

	messages := cm.GetMessages()
	if len(messages) < 2 {
		t.Errorf("expected at least 2 messages after compact, got %d", len(messages))
	}
}

func TestContextManagerDetectUserTaskImplement(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "implement a new feature"})

	task := cm.GetActiveTask()
	if task == nil {
		t.Fatal("expected task to be detected")
	}
	if !strings.Contains(strings.ToLower(task.Goal), "implement") {
		t.Errorf("task goal = %q, should contain 'implement'", task.Goal)
	}
}

func TestContextManagerDetectUserTaskFix(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "fix the bug in login"})

	task := cm.GetActiveTask()
	if task == nil {
		t.Fatal("expected task to be detected")
	}
	if !strings.Contains(strings.ToLower(task.Goal), "fix") {
		t.Errorf("task goal = %q, should contain 'fix'", task.Goal)
	}
}

func TestContextManagerTaskHistory(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.SetActiveTask("first task")
	cm.CompleteCurrentTask("first task done")

	cm.SetActiveTask("second task")
	cm.CompleteCurrentTask("second task done")

	if len(cm.tasks.history) != 2 {
		t.Errorf("task history length = %d, want 2", len(cm.tasks.history))
	}
}

func TestContextManagerSystemPromptContainsTools(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{
			{
				Type: "function",
				Function: clients.ToolFunc{
					Name:        "test_tool",
					Description: "A test tool",
				},
			},
		}
	}

	cm := NewContextManager(10000, provider)
	messages := cm.GetMessages()

	systemPrompt := messages[0].Content
	if !strings.Contains(systemPrompt, "test_tool") {
		t.Error("system prompt should contain tool schema")
	}
}

func TestContextManagerPlanModePrompt(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.SetPlanMode(true)

	messages := cm.GetMessages()
	planPrompt := messages[0].Content

	if !strings.Contains(planPrompt, "PLAN MODE") {
		t.Error("plan mode prompt should contain 'PLAN MODE'")
	}
}

func TestContextManagerConciseModeInstructions(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.SetConciseMode(true)

	messages := cm.GetMessages()
	systemPrompt := messages[0].Content

	if !strings.Contains(systemPrompt, "CONCISE") {
		t.Error("system prompt should contain concise mode instructions")
	}
}

func TestContextManagerMultipleTasksSequential(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.SetActiveTask("task one")
	task1 := cm.GetActiveTask()
	if task1 == nil {
		t.Fatal("expected task one")
	}

	cm.SetActiveTask("task two")
	task2 := cm.GetActiveTask()
	if task2 == nil {
		t.Fatal("expected task two")
	}
	if task2.Goal != "task two" {
		t.Errorf("task two goal = %q, want %q", task2.Goal, "task two")
	}
}

func TestContextManagerClearResetsMessages(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "message 1"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "response 1"})
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "message 2"})

	cm.Clear()

	messages := cm.GetMessages()
	if len(messages) != 1 {
		t.Errorf("expected 1 message after clear (system prompt), got %d", len(messages))
	}
}

func TestContextManagerClearKeepsSystemPrompt(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "test"})
	cm.Clear()

	messages := cm.GetMessages()
	if messages[0].Role != "system" {
		t.Errorf("first message role = %q, want %q", messages[0].Role, "system")
	}
}

func TestContextManagerToolCallPreservation(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{
		Role:    "assistant",
		Content: "I'll use tools",
		ToolCalls: []clients.ToolCall{
			{
				ID: "call_1",
				Function: clients.ToolFunc{
					Name:      "read_file",
					Arguments: map[string]interface{}{"path": "test.txt"},
				},
			},
		},
	})

	messages := cm.GetMessages()
	assistantMsg := messages[1]

	if len(assistantMsg.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
	}
	if assistantMsg.ToolCalls[0].Function.Name != "read_file" {
		t.Errorf("tool name = %q, want %q", assistantMsg.ToolCalls[0].Function.Name, "read_file")
	}
}

func TestContextManagerTokenEstimation(t *testing.T) {
	provider := func() []clients.Tool {
		return []clients.Tool{}
	}

	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "hello world"})

	tokens, _, _ := cm.GetStats()
	if tokens == 0 {
		t.Error("token count should be > 0")
	}

	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "hi there"})

	tokens2, _, _ := cm.GetStats()
	if tokens2 <= tokens {
		t.Error("token count should increase after adding message")
	}
}

func TestContextManagerUpdateTokenCount(t *testing.T) {
	provider := func() []clients.Tool { return nil }
	cm := NewContextManager(10000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "hello"})
	estimate, _, _ := cm.GetStats()
	if estimate == 0 {
		t.Fatal("heuristic estimate should be > 0")
	}

	cm.UpdateTokenCount(1234)
	real, _, _ := cm.GetStats()
	if real != 1234 {
		t.Errorf("GetStats() = %d, want 1234 (real count)", real)
	}

	cm.UpdateTokenCount(0)
	after, _, _ := cm.GetStats()
	if after != 1234 {
		t.Errorf("GetStats() = %d after zero update, want 1234", after)
	}
}

func TestContextManagerCanCompact(t *testing.T) {
	provider := func() []clients.Tool { return nil }
	cm := NewContextManager(1000, provider)

	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "msg"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "ok"})
	cm.AddMessage(clients.ChatMessage{Role: "user", Content: "msg"})
	cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "ok"})
	cm.UpdateTokenCount(800)
	if cm.CanCompact() {
		t.Error("CanCompact() = true with only 5 messages (system+4), want false")
	}

	for i := 0; i < 6; i++ {
		cm.AddMessage(clients.ChatMessage{Role: "user", Content: "msg"})
		cm.AddMessage(clients.ChatMessage{Role: "assistant", Content: "ok"})
	}
	cm.UpdateTokenCount(800)
	if !cm.CanCompact() {
		t.Error("CanCompact() = false with real count at 80% and enough messages, want true")
	}

	cm.UpdateTokenCount(500)
	if cm.CanCompact() {
		t.Error("CanCompact() = true with real count at 50%, want false")
	}
}

// TestContextManagerConcurrentAccess exercises the RWMutex added to
// ContextManager. Run with -race to surface any remaining sharing of state.
func TestContextManagerConcurrentAccess(t *testing.T) {
	provider := func() []clients.Tool { return nil }
	cm := NewContextManager(10000, provider)

	var wg sync.WaitGroup
	const goroutines = 8
	const iterations = 100
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				cm.AddMessage(clients.ChatMessage{Role: "user", Content: "hello"})
				_ = cm.GetMessages()
				_, _, _ = cm.GetStats()
				cm.SetPlanMode(id%2 == 0)
				_ = cm.IsInPlanMode()
				cm.SetActiveTask("test")
				_ = cm.GetActiveTask()
			}
		}(g)
	}
	wg.Wait()
}

// TestDetectCompletionPhraseNoActiveTask makes sure completion phrase detection
// without an active task is a no-op rather than a crash.
func TestDetectCompletionPhraseNoActiveTask(t *testing.T) {
	provider := func() []clients.Tool { return nil }
	cm := NewContextManager(10000, provider)
	cm.AddMessage(clients.ChatMessage{
		Role:    "assistant",
		Content: "analysis complete - here are the findings",
	})
	if task := cm.GetActiveTask(); task != nil {
		t.Errorf("expected no active task, got %+v", task)
	}
}
