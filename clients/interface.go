package clients

import "time"

// ChatMessage represents a message in the conversation
type ChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall represents a function call request from the LLM
type ToolCall struct {
	ID       string   `json:"id,omitempty"`
	Type     string   `json:"type,omitempty"`
	Function ToolFunc `json:"function"`
}

// ToolFunc represents the function details in a tool call
type ToolFunc struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Arguments   map[string]interface{} `json:"arguments,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function ToolFunc `json:"function"`
}

// UserTask represents an active user task
type UserTask struct {
	ID        string    `json:"id"`
	Goal      string    `json:"goal"`
	Context   string    `json:"context"`
	CreatedAt time.Time `json:"created_at"`
	Status    string    `json:"status"` // "active", "completed", "abandoned"
	SubTasks  []string  `json:"sub_tasks"`
}

// ToolExecutor is the callback for executing tools
type ToolExecutor func(toolCall ToolCall, planMode bool) (string, error)

// ToolSchemaProvider provides the list of available tools
type ToolSchemaProvider func() []Tool

// CompactFunc is the callback for compacting messages
type CompactFunc func(messages []ChatMessage) (string, error)

// ContextManager defines the interface for conversation context management
type ContextManager interface {
	AddMessage(ChatMessage)
	GetMessages() []ChatMessage
	GetStats() (int, int, int) // currentTokens, messageCount, maxTokens
	SetMaxTokens(int)
	Clear()
	CompactContext(compactFunc CompactFunc) error
	CanCompact() bool
	SetPlanMode(bool)
	IsInPlanMode() bool
	SetConciseMode(bool)
	IsInConciseMode() bool
	GetActiveTask() *UserTask
	SetActiveTask(string)
	CompleteCurrentTask(string)
}

// Chat is the streaming-chat surface a client must implement.
type Chat interface {
	StreamChat(userInput string) error
	StreamChatWithHistory(messages []ChatMessage) error
}

// ContextLifecycle covers conversation state and compaction.
type ContextLifecycle interface {
	ClearContext()
	GetStats() (int, int, int) // currentTokens, messageCount, maxTokens
	CanCompact() bool
	CompactContext() error
}

// Modes toggles plan and concise modes.
type Modes interface {
	IsInPlanMode() bool
	EnablePlanMode()
	DisablePlanMode()
	IsInConciseMode() bool
	EnableConciseMode()
	DisableConciseMode()
}

// Tasks exposes user-task tracking.
type Tasks interface {
	SetActiveTask(task string)
	GetActiveTask() string
	CompleteCurrentTask()
}

// Interruptible lets callers cancel an in-flight stream.
type Interruptible interface {
	Interrupt()
	IsResponseActive() bool
}

// WindowDetector reports the model's usable context window in tokens.
type WindowDetector interface {
	DetectContextWindow() (int, error)
}

// LLMClient is the union surface every concrete client implements. Consumers
// that need only a subset should depend on the narrower interfaces above.
type LLMClient interface {
	Chat
	ContextLifecycle
	Modes
	Tasks
	Interruptible
	WindowDetector
	SetDebug(enabled bool)
	SetTruncator(TruncationPolicy)
}

// ClientConfig holds configuration for creating clients
type ClientConfig struct {
	APIType     string
	BaseURL     string
	ModelName   string
	BearerToken string
	Debug       bool
}

// ChatResponse represents a streaming response chunk
type ChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt time.Time   `json:"created_at"`
	Message   ChatMessage `json:"message"`
	Done      bool        `json:"done"`
}
