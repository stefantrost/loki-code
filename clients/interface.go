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

// LLMClient defines the interface that all LLM clients must implement
type LLMClient interface {
	// Core chat functionality
	StreamChat(userInput string) error
	StreamChatWithHistory(messages []ChatMessage) error

	// Context and conversation management
	ClearContext()
	GetStats() (int, int, int) // currentTokens, messageCount, maxTokens
	CanCompact() bool
	CompactContext() error

	// Mode management
	IsInPlanMode() bool
	EnablePlanMode()
	DisablePlanMode()
	IsInConciseMode() bool
	EnableConciseMode()
	DisableConciseMode()

	// Task management
	SetActiveTask(task string)
	GetActiveTask() string
	CompleteCurrentTask()

	// Configuration
	SetDebug(enabled bool)

	// Interruption support
	Interrupt()
	IsResponseActive() bool

	// Context window detection
	DetectContextWindow() (int, error)
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
