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
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents the function details in a tool call
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Arguments   map[string]interface{} `json:"arguments,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// UserTask represents an active user task
type UserTask struct {
	Goal   string
	Status string
}

// ContextManager defines the interface for conversation context management
type ContextManager interface {
	AddMessage(ChatMessage)
	GetMessages() []ChatMessage
	GetStats() (int, int, int)
	SetMaxTokens(int)
	ClearMessages()
	CompactContext(func([]ChatMessage) (string, error)) error
	CanCompact() bool
	EnablePlanMode()
	DisablePlanMode()
	EnableConciseMode()
	DisableConciseMode()
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

// MainPackageIntegration defines the interface for main package functions
type MainPackageIntegration interface {
	NewContextManager(maxTokens int) ContextManager
	GetAvailableTools() []Tool
	ExecuteToolWithPlanMode(toolCall ToolCall, planMode bool) (string, error)
}

// Global integration instance
var mainPackage MainPackageIntegration

// SetMainPackageIntegration sets up the integration with main package
func SetMainPackageIntegration(integration MainPackageIntegration) {
	mainPackage = integration
}

// ChatResponse represents a streaming response chunk
type ChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt time.Time   `json:"created_at"`
	Message   ChatMessage `json:"message"`
	Done      bool        `json:"done"`
}