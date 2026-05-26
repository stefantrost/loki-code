package clients

import (
	"testing"
)

func TestDetectAPITypeOllama(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		wantType string
	}{
		{"ollama default", "http://localhost:11434", "ollama"},
		{"ollama with port", "http://localhost:11434", "ollama"},
		{"ollama hostname", "http://ollama.local:11434", "ollama"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectAPIType(tt.baseURL)
			if got != tt.wantType {
				t.Errorf("DetectAPIType(%q) = %q, want %q", tt.baseURL, got, tt.wantType)
			}
		})
	}
}

func TestDetectAPITypeOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		wantType string
	}{
		{"openai api", "https://api.openai.com/v1", "openai-compatible"},
		{"openai hostname", "https://openai.example.com", "openai-compatible"},
		{"chat completions endpoint", "https://api.example.com/chat/completions", "openai-compatible"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectAPIType(tt.baseURL)
			if got != tt.wantType {
				t.Errorf("DetectAPIType(%q) = %q, want %q", tt.baseURL, got, tt.wantType)
			}
		})
	}
}

func TestValidateConfigOllama(t *testing.T) {
	config := ClientConfig{
		APIType:   "ollama",
		BaseURL:   "http://localhost:11434",
		ModelName: "qwen3:32b",
	}

	err := ValidateConfig(config)
	if err != nil {
		t.Errorf("ValidateConfig(ollama) error = %v", err)
	}
}

func TestValidateConfigOpenAI(t *testing.T) {
	config := ClientConfig{
		APIType:    "openai",
		BaseURL:    "https://api.openai.com/v1",
		ModelName:  "gpt-4",
		BearerToken: "sk-test123",
	}

	err := ValidateConfig(config)
	if err != nil {
		t.Errorf("ValidateConfig(openai) error = %v", err)
	}
}

func TestValidateConfigOpenAICompat(t *testing.T) {
	config := ClientConfig{
		APIType:    "openai-compatible",
		BaseURL:    "http://localhost:1234/v1",
		ModelName:  "qwen/qwen3.6-35b-a3b",
		BearerToken: "",
	}

	err := ValidateConfig(config)
	if err != nil {
		t.Errorf("ValidateConfig(openai-compatible) error = %v", err)
	}
}

func TestValidateConfigMissingAPIType(t *testing.T) {
	config := ClientConfig{
		BaseURL:   "http://localhost:11434",
		ModelName: "qwen3:32b",
	}

	err := ValidateConfig(config)
	if err == nil {
		t.Error("ValidateConfig expected error for missing API type")
	}
}

func TestValidateConfigMissingBaseURL(t *testing.T) {
	config := ClientConfig{
		APIType:   "openai",
		ModelName: "gpt-4",
	}

	err := ValidateConfig(config)
	if err == nil {
		t.Error("ValidateConfig expected error for missing base URL")
	}
}

func TestValidateConfigMissingModelName(t *testing.T) {
	config := ClientConfig{
		APIType:   "openai",
		BaseURL:   "https://api.openai.com/v1",
	}

	err := ValidateConfig(config)
	if err == nil {
		t.Error("ValidateConfig expected error for missing model name")
	}
}

func TestGetDefaultConfigOllama(t *testing.T) {
	config := GetDefaultConfig("ollama")

	if config.APIType != "ollama" {
		t.Errorf("APIType = %q, want %q", config.APIType, "ollama")
	}
	if config.BaseURL != "http://localhost:11434" {
		t.Errorf("BaseURL = %q, want %q", config.BaseURL, "http://localhost:11434")
	}
	if config.ModelName != "qwen3:32b" {
		t.Errorf("ModelName = %q, want %q", config.ModelName, "qwen3:32b")
	}
}

func TestGetDefaultConfigOpenAI(t *testing.T) {
	config := GetDefaultConfig("openai")

	if config.APIType != "openai" {
		t.Errorf("APIType = %q, want %q", config.APIType, "openai")
	}
	if config.BaseURL != "https://api.openai.com/v1" {
		t.Errorf("BaseURL = %q, want %q", config.BaseURL, "https://api.openai.com/v1")
	}
	if config.ModelName != "gpt-3.5-turbo" {
		t.Errorf("ModelName = %q, want %q", config.ModelName, "gpt-3.5-turbo")
	}
}

func TestGetDefaultConfigOpenAICompatible(t *testing.T) {
	config := GetDefaultConfig("openai-compatible")

	if config.APIType != "openai-compatible" {
		t.Errorf("APIType = %q, want %q", config.APIType, "openai-compatible")
	}
}

func TestGetDefaultConfigInvalid(t *testing.T) {
	config := GetDefaultConfig("invalid")

	// Should fall back to ollama defaults
	if config.APIType != "ollama" {
		t.Errorf("APIType = %q, want %q (fallback)", config.APIType, "ollama")
	}
}

func TestToolFuncJSONMarshaling(t *testing.T) {
	tool := ToolFunc{
		Name:        "test_tool",
		Description: "A test tool",
		Arguments: map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "File path",
			},
		},
	}

	if tool.Name != "test_tool" {
		t.Errorf("Name = %q, want %q", tool.Name, "test_tool")
	}
	if tool.Description != "A test tool" {
		t.Errorf("Description = %q, want %q", tool.Description, "A test tool")
	}
}

func TestToolStruct(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: ToolFunc{
			Name:        "test_tool",
			Description: "A test tool",
		},
	}

	if tool.Type != "function" {
		t.Errorf("Type = %q, want %q", tool.Type, "function")
	}
	if tool.Function.Name != "test_tool" {
		t.Errorf("Function.Name = %q, want %q", tool.Function.Name, "test_tool")
	}
}

func TestChatMessageStruct(t *testing.T) {
	msg := ChatMessage{
		Role:    "user",
		Content: "hello",
	}

	if msg.Role != "user" {
		t.Errorf("Role = %q, want %q", msg.Role, "user")
	}
	if msg.Content != "hello" {
		t.Errorf("Content = %q, want %q", msg.Content, "hello")
	}
}

func TestToolCallStruct(t *testing.T) {
	call := ToolCall{
		ID:   "call_123",
		Type: "function",
		Function: ToolFunc{
			Name:      "read_file",
			Arguments: map[string]interface{}{"path": "test.txt"},
		},
	}

	if call.ID != "call_123" {
		t.Errorf("ID = %q, want %q", call.ID, "call_123")
	}
	if call.Type != "function" {
		t.Errorf("Type = %q, want %q", call.Type, "function")
	}
	if call.Function.Name != "read_file" {
		t.Errorf("Function.Name = %q, want %q", call.Function.Name, "read_file")
	}
}

func TestUserTaskStruct(t *testing.T) {
	task := UserTask{
		ID:        "task_1",
		Goal:      "implement login",
		Status:    "active",
		Context:   "",
		SubTasks:  []string{},
	}

	if task.ID != "task_1" {
		t.Errorf("ID = %q, want %q", task.ID, "task_1")
	}
	if task.Goal != "implement login" {
		t.Errorf("Goal = %q, want %q", task.Goal, "implement login")
	}
	if task.Status != "active" {
		t.Errorf("Status = %q, want %q", task.Status, "active")
	}
}

func TestClientConfigStruct(t *testing.T) {
	config := ClientConfig{
		APIType:    "openai-compatible",
		BaseURL:    "http://localhost:1234/v1",
		ModelName:  "qwen/qwen3.6-35b-a3b",
		BearerToken: "",
		Debug:      false,
	}

	if config.APIType != "openai-compatible" {
		t.Errorf("APIType = %q, want %q", config.APIType, "openai-compatible")
	}
	if config.BaseURL != "http://localhost:1234/v1" {
		t.Errorf("BaseURL = %q, want %q", config.BaseURL, "http://localhost:1234/v1")
	}
	if config.ModelName != "qwen/qwen3.6-35b-a3b" {
		t.Errorf("ModelName = %q, want %q", config.ModelName, "qwen/qwen3.6-35b-a3b")
	}
}
