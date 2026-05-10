package clients

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// OllamaClient implements the LLMClient interface for Ollama API
type OllamaClient struct {
	baseURL          string
	modelName        string
	toolExecutor     ToolExecutor
	toolSchemaProvider ToolSchemaProvider
	planMode         bool
	conciseMode      bool
	debug            bool
	responseActive   bool
	interruptChan    chan struct{}
	contextManager   ContextManager
	httpClient       *http.Client
}

// Ollama-specific request/response structures
type OllamaRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	Tools    []Tool        `json:"tools,omitempty"`
}

type OllamaModelShowRequest struct {
	Model string `json:"model"`
}

type OllamaModelShowResponse struct {
	ModelInfo map[string]interface{} `json:"model_info"`
}

// NewOllamaClient creates a new Ollama client with the given configuration
func NewOllamaClient(baseURL, modelName string, ctxMgr ContextManager, toolExec ToolExecutor, toolProvider ToolSchemaProvider) *OllamaClient {
	return &OllamaClient{
		baseURL:          baseURL,
		modelName:        modelName,
		toolExecutor:     toolExec,
		toolSchemaProvider: toolProvider,
		contextManager:   ctxMgr,
		interruptChan:    make(chan struct{}),
		httpClient: &http.Client{
			Timeout: 5 * time.Minute, // Hard timeout for streaming
		},
	}
}

// StreamChat implements LLMClient interface
func (c *OllamaClient) StreamChat(userInput string) error {
	userMessage := ChatMessage{
		Role:    "user",
		Content: userInput,
	}
	c.contextManager.AddMessage(userMessage)

	messages := c.contextManager.GetMessages()
	return c.StreamChatWithHistory(messages)
}

// StreamChatWithHistory implements LLMClient interface
func (c *OllamaClient) StreamChatWithHistory(messages []ChatMessage) error {
	currentTokens, messageCount, maxTokens := c.contextManager.GetStats()
	fmt.Printf("[Context: %d/%d tokens, %d messages]\n", currentTokens, maxTokens, messageCount-1)

	c.debugLog("Sending %d messages to model %s", len(messages), c.modelName)
	for i, msg := range messages {
		c.debugLog("Message %d: role=%s, content_length=%d, tool_calls=%d",
			i+1, msg.Role, len(msg.Content), len(msg.ToolCalls))
		if c.debug && len(msg.Content) > 0 {
			c.debugLog("  Content: %s", msg.Content)
		}
	}

	request := OllamaRequest{
		Model:    c.modelName,
		Messages: messages,
		Stream:   true,
		Tools:    c.toolSchemaProvider(),
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	if c.debug {
		c.debugLog("Full request JSON: %s", string(jsonData))
	}

	resp, err := c.httpClient.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("API returned status %s and failed to read error details: %v", resp.Status, err)
		}
		errorBody := string(bodyBytes)

		if resp.StatusCode == http.StatusBadRequest {
			slog.Error("Ollama API error", "status", resp.Status, "error", errorBody)
		}

		return fmt.Errorf("API returned status: %s, details: %s", resp.Status, errorBody)
	}

	c.responseActive = true
	defer func() { c.responseActive = false }()

	scanner := bufio.NewScanner(resp.Body)
	var currentMessage ChatMessage
	var hasToolCalls bool

	for scanner.Scan() {
		select {
		case <-c.interruptChan:
			fmt.Println("\n[Interrupted]")
			return nil
		default:
		}

		line := scanner.Text()
		if line == "" {
			continue
		}

		var chatResponse ChatResponse
		if err := json.Unmarshal([]byte(line), &chatResponse); err != nil {
			c.debugLog("Failed to parse response line: %q, error: %v", line, err)
			continue
		}

		if chatResponse.Message.Content != "" {
			fmt.Print(chatResponse.Message.Content)
			currentMessage.Content += chatResponse.Message.Content
			c.debugLog("Streaming chunk received: %q", chatResponse.Message.Content)
		}

		if len(chatResponse.Message.ToolCalls) > 0 {
			currentMessage.ToolCalls = append(currentMessage.ToolCalls, chatResponse.Message.ToolCalls...)
			hasToolCalls = true
		}

		if chatResponse.Done {
			fmt.Println()
			currentMessage.Role = "assistant"
			c.debugLog("Streaming complete. Final message content: %q", currentMessage.Content)
			break
		}
	}

	if hasToolCalls {
		c.contextManager.AddMessage(currentMessage)
		c.debugLog("Found %d tool calls, processing...", len(currentMessage.ToolCalls))
		return c.handleToolCalls(currentMessage)
	}

	c.contextManager.AddMessage(currentMessage)
	return nil
}

func (c *OllamaClient) handleToolCalls(assistantMessage ChatMessage) error {
	for _, toolCall := range assistantMessage.ToolCalls {
		c.debugLog("Processing tool call: %s with function %s", toolCall.ID, toolCall.Function.Name)

		fmt.Printf("🔧 Executing tools...\n")
		fmt.Printf("Calling %s...\n", toolCall.Function.Name)

		result, err := c.toolExecutor(toolCall, c.planMode)
		if err != nil {
			result = fmt.Sprintf("Error: %v", err)
		}

		result = c.truncateToolResult(result, toolCall.Function.Name)

		c.debugLog("Tool %s result length: %d characters", toolCall.Function.Name, len(result))

		fmt.Printf("✓ %s completed\n", toolCall.Function.Name)

		toolMessage := ChatMessage{
			Role:    "tool",
			Content: result,
		}
		c.contextManager.AddMessage(toolMessage)
	}

	messages := c.contextManager.GetMessages()
	return c.StreamChatWithHistory(messages)
}

func (c *OllamaClient) truncateToolResult(result, toolName string) string {
	const maxLength = 5000

	if len(result) <= maxLength {
		return result
	}

	switch toolName {
	case "read_file":
		truncated := result[:maxLength-100]
		truncated += fmt.Sprintf("\n\n... (file content truncated - showing first %d characters of %d total)",
			maxLength-100, len(result))
		return truncated

	case "list_files", "find_files":
		lines := strings.Split(result, "\n")
		var truncated []string
		currentLength := 0

		for _, line := range lines {
			if currentLength+len(line)+1 > maxLength-100 {
				break
			}
			truncated = append(truncated, line)
			currentLength += len(line) + 1
		}

		truncatedResult := strings.Join(truncated, "\n")
		if len(truncated) < len(lines) {
			truncatedResult += fmt.Sprintf("\n... (output truncated - showing %d of %d lines)",
				len(truncated), len(lines))
		}
		return truncatedResult

	default:
		truncated := result[:maxLength-50]
		truncated += fmt.Sprintf("... (truncated at %d characters)", maxLength-50)
		return truncated
	}
}

// LLMClient interface implementations

func (c *OllamaClient) ClearContext() {
	c.contextManager.Clear()
}

func (c *OllamaClient) GetStats() (int, int, int) {
	return c.contextManager.GetStats()
}

func (c *OllamaClient) CanCompact() bool {
	return c.contextManager.CanCompact()
}

func (c *OllamaClient) CompactContext() error {
	return c.contextManager.CompactContext(c.CompactMessages)
}

func (c *OllamaClient) CompactMessages(messages []ChatMessage) (string, error) {
	compactPrompt := `Please provide a concise summary of the conversation above. Focus on:
1. Key decisions made
2. Important code changes or implementations
3. Current task status
4. Any unresolved issues

Keep the summary under 200 words while preserving essential context.`

	request := OllamaRequest{
		Model: c.modelName,
		Messages: append(messages, ChatMessage{
			Role:    "user",
			Content: compactPrompt,
		}),
		Stream: false,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := c.httpClient.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API returned status: %s", resp.Status)
	}

	var chatResponse ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResponse); err != nil {
		return "", fmt.Errorf("error decoding response: %v", err)
	}

	return chatResponse.Message.Content, nil
}

func (c *OllamaClient) EnablePlanMode() {
	c.planMode = true
	c.contextManager.SetPlanMode(true)
}

func (c *OllamaClient) DisablePlanMode() {
	c.planMode = false
	c.contextManager.SetPlanMode(false)
}

func (c *OllamaClient) IsInPlanMode() bool {
	return c.planMode
}

func (c *OllamaClient) EnableConciseMode() {
	c.conciseMode = true
	c.contextManager.SetConciseMode(true)
}

func (c *OllamaClient) DisableConciseMode() {
	c.conciseMode = false
	c.contextManager.SetConciseMode(false)
}

func (c *OllamaClient) IsInConciseMode() bool {
	return c.conciseMode
}

func (c *OllamaClient) SetActiveTask(task string) {
	c.contextManager.SetActiveTask(task)
}

func (c *OllamaClient) GetActiveTask() string {
	if task := c.contextManager.GetActiveTask(); task != nil {
		return task.Goal
	}
	return ""
}

func (c *OllamaClient) CompleteCurrentTask() {
	c.contextManager.CompleteCurrentTask("")
}

func (c *OllamaClient) Interrupt() {
	select {
	case c.interruptChan <- struct{}{}:
	default:
	}
}

func (c *OllamaClient) IsResponseActive() bool {
	return c.responseActive
}

func (c *OllamaClient) DetectContextWindow() (int, error) {
	request := OllamaModelShowRequest{
		Model: c.modelName,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return 0, fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := c.httpClient.Post(c.baseURL+"/api/show", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return 0, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("API returned status: %s", resp.Status)
	}

	var modelResponse OllamaModelShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelResponse); err != nil {
		return 0, fmt.Errorf("error decoding response: %v", err)
	}

	if modelInfo, exists := modelResponse.ModelInfo["general.parameter_count"]; exists {
		if paramCount, ok := modelInfo.(float64); ok {
			if paramCount > 30000000000 {
				return 32768, nil
			} else if paramCount > 10000000000 {
				return 16384, nil
			} else if paramCount > 5000000000 {
				return 8192, nil
			}
		}
	}

	return 4096, nil
}

func (c *OllamaClient) SetDebug(enabled bool) {
	c.debug = enabled
	if enabled {
		slog.Info("Debug logging enabled for Ollama client")
	}
}

func (c *OllamaClient) debugLog(format string, args ...interface{}) {
	if c.debug {
		slog.Debug(fmt.Sprintf(format, args...))
	}
}
