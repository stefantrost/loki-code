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

// OpenAIClient implements the LLMClient interface for OpenAI-compatible APIs
type OpenAIClient struct {
	baseURL          string
	bearerToken      string
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

// OpenAI-compatible request/response structures
type OpenAIRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`
	Tools    []Tool        `json:"tools,omitempty"`
}

type OpenAIResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []OpenAIChoice         `json:"choices"`
	Usage   map[string]interface{} `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message,omitempty"`
	Delta        ChatMessage `json:"delta,omitempty"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

type OpenAIStreamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
}

// NewOpenAIClient creates a new OpenAI-compatible client
func NewOpenAIClient(baseURL, bearerToken, modelName string, ctxMgr ContextManager, toolExec ToolExecutor, toolProvider ToolSchemaProvider) *OpenAIClient {
	return &OpenAIClient{
		baseURL:          baseURL,
		bearerToken:      bearerToken,
		modelName:        modelName,
		toolExecutor:     toolExec,
		toolSchemaProvider: toolProvider,
		contextManager:   ctxMgr,
		interruptChan:    make(chan struct{}),
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

// StreamChat implements LLMClient interface
func (c *OpenAIClient) StreamChat(userInput string) error {
	userMessage := ChatMessage{
		Role:    "user",
		Content: userInput,
	}
	c.contextManager.AddMessage(userMessage)

	messages := c.contextManager.GetMessages()
	return c.StreamChatWithHistory(messages)
}

// StreamChatWithHistory implements LLMClient interface
func (c *OpenAIClient) StreamChatWithHistory(messages []ChatMessage) error {
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

	request := OpenAIRequest{
		Model:    c.modelName,
		Messages: messages,
		Stream:   true,
		Tools:    c.toolSchemaProvider(),
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	if c.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.handleHTTPError(resp)
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

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		jsonData := strings.TrimPrefix(line, "data: ")

		if jsonData == "[DONE]" {
			break
		}

		var streamResponse OpenAIStreamResponse
			if err := json.Unmarshal([]byte(jsonData), &streamResponse); err != nil {
				c.debugLog("Failed to parse response line: %q, error: %v", jsonData, err)
				if c.debug {
					slog.Error("Stream parse error", "json", jsonData, "error", err)
				}
				continue
			}

		for _, choice := range streamResponse.Choices {
			if choice.Delta.Content != "" {
				fmt.Print(choice.Delta.Content)
				currentMessage.Content += choice.Delta.Content
				c.debugLog("Streaming chunk received: %q", choice.Delta.Content)
			}

			if len(choice.Delta.ToolCalls) > 0 {
				currentMessage.ToolCalls = append(currentMessage.ToolCalls, choice.Delta.ToolCalls...)
				hasToolCalls = true
			}

			if choice.FinishReason == "stop" || choice.FinishReason == "tool_calls" {
				fmt.Println()
				currentMessage.Role = "assistant"
				c.debugLog("Streaming complete. Final message content: %q", currentMessage.Content)

				if hasToolCalls {
					c.contextManager.AddMessage(currentMessage)
					c.debugLog("Found %d tool calls, processing...", len(currentMessage.ToolCalls))
					return c.handleToolCalls(currentMessage)
				}

				c.contextManager.AddMessage(currentMessage)
				return nil
			}
		}
	}

	if !hasToolCalls {
		currentMessage.Role = "assistant"
		c.contextManager.AddMessage(currentMessage)
	}

	return nil
}

func (c *OpenAIClient) handleToolCalls(assistantMessage ChatMessage) error {
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

func (c *OpenAIClient) truncateToolResult(result, toolName string) string {
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

func (c *OpenAIClient) ClearContext() {
	c.contextManager.Clear()
}

func (c *OpenAIClient) GetStats() (int, int, int) {
	return c.contextManager.GetStats()
}

func (c *OpenAIClient) CanCompact() bool {
	return c.contextManager.CanCompact()
}

func (c *OpenAIClient) CompactContext() error {
	return c.contextManager.CompactContext(c.CompactMessages)
}

func (c *OpenAIClient) CompactMessages(messages []ChatMessage) (string, error) {
	compactPrompt := `Please provide a concise summary of the conversation above. Focus on:
1. Key decisions made
2. Important code changes or implementations
3. Current task status
4. Any unresolved issues

Keep the summary under 200 words while preserving essential context.`

	compactMessages := append(messages, ChatMessage{
		Role:    "user",
		Content: compactPrompt,
	})

	request := OpenAIRequest{
		Model:    c.modelName,
		Messages: compactMessages,
		Stream:   false,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	if c.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		httpErr := c.handleHTTPError(resp)
		return "", httpErr
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response body: %v", err)
	}

	var apiResponse OpenAIResponse
	if err := json.Unmarshal(bodyBytes, &apiResponse); err != nil {
		return "", c.handleParseError(err, bodyBytes)
	}

	if len(apiResponse.Choices) == 0 {
		return "", fmt.Errorf("no response choices returned")
	}

	return apiResponse.Choices[0].Message.Content, nil
}

func (c *OpenAIClient) EnablePlanMode() {
	c.planMode = true
	c.contextManager.SetPlanMode(true)
}

func (c *OpenAIClient) DisablePlanMode() {
	c.planMode = false
	c.contextManager.SetPlanMode(false)
}

func (c *OpenAIClient) IsInPlanMode() bool {
	return c.planMode
}

func (c *OpenAIClient) EnableConciseMode() {
	c.conciseMode = true
	c.contextManager.SetConciseMode(true)
}

func (c *OpenAIClient) DisableConciseMode() {
	c.conciseMode = false
	c.contextManager.SetConciseMode(false)
}

func (c *OpenAIClient) IsInConciseMode() bool {
	return c.conciseMode
}

func (c *OpenAIClient) SetActiveTask(task string) {
	c.contextManager.SetActiveTask(task)
}

func (c *OpenAIClient) GetActiveTask() string {
	if task := c.contextManager.GetActiveTask(); task != nil {
		return task.Goal
	}
	return ""
}

func (c *OpenAIClient) CompleteCurrentTask() {
	c.contextManager.CompleteCurrentTask("")
}

func (c *OpenAIClient) Interrupt() {
	select {
	case c.interruptChan <- struct{}{}:
	default:
	}
}

func (c *OpenAIClient) IsResponseActive() bool {
	return c.responseActive
}

func (c *OpenAIClient) DetectContextWindow() (int, error) {
	switch strings.ToLower(c.modelName) {
	case "gpt-4", "gpt-4-turbo":
		return 128000, nil
	case "gpt-3.5-turbo":
		return 16385, nil
	case "gemini-2.5-pro":
		return 1000000, nil
	default:
		return 16000, nil
	}
}

func (c *OpenAIClient) SetDebug(enabled bool) {
	c.debug = enabled
	if enabled {
		slog.Info("Debug logging enabled for OpenAI client")
	}
}

func (c *OpenAIClient) debugLog(format string, args ...interface{}) {
	if c.debug {
		slog.Debug(fmt.Sprintf(format, args...))
	}
}

func (c *OpenAIClient) handleHTTPError(resp *http.Response) error {
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("API returned status %s and failed to read response body: %v", resp.Status, err)
	}

	bodyStr := string(bodyBytes)

	if resp.StatusCode != http.StatusOK {
		slog.Error("API error", "status", resp.Status, "error", bodyStr)
		return fmt.Errorf("API returned status %s: %s", resp.Status, bodyStr)
	}

	if c.debug {
		slog.Error("Parse error", "response", bodyStr)
	}
	return fmt.Errorf("failed to parse API response")
}

func (c *OpenAIClient) handleParseError(parseErr error, responseBody []byte) error {
	if c.debug {
		slog.Error("Parse error", "error", parseErr, "response", string(responseBody))
	} else {
		slog.Error("API error (parse failed)", "response", string(responseBody))
	}
	return fmt.Errorf("error decoding response: %v", parseErr)
}
