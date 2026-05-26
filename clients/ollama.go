package clients

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// OllamaClient implements the LLMClient interface for Ollama API
type OllamaClient struct {
	*baseClient
	baseURL string
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
		baseClient: &baseClient{
			modelName:          modelName,
			toolExecutor:       toolExec,
			toolSchemaProvider: toolProvider,
			contextManager:     ctxMgr,
			interruptChan:      make(chan struct{}),
			httpClient: &http.Client{
				Timeout: 5 * time.Minute, // Hard timeout for streaming
			},
		},
		baseURL: baseURL,
	}
}

func (c *OllamaClient) planMode() bool { return c.IsInPlanMode() }

// StreamChat implements LLMClient interface
func (c *OllamaClient) StreamChat(userInput string) error {
	slog.Debug("StreamChat started", "user_input_length", len(userInput), "user_input_preview", truncateString(userInput, 200))

	userMessage := ChatMessage{
		Role:    "user",
		Content: userInput,
	}
	c.contextManager.AddMessage(userMessage)
	_, msgs, _ := c.contextManager.GetStats()
	slog.Debug("User message added to context", "total_messages", msgs)

	messages := c.contextManager.GetMessages()
	slog.Debug("Retrieved messages for API call", "message_count", len(messages))
	return c.StreamChatWithHistory(messages)
}

// StreamChatWithHistory implements LLMClient interface
func (c *OllamaClient) StreamChatWithHistory(messages []ChatMessage) error {
	currentTokens, messageCount, maxTokens := c.contextManager.GetStats()
	slog.Debug("Starting API request", "model", c.modelName, "message_count", len(messages),
		"current_tokens", currentTokens, "max_tokens", maxTokens)
	fmt.Printf("[Context: %d/%d tokens, %d messages]\n", currentTokens, maxTokens, messageCount-1)

	slog.Debug("Building request with all messages", "message_count", len(messages))
	for i, msg := range messages {
		slog.Debug("Message in request", "index", i, "role", msg.Role, "content_length", len(msg.Content),
			"tool_calls_count", len(msg.ToolCalls))
		if msg.Role == "user" || msg.Role == "assistant" {
			slog.Debug("Message content preview", "index", i, "role", msg.Role, "preview", truncateString(msg.Content, 300))
		}
		if len(msg.ToolCalls) > 0 {
			slog.Debug("Tool calls in message", "index", i, "tool_calls", msg.ToolCalls)
		}
	}

	tools := c.toolSchemaProvider()
	slog.Debug("Retrieved tool schemas", "tool_count", len(tools))
	for _, tool := range tools {
		slog.Debug("Tool schema", "tool_name", tool.Function.Name, "tool_description", truncateString(tool.Function.Description, 200),
			"arguments", tool.Function.Arguments)
	}

	request := OllamaRequest{
		Model:    c.modelName,
		Messages: messages,
		Stream:   true,
		Tools:    tools,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		slog.Error("Failed to marshal request", "error", err)
		return fmt.Errorf("error marshaling request: %v", err)
	}

	slog.Debug("Request JSON size", "bytes", len(jsonData))
	if c.debug {
		c.debugLog("Full request JSON: %s", string(jsonData))
	}

	slog.Debug("Sending HTTP POST request", "url", c.baseURL+"/api/chat")
	resp, err := c.httpClient.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		slog.Error("HTTP request failed", "error", err)
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	slog.Debug("Received HTTP response", "status", resp.Status, "status_code", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			slog.Error("Failed to read error response body", "error", err)
			return fmt.Errorf("API returned status %s and failed to read error details: %v", resp.Status, err)
		}
		errorBody := string(bodyBytes)
		slog.Error("API returned error response", "status", resp.Status, "error_body", errorBody)

		if resp.StatusCode == http.StatusBadRequest {
			slog.Error("Ollama API error", "status", resp.Status, "error", errorBody)
		}

		return fmt.Errorf("API returned status: %s, details: %s", resp.Status, errorBody)
	}

	slog.Debug("HTTP request successful, starting response stream")
	c.responseActive.Store(true)
	defer func() {
		c.responseActive.Store(false)
		slog.Debug("Response stream ended")
	}()

	scanner := bufio.NewScanner(resp.Body)
	var currentMessage ChatMessage
	var hasToolCalls bool
	var totalChunks int

	for scanner.Scan() {
		totalChunks++
		select {
		case <-c.interruptChan:
			slog.Debug("Interrupt received during streaming")
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
			slog.Debug("Failed to parse response line", "error", err, "line_preview", truncateString(line, 200))
			c.debugLog("Failed to parse response line: %q, error: %v", line, err)
			continue
		}

		slog.Debug("Parsed response chunk", "done", chatResponse.Done, "content_length", len(chatResponse.Message.Content),
			"tool_calls_count", len(chatResponse.Message.ToolCalls))

		if chatResponse.Message.Content != "" {
			slog.Debug("Streaming content chunk", "content", chatResponse.Message.Content)
			fmt.Print(chatResponse.Message.Content)
			currentMessage.Content += chatResponse.Message.Content
			c.debugLog("Streaming chunk received: %q", chatResponse.Message.Content)
		}

		if len(chatResponse.Message.ToolCalls) > 0 {
			slog.Debug("Tool calls detected in response", "tool_calls", chatResponse.Message.ToolCalls)
			currentMessage.ToolCalls = mergeToolCallDeltas(currentMessage.ToolCalls, chatResponse.Message.ToolCalls)
			hasToolCalls = true
			for _, tc := range chatResponse.Message.ToolCalls {
				slog.Debug("Tool call details", "tool_id", tc.ID, "tool_name", tc.Function.Name,
					"tool_args", tc.Function.Arguments)
			}
		}

		if chatResponse.Done {
			slog.Debug("Streaming complete", "total_chunks", totalChunks, "final_content_length", len(currentMessage.Content),
				"has_tool_calls", hasToolCalls, "tool_calls_count", len(currentMessage.ToolCalls))
			if chatResponse.PromptEvalCount > 0 {
				c.contextManager.UpdateTokenCount(chatResponse.PromptEvalCount)
			}
			fmt.Println()
			currentMessage.Role = "assistant"
			c.debugLog("Streaming complete. Final message content: %q", currentMessage.Content)
			break
		}
	}

	slog.Debug("Stream processing finished", "total_chunks_processed", totalChunks, "has_tool_calls", hasToolCalls)

	if hasToolCalls {
		slog.Debug("Processing tool calls", "tool_calls_count", len(currentMessage.ToolCalls))
		c.contextManager.AddMessage(currentMessage)
		c.debugLog("Found %d tool calls, processing...", len(currentMessage.ToolCalls))
		return c.handleToolCalls(c, currentMessage)
	}

	slog.Debug("No tool calls, adding assistant message to context")
	c.contextManager.AddMessage(currentMessage)
	return nil
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

