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
	*baseClient
	baseURL     string
	bearerToken string
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
		baseClient: &baseClient{
			modelName:          modelName,
			toolExecutor:       toolExec,
			toolSchemaProvider: toolProvider,
			contextManager:     ctxMgr,
			interruptChan:      make(chan struct{}),
			httpClient: &http.Client{
				Timeout: 5 * time.Minute,
			},
		},
		baseURL:     baseURL,
		bearerToken: bearerToken,
	}
}

func (c *OpenAIClient) planMode() bool { return c.IsInPlanMode() }

// StreamChat implements LLMClient interface
func (c *OpenAIClient) StreamChat(userInput string) error {
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
func (c *OpenAIClient) StreamChatWithHistory(messages []ChatMessage) error {
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

	request := OpenAIRequest{
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

	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		slog.Error("Failed to create HTTP request", "error", err)
		return fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	if c.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
		slog.Debug("Bearer token set for authentication")
	} else {
		slog.Debug("No bearer token configured")
	}

	slog.Debug("Sending HTTP POST request", "url", c.baseURL+"/chat/completions")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		slog.Error("HTTP request failed", "error", err)
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	slog.Debug("Received HTTP response", "status", resp.Status, "status_code", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		return c.handleHTTPError(resp)
	}

	slog.Debug("HTTP request successful, starting SSE stream")
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

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		jsonData := strings.TrimPrefix(line, "data: ")

		if jsonData == "[DONE]" {
			slog.Debug("Received SSE [DONE] marker")
			break
		}

		var streamResponse OpenAIStreamResponse
		if err := json.Unmarshal([]byte(jsonData), &streamResponse); err != nil {
			slog.Debug("Failed to parse SSE line", "error", err, "line_preview", truncateString(jsonData, 200))
			c.debugLog("Failed to parse response line: %q, error: %v", jsonData, err)
			if c.debug {
				slog.Error("Stream parse error", "json", jsonData, "error", err)
			}
			continue
		}

		slog.Debug("Parsed SSE chunk", "choices_count", len(streamResponse.Choices), "finish_reasons",
			func() []string {
				reasons := make([]string, len(streamResponse.Choices))
				for i, c := range streamResponse.Choices {
					reasons[i] = c.FinishReason
				}
				return reasons
			}())

		for _, choice := range streamResponse.Choices {
			if choice.Delta.Content != "" {
				slog.Debug("Streaming content chunk", "content", choice.Delta.Content)
				fmt.Print(choice.Delta.Content)
				currentMessage.Content += choice.Delta.Content
				c.debugLog("Streaming chunk received: %q", choice.Delta.Content)
			}

			if len(choice.Delta.ToolCalls) > 0 {
				slog.Debug("Tool calls detected in SSE chunk", "tool_calls", choice.Delta.ToolCalls)
				currentMessage.ToolCalls = mergeToolCallDeltas(currentMessage.ToolCalls, choice.Delta.ToolCalls)
				hasToolCalls = true
				for _, tc := range choice.Delta.ToolCalls {
					slog.Debug("Tool call details", "tool_id", tc.ID, "tool_name", tc.Function.Name,
						"tool_args", tc.Function.Arguments)
				}
			}

			if choice.FinishReason == "stop" || choice.FinishReason == "tool_calls" {
				slog.Debug("Streaming complete", "finish_reason", choice.FinishReason, "total_chunks", totalChunks,
					"final_content_length", len(currentMessage.Content), "has_tool_calls", hasToolCalls,
					"tool_calls_count", len(currentMessage.ToolCalls))
				fmt.Println()
				currentMessage.Role = "assistant"
				c.debugLog("Streaming complete. Final message content: %q", currentMessage.Content)

				if hasToolCalls {
					slog.Debug("Tool calls detected, processing them")
					c.contextManager.AddMessage(currentMessage)
					c.debugLog("Found %d tool calls, processing...", len(currentMessage.ToolCalls))
					return c.handleToolCalls(c, currentMessage)
				}

				slog.Debug("No tool calls, adding assistant message to context")
				c.contextManager.AddMessage(currentMessage)
				return nil
			}
		}
	}

	slog.Debug("SSE stream processing finished", "total_chunks_processed", totalChunks, "has_tool_calls", hasToolCalls)

	if !hasToolCalls {
		slog.Debug("No tool calls found, adding assistant message to context")
		currentMessage.Role = "assistant"
		c.contextManager.AddMessage(currentMessage)
	}

	return nil
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
