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

// openAIOutFunc / openAIOutToolCall / openAIOutMessage are the wire types used
// when sending messages TO the API. Tool-call arguments must be a JSON string
// per the OpenAI spec; the shared ToolFunc.Arguments map is marshaled here.
type openAIOutFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
type openAIOutToolCall struct {
	ID       string        `json:"id,omitempty"`
	Type     string        `json:"type,omitempty"`
	Function openAIOutFunc `json:"function"`
}
type openAIOutMessage struct {
	Role       string              `json:"role"`
	Content    string              `json:"content,omitempty"`
	ToolCalls  []openAIOutToolCall `json:"tool_calls,omitempty"`
	ToolCallID string              `json:"tool_call_id,omitempty"`
}

// toOpenAIMessages converts the shared ChatMessage slice to the OpenAI wire
// format, marshaling each tool-call arguments map to a JSON string.
func toOpenAIMessages(msgs []ChatMessage) []openAIOutMessage {
	out := make([]openAIOutMessage, len(msgs))
	for i, m := range msgs {
		om := openAIOutMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			argsJSON, _ := json.Marshal(tc.Function.Arguments)
			om.ToolCalls = append(om.ToolCalls, openAIOutToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: openAIOutFunc{
					Name:      tc.Function.Name,
					Arguments: string(argsJSON),
				},
			})
		}
		out[i] = om
	}
	return out
}

// openAIToolCallChunk is a single tool-call delta as it arrives over SSE.
// Arguments accumulates as a raw JSON string across multiple chunks.
type openAIToolCallChunk struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// openAIDelta is the delta field inside each SSE chunk. It uses a separate
// type from ChatMessage so that Arguments stays as a string during streaming.
type openAIDelta struct {
	Role             string                `json:"role,omitempty"`
	Content          string                `json:"content,omitempty"`
	ReasoningContent string                `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIToolCallChunk `json:"tool_calls,omitempty"`
}

type openAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIRequest struct {
	Model         string               `json:"model"`
	Messages      []openAIOutMessage   `json:"messages"`
	Stream        bool                 `json:"stream,omitempty"`
	StreamOptions *openAIStreamOptions `json:"stream_options,omitempty"`
	Tools         []Tool               `json:"tools,omitempty"`
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
	Delta        openAIDelta `json:"delta,omitempty"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

type OpenAIStreamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
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
	c.notifyStreamStart()
	currentTokens, messageCount, maxTokens := c.contextManager.GetStats()
	slog.Debug("Starting API request", "model", c.modelName, "message_count", len(messages),
		"current_tokens", currentTokens, "max_tokens", maxTokens)
	_ = messageCount // consumed by agent layer via GetStats

	tools := c.toolSchemaProvider()
	slog.Debug("Retrieved tool schemas", "tool_count", len(tools))

	request := OpenAIRequest{
		Model:         c.modelName,
		Messages:      toOpenAIMessages(messages),
		Stream:        true,
		StreamOptions: &openAIStreamOptions{IncludeUsage: true},
		Tools:         tools,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		slog.Error("Failed to marshal request", "error", err)
		return fmt.Errorf("error marshaling request: %v", err)
	}

	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		slog.Error("Failed to create HTTP request", "error", err)
		return fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		slog.Error("HTTP request failed", "error", err)
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.handleHTTPError(resp)
	}

	c.responseActive.Store(true)
	defer c.responseActive.Store(false)

	// toolChunks accumulates per-index tool-call deltas. Only the first chunk
	// for a given index carries the ID and name; subsequent ones carry more
	// argument fragments. We merge by index, then parse args at the end.
	toolChunks := map[int]*openAIToolCallChunk{}

	scanner := bufio.NewScanner(resp.Body)
	var currentMessage ChatMessage
	var totalChunks int

	for scanner.Scan() {
		totalChunks++
		select {
		case <-c.interruptChan:
			fmt.Fprintln(c.getOutputWriter(), "\n[Interrupted]")
			return nil
		default:
		}

		line := scanner.Text()
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			slog.Debug("Received SSE [DONE]", "total_chunks", totalChunks)
			break
		}

		var chunk OpenAIStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			slog.Debug("Failed to parse SSE chunk", "error", err)
			continue
		}

		// Usage chunk: choices is empty, usage carries real token counts.
		if len(chunk.Choices) == 0 && chunk.Usage != nil {
			c.contextManager.UpdateTokenCount(chunk.Usage.PromptTokens)
			continue
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.ReasoningContent != "" {
				fmt.Fprint(c.getThinkingWriter(), choice.Delta.ReasoningContent)
			}
			if choice.Delta.Content != "" {
				fmt.Fprint(c.getOutputWriter(), choice.Delta.Content)
				currentMessage.Content += choice.Delta.Content
			}

			accumulateToolDeltas(toolChunks, choice.Delta.ToolCalls)

			if choice.FinishReason == "stop" || choice.FinishReason == "tool_calls" {
				fmt.Fprintln(c.getOutputWriter())
				currentMessage.Role = "assistant"
				currentMessage.ToolCalls = buildToolCalls(toolChunks)
				slog.Debug("Stream complete", "finish_reason", choice.FinishReason,
					"tool_calls", len(currentMessage.ToolCalls))

				if len(currentMessage.ToolCalls) > 0 {
					c.contextManager.AddMessage(currentMessage)
					return c.handleToolCalls(c, currentMessage)
				}
				c.contextManager.AddMessage(currentMessage)
				return nil
			} else if choice.FinishReason == "length" {
				slog.Warn("Stream truncated by context window", "finish_reason", "length",
					"partial_content_len", len(currentMessage.Content))
				fmt.Fprintf(c.getSystemWriter(), "⚠️  Response truncated (context window full) — task may be incomplete\n")
				break
			} else if choice.FinishReason != "" {
				slog.Debug("Unrecognised finish_reason", "finish_reason", choice.FinishReason)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("SSE scanner error", "error", err)
		return fmt.Errorf("stream read error: %w", err)
	}

	// [DONE] path — finish_reason may not have fired (some providers omit it).
	fmt.Fprintln(c.getOutputWriter())
	currentMessage.Role = "assistant"
	currentMessage.ToolCalls = buildToolCalls(toolChunks)
	c.contextManager.AddMessage(currentMessage)
	if len(currentMessage.ToolCalls) > 0 {
		return c.handleToolCalls(c, currentMessage)
	}
	return nil
}

// accumulateToolDeltas merges incoming tool-call delta fragments into the index-keyed accumulator.
func accumulateToolDeltas(chunks map[int]*openAIToolCallChunk, deltas []openAIToolCallChunk) {
	for _, tc := range deltas {
		acc, exists := chunks[tc.Index]
		if !exists {
			acc = &openAIToolCallChunk{Index: tc.Index}
			chunks[tc.Index] = acc
		}
		if tc.ID != "" {
			acc.ID = tc.ID
		}
		if tc.Type != "" {
			acc.Type = tc.Type
		}
		if tc.Function.Name != "" {
			acc.Function.Name = tc.Function.Name
		}
		acc.Function.Arguments += tc.Function.Arguments
	}
}

// buildToolCalls converts the index-keyed accumulator into a ToolCall slice,
// parsing each accumulated JSON-string argument into a map.
func buildToolCalls(chunks map[int]*openAIToolCallChunk) []ToolCall {
	if len(chunks) == 0 {
		return nil
	}
	calls := make([]ToolCall, 0, len(chunks))
	for i := 0; i < len(chunks); i++ {
		chunk, ok := chunks[i]
		if !ok {
			continue
		}
		var args map[string]interface{}
		if chunk.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(chunk.Function.Arguments), &args); err != nil {
				slog.Debug("Failed to parse tool call arguments", "index", i, "args", chunk.Function.Arguments, "error", err)
				args = map[string]interface{}{}
			}
		}
		calls = append(calls, ToolCall{
			ID:    chunk.ID,
			Type:  chunk.Type,
			Index: chunk.Index,
			Function: ToolFunc{
				Name:      chunk.Function.Name,
				Arguments: args,
			},
		})
	}
	return calls
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
		Messages: toOpenAIMessages(compactMessages),
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
	if n, err := c.queryContextWindowFromAPI(); err == nil && n > 0 {
		return n, nil
	}
	// Static fallback for well-known OpenAI models.
	switch strings.ToLower(c.modelName) {
	case "gpt-4", "gpt-4-turbo", "gpt-4o":
		return 128000, nil
	case "gpt-3.5-turbo":
		return 16385, nil
	case "gemini-2.5-pro":
		return 1000000, nil
	default:
		return 0, fmt.Errorf("unknown model context window for %q", c.modelName)
	}
}

// queryContextWindowFromAPI tries two discovery paths:
//  1. Standard OpenAI-compat /v1/models — some providers include context_length / context_window.
//  2. LM Studio extended /api/v0/models — returns loaded_context_length per model.
//
// Returns 0 if neither path yields a result for the configured model.
func (c *OpenAIClient) queryContextWindowFromAPI() (int, error) {
	// Path 1: standard endpoint.
	if n := c.fetchContextFromModelsEndpoint(c.baseURL + "/models"); n > 0 {
		return n, nil
	}
	// Path 2: LM Studio extended API at the same host, swapping /v1 for /api/v0.
	lmsURL := strings.TrimSuffix(c.baseURL, "/v1") + "/api/v0/models"
	if n := c.fetchContextFromModelsEndpoint(lmsURL); n > 0 {
		return n, nil
	}
	return 0, fmt.Errorf("context window not found via API")
}

// fetchContextFromModelsEndpoint GETs the given URL and looks for the configured
// model in a list response. It tries context_length, context_window,
// loaded_context_length, and max_context_length keys, in that order.
func (c *OpenAIClient) fetchContextFromModelsEndpoint(url string) int {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return 0
	}
	if c.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0
	}

	var envelope struct {
		Data []map[string]interface{} `json:"data"`
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0
	}
	if err := json.Unmarshal(body, &envelope); err != nil || len(envelope.Data) == 0 {
		return 0
	}

	for _, m := range envelope.Data {
		id, _ := m["id"].(string)
		if id != c.modelName {
			continue
		}
		for _, key := range []string{"loaded_context_length", "context_length", "context_window", "max_context_length"} {
			if v, ok := m[key]; ok {
				switch n := v.(type) {
				case float64:
					if int(n) > 0 {
						return int(n)
					}
				}
			}
		}
	}
	return 0
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
