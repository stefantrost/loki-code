package clients

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAIClient implements the LLMClient interface for OpenAI-compatible APIs
type OpenAIClient struct {
	baseURL        string
	bearerToken    string
	client         *http.Client
	contextManager ContextManager
	planMode       bool
	modelName      string
	debug          bool
	conciseMode    bool
	responseActive bool
	interruptChan  chan struct{}
}

// OpenAI-compatible request/response structures
type OpenAIRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`
	Tools    []Tool        `json:"tools,omitempty"`
}

// LocalMind-specific request structures
type LocalMindRequest struct {
	ApiKey string        `json:"api_key"`
	Chat   LocalMindChat `json:"chat"`
	Model  string        `json:"model,omitempty"`
	Stream bool          `json:"stream,omitempty"`
}

type LocalMindChat struct {
	Messages []ChatMessage `json:"messages"`
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
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
}

// NewOpenAIClient creates a new OpenAI-compatible client
func NewOpenAIClient(baseURL, bearerToken, modelName string) *OpenAIClient {
	client := &OpenAIClient{
		baseURL:     baseURL,
		bearerToken: bearerToken,
		client: &http.Client{
			Timeout: 0, // No timeout for streaming responses
		},
		contextManager: mainPackage.NewContextManager(4000), // Default context limit
		modelName:      modelName,
		interruptChan:  make(chan struct{}),
	}
	
	// OpenAI-compatible APIs typically have larger context windows
	// Set a reasonable default - this could be configurable
	client.contextManager.SetMaxTokens(16000) // Conservative default
	fmt.Printf("✓ OpenAI client initialized with model: %s\n", modelName)
	fmt.Printf("✓ Context limit: 16,000 tokens\n")
	
	return client
}

// StreamChat implements LLMClient interface
func (c *OpenAIClient) StreamChat(userInput string) error {
	// Add user message to context
	userMessage := ChatMessage{
		Role:    "user",
		Content: userInput,
	}
	c.contextManager.AddMessage(userMessage)
	
	// Get all messages including system prompt
	messages := c.contextManager.GetMessages()
	return c.StreamChatWithHistory(messages)
}

// StreamChatWithHistory implements LLMClient interface
func (c *OpenAIClient) StreamChatWithHistory(messages []ChatMessage) error {
	// Display context stats
	currentTokens, messageCount, maxTokens := c.contextManager.GetStats()
	fmt.Printf("[Context: %d/%d tokens, %d messages]\n", currentTokens, maxTokens, messageCount-1) // -1 for system prompt
	
	c.debugLog("Sending %d messages to model %s", len(messages), c.modelName)
	for i, msg := range messages {
		c.debugLog("Message %d: role=%s, content_length=%d, tool_calls=%d", 
			i+1, msg.Role, len(msg.Content), len(msg.ToolCalls))
		if c.debug && len(msg.Content) > 0 {
			c.debugLog("  Content: %s", msg.Content)
		}
	}

	// Build request based on API type
	var jsonData []byte
	var err error
	
	if c.isLocalMindAPI() {
		// Build LocalMind-specific request
		request := LocalMindRequest{
			ApiKey: c.bearerToken,
			Chat: LocalMindChat{
				Messages: messages,
				Tools:    mainPackage.GetAvailableTools(),
			},
			Model:  c.modelName,
			Stream: true,
		}
		c.debugLog("Building LocalMind API request format")
		jsonData, err = json.Marshal(request)
	} else {
		// Build standard OpenAI-compatible request
		request := OpenAIRequest{
			Model:    c.modelName,
			Messages: messages,
			Stream:   true,
			Tools:    mainPackage.GetAvailableTools(),
		}
		c.debugLog("Building OpenAI API request format")
		jsonData, err = json.Marshal(request)
	}
	
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	// Create HTTP request with authentication
	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error creating request: %v", err)
	}

	// Add headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	
	// For non-LocalMind APIs, use Authorization header
	if c.bearerToken != "" && !c.isLocalMindAPI() {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}
	// For LocalMind, the api_key is already in the request body

	// Execute request
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.handleHTTPError(resp)
	}

	// Process streaming response
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

		// OpenAI streaming format uses "data: " prefix
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		
		// Remove "data: " prefix
		jsonData := strings.TrimPrefix(line, "data: ")
		
		// Check for end of stream
		if jsonData == "[DONE]" {
			break
		}

		var streamResponse OpenAIStreamResponse
		if err := json.Unmarshal([]byte(jsonData), &streamResponse); err != nil {
			c.debugLog("Failed to parse response line: %q, error: %v", jsonData, err)
			if c.debug {
				fmt.Printf("❌ Stream Parse Error - Raw JSON: %s\n", jsonData)
			}
			continue
		}

		// Process choices (typically only one choice)
		for _, choice := range streamResponse.Choices {
			// Handle delta content (streaming)
			if choice.Delta.Content != "" {
				fmt.Print(choice.Delta.Content)
				currentMessage.Content += choice.Delta.Content
				c.debugLog("Streaming chunk received: %q", choice.Delta.Content)
			}

			// Handle tool calls
			if len(choice.Delta.ToolCalls) > 0 {
				currentMessage.ToolCalls = append(currentMessage.ToolCalls, choice.Delta.ToolCalls...)
				hasToolCalls = true
			}

			// Check for completion
			if choice.FinishReason == "stop" || choice.FinishReason == "tool_calls" {
				fmt.Println()
				currentMessage.Role = "assistant"
				c.debugLog("Streaming complete. Final message content: %q", currentMessage.Content)
				c.debugLog("Final message length: %d characters", len(currentMessage.Content))
				
				// Process tool calls if present
				if hasToolCalls {
					c.contextManager.AddMessage(currentMessage)
					c.debugLog("Found %d tool calls, processing...", len(currentMessage.ToolCalls))
					return c.handleToolCalls(currentMessage)
				}
				
				// Add the final assistant message to context
				c.contextManager.AddMessage(currentMessage)
				return nil
			}
		}
	}

	// Add the final assistant message to context if not already added
	if !hasToolCalls {
		currentMessage.Role = "assistant"
		c.contextManager.AddMessage(currentMessage)
	}

	return nil
}

// handleToolCalls processes tool calls from the assistant (same logic as Ollama)
func (c *OpenAIClient) handleToolCalls(assistantMessage ChatMessage) error {
	for _, toolCall := range assistantMessage.ToolCalls {
		c.debugLog("Processing tool call: %s with function %s", toolCall.ID, toolCall.Function.Name)
		c.debugLog("Tool arguments: %v", getKeys(toolCall.Function.Arguments))
		
		fmt.Printf("🔧 Executing tools...\n")
		fmt.Printf("Calling %s...\n", toolCall.Function.Name)
		
		// Execute the tool
		result, err := mainPackage.ExecuteToolWithPlanMode(toolCall, c.planMode)
		if err != nil {
			result = fmt.Sprintf("Error: %v", err)
		}
		
		// Truncate long results to prevent context overflow
		result = c.truncateToolResult(result, toolCall.Function.Name)
		
		c.debugLog("Tool %s result length: %d characters", toolCall.Function.Name, len(result))
		
		fmt.Printf("✓ %s completed\n", toolCall.Function.Name)
		
		// Add tool result as a message
		toolMessage := ChatMessage{
			Role:    "tool",
			Content: result,
		}
		c.contextManager.AddMessage(toolMessage)
	}

	// Continue conversation with tool results
	messages := c.contextManager.GetMessages()
	return c.StreamChatWithHistory(messages)
}

// truncateToolResult - same implementation as Ollama client
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

// ClearContext implements LLMClient interface
func (c *OpenAIClient) ClearContext() {
	c.contextManager.ClearMessages()
}

// GetStats implements LLMClient interface
func (c *OpenAIClient) GetStats() (int, int, int) {
	return c.contextManager.GetStats()
}

// CanCompact implements LLMClient interface
func (c *OpenAIClient) CanCompact() bool {
	return c.contextManager.CanCompact()
}

// CompactContext implements LLMClient interface
func (c *OpenAIClient) CompactContext() error {
	return c.contextManager.CompactContext(c.CompactMessages)
}

// CompactMessages compacts conversation using OpenAI API
func (c *OpenAIClient) CompactMessages(messages []ChatMessage) (string, error) {
	compactPrompt := `Please provide a concise summary of the conversation above. Focus on:
1. Key decisions made
2. Important code changes or implementations
3. Current task status
4. Any unresolved issues

Keep the summary under 200 words while preserving essential context.`

	// Create a non-streaming request for compaction
	compactMessages := append(messages, ChatMessage{
		Role:    "user",
		Content: compactPrompt,
	})
	
	var jsonData []byte
	var err error
	
	if c.isLocalMindAPI() {
		// Build LocalMind-specific request
		request := LocalMindRequest{
			ApiKey: c.bearerToken,
			Chat: LocalMindChat{
				Messages: compactMessages,
			},
			Model:  c.modelName,
			Stream: false,
		}
		jsonData, err = json.Marshal(request)
	} else {
		// Build standard OpenAI-compatible request
		request := OpenAIRequest{
			Model:    c.modelName,
			Messages: compactMessages,
			Stream:   false,
		}
		jsonData, err = json.Marshal(request)
	}
	
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	// Create authenticated request
	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	
	// For non-LocalMind APIs, use Authorization header
	if c.bearerToken != "" && !c.isLocalMindAPI() {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}
	// For LocalMind, the api_key is already in the request body

	resp, err := c.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		httpErr := c.handleHTTPError(resp)
		return "", httpErr
	}

	// Read response body for better error handling
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

// EnablePlanMode implements LLMClient interface
func (c *OpenAIClient) EnablePlanMode() {
	c.planMode = true
	c.contextManager.EnablePlanMode()
}

// DisablePlanMode implements LLMClient interface
func (c *OpenAIClient) DisablePlanMode() {
	c.planMode = false
	c.contextManager.DisablePlanMode()
}

// IsInPlanMode implements LLMClient interface
func (c *OpenAIClient) IsInPlanMode() bool {
	return c.planMode
}

// EnableConciseMode implements LLMClient interface
func (c *OpenAIClient) EnableConciseMode() {
	c.conciseMode = true
	c.contextManager.EnableConciseMode()
}

// DisableConciseMode implements LLMClient interface
func (c *OpenAIClient) DisableConciseMode() {
	c.conciseMode = false
	c.contextManager.DisableConciseMode()
}

// IsInConciseMode implements LLMClient interface
func (c *OpenAIClient) IsInConciseMode() bool {
	return c.conciseMode
}

// SetActiveTask implements LLMClient interface
func (c *OpenAIClient) SetActiveTask(task string) {
	c.contextManager.SetActiveTask(task)
}

// GetActiveTask implements LLMClient interface
func (c *OpenAIClient) GetActiveTask() string {
	if task := c.contextManager.GetActiveTask(); task != nil {
		return task.Goal
	}
	return ""
}

// CompleteCurrentTask implements LLMClient interface
func (c *OpenAIClient) CompleteCurrentTask() {
	c.contextManager.CompleteCurrentTask("")
}

// Interrupt implements LLMClient interface
func (c *OpenAIClient) Interrupt() {
	select {
	case c.interruptChan <- struct{}{}:
	default:
	}
}

// IsResponseActive implements LLMClient interface
func (c *OpenAIClient) IsResponseActive() bool {
	return c.responseActive
}

// DetectContextWindow implements LLMClient interface
func (c *OpenAIClient) DetectContextWindow() (int, error) {
	// For OpenAI-compatible APIs, we'll use reasonable defaults based on common models
	// This could be made configurable or determined by API calls if the provider supports it
	switch strings.ToLower(c.modelName) {
	case "gpt-4", "gpt-4-turbo":
		return 128000, nil
	case "gpt-3.5-turbo":
		return 16385, nil
	case "gemini-2.5-pro":
		return 1000000, nil // Gemini 2.5 Pro has 1M token context window
	case "localmind-ultra":
		return 32000, nil // Based on user's example
	default:
		// Conservative default for unknown models
		return 16000, nil
	}
}

// SetDebug implements LLMClient interface
func (c *OpenAIClient) SetDebug(enabled bool) {
	c.debug = enabled
	if enabled {
		fmt.Println("[DEBUG] Debug logging enabled for OpenAI client")
	}
}

// debugLog prints debug messages when debug mode is enabled
func (c *OpenAIClient) debugLog(format string, args ...interface{}) {
	if c.debug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

// isLocalMindAPI detects if we're connecting to a LocalMind API
func (c *OpenAIClient) isLocalMindAPI() bool {
	return strings.Contains(strings.ToLower(c.baseURL), "localmind")
}

// handleHTTPError handles HTTP error responses with enhanced debug output
func (c *OpenAIClient) handleHTTPError(resp *http.Response) error {
	// Read the response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("API returned status %s and failed to read response body: %v", resp.Status, err)
	}
	
	bodyStr := string(bodyBytes)
	
	if resp.StatusCode != http.StatusOK {
		// For non-200 responses, show error in both normal and debug mode
		fmt.Printf("❌ API Error (%s): %s\n", resp.Status, bodyStr)
		return fmt.Errorf("API returned status %s: %s", resp.Status, bodyStr)
	}
	
	// For 200 responses that couldn't be parsed, show different messages
	// based on debug mode (this will be called from parse error handlers)
	if c.debug {
		fmt.Printf("❌ Parse Error - Response body: %s\n", bodyStr)
	}
	return fmt.Errorf("failed to parse API response")
}

// handleParseError handles JSON parsing errors for 200 responses
func (c *OpenAIClient) handleParseError(parseErr error, responseBody []byte) error {
	// In normal mode: show the response body (likely contains error details)
	// In debug mode: show additional debug context plus the response body
	if c.debug {
		fmt.Printf("❌ Parse Error (Debug) - Parse failed: %v\n", parseErr)
		fmt.Printf("❌ Parse Error (Debug) - Response body: %s\n", string(responseBody))
	} else {
		// Normal mode: just show the response body which likely contains the actual error
		fmt.Printf("❌ API Response: %s\n", string(responseBody))
	}
	return fmt.Errorf("error decoding response: %v", parseErr)
}

