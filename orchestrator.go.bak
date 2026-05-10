package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type OllamaClient struct {
	baseURL        string
	client         *http.Client
	contextManager *ContextManager
	planMode       bool
	modelName      string
	debug          bool
	parser         *ToolCallParser
	conciseMode    bool
	responseActive bool
	interruptChan  chan struct{}
}

type ChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	Tools    []Tool        `json:"tools,omitempty"`
}

type ChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt time.Time   `json:"created_at"`
	Message   ChatMessage `json:"message"`
	Done      bool        `json:"done"`
}

type ModelShowRequest struct {
	Model string `json:"model"`
}

type ModelShowResponse struct {
	ModelInfo map[string]interface{} `json:"model_info"`
}

// debugLog prints debug messages when debug mode is enabled
func (c *OllamaClient) debugLog(format string, args ...interface{}) {
	if c.debug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

// SetDebug enables or disables debug logging
func (c *OllamaClient) SetDebug(enabled bool) {
	c.debug = enabled
	c.parser = NewToolCallParser(enabled) // Update parser with debug setting
	if enabled {
		fmt.Println("[DEBUG] Debug logging enabled")
	}
}

// getKeys returns the keys of a map for debugging
func getKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func NewOllamaClient(baseURL, modelName string) *OllamaClient {
	client := &OllamaClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 0, // No timeout for streaming responses
		},
		contextManager: NewContextManager(4000), // Default fallback
		modelName:      modelName,
		parser:         NewToolCallParser(false), // Will be updated in SetDebug
		interruptChan:  make(chan struct{}),
	}
	
	// Try to detect the actual context window
	if contextWindow, err := client.DetectContextWindow(); err == nil {
		optimalLimit := calculateOptimalContextLimit(contextWindow)
		client.contextManager.SetMaxTokens(optimalLimit)
		fmt.Printf("✓ Detected context window: %d tokens\n", contextWindow)
		fmt.Printf("✓ Set context limit: %d tokens (75%% utilization)\n", optimalLimit)
	} else {
		fmt.Printf("⚠️ Could not detect context window: %v\n", err)
		fmt.Printf("✓ Using default context limit: 4,000 tokens\n")
	}
	
	return client
}

func (c *OllamaClient) StreamChat(userInput string) error {
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

func (c *OllamaClient) StreamChatWithHistory(messages []ChatMessage) error {
	// Display context stats
	currentTokens, messageCount, maxTokens := c.contextManager.GetStats()
	fmt.Printf("[Context: %d/%d tokens, %d messages]\n", currentTokens, maxTokens, messageCount-1) // -1 for system prompt
	
	c.debugLog("Sending %d messages to model %s", len(messages), c.modelName)
	for i, msg := range messages {
		c.debugLog("Message %d: role=%s, content_length=%d, tool_calls=%d", 
			i+1, msg.Role, len(msg.Content), len(msg.ToolCalls))
		if c.debug && len(msg.Content) > 0 {
			// Show full content in debug
			c.debugLog("  Content: %s", msg.Content)
		}
	}
	
	request := ChatRequest{
		Model:    c.modelName,
		Messages: messages,
		Stream:   true,
		Tools:    GetAvailableTools(),
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := c.client.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API returned status: %s", resp.Status)
	}

	// Mark response as active for interrupt handling
	c.SetResponseActive(true)
	defer c.SetResponseActive(false)

	var currentMessage ChatMessage
	var hasToolCalls bool

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		// Check for interrupt signal
		select {
		case <-c.interruptChan:
			fmt.Println("\n[Response interrupted by user]")
			return fmt.Errorf("response interrupted by user")
		default:
			// Continue normal streaming
		}
		
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chatResponse ChatResponse
		if err := json.Unmarshal([]byte(line), &chatResponse); err != nil {
			continue
		}

		// Display content as it streams
		if chatResponse.Message.Content != "" {
			fmt.Print(chatResponse.Message.Content)
			currentMessage.Content += chatResponse.Message.Content
			c.debugLog("Streaming chunk received: %q", chatResponse.Message.Content)
			c.debugLog("Current total content length: %d", len(currentMessage.Content))
		}

		// Check for tool calls
		if len(chatResponse.Message.ToolCalls) > 0 {
			currentMessage.ToolCalls = append(currentMessage.ToolCalls, chatResponse.Message.ToolCalls...)
			hasToolCalls = true
		}

		if chatResponse.Done {
			fmt.Println()
			currentMessage.Role = "assistant"
			c.debugLog("Streaming complete. Final message content: %q", currentMessage.Content)
			c.debugLog("Final message length: %d characters", len(currentMessage.Content))
			break
		}
	}

	// Handle tool calls if present
	if hasToolCalls {
		// Add assistant message to context manager
		c.contextManager.AddMessage(currentMessage)
		return c.handleToolCalls(currentMessage)
	} else if currentMessage.Content != "" {
		c.debugLog("No native tool calls found, checking content for JSON tool calls")
		c.debugLog("About to parse content with length %d: %q", len(currentMessage.Content), currentMessage.Content)
		// Check for tool calls in content (fallback for models that don't use native tool calling)
		contentToolCalls := c.parser.ParseToolCallsFromContent(currentMessage.Content)
		if len(contentToolCalls) > 0 {
			c.debugLog("Found %d tool calls in content", len(contentToolCalls))
			// Add tool calls to the message and handle them
			currentMessage.ToolCalls = contentToolCalls
			c.contextManager.AddMessage(currentMessage)
			return c.handleToolCalls(currentMessage)
		}
		c.debugLog("No tool calls found, treating as regular response")
		// Add regular assistant response to context
		c.contextManager.AddMessage(currentMessage)
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading response: %v", err)
	}

	return nil
}

func (c *OllamaClient) handleToolCalls(assistantMessage ChatMessage) error {
	fmt.Println("\n🔧 Executing tools...")
	c.debugLog("Handling %d tool calls", len(assistantMessage.ToolCalls))
	
	// Execute each tool call
	for i, toolCall := range assistantMessage.ToolCalls {
		fmt.Printf("Calling %s...\n", toolCall.Function.Name)
		c.debugLog("Tool call %d: %s with args %v", i+1, toolCall.Function.Name, toolCall.Function.Arguments)
		
		result, err := ExecuteToolWithPlanMode(toolCall, c.planMode)
		if err != nil {
			result = fmt.Sprintf("Error: %v", err)
			c.debugLog("Tool execution failed: %v", err)
		} else {
			c.debugLog("Tool result length: %d characters", len(result))
		}
		
		// Truncate large tool results to prevent context overflow
		truncatedResult := c.truncateToolResult(result, toolCall.Function.Name)
		
		// Add tool result to context manager
		toolMessage := ChatMessage{
			Role:    "tool",
			Content: truncatedResult,
		}
		c.contextManager.AddMessage(toolMessage)
		c.debugLog("Added tool result to context with role 'tool' (original: %d chars, truncated: %d chars)", 
			len(result), len(truncatedResult))
		
		fmt.Printf("✓ %s completed\n", toolCall.Function.Name)
	}
	
	fmt.Println("\n🤖 Assistant response:")
	
	// Continue conversation with updated context
	messages := c.contextManager.GetMessages()
	c.debugLog("Continuing conversation with %d messages after tool execution", len(messages))
	return c.StreamChatWithHistory(messages)
}

func (c *OllamaClient) ClearContext() {
	c.contextManager.Clear()
}

func (c *OllamaClient) GetContextStats() (int, int, int) {
	return c.contextManager.GetStats()
}

func (c *OllamaClient) SetMaxTokens(maxTokens int) {
	c.contextManager.SetMaxTokens(maxTokens)
}

func (c *OllamaClient) CompactMessages(messages []ChatMessage) (string, error) {
	// Create compacting prompt
	prompt := c.contextManager.CreateCompactingPrompt(messages)
	
	// Build the messages for summarization
	var summaryMessages []ChatMessage
	
	// Add system prompt for summarization
	systemPrompt := ChatMessage{
		Role:    "system",
		Content: "You are a helpful assistant that creates concise conversation summaries. Focus on preserving key information, decisions, and context while being brief.",
	}
	summaryMessages = append(summaryMessages, systemPrompt)
	
	// Add the compacting prompt as user message
	summaryMessages = append(summaryMessages, ChatMessage{
		Role:    "user",
		Content: prompt,
	})
	
	// Add all messages to compact
	for _, msg := range messages {
		summaryMessages = append(summaryMessages, msg)
	}
	
	// Add final instruction
	summaryMessages = append(summaryMessages, ChatMessage{
		Role:    "user",
		Content: "SUMMARY:",
	})
	
	// Create request for summarization
	request := ChatRequest{
		Model:    c.modelName,
		Messages: summaryMessages,
		Stream:   false, // We want complete response
	}
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}
	
	resp, err := c.client.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
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
	
	if !chatResponse.Done {
		return "", fmt.Errorf("incomplete response received")
	}
	
	return chatResponse.Message.Content, nil
}

func (c *OllamaClient) CompactContext() error {
	return c.contextManager.CompactContext(c)
}

func (c *OllamaClient) CanCompact() bool {
	return c.contextManager.CanCompact()
}

func (c *OllamaClient) EnablePlanMode() {
	c.planMode = true
	// Update context manager with plan mode system prompt
	c.contextManager.SetPlanMode(true)
}

func (c *OllamaClient) DisablePlanMode() {
	c.planMode = false
	// Update context manager with normal system prompt
	c.contextManager.SetPlanMode(false)
}

func (c *OllamaClient) IsInPlanMode() bool {
	return c.planMode
}

func (c *OllamaClient) SetConciseMode(enabled bool) {
	c.conciseMode = enabled
	// Update context manager with concise mode setting
	c.contextManager.SetConciseMode(enabled)
}

func (c *OllamaClient) IsInConciseMode() bool {
	return c.conciseMode
}

func (c *OllamaClient) SetResponseActive(active bool) {
	c.responseActive = active
}

func (c *OllamaClient) IsResponseActive() bool {
	return c.responseActive
}

func (c *OllamaClient) InterruptResponse() {
	select {
	case c.interruptChan <- struct{}{}:
		// Interrupt signal sent
	default:
		// Channel is full, interrupt already pending
	}
}

func (c *OllamaClient) truncateToolResult(result, toolName string) string {
	maxChars := 3000 // Maximum characters for tool results in context
	
	if len(result) <= maxChars {
		return result
	}
	
	// Different truncation strategies based on tool type
	switch toolName {
	case "read_file":
		// For file contents, show beginning and end with indicator
		prefixLength := maxChars / 2
		suffixLength := maxChars / 4
		
		prefix := result[:prefixLength]
		suffix := result[len(result)-suffixLength:]
		
		// Count lines for better context
		prefixLines := strings.Count(prefix, "\n")
		totalLines := strings.Count(result, "\n")
		
		return fmt.Sprintf("%s\n\n... [FILE TRUNCATED: %d lines shown, %d total lines, %d chars] ...\n\n%s",
			prefix, prefixLines, totalLines, len(result), suffix)
			
	case "list_files", "find_files":
		// For file lists, truncate with count
		lines := strings.Split(result, "\n")
		if len(lines) > 100 {
			truncated := strings.Join(lines[:100], "\n")
			return fmt.Sprintf("%s\n\n... [LIST TRUNCATED: showing 100 of %d items] ...", 
				truncated, len(lines))
		}
		return result
		
	case "grep_content":
		// For grep results, show first results with count
		lines := strings.Split(result, "\n")
		if len(lines) > 50 {
			truncated := strings.Join(lines[:50], "\n")
			return fmt.Sprintf("%s\n\n... [SEARCH RESULTS TRUNCATED: showing 50 of %d matches] ...", 
				truncated, len(lines))
		}
		return result
		
	default:
		// Generic truncation for other tools
		return result[:maxChars] + "\n\n... [OUTPUT TRUNCATED] ..."
	}
}

func (c *OllamaClient) GetActiveTask() *UserTask {
	return c.contextManager.GetActiveTask()
}

func (c *OllamaClient) SetActiveTask(goal string) {
	c.contextManager.SetActiveTask(goal)
}

func (c *OllamaClient) CompleteCurrentTask(summary string) {
	c.contextManager.CompleteCurrentTask(summary)
}

func (c *OllamaClient) DetectContextWindow() (int, error) {
	request := ModelShowRequest{
		Model: c.modelName,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return 0, fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := c.client.Post(c.baseURL+"/api/show", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return 0, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("API returned status: %s", resp.Status)
	}

	var showResponse ModelShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&showResponse); err != nil {
		return 0, fmt.Errorf("error decoding response: %v", err)
	}

	// Look for any key ending with ".context_length" (dynamic detection)
	for key, value := range showResponse.ModelInfo {
		if strings.HasSuffix(key, ".context_length") {
			if contextLength, ok := value.(float64); ok {
				fmt.Printf("🔍 Found context length key: %s = %d\n", key, int(contextLength))
				return int(contextLength), nil
			}
		}
	}

	// Fallback: try some generic field names
	fallbackFields := []string{
		"context_length", 
		"max_context_length",
		"num_ctx_max",
	}

	for _, field := range fallbackFields {
		if value, exists := showResponse.ModelInfo[field]; exists {
			if contextLength, ok := value.(float64); ok {
				fmt.Printf("🔍 Found context length (fallback): %s = %d\n", field, int(contextLength))
				return int(contextLength), nil
			}
		}
	}

	return 0, fmt.Errorf("context length not found in model info")
}

func calculateOptimalContextLimit(maxContext int) int {
	// Reserve 25% for responses and JSON overhead
	usableContext := int(float64(maxContext) * 0.75)
	
	// Minimum safety limit
	if usableContext < 2000 {
		return 2000
	}
	
	// Maximum practical limit (avoid memory issues)
	if usableContext > 50000 {
		return 50000
	}
	
	return usableContext
}