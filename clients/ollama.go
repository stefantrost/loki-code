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

// OllamaClient implements the LLMClient interface for Ollama API
type OllamaClient struct {
	baseURL        string
	client         *http.Client
	contextManager ContextManager
	planMode       bool
	modelName      string
	debug          bool
	conciseMode    bool
	responseActive bool
	interruptChan  chan struct{}
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

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(baseURL, modelName string) *OllamaClient {
	client := &OllamaClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 0, // No timeout for streaming responses
		},
		contextManager: mainPackage.NewContextManager(4000), // Default fallback
		modelName:      modelName,
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

// StreamChat implements LLMClient interface
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

// StreamChatWithHistory implements LLMClient interface
func (c *OllamaClient) StreamChatWithHistory(messages []ChatMessage) error {
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

	// Build request
	request := OllamaRequest{
		Model:    c.modelName,
		Messages: messages,
		Stream:   true,
		Tools:    mainPackage.GetAvailableTools(),
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	if c.debug {
		c.debugLog("Full request JSON: %s", string(jsonData))
	}

	resp, err := c.client.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Read the response body to get detailed error information
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("API returned status %s and failed to read error details: %v", resp.Status, err)
		}
		errorBody := string(bodyBytes)
		
		// Show error details in both debug and normal mode for 400 errors
		if resp.StatusCode == http.StatusBadRequest {
			fmt.Printf("❌ Ollama API Error (%s): %s\n", resp.Status, errorBody)
		}
		
		return fmt.Errorf("API returned status: %s, details: %s", resp.Status, errorBody)
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

		var chatResponse ChatResponse
		if err := json.Unmarshal([]byte(line), &chatResponse); err != nil {
			c.debugLog("Failed to parse response line: %q, error: %v", line, err)
			continue
		}

		// Display content as it streams
		if chatResponse.Message.Content != "" {
			fmt.Print(chatResponse.Message.Content)
			currentMessage.Content += chatResponse.Message.Content
			c.debugLog("Streaming chunk received: %q", chatResponse.Message.Content)
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
			break
		}
	}

	// Handle tool calls if present
	if hasToolCalls {
		c.contextManager.AddMessage(currentMessage)
		c.debugLog("Found %d tool calls, processing...", len(currentMessage.ToolCalls))
		return c.handleToolCalls(currentMessage)
	}

	// Add the final assistant message to context
	c.contextManager.AddMessage(currentMessage)
	return nil
}

// handleToolCalls processes tool calls from the assistant
func (c *OllamaClient) handleToolCalls(assistantMessage ChatMessage) error {
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

// truncateToolResult truncates long tool results to prevent context overflow
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

// ClearContext implements LLMClient interface
func (c *OllamaClient) ClearContext() {
	c.contextManager.ClearMessages()
}

// GetStats implements LLMClient interface
func (c *OllamaClient) GetStats() (int, int, int) {
	return c.contextManager.GetStats()
}

// CanCompact implements LLMClient interface
func (c *OllamaClient) CanCompact() bool {
	return c.contextManager.CanCompact()
}

// CompactContext implements LLMClient interface
func (c *OllamaClient) CompactContext() error {
	return c.contextManager.CompactContext(c.CompactMessages)
}

// CompactMessages compacts conversation using Ollama API
func (c *OllamaClient) CompactMessages(messages []ChatMessage) (string, error) {
	compactPrompt := `Please provide a concise summary of the conversation above. Focus on:
1. Key decisions made
2. Important code changes or implementations
3. Current task status
4. Any unresolved issues

Keep the summary under 200 words while preserving essential context.`

	// Create a non-streaming request for compaction
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
	
	return chatResponse.Message.Content, nil
}

// EnablePlanMode implements LLMClient interface
func (c *OllamaClient) EnablePlanMode() {
	c.planMode = true
	c.contextManager.EnablePlanMode()
}

// DisablePlanMode implements LLMClient interface
func (c *OllamaClient) DisablePlanMode() {
	c.planMode = false
	c.contextManager.DisablePlanMode()
}

// IsInPlanMode implements LLMClient interface
func (c *OllamaClient) IsInPlanMode() bool {
	return c.planMode
}

// EnableConciseMode implements LLMClient interface
func (c *OllamaClient) EnableConciseMode() {
	c.conciseMode = true
	c.contextManager.EnableConciseMode()
}

// DisableConciseMode implements LLMClient interface
func (c *OllamaClient) DisableConciseMode() {
	c.conciseMode = false
	c.contextManager.DisableConciseMode()
}

// IsInConciseMode implements LLMClient interface
func (c *OllamaClient) IsInConciseMode() bool {
	return c.conciseMode
}

// SetActiveTask implements LLMClient interface
func (c *OllamaClient) SetActiveTask(task string) {
	c.contextManager.SetActiveTask(task)
}

// GetActiveTask implements LLMClient interface
func (c *OllamaClient) GetActiveTask() string {
	if task := c.contextManager.GetActiveTask(); task != nil {
		return task.Goal
	}
	return ""
}

// CompleteCurrentTask implements LLMClient interface
func (c *OllamaClient) CompleteCurrentTask() {
	c.contextManager.CompleteCurrentTask("")
}

// Interrupt implements LLMClient interface
func (c *OllamaClient) Interrupt() {
	select {
	case c.interruptChan <- struct{}{}:
	default:
	}
}

// IsResponseActive implements LLMClient interface
func (c *OllamaClient) IsResponseActive() bool {
	return c.responseActive
}

// DetectContextWindow implements LLMClient interface
func (c *OllamaClient) DetectContextWindow() (int, error) {
	request := OllamaModelShowRequest{
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

	var modelResponse OllamaModelShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelResponse); err != nil {
		return 0, fmt.Errorf("error decoding response: %v", err)
	}

	// Try to extract context window from model info
	if modelInfo, exists := modelResponse.ModelInfo["general.parameter_count"]; exists {
		if paramCount, ok := modelInfo.(float64); ok {
			// Very rough estimation: larger models tend to have larger context windows
			if paramCount > 30000000000 { // 30B+ parameters
				return 32768, nil
			} else if paramCount > 10000000000 { // 10B+ parameters  
				return 16384, nil
			} else if paramCount > 5000000000 { // 5B+ parameters
				return 8192, nil
			}
		}
	}

	// Default fallback
	return 4096, nil
}

// SetDebug implements LLMClient interface
func (c *OllamaClient) SetDebug(enabled bool) {
	c.debug = enabled
	if enabled {
		fmt.Println("[DEBUG] Debug logging enabled for Ollama client")
	}
}

// debugLog prints debug messages when debug mode is enabled
func (c *OllamaClient) debugLog(format string, args ...interface{}) {
	if c.debug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
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

// calculateOptimalContextLimit calculates optimal context limit (75% of total)
func calculateOptimalContextLimit(totalContext int) int {
	return int(float64(totalContext) * 0.75)
}