package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type OllamaClient struct {
	baseURL        string
	client         *http.Client
	contextManager *ContextManager
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

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 0, // No timeout for streaming responses
		},
		contextManager: NewContextManager(4000), // 4K tokens for qwen3:32b
	}
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
	request := ChatRequest{
		Model:    "qwen3:32b",
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

	var currentMessage ChatMessage
	var hasToolCalls bool

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
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
		}

		// Check for tool calls
		if len(chatResponse.Message.ToolCalls) > 0 {
			currentMessage.ToolCalls = append(currentMessage.ToolCalls, chatResponse.Message.ToolCalls...)
			hasToolCalls = true
		}

		if chatResponse.Done {
			fmt.Println()
			currentMessage.Role = "assistant"
			break
		}
	}

	// Handle tool calls if present
	if hasToolCalls {
		// Add assistant message to context manager
		c.contextManager.AddMessage(currentMessage)
		return c.handleToolCalls(currentMessage)
	} else if currentMessage.Content != "" {
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
	
	// Execute each tool call
	for _, toolCall := range assistantMessage.ToolCalls {
		fmt.Printf("Calling %s...\n", toolCall.Function.Name)
		
		result, err := ExecuteTool(toolCall)
		if err != nil {
			result = fmt.Sprintf("Error: %v", err)
		}
		
		// Add tool result to context manager
		toolMessage := ChatMessage{
			Role:    "tool",
			Content: result,
		}
		c.contextManager.AddMessage(toolMessage)
		
		fmt.Printf("✓ %s completed\n", toolCall.Function.Name)
	}
	
	fmt.Println("\n🤖 Assistant response:")
	
	// Continue conversation with updated context
	messages := c.contextManager.GetMessages()
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
		Model:    "qwen3:32b",
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