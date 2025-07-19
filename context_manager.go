package main

import (
	"fmt"
)

type ContextManager struct {
	messages      []ChatMessage
	maxTokens     int
	systemPrompt  ChatMessage
	retainCount   int // Number of recent exchanges to retain
}

func NewContextManager(maxTokens int) *ContextManager {
	systemPrompt := ChatMessage{
		Role: "system",
		Content: `You are Loki Code, an AI coding assistant specialized in helping developers with programming tasks, code analysis, and file operations.

GUIDELINES:
- Keep responses brief, focused, and actionable
- Prioritize practical solutions over lengthy explanations
- Use available tools efficiently when users request file operations
- Provide concise code examples when helpful
- Ask clarifying questions if requirements are unclear

AVAILABLE TOOLS:
- create_file: Create new files with specified content
- read_file: Read contents of existing files  
- update_file: Modify existing files with new content
- delete_file: Remove files
- list_files: List directory contents

BEHAVIOR:
- Be direct and helpful
- Focus on solving the immediate problem
- Use tools when appropriate for file operations
- Explain your reasoning briefly when using tools
- Maintain conversation context efficiently`,
	}

	return &ContextManager{
		messages:     []ChatMessage{},
		maxTokens:    maxTokens,
		systemPrompt: systemPrompt,
		retainCount:  6, // Keep last 3 user+assistant exchanges
	}
}

func (cm *ContextManager) AddMessage(message ChatMessage) {
	cm.messages = append(cm.messages, message)
	cm.trimIfNeeded()
}

func (cm *ContextManager) GetMessages() []ChatMessage {
	// Always start with system prompt
	result := []ChatMessage{cm.systemPrompt}
	result = append(result, cm.messages...)
	return result
}

func (cm *ContextManager) GetSystemPrompt() ChatMessage {
	return cm.systemPrompt
}

func (cm *ContextManager) trimIfNeeded() {
	allMessages := cm.GetMessages()
	currentTokens := cm.estimateTokens(allMessages)
	
	if currentTokens <= cm.maxTokens {
		return
	}

	fmt.Printf("\n⚠️ Context approaching limit (%d tokens), trimming...\n", currentTokens)
	
	// Find the cutoff point while preserving tool call sequences
	cm.smartTrim()
	
	trimmedTokens := cm.estimateTokens(cm.GetMessages())
	fmt.Printf("✓ Context trimmed to %d tokens\n", trimmedTokens)
}

func (cm *ContextManager) smartTrim() {
	if len(cm.messages) <= cm.retainCount {
		return // Don't trim if we have few messages
	}

	// Always keep the last retainCount messages to maintain recent context
	keepFromIndex := len(cm.messages) - cm.retainCount
	
	// Look backwards to find a safe cutoff point (avoid breaking tool sequences)
	cutoffIndex := cm.findSafeCutoff(keepFromIndex)
	
	if cutoffIndex > 0 {
		cm.messages = cm.messages[cutoffIndex:]
	}
}

func (cm *ContextManager) findSafeCutoff(preferredIndex int) int {
	// Start from preferred index and look backwards for a safe cut
	for i := preferredIndex; i > 0; i-- {
		// Safe to cut after assistant messages that don't have tool calls
		if i < len(cm.messages) && 
		   cm.messages[i-1].Role == "assistant" && 
		   len(cm.messages[i-1].ToolCalls) == 0 {
			return i
		}
		
		// Also safe to cut after tool messages (tool sequences are complete)
		if i < len(cm.messages) && cm.messages[i-1].Role == "tool" {
			return i
		}
	}
	
	// If no safe cutoff found, cut at preferred index anyway
	return preferredIndex
}

func (cm *ContextManager) estimateTokens(messages []ChatMessage) int {
	totalChars := 0
	
	for _, msg := range messages {
		// Count message content
		totalChars += len(msg.Content)
		
		// Count tool calls (these can be significant)
		for _, toolCall := range msg.ToolCalls {
			totalChars += len(toolCall.Function.Name)
			// Estimate size of arguments (JSON structure)
			for key, value := range toolCall.Function.Arguments {
				totalChars += len(key)
				totalChars += len(fmt.Sprintf("%v", value))
			}
		}
		
		// Add overhead for JSON structure
		totalChars += 50 // Rough estimate for role, timestamps, etc.
	}
	
	// Convert chars to tokens (rough approximation: ~4 chars per token)
	estimatedTokens := totalChars / 4
	
	// Add buffer for safety (30% overhead)
	return int(float64(estimatedTokens) * 1.3)
}

func (cm *ContextManager) GetStats() (int, int, int) {
	currentTokens := cm.estimateTokens(cm.GetMessages())
	messageCount := len(cm.messages)
	return currentTokens, messageCount, cm.maxTokens
}

func (cm *ContextManager) SetMaxTokens(maxTokens int) {
	cm.maxTokens = maxTokens
	cm.trimIfNeeded()
}

func (cm *ContextManager) Clear() {
	cm.messages = []ChatMessage{}
}

func (cm *ContextManager) GetLastUserMessage() *ChatMessage {
	for i := len(cm.messages) - 1; i >= 0; i-- {
		if cm.messages[i].Role == "user" {
			return &cm.messages[i]
		}
	}
	return nil
}

func (cm *ContextManager) HasToolCallsInProgress() bool {
	// Check if the last assistant message has tool calls that haven't been resolved
	for i := len(cm.messages) - 1; i >= 0; i-- {
		msg := cm.messages[i]
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			// Look for corresponding tool results
			for j := i + 1; j < len(cm.messages); j++ {
				if cm.messages[j].Role == "tool" {
					return false // Found tool results, so calls are resolved
				}
			}
			return true // Found tool calls without results
		}
		if msg.Role == "user" {
			break // Reached next user message, no pending tool calls
		}
	}
	return false
}

// CompactClient interface to avoid circular dependency
type CompactClient interface {
	CompactMessages(messages []ChatMessage) (string, error)
}

func (cm *ContextManager) CanCompact() bool {
	// Need at least 8 messages to make compacting worthwhile
	// (excluding system prompt, so at least 9 total)
	return len(cm.messages) >= 8
}

func (cm *ContextManager) CompactContext(client CompactClient) error {
	if !cm.CanCompact() {
		return fmt.Errorf("not enough messages to compact (need at least 8, have %d)", len(cm.messages))
	}

	if cm.HasToolCallsInProgress() {
		return fmt.Errorf("cannot compact while tool calls are in progress")
	}

	// Identify compactable range
	recentCount := 6 // Keep last 6 messages
	if len(cm.messages) <= recentCount {
		return fmt.Errorf("not enough messages to compact")
	}

	compactEndIndex := len(cm.messages) - recentCount
	messagesToCompact := cm.messages[:compactEndIndex]
	recentMessages := cm.messages[compactEndIndex:]

	// Get summary from LLM
	fmt.Println("🔄 Generating conversation summary...")
	summary, err := client.CompactMessages(messagesToCompact)
	if err != nil {
		return fmt.Errorf("failed to generate summary: %v", err)
	}

	// Create summary message
	summaryMessage := ChatMessage{
		Role:    "assistant",
		Content: fmt.Sprintf("📋 Context Summary: %s", summary),
	}

	// Calculate token savings
	oldTokens := cm.estimateTokens(append([]ChatMessage{cm.systemPrompt}, cm.messages...))
	
	// Replace compacted messages with summary
	cm.messages = append([]ChatMessage{summaryMessage}, recentMessages...)
	
	newTokens := cm.estimateTokens(cm.GetMessages())
	savedTokens := oldTokens - newTokens

	fmt.Printf("✓ Context compacted: %d → %d tokens (saved %d tokens)\n", 
		oldTokens, newTokens, savedTokens)
	
	return nil
}

func (cm *ContextManager) CreateCompactingPrompt(messages []ChatMessage) string {
	return `CONTEXT SUMMARIZATION REQUEST

Please provide a concise summary of the following conversation history. Focus on:
- Key decisions made and conclusions reached
- Important information discovered or discussed
- File operations performed and their results
- Code solutions or technical details discussed
- Any ongoing tasks, context, or important state

Keep the summary brief but preserve essential context for continuing the conversation.
Organize the summary logically and use clear, concise language.

CONVERSATION TO SUMMARIZE:`
}