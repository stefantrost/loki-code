package main

import (
	"fmt"
)

type ContextManager struct {
	messages          []ChatMessage
	maxTokens         int
	systemPrompt      ChatMessage
	planModePrompt    ChatMessage
	retainCount       int  // Number of recent exchanges to retain
	planMode          bool // Whether plan mode is active
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

	planModePrompt := ChatMessage{
		Role: "system",
		Content: `You are Loki Code, an AI coding assistant in PLAN MODE. You specialize in analyzing tasks and creating detailed execution plans.

🎯 PLAN MODE ACTIVE - You can:
- Read files and analyze code (read_file, list_files)
- Create detailed multi-step execution plans
- Break down complex tasks into actionable steps
- Analyze codebases and provide insights

⚠️ PLAN MODE RESTRICTIONS - You CANNOT:
- Create, update, or delete files
- Execute any write operations
- Make changes to the codebase

BEHAVIOR:
- When given a task, break it down into a clear, numbered plan
- Use read-only tools to analyze the current state
- Focus on creating comprehensive, actionable plans
- Explain the reasoning behind each step
- Identify dependencies and potential issues
- Structure plans with clear sections: Overview, Analysis, Steps, Considerations`,
	}

	return &ContextManager{
		messages:       []ChatMessage{},
		maxTokens:      maxTokens,
		systemPrompt:   systemPrompt,
		planModePrompt: planModePrompt,
		retainCount:    6, // Keep last 3 user+assistant exchanges
		planMode:       false,
	}
}

func (cm *ContextManager) AddMessage(message ChatMessage) {
	cm.messages = append(cm.messages, message)
	cm.trimIfNeeded()
}

func (cm *ContextManager) GetMessages() []ChatMessage {
	// Always start with appropriate system prompt
	var activePrompt ChatMessage
	if cm.planMode {
		activePrompt = cm.planModePrompt
	} else {
		activePrompt = cm.systemPrompt
	}
	
	result := []ChatMessage{activePrompt}
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
	// Check if we're using enough context to warrant compacting
	currentTokens := cm.estimateTokens(cm.GetMessages())
	
	// Only compact if we're using at least 60% of available context
	threshold := int(float64(cm.maxTokens) * 0.6)
	
	// Also require minimum message count to ensure there's meaningful content
	minMessages := 6
	
	return currentTokens >= threshold && len(cm.messages) >= minMessages
}

func (cm *ContextManager) CompactContext(client CompactClient) error {
	currentTokens := cm.estimateTokens(cm.GetMessages())
	if !cm.CanCompact() {
		threshold := int(float64(cm.maxTokens) * 0.6)
		return fmt.Errorf("context not ready for compacting (using %d/%d tokens, need %d+ tokens)", 
			currentTokens, cm.maxTokens, threshold)
	}

	if cm.HasToolCallsInProgress() {
		return fmt.Errorf("cannot compact while tool calls are in progress")
	}

	// Calculate how much context to keep recent (aim for ~25% of max tokens)
	targetRecentTokens := int(float64(cm.maxTokens) * 0.25)
	
	// Find cutoff point by working backwards and estimating tokens
	recentMessages := []ChatMessage{}
	
	for i := len(cm.messages) - 1; i >= 0; i-- {
		testMessages := append([]ChatMessage{cm.messages[i]}, recentMessages...)
		testTokens := cm.estimateTokens(testMessages)
		
		if testTokens > targetRecentTokens && len(recentMessages) > 0 {
			break // Stop before exceeding target
		}
		
		recentMessages = testMessages
	}
	
	// Ensure we have something to compact
	compactEndIndex := len(cm.messages) - len(recentMessages)
	if compactEndIndex <= 1 {
		return fmt.Errorf("not enough content to compact (would keep %d recent messages)", len(recentMessages))
	}
	
	messagesToCompact := cm.messages[:compactEndIndex]

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
	compactedMessageCount := len(messagesToCompact)
	keptMessageCount := len(recentMessages)

	fmt.Printf("✓ Context compacted: %d → %d tokens (saved %d tokens)\n", 
		oldTokens, newTokens, savedTokens)
	fmt.Printf("📊 Compacted %d messages → 1 summary, kept %d recent messages\n", 
		compactedMessageCount, keptMessageCount)
	
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

func (cm *ContextManager) SetPlanMode(enabled bool) {
	cm.planMode = enabled
}

func (cm *ContextManager) IsInPlanMode() bool {
	return cm.planMode
}