package main

import (
	"encoding/json"
	"fmt"
	"loki-code/clients"
	"strings"
	"time"
)

// ContextManager handles conversation context with token management
type ContextManager struct {
	messages       []clients.ChatMessage
	maxTokens      int
	systemPrompt   clients.ChatMessage
	planModePrompt clients.ChatMessage
	retainCount    int
	planMode       bool
	conciseMode    bool
	activeTask     *clients.UserTask
	taskHistory    []clients.UserTask
}

func NewContextManager(maxTokens int, toolProvider clients.ToolSchemaProvider) *ContextManager {
	systemPrompt := clients.ChatMessage{
		Role: "system",
		Content: fmt.Sprintf(`You are Loki Code, an AI coding assistant that implements solutions intelligently.

SMART EXECUTION WORKFLOW:
1. ANALYZE: Read relevant files to understand current state
2. PLAN: Create brief implementation strategy (1-3 steps)
3. EXECUTE: Use tools to implement the solution
4. DELIVER: Process tool results to complete the user's request
5. VERIFY: Ensure changes are complete and logical

WHEN TO USE TOOLS:
- User requests code changes → READ context, PLAN approach, IMPLEMENT with tools
- User wants new features → ANALYZE requirements, DESIGN solution, CREATE files
- User reports bugs → READ code, IDENTIFY root cause, FIX with updates
- User asks for implementation → UNDERSTAND needs, PLAN structure, BUILD it
- User requests information → FETCH data with tools, PROCESS results, DELIVER answer
- User wants summaries → GET content with tools, ANALYZE data, PROVIDE summary
- User needs analysis → GATHER information, EXAMINE details, PRESENT findings

EXECUTION PRINCIPLES:
- Always READ before WRITING (understand context first)
- Always PLAN before IMPLEMENTING (think through the approach)
- Always USE TOOLS when code changes are needed (don't just describe)
- Always COMPLETE the user's original request after using tools
- Always VERIFY your work makes sense

COMPLETE USER REQUESTS:
After using tools, immediately process the results to fulfill the user's original request:
- If user asked to "summarize X" → fetch X, then provide the summary
- If user asked to "analyze Y" → gather Y data, then present analysis
- If user asked to "explain Z" → retrieve Z info, then give explanation
- NEVER leave the user hanging after fetching data - complete the request!

EXECUTION EXAMPLES:

CODING TASKS:
❌ "Here's the code you need: [shows code]"
✅ "I'll implement that: [reads files] → [brief plan] → [creates/updates files]"

❌ "You should update the function to handle this case"
✅ "I'll add that handling: [reads current code] → [explains approach] → [updates file]"

❌ "To implement this feature, you would need to..."
✅ "I'll implement this feature: [analyzes codebase] → [plans structure] → [creates files]"

INFORMATION TASKS:
❌ [fetches webpage] → "The data is now available. You might want to analyze this..."
✅ [fetches webpage] → "Based on this content, here's the summary you requested: [provides summary]"

❌ [gets API data] → "Here's the API response. What would you like to know about it?"
✅ [gets API data] → "Here's the analysis you asked for: [analyzes and explains the data]"

❌ [reads file] → "I found some information. Let me know if you need anything else."
✅ [reads file] → "Based on the file content, here are the key points: [delivers requested information]"

%s

CRITICAL: You are in EXECUTE MODE - make actual changes while being thoughtful about implementation!`, generateToolSchema(toolProvider)),
	}

	planModePrompt := clients.ChatMessage{
		Role: "system",
		Content: fmt.Sprintf(`You are Loki Code in PLAN MODE - you create detailed execution plans WITHOUT making changes.

🎯 PLAN MODE CAPABILITIES:
- READ files to analyze current state (read_file, list_files, find_files, grep_content) ✅
- FETCH information from web (http_request) for research and analysis ✅
- ANALYZE code structure and requirements ✅
- CREATE comprehensive implementation plans ✅
- IDENTIFY dependencies and potential issues ✅
- RESEARCH project patterns and conventions ✅
- COMPLETE information requests (summaries, analysis) directly ✅

⚠️ PLAN MODE RESTRICTIONS:
- Make actual file changes (create_file, update_file) ❌
- Execute write operations ❌
- Implement solutions directly ❌

PLANNING WORKFLOW:
1. ANALYZE: Use read-only tools to understand current state
2. RESEARCH: Study existing code patterns and structure
3. DESIGN: Create detailed implementation strategy
4. PLAN: Break down into executable steps with specific file operations
5. VALIDATE: Ensure plan addresses all requirements

IMPORTANT: For information requests (summaries, analysis), complete them directly using read-only tools:
- If user asks to "summarize X", fetch X and provide the summary immediately
- If user asks to "analyze Y", gather Y data and deliver the analysis
- Don't create plans for information tasks - complete them directly!

For implementation tasks: Create comprehensive plans that others can execute in EXECUTE MODE.

%s

CRITICAL: You are in PLAN MODE - analyze and plan thoroughly, but don't make actual changes!`, generateToolSchema(toolProvider)),
	}

	return &ContextManager{
		messages:       []clients.ChatMessage{},
		maxTokens:      maxTokens,
		systemPrompt:   systemPrompt,
		planModePrompt: planModePrompt,
		retainCount:    16,
		planMode:       false,
	}
}

func (cm *ContextManager) AddMessage(message clients.ChatMessage) {
	if message.Role == "user" {
		detectedTask := cm.detectUserTask(message)
		if detectedTask != nil {
			if cm.activeTask == nil || cm.activeTask.Status != "active" {
				cm.activeTask = detectedTask
				fmt.Printf("🎯 New task detected: %s\n", detectedTask.Goal)
			}
		}
	}

	cm.messages = append(cm.messages, message)

	// Check for task completion in assistant messages
	if message.Role == "assistant" {
		cm.checkTaskCompletion(message)
	}

	cm.trimIfNeeded()
}

func (cm *ContextManager) GetMessages() []clients.ChatMessage {
	var activePrompt clients.ChatMessage
	if cm.planMode {
		activePrompt = cm.planModePrompt
	} else {
		activePrompt = cm.systemPrompt
	}

	if cm.activeTask != nil && cm.activeTask.Status == "active" {
		activePrompt = cm.enhancePromptWithTask(activePrompt)
	}

	if cm.conciseMode {
		activePrompt = cm.addConciseModeInstructions(activePrompt)
	}

	result := []clients.ChatMessage{activePrompt}
	result = append(result, cm.messages...)
	return result
}

func (cm *ContextManager) GetSystemPrompt() clients.ChatMessage {
	return cm.systemPrompt
}

func (cm *ContextManager) trimIfNeeded() {
	allMessages := cm.GetMessages()
	currentTokens := cm.estimateTokens(allMessages)

	if currentTokens <= cm.maxTokens {
		return
	}

	if cm.HasToolCallsInProgress() {
		fmt.Printf("\n🔧 Context limit reached (%d/%d tokens) but tool calls in progress - delaying trim\n",
			currentTokens, cm.maxTokens)
		return
	}

	beforeMessageCount := len(cm.messages)
	fmt.Printf("\n⚠️ Context approaching limit (%d/%d tokens, %d messages), trimming...\n",
		currentTokens, cm.maxTokens, beforeMessageCount)

	cm.smartTrim()

	afterMessageCount := len(cm.messages)
	trimmedTokens := cm.estimateTokens(cm.GetMessages())
	fmt.Printf("✓ Context trimmed to %d tokens (%d messages, removed %d messages)\n",
		trimmedTokens, afterMessageCount, beforeMessageCount-afterMessageCount)
}

func (cm *ContextManager) smartTrim() {
	if len(cm.messages) <= cm.retainCount {
		return
	}

	conservativeRetainCount := cm.retainCount + 4
	if len(cm.messages) <= conservativeRetainCount {
		return
	}

	keepFromIndex := len(cm.messages) - conservativeRetainCount
	cutoffIndex := cm.findSafeCutoff(keepFromIndex)

	if cutoffIndex > 0 {
		cm.messages = cm.messages[cutoffIndex:]
	}
}

func (cm *ContextManager) findSafeCutoff(preferredIndex int) int {
	for i := preferredIndex; i > 0; i-- {
		if i < len(cm.messages) &&
			cm.messages[i-1].Role == "assistant" &&
			len(cm.messages[i-1].ToolCalls) == 0 {
			return i
		}

		if i < len(cm.messages) && cm.messages[i-1].Role == "tool" {
			return i
		}
	}

	return preferredIndex
}

func (cm *ContextManager) estimateTokens(messages []clients.ChatMessage) int {
	totalChars := 0

	for _, msg := range messages {
		contentChars := len(msg.Content)

		if msg.Role == "tool" {
			totalChars += int(float64(contentChars) * 1.2)
		} else {
			totalChars += contentChars
		}

		for _, toolCall := range msg.ToolCalls {
			totalChars += len(toolCall.Function.Name)
			for key, value := range toolCall.Function.Arguments {
				totalChars += len(key)
				totalChars += len(fmt.Sprintf("%v", value))
			}
		}

		totalChars += 50
	}

	estimatedTokens := totalChars / 4
	return int(float64(estimatedTokens) * 1.15)
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
	cm.messages = []clients.ChatMessage{}
}

func (cm *ContextManager) GetLastUserMessage() *clients.ChatMessage {
	for i := len(cm.messages) - 1; i >= 0; i-- {
		if cm.messages[i].Role == "user" {
			return &cm.messages[i]
		}
	}
	return nil
}

func (cm *ContextManager) HasToolCallsInProgress() bool {
	for i := len(cm.messages) - 1; i >= 0; i-- {
		msg := cm.messages[i]
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			for j := i + 1; j < len(cm.messages); j++ {
				if cm.messages[j].Role == "tool" {
					return false
				}
			}
			return true
		}
		if msg.Role == "user" {
			break
		}
	}
	return false
}

func (cm *ContextManager) CanCompact() bool {
	currentTokens := cm.estimateTokens(cm.GetMessages())
	threshold := int(float64(cm.maxTokens) * 0.6)
	minMessages := 6

	return currentTokens >= threshold && len(cm.messages) >= minMessages
}

func (cm *ContextManager) CompactContext(compactFunc clients.CompactFunc) error {
	currentTokens := cm.estimateTokens(cm.GetMessages())
	if !cm.CanCompact() {
		threshold := int(float64(cm.maxTokens) * 0.6)
		return fmt.Errorf("context not ready for compacting (using %d/%d tokens, need %d+ tokens)",
			currentTokens, cm.maxTokens, threshold)
	}

	if cm.HasToolCallsInProgress() {
		return fmt.Errorf("cannot compact while tool calls are in progress")
	}

	targetRecentTokens := int(float64(cm.maxTokens) * 0.25)

	recentMessages := []clients.ChatMessage{}

	for i := len(cm.messages) - 1; i >= 0; i-- {
		testMessages := append([]clients.ChatMessage{cm.messages[i]}, recentMessages...)
		testTokens := cm.estimateTokens(testMessages)

		if testTokens > targetRecentTokens && len(recentMessages) > 0 {
			break
		}

		recentMessages = testMessages
	}

	compactEndIndex := len(cm.messages) - len(recentMessages)
	if compactEndIndex <= 1 {
		return fmt.Errorf("not enough content to compact (would keep %d recent messages)", len(recentMessages))
	}

	messagesToCompact := cm.messages[:compactEndIndex]

	fmt.Println("🔄 Generating conversation summary...")
	summary, err := compactFunc(messagesToCompact)
	if err != nil {
		return fmt.Errorf("failed to generate summary: %v", err)
	}

	summaryMessage := clients.ChatMessage{
		Role:    "assistant",
		Content: fmt.Sprintf("📋 Context Summary: %s", summary),
	}

	oldTokens := cm.estimateTokens(append([]clients.ChatMessage{cm.systemPrompt}, cm.messages...))

	cm.messages = append([]clients.ChatMessage{summaryMessage}, recentMessages...)

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

func (cm *ContextManager) SetPlanMode(enabled bool) {
	cm.planMode = enabled
}

func (cm *ContextManager) IsInPlanMode() bool {
	return cm.planMode
}

func (cm *ContextManager) SetConciseMode(enabled bool) {
	cm.conciseMode = enabled
}

func (cm *ContextManager) IsInConciseMode() bool {
	return cm.conciseMode
}

func (cm *ContextManager) addConciseModeInstructions(prompt clients.ChatMessage) clients.ChatMessage {
	conciseInstructions := `

RESPONSE MODE: CONCISE
- Keep all responses brief and to-the-point
- Provide essential information only
- Use bullet points when appropriate
- Avoid lengthy explanations unless specifically requested
- Focus on direct answers and immediate next steps
- When using tools, be efficient and purposeful`

	modifiedPrompt := prompt
	modifiedPrompt.Content = prompt.Content + conciseInstructions
	return modifiedPrompt
}

// Task management functions
func generateTaskID() string {
	return fmt.Sprintf("task_%d", time.Now().Unix())
}

func (cm *ContextManager) detectUserTask(message clients.ChatMessage) *clients.UserTask {
	if message.Role != "user" {
		return nil
	}

	content := strings.ToLower(message.Content)

	significantKeywords := []string{
		"implement", "add", "create", "build", "make", "write",
		"fix", "debug", "solve", "resolve", "repair",
		"refactor", "improve", "optimize", "enhance", "update",
		"analyze", "review", "explain", "understand", "find",
		"remove", "delete", "change", "modify", "replace",
	}

	hasSignificantKeyword := false
	for _, keyword := range significantKeywords {
		if strings.Contains(content, keyword) {
			hasSignificantKeyword = true
			break
		}
	}

	taskPatterns := []string{
		"help me", "can you", "i want", "i need", "could you",
		"let's", "how to", "what's wrong", "why is",
	}

	hasTaskPattern := false
	for _, pattern := range taskPatterns {
		if strings.Contains(content, pattern) {
			hasTaskPattern = true
			break
		}
	}

	simplePatterns := []string{
		"what is", "who is", "when is", "where is",
		"/", "exit", "quit", "help", "status",
	}

	isSimple := false
	for _, pattern := range simplePatterns {
		if strings.Contains(content, pattern) {
			isSimple = true
			break
		}
	}

	if (hasSignificantKeyword || hasTaskPattern) && !isSimple && len(message.Content) > 10 {
		goal := message.Content
		if len(goal) > 200 {
			goal = goal[:200] + "..."
		}

		return &clients.UserTask{
			ID:        generateTaskID(),
			Goal:      goal,
			Context:   "",
			CreatedAt: time.Now(),
			Status:    "active",
			SubTasks:  []string{},
		}
	}

	return nil
}

func (cm *ContextManager) SetActiveTask(goal string) {
	cm.activeTask = &clients.UserTask{
		ID:        generateTaskID(),
		Goal:      goal,
		Context:   "",
		CreatedAt: time.Now(),
		Status:    "active",
		SubTasks:  []string{},
	}
	fmt.Printf("🎯 Active task set: %s\n", goal)
}

func (cm *ContextManager) GetActiveTask() *clients.UserTask {
	return cm.activeTask
}

func (cm *ContextManager) CompleteCurrentTask(summary string) {
	if cm.activeTask != nil && cm.activeTask.Status == "active" {
		cm.activeTask.Status = "completed"
		cm.taskHistory = append(cm.taskHistory, *cm.activeTask)
		fmt.Printf("✅ Task completed: %s\n", summary)
		cm.activeTask = nil
	}
}

func (cm *ContextManager) enhancePromptWithTask(prompt clients.ChatMessage) clients.ChatMessage {
	if cm.activeTask == nil || cm.activeTask.Status != "active" {
		return prompt
	}

	taskContext := fmt.Sprintf(`

🎯 CURRENT ACTIVE TASK:
Goal: %s
Created: %s
Status: %s

CRITICAL TASK REQUIREMENTS:
- Remember this task throughout ALL tool operations and responses
- After using tools, immediately process results to fulfill this specific request
- Don't ask what to do next - complete the task based on the original goal
- If the task was to "summarize X", provide the summary after fetching X
- If the task was to "analyze Y", provide the analysis after gathering Y data
- Stay focused on delivering what was originally requested

When you believe this task is fully completed, end your response with "TASK COMPLETED: [brief summary]"`,
		cm.activeTask.Goal,
		cm.activeTask.CreatedAt.Format("15:04:05"),
		cm.activeTask.Status)

	enhanced := prompt
	enhanced.Content = prompt.Content + taskContext
	return enhanced
}

func (cm *ContextManager) checkTaskCompletion(assistantMessage clients.ChatMessage) {
	if cm.activeTask == nil || cm.activeTask.Status != "active" {
		return
	}

	content := strings.ToLower(assistantMessage.Content)

	completionPhrases := []string{
		"task completed",
		"implementation complete",
		"implementation is complete",
		"task is done",
		"task finished",
		"successfully completed",
		"implementation finished",
		"summary complete",
		"analysis complete",
		"here's the summary",
		"summary of the website",
		"summary of the content",
		"analysis of the",
		"based on this content",
		"the summary you requested",
		"analysis you asked for",
	}

	for _, phrase := range completionPhrases {
		if strings.Contains(content, phrase) {
			summary := cm.extractCompletionSummary(assistantMessage.Content)
			cm.CompleteCurrentTask(summary)
			return
		}
	}
}

func (cm *ContextManager) extractCompletionSummary(content string) string {
	taskCompletedIndex := strings.Index(strings.ToLower(content), "task completed")
	if taskCompletedIndex != -1 {
		remaining := content[taskCompletedIndex:]
		colonIndex := strings.Index(remaining, ":")
		if colonIndex != -1 && len(remaining) > colonIndex+1 {
			summary := strings.TrimSpace(remaining[colonIndex+1:])
			if len(summary) > 100 {
				summary = summary[:100] + "..."
			}
			return summary
		}
	}

	if cm.activeTask != nil {
		return cm.activeTask.Goal
	}

	return "Task completed"
}

// generateToolSchema creates a formatted JSON string representation of available tools
func generateToolSchema(toolProvider clients.ToolSchemaProvider) string {
	tools := toolProvider()

	var schema strings.Builder
	schema.WriteString("Available tools (use exact JSON format):\n\n")

	for i, tool := range tools {
		toolJSON, err := json.MarshalIndent(tool, "", "  ")
		if err != nil {
			continue
		}

		schema.WriteString(fmt.Sprintf("%d. %s\n", i+1, tool.Function.Name))
		schema.WriteString(fmt.Sprintf("   %s\n", tool.Function.Description))
		schema.WriteString(fmt.Sprintf("   Schema: %s\n\n", string(toolJSON)))
	}

	schema.WriteString("SMART TOOL USAGE PATTERNS:\n\n")
	schema.WriteString("1. Code Implementation: list_files → read_file → [plan] → create_file/update_file\n")
	schema.WriteString("2. Bug Fixes: read_file → grep_content → [analyze] → update_file\n")
	schema.WriteString("3. New Features: find_files → read_file → [design] → create_file\n")
	schema.WriteString("4. Code Analysis: read_file → grep_content → analyze_code → [insights]\n")
	schema.WriteString("5. Project Understanding: list_files → read_file → get_pwd → [context]\n\n")

	schema.WriteString("EXECUTION WORKFLOW:\n")
	schema.WriteString("- READ first to understand context\n")
	schema.WriteString("- PLAN your implementation approach\n")
	schema.WriteString("- USE tools to implement changes\n")
	schema.WriteString("- VERIFY the implementation is complete\n\n")

	schema.WriteString("TOOL CALL FORMAT EXAMPLES:\n\n")
	schema.WriteString("Standard format (preferred):\n")
	schema.WriteString(`{"name": "read_file", "arguments": {"path": "example.go"}}`)
	schema.WriteString("\n\nAlternative format (also supported):\n")
	schema.WriteString(`{"function": {"name": "read_file", "arguments": {"path": "example.go"}}}`)
	schema.WriteString("\n\nIMPORTANT:\n")
	schema.WriteString("- Use valid JSON syntax\n")
	schema.WriteString("- Include all required parameters\n")
	schema.WriteString("- Wrap JSON in explanation text if needed\n")
	schema.WriteString("- Always follow the smart usage patterns above\n")

	return schema.String()
}
