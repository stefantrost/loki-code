package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"loki-code/clients"
	"strings"
	"sync"
)

// Context-management tuning constants. Token estimation is a heuristic
// (chars/4 with a per-message and per-tool overhead nudge); these multipliers
// were picked empirically against qwen3:32b and gpt-4-class tokenizers and
// will drift on tool-call-heavy turns. Treat them as approximate.
const (
	// toolMessageTokenInflation accounts for the wrapping JSON/tool-result
	// metadata that providers add around the raw content we count.
	toolMessageTokenInflation = 1.2
	// perMessageOverheadChars approximates per-message JSON envelope size
	// (role, separators, etc.) before dividing by charsPerToken.
	perMessageOverheadChars = 50
	// charsPerToken is the rough character-to-token ratio for English+code.
	// Used only as a cold-start fallback before the first real API token count.
	charsPerToken = 4
	// tokenSafetyFactor inflates the cold-start estimate to leave headroom for
	// tokenizer differences across models.
	tokenSafetyFactor = 1.15
	// compactionThresholdRatio is the fraction of maxTokens at which
	// auto-compact triggers (and /compact is permitted manually).
	compactionThresholdRatio = 0.75
	// compactionRecentRatio controls how much of maxTokens to keep verbatim
	// at the tail of a compaction (the rest gets summarized).
	compactionRecentRatio = 0.25
	// compactionMinMessages is the minimum message count before compaction runs.
	compactionMinMessages = 6
)

// ContextManager handles conversation context with token management.
// All public methods are safe for concurrent use; internal helpers ending in
// "Locked" assume the caller already holds mu.
type ContextManager struct {
	mu               sync.RWMutex
	messages         []clients.ChatMessage
	maxTokens        int
	systemPrompt     clients.ChatMessage
	planModePrompt   clients.ChatMessage
	planMode         bool
	conciseMode      bool
	tasks            taskTracker
	// lastKnownTokens holds the real prompt-token count from the most recent
	// API response. -1 means no real count has arrived yet (cold-start).
	lastKnownTokens  int
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
		messages:        []clients.ChatMessage{},
		maxTokens:       maxTokens,
		systemPrompt:    systemPrompt,
		planModePrompt:  planModePrompt,
		lastKnownTokens: -1,
	}
}

func (cm *ContextManager) AddMessage(message clients.ChatMessage) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	slog.Debug("AddMessage called", "role", message.Role, "content_length", len(message.Content),
		"tool_calls_count", len(message.ToolCalls), "messages_before", len(cm.messages))

	if message.Role == "user" {
		if detectedTask := detectTaskFromUser(message); detectedTask != nil {
			slog.Debug("Task detected from user message", "task_goal", detectedTask.Goal)
			if cm.tasks.active == nil || cm.tasks.active.Status != "active" {
				cm.tasks.active = detectedTask
				slog.Info("Active task set", "task_goal", detectedTask.Goal)
				fmt.Printf("🎯 New task detected: %s\n", detectedTask.Goal)
			}
		}
	}

	cm.messages = append(cm.messages, message)
	slog.Debug("Message appended to context", "messages_after", len(cm.messages))

	// Check for task completion in assistant messages
	if message.Role == "assistant" {
		cm.checkTaskCompletionLocked(message)
	}
}

func (cm *ContextManager) GetMessages() []clients.ChatMessage {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.getMessagesLocked()
}

func (cm *ContextManager) getMessagesLocked() []clients.ChatMessage {
	slog.Debug("GetMessages called", "plan_mode", cm.planMode, "concise_mode", cm.conciseMode,
		"active_task", cm.tasks.active != nil, "messages_count", len(cm.messages))

	var activePrompt clients.ChatMessage
	if cm.planMode {
		slog.Debug("Using plan mode prompt")
		activePrompt = cm.planModePrompt
	} else {
		slog.Debug("Using system prompt")
		activePrompt = cm.systemPrompt
	}

	if cm.tasks.active != nil && cm.tasks.active.Status == "active" {
		slog.Debug("Enhancing prompt with active task", "task_goal", cm.tasks.active.Goal)
		activePrompt = cm.enhancePromptWithTask(activePrompt)
	}

	if cm.conciseMode {
		slog.Debug("Adding concise mode instructions")
		activePrompt = cm.addConciseModeInstructions(activePrompt)
	}

	result := []clients.ChatMessage{activePrompt}
	result = append(result, cm.messages...)
	slog.Debug("Messages retrieved", "total_count", len(result), "prompt_length", len(activePrompt.Content))
	return result
}

func (cm *ContextManager) GetSystemPrompt() clients.ChatMessage {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.systemPrompt
}


func (cm *ContextManager) estimateTokens(messages []clients.ChatMessage) int {
	totalChars := 0

	for _, msg := range messages {
		contentChars := len(msg.Content)

		if msg.Role == "tool" {
			totalChars += int(float64(contentChars) * toolMessageTokenInflation)
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

		totalChars += perMessageOverheadChars
	}

	estimatedTokens := totalChars / charsPerToken
	return int(float64(estimatedTokens) * tokenSafetyFactor)
}

func (cm *ContextManager) GetStats() (int, int, int) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	var currentTokens int
	if cm.lastKnownTokens >= 0 {
		currentTokens = cm.lastKnownTokens
	} else {
		currentTokens = cm.estimateTokens(cm.getMessagesLocked())
	}
	return currentTokens, len(cm.messages), cm.maxTokens
}

// UpdateTokenCount stores the real prompt-token count received from the
// provider after a completed response. Replaces the heuristic estimate.
func (cm *ContextManager) UpdateTokenCount(promptTokens int) {
	if promptTokens <= 0 {
		return
	}
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.lastKnownTokens = promptTokens
	slog.Debug("Token count updated from API", "prompt_tokens", promptTokens)
}

func (cm *ContextManager) SetMaxTokens(maxTokens int) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.maxTokens = maxTokens
}

func (cm *ContextManager) Clear() {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.messages = []clients.ChatMessage{}
}

func (cm *ContextManager) GetLastUserMessage() *clients.ChatMessage {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	for i := len(cm.messages) - 1; i >= 0; i-- {
		if cm.messages[i].Role == "user" {
			msg := cm.messages[i]
			return &msg
		}
	}
	return nil
}

func (cm *ContextManager) HasToolCallsInProgress() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.hasToolCallsInProgressLocked()
}

func (cm *ContextManager) hasToolCallsInProgressLocked() bool {
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
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.canCompactLocked()
}

func (cm *ContextManager) canCompactLocked() bool {
	var currentTokens int
	if cm.lastKnownTokens >= 0 {
		currentTokens = cm.lastKnownTokens
	} else {
		currentTokens = cm.estimateTokens(cm.getMessagesLocked())
	}
	threshold := int(float64(cm.maxTokens) * compactionThresholdRatio)
	return currentTokens >= threshold && len(cm.messages) >= compactionMinMessages
}

func (cm *ContextManager) CompactContext(compactFunc clients.CompactFunc) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	currentTokens := cm.estimateTokens(cm.getMessagesLocked())
	if !cm.canCompactLocked() {
		threshold := int(float64(cm.maxTokens) * compactionThresholdRatio)
		return fmt.Errorf("context not ready for compacting (using %d/%d tokens, need %d+ tokens)",
			currentTokens, cm.maxTokens, threshold)
	}

	if cm.hasToolCallsInProgressLocked() {
		return fmt.Errorf("cannot compact while tool calls are in progress")
	}

	targetRecentTokens := int(float64(cm.maxTokens) * compactionRecentRatio)
	slog.Debug("Starting context compaction", "total_messages", len(cm.messages), "max_tokens", cm.maxTokens,
		"target_recent_tokens", targetRecentTokens)

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
	slog.Debug("Calculated compaction boundaries", "compact_end_index", compactEndIndex,
		"recent_messages_count", len(recentMessages), "messages_to_compact", compactEndIndex)

	if compactEndIndex <= 1 {
		slog.Warn("Not enough content to compact", "recent_messages", len(recentMessages))
		return fmt.Errorf("not enough content to compact (would keep %d recent messages)", len(recentMessages))
	}

	messagesToCompact := cm.messages[:compactEndIndex]
	slog.Debug("Messages selected for compaction", "count", len(messagesToCompact))

	fmt.Println("🔄 Generating conversation summary...")
	slog.Debug("Calling compact function to generate summary")
	summary, err := compactFunc(messagesToCompact)
	if err != nil {
		slog.Error("Failed to generate summary", "error", err)
		return fmt.Errorf("failed to generate summary: %v", err)
	}

	slog.Debug("Summary generated", "summary_length", len(summary))

	summaryMessage := clients.ChatMessage{
		Role:    "assistant",
		Content: fmt.Sprintf("📋 Context Summary: %s", summary),
	}

	oldTokens := cm.estimateTokens(append([]clients.ChatMessage{cm.systemPrompt}, cm.messages...))
	slog.Debug("Old token count calculated", "old_tokens", oldTokens)

	cm.messages = append([]clients.ChatMessage{summaryMessage}, recentMessages...)

	newTokens := cm.estimateTokens(cm.getMessagesLocked())
	savedTokens := oldTokens - newTokens
	compactedMessageCount := len(messagesToCompact)
	keptMessageCount := len(recentMessages)

	slog.Info("Context compaction completed", "old_tokens", oldTokens, "new_tokens", newTokens,
		"saved_tokens", savedTokens, "compacted_messages", compactedMessageCount, "kept_messages", keptMessageCount)

	fmt.Printf("✓ Context compacted: %d → %d tokens (saved %d tokens)\n",
		oldTokens, newTokens, savedTokens)
	fmt.Printf("📊 Compacted %d messages → 1 summary, kept %d recent messages\n",
		compactedMessageCount, keptMessageCount)

	return nil
}

func (cm *ContextManager) SetPlanMode(enabled bool) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.planMode = enabled
}

func (cm *ContextManager) IsInPlanMode() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.planMode
}

func (cm *ContextManager) SetConciseMode(enabled bool) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.conciseMode = enabled
}

func (cm *ContextManager) IsInConciseMode() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
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

func (cm *ContextManager) SetActiveTask(goal string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.tasks.set(goal)
}

func (cm *ContextManager) GetActiveTask() *clients.UserTask {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.tasks.get()
}

func (cm *ContextManager) CompleteCurrentTask(summary string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.completeCurrentTaskLocked(summary)
}

func (cm *ContextManager) completeCurrentTaskLocked(summary string) {
	cm.tasks.complete(summary)
}

func (cm *ContextManager) enhancePromptWithTask(prompt clients.ChatMessage) clients.ChatMessage {
	if cm.tasks.active == nil || cm.tasks.active.Status != "active" {
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
		cm.tasks.active.Goal,
		cm.tasks.active.CreatedAt.Format("15:04:05"),
		cm.tasks.active.Status)

	enhanced := prompt
	enhanced.Content = prompt.Content + taskContext
	return enhanced
}

func (cm *ContextManager) checkTaskCompletionLocked(assistantMessage clients.ChatMessage) {
	if cm.tasks.active == nil || cm.tasks.active.Status != "active" {
		return
	}
	if detectCompletionPhrase(assistantMessage.Content) {
		summary := cm.extractCompletionSummary(assistantMessage.Content)
		cm.completeCurrentTaskLocked(summary)
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

	if cm.tasks.active != nil {
		return cm.tasks.active.Goal
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

		fmt.Fprintf(&schema, "%d. %s\n", i+1, tool.Function.Name)
		fmt.Fprintf(&schema, "   %s\n", tool.Function.Description)
		fmt.Fprintf(&schema, "   Schema: %s\n\n", string(toolJSON))
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
