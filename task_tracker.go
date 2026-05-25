package main

import (
	"fmt"
	"log/slog"
	"loki-code/clients"
	"strings"
	"time"
)

// taskTracker owns user-task detection, lifecycle, and the assistant-message
// "completion phrase" heuristic. It is not goroutine-safe on its own; the
// ContextManager that embeds it serializes access via its mutex.
type taskTracker struct {
	active  *clients.UserTask
	history []clients.UserTask
}

func generateTaskID() string {
	return fmt.Sprintf("task_%d", time.Now().Unix())
}

func (t *taskTracker) set(goal string) {
	slog.Debug("taskTracker.set", "goal", goal)
	t.active = &clients.UserTask{
		ID:        generateTaskID(),
		Goal:      goal,
		Context:   "",
		CreatedAt: time.Now(),
		Status:    "active",
		SubTasks:  []string{},
	}
	slog.Info("Active task set", "task_id", t.active.ID, "goal", goal)
	fmt.Printf("🎯 Active task set: %s\n", goal)
}

func (t *taskTracker) complete(summary string) {
	if t.active == nil || t.active.Status != "active" {
		return
	}
	slog.Info("Completing active task", "task_id", t.active.ID, "goal", t.active.Goal)
	t.active.Status = "completed"
	t.history = append(t.history, *t.active)
	fmt.Printf("✅ Task completed: %s\n", summary)
	t.active = nil
}

// get returns a copy of the active task, or nil if none is active.
func (t *taskTracker) get() *clients.UserTask {
	if t.active == nil {
		return nil
	}
	task := *t.active
	return &task
}

// detectFromUser inspects an incoming user message and returns a new task if
// the message looks like a coding request rather than a quick question. This
// is a deliberately fuzzy keyword heuristic; callers decide whether to adopt
// the returned task as the active one.
func detectTaskFromUser(message clients.ChatMessage) *clients.UserTask {
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
	hasSignificantKeyword := containsAny(content, significantKeywords)

	taskPatterns := []string{
		"help me", "can you", "i want", "i need", "could you",
		"let's", "how to", "what's wrong", "why is",
	}
	hasTaskPattern := containsAny(content, taskPatterns)

	simplePatterns := []string{
		"what is", "who is", "when is", "where is",
		"/", "exit", "quit", "help", "status",
	}
	isSimple := containsAny(content, simplePatterns)

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

func containsAny(haystack string, needles []string) bool {
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			return true
		}
	}
	return false
}

// completionPhrases are case-insensitive substrings that, when present in an
// assistant message, are treated as "the active task is done".
var completionPhrases = []string{
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

// detectCompletionPhrase reports whether the assistant content contains any
// of the recognized "task done" phrases.
func detectCompletionPhrase(content string) bool {
	lower := strings.ToLower(content)
	for _, phrase := range completionPhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}
