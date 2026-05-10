package main

import "loki-code/clients"

// Integration between main package and clients package
// This file bridges the gap by providing main package implementations
// to the clients package

func init() {
	// Set up the function pointers in the clients package
	clients.SetMainPackageIntegration(MainPackageIntegration{})
}

// MainPackageIntegration implements the bridge interface
type MainPackageIntegration struct{}

func (m MainPackageIntegration) NewContextManager(maxTokens int) clients.ContextManager {
	cm := NewContextManager(maxTokens)
	return &ContextManagerWrapper{cm: cm}
}

func (m MainPackageIntegration) GetAvailableTools() []clients.Tool {
	tools := GetAvailableTools()
	// Convert main package Tools to client Tools
	clientTools := make([]clients.Tool, len(tools))
	for i, tool := range tools {
		clientTools[i] = clients.Tool{
			Type: tool.Type,
			Function: clients.Function{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return clientTools
}

func (m MainPackageIntegration) ExecuteToolWithPlanMode(toolCall clients.ToolCall, planMode bool) (string, error) {
	// Convert client ToolCall to main package ToolCall
	mainToolCall := ToolCall{
		Function: ToolFunction{
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		},
	}
	return ExecuteToolWithPlanMode(mainToolCall, planMode)
}

// ContextManagerWrapper wraps the main package ContextManager to implement the client interface
type ContextManagerWrapper struct {
	cm *ContextManager
}

func (w *ContextManagerWrapper) AddMessage(msg clients.ChatMessage) {
	mainMsg := ChatMessage{
		Role:    msg.Role,
		Content: msg.Content,
	}
	// Convert tool calls if present
	if len(msg.ToolCalls) > 0 {
		mainMsg.ToolCalls = make([]ToolCall, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			mainMsg.ToolCalls[i] = ToolCall{
				Function: ToolFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
		}
	}
	w.cm.AddMessage(mainMsg)
}

func (w *ContextManagerWrapper) GetMessages() []clients.ChatMessage {
	mainMsgs := w.cm.GetMessages()
	clientMsgs := make([]clients.ChatMessage, len(mainMsgs))
	for i, msg := range mainMsgs {
		clientMsgs[i] = clients.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
		// Convert tool calls if present
		if len(msg.ToolCalls) > 0 {
			clientMsgs[i].ToolCalls = make([]clients.ToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				clientMsgs[i].ToolCalls[j] = clients.ToolCall{
					Function: clients.Function{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}
	}
	return clientMsgs
}

func (w *ContextManagerWrapper) GetStats() (int, int, int) {
	return w.cm.GetStats()
}

func (w *ContextManagerWrapper) SetMaxTokens(maxTokens int) {
	w.cm.SetMaxTokens(maxTokens)
}

func (w *ContextManagerWrapper) ClearMessages() {
	w.cm.Clear()
}

func (w *ContextManagerWrapper) CompactContext(compactFunc func([]clients.ChatMessage) (string, error)) error {
	// Create a bridge function that converts between types
	bridgeFunc := func(msgs []ChatMessage) (string, error) {
		clientMsgs := make([]clients.ChatMessage, len(msgs))
		for i, msg := range msgs {
			clientMsgs[i] = clients.ChatMessage{
				Role:    msg.Role,
				Content: msg.Content,
			}
		}
		return compactFunc(clientMsgs)
	}
	
	return w.cm.CompactContext(&ContextManagerCompactWrapper{compactFunc: bridgeFunc})
}

// ContextManagerCompactWrapper implements the CompactClient interface
type ContextManagerCompactWrapper struct {
	compactFunc func([]ChatMessage) (string, error)
}

func (w *ContextManagerCompactWrapper) CompactMessages(messages []ChatMessage) (string, error) {
	return w.compactFunc(messages)
}

func (w *ContextManagerWrapper) CanCompact() bool {
	return w.cm.CanCompact()
}

func (w *ContextManagerWrapper) EnablePlanMode() {
	w.cm.SetPlanMode(true)
}

func (w *ContextManagerWrapper) DisablePlanMode() {
	w.cm.SetPlanMode(false)
}

func (w *ContextManagerWrapper) EnableConciseMode() {
	w.cm.SetConciseMode(true)
}

func (w *ContextManagerWrapper) DisableConciseMode() {
	w.cm.SetConciseMode(false)
}

func (w *ContextManagerWrapper) GetActiveTask() *clients.UserTask {
	task := w.cm.GetActiveTask()
	if task == nil {
		return nil
	}
	return &clients.UserTask{
		Goal:   task.Goal,
		Status: task.Status,
	}
}

func (w *ContextManagerWrapper) SetActiveTask(goal string) {
	w.cm.SetActiveTask(goal)
}

func (w *ContextManagerWrapper) CompleteCurrentTask(summary string) {
	w.cm.CompleteCurrentTask(summary)
}