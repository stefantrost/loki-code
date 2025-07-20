package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ToolCallParser handles parsing tool calls from various formats
type ToolCallParser struct {
	debug bool
}

// NewToolCallParser creates a new parser instance
func NewToolCallParser(debug bool) *ToolCallParser {
	return &ToolCallParser{
		debug: debug,
	}
}

// debugLog prints debug messages when debug mode is enabled
func (p *ToolCallParser) debugLog(format string, args ...interface{}) {
	if p.debug {
		fmt.Printf("[PARSER DEBUG] "+format+"\n", args...)
	}
}

// ParseToolCallsFromContent attempts to extract tool calls from message content
func (p *ToolCallParser) ParseToolCallsFromContent(content string) []ToolCall {
	var toolCalls []ToolCall
	
	p.debugLog("Parsing content for tool calls: %s", content)
	
	// Find all potential JSON objects in the content
	jsonObjects := p.extractJSONObjects(content)
	p.debugLog("Found %d potential JSON objects", len(jsonObjects))
	
	for i, jsonStr := range jsonObjects {
		p.debugLog("Attempting to parse JSON object %d: %s", i+1, jsonStr)
		
		var jsonData map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &jsonData); err != nil {
			p.debugLog("Failed to parse JSON object %d: %v", i+1, err)
			continue
		}
		
		p.debugLog("Successfully parsed JSON with keys: %v", getKeys(jsonData))
		
		// Smart format detection
		var name string
		var arguments map[string]interface{}
		
		// Check if it has "function" wrapper (Ollama native format)
		if functionData, hasFunction := jsonData["function"]; hasFunction {
			p.debugLog("Found 'function' wrapper - using Ollama native format")
			if funcMap, ok := functionData.(map[string]interface{}); ok {
				if n, hasName := funcMap["name"].(string); hasName {
					name = n
				}
				if args, hasArgs := funcMap["arguments"].(map[string]interface{}); hasArgs {
					arguments = args
				}
			}
		} else if jsonName, hasName := jsonData["name"].(string); hasName {
			p.debugLog("Found direct 'name' field - using flat JSON format")
			// Direct format (flat JSON)
			name = jsonName
			if args, hasArgs := jsonData["arguments"].(map[string]interface{}); hasArgs {
				arguments = args
			}
		}
		
		// If we successfully extracted name and arguments, create tool call
		if name != "" && arguments != nil {
			p.debugLog("Creating tool call: name=%s, args=%v", name, arguments)
			toolCall := ToolCall{
				Function: ToolFunction{
					Name:      name,
					Arguments: arguments,
				},
			}
			toolCalls = append(toolCalls, toolCall)
		} else {
			p.debugLog("JSON object %d is not a valid tool call - name='%s', arguments=%v", i+1, name, arguments)
		}
	}
	
	return toolCalls
}

// extractJSONObjects finds JSON objects in content using simple first-to-last brace extraction
func (p *ToolCallParser) extractJSONObjects(content string) []string {
	var jsonObjects []string
	
	p.debugLog("Extracting JSON from content using simple brace matching")
	
	// Simple approach: find first { to last }
	firstBrace := strings.Index(content, "{")
	lastBrace := strings.LastIndex(content, "}")
	
	if firstBrace != -1 && lastBrace != -1 && lastBrace > firstBrace {
		jsonStr := content[firstBrace:lastBrace+1]
		p.debugLog("Found JSON candidate from position %d to %d: %s", firstBrace, lastBrace, jsonStr)
		jsonObjects = append(jsonObjects, jsonStr)
	} else {
		p.debugLog("No JSON braces found in content")
	}
	
	return jsonObjects
}