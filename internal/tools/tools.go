package tools

import (
	"fmt"
	"log/slog"
	"strings"

	"loki-code/clients"
	"loki-code/internal/ui"
)

// Confirmation hooks for mutating tools. Tests override these to bypass
// interactive prompts; the defaults read from stdin via the ui helpers.
// In TUI mode main.go calls SetConfirmHooks to replace them with the
// view's suspend-aware versions so the terminal is handed back correctly.
var (
	confirmYN   = ui.PromptUser
	confirmDiff = ui.ShowDiffAndConfirm
)

// SetConfirmHooks replaces the confirmation callbacks used by mutating tools.
// Call this after view selection: in TUI mode pass v.Confirm and
// v.ShowDiffAndConfirm so the alt-screen is suspended around each prompt.
func SetConfirmHooks(
	yn func(string) (bool, error),
	diff func(string, string, string) (bool, error),
) {
	confirmYN = yn
	confirmDiff = diff
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// SmartTruncate is a TruncationPolicy that knows which tools produce
// line-oriented output (list_files / find_files) and trims at line boundaries
// for them while falling back to byte-truncation for everything else.
// main.go registers this with each LLM client so clients/ stays generic.
func SmartTruncate(result, toolName string) string {
	const maxLength = clients.MaxToolResultChars
	if len(result) <= maxLength {
		return result
	}
	switch toolName {
	case "read_file":
		return result[:maxLength-150] + fmt.Sprintf(
			"\n\n... (truncated at %d chars — call read_file again with offset=%d to continue)",
			maxLength-150, maxLength-150)
	case "list_files", "find_files":
		lines := strings.Split(result, "\n")
		var kept []string
		used := 0
		for _, line := range lines {
			if used+len(line)+1 > maxLength-100 {
				break
			}
			kept = append(kept, line)
			used += len(line) + 1
		}
		out := strings.Join(kept, "\n")
		if len(kept) < len(lines) {
			out += fmt.Sprintf("\n... (output truncated - showing %d of %d lines)",
				len(kept), len(lines))
		}
		return out
	default:
		return result[:maxLength-50] + fmt.Sprintf("... (truncated at %d characters)", maxLength-50)
	}
}

// GetAvailableTools returns the JSON-schema list shown to the model. The
// per-tool executor and plan-mode policy live in toolRegistry; the two must
// stay in sync (enforced by TestArchitecture_RegistryCompleteness).
func GetAvailableTools() []clients.Tool {
	baseTools := []clients.Tool{
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "create_file",
				Description: "Create a new file with content. Use after analyzing project structure and planning the implementation.",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to create",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Content to write to the file",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "read_file",
				Description: "Read file contents. Use offset and length to page through large files.",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to read",
					},
					"offset": map[string]interface{}{
						"type":        "integer",
						"description": "Character offset to start reading from (0-based). Default: 0.",
					},
					"length": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of characters to read. Default: read the whole file.",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "update_file",
				Description: "Update existing file content. Always read the file first to understand current implementation before making changes.",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to update",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "New content for the file",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "delete_file",
				Description: "Delete a file",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to delete",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "list_files",
				Description: "List files in a directory",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Directory path to list (default: current directory)",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "exec_command",
				Description: "Execute shell commands safely (whitelist of allowed commands)",
				Arguments: map[string]interface{}{
					"command": map[string]interface{}{
						"type":        "string",
						"description": "Command to execute (from whitelist: find, grep, pwd, tree, wc, sort, uniq, whoami, date, which)",
					},
					"args": map[string]interface{}{
						"type":        "array",
						"description": "Command arguments as array of strings",
						"items":       map[string]interface{}{"type": "string"},
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "find_files",
				Description: "Find files by name pattern using find command",
				Arguments: map[string]interface{}{
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "File name pattern to search for (supports wildcards like *.go)",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Directory to search in (default: current directory)",
					},
					"type": map[string]interface{}{
						"type":        "string",
						"description": "File type filter: 'f' for files only, 'd' for directories only",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "grep_content",
				Description: "Search for patterns in file contents using grep",
				Arguments: map[string]interface{}{
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "Pattern to search for in files",
					},
					"files": map[string]interface{}{
						"type":        "string",
						"description": "File path or pattern (e.g., '*.go', 'src/*.py')",
					},
					"options": map[string]interface{}{
						"type":        "string",
						"description": "Grep options: -i (ignore case), -n (line numbers), -r (recursive), -l (files only)",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "get_pwd",
				Description: "Get current working directory",
				Arguments:   map[string]interface{}{},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "tree_view",
				Description: "Show directory tree structure",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Directory path to show tree for (default: current directory)",
					},
					"depth": map[string]interface{}{
						"type":        "number",
						"description": "Maximum depth to show (default: 3)",
					},
				},
			},
		},
		{
			Type: "function",
			Function: clients.ToolFunc{
				Name:        "http_request",
				Description: "Make HTTP requests using curl. Execute HTTP operations to interact with APIs and web services.",
				Arguments: map[string]interface{}{
					"url": map[string]interface{}{
						"type":        "string",
						"description": "Target URL (must be http:// or https://)",
					},
					"method": map[string]interface{}{
						"type":        "string",
						"description": "HTTP method: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS (default: GET)",
					},
					"headers": map[string]interface{}{
						"type":        "object",
						"description": "Custom headers as key-value pairs (e.g., {\"Content-Type\": \"application/json\"})",
					},
					"data": map[string]interface{}{
						"type":        "string",
						"description": "Request body data for POST/PUT requests",
					},
					"timeout": map[string]interface{}{
						"type":        "number",
						"description": "Request timeout in seconds (max: 30, default: 10)",
					},
					"follow_redirects": map[string]interface{}{
						"type":        "boolean",
						"description": "Follow HTTP redirects (default: true)",
					},
				},
			},
		},
	}

	projectInfo := detectProject()
	if len(projectInfo.Analyzers) > 0 {
		baseTools = append(baseTools, createAnalyzerTool(projectInfo))
	}

	return baseTools
}

func createAnalyzerTool(projectInfo ProjectInfo) clients.Tool {
	availableAnalyzers := getAnalyzerNames(projectInfo.Analyzers)
	description := fmt.Sprintf("Run static analysis for %s project (available: %s)",
		projectInfo.Language,
		strings.Join(availableAnalyzers, ", "))

	return clients.Tool{
		Type: "function",
		Function: clients.ToolFunc{
			Name:        "analyze_code",
			Description: description,
			Arguments: map[string]interface{}{
				"scope": map[string]interface{}{
					"type":        "string",
					"description": "Analysis scope: 'file' (specific file), 'package' (current package/directory), or 'all' (entire project)",
				},
				"file_path": map[string]interface{}{
					"type":        "string",
					"description": "Specific file to analyze (required for scope='file')",
				},
				"analyzer": map[string]interface{}{
					"type":        "string",
					"description": fmt.Sprintf("Analyzer to use: %s (default: auto-select best)", strings.Join(availableAnalyzers, ", ")),
				},
				"fix": map[string]interface{}{
					"type":        "boolean",
					"description": "Auto-fix issues when supported (default: false)",
				},
			},
		},
	}
}

// ExecuteToolWithPlanMode executes a tool call with optional plan mode restriction.
// Dispatch and plan-mode policy come from toolRegistry; adding a tool there is
// the only edit needed.
func ExecuteToolWithPlanMode(toolCall clients.ToolCall, planMode bool) (string, error) {
	name := toolCall.Function.Name
	slog.Debug("ExecuteToolWithPlanMode called", "tool_name", name,
		"tool_id", toolCall.ID, "plan_mode", planMode, "arguments", toolCall.Function.Arguments)

	entry, ok := toolRegistry[name]
	if !ok {
		slog.Error("Unknown tool requested", "tool_name", name)
		return "", fmt.Errorf("unknown tool: %s", name)
	}

	if planMode && !entry.planAllowed {
		slog.Warn("Tool execution blocked in plan mode", "tool_name", name)
		return fmt.Sprintf("⚠️ Plan Mode: Cannot execute '%s'. This tool is restricted in plan mode.\n\nConsider adding this operation to your execution plan:\n- %s with the specified parameters",
			name, name), nil
	}

	slog.Debug("Executing tool", "tool_name", name)
	result, err := entry.exec(toolCall.Function.Arguments)
	if err != nil {
		slog.Error("Tool execution failed", "tool_name", name, "error", err)
		return result, err
	}

	slog.Debug("Tool execution completed successfully", "tool_name", name,
		"result_length", len(result), "result_preview", truncateString(result, 500))
	return result, nil
}

func isToolAllowedInPlanMode(name string) bool {
	if entry, ok := toolRegistry[name]; ok {
		return entry.planAllowed
	}
	return false
}
