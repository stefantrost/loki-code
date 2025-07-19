package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type ToolCall struct {
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

func GetAvailableTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "create_file",
				Description: "Create a new file with specified content",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "File path to create",
						},
						"content": map[string]interface{}{
							"type":        "string",
							"description": "Content to write to the file",
						},
					},
					"required": []string{"path", "content"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "read_file",
				Description: "Read the contents of a file",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "File path to read",
						},
					},
					"required": []string{"path"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "update_file",
				Description: "Update an existing file with new content",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "File path to update",
						},
						"content": map[string]interface{}{
							"type":        "string",
							"description": "New content for the file",
						},
					},
					"required": []string{"path", "content"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "delete_file",
				Description: "Delete a file",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "File path to delete",
						},
					},
					"required": []string{"path"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "list_files",
				Description: "List files in a directory",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Directory path to list (default: current directory)",
						},
					},
					"required": []string{},
				},
			},
		},
	}
}

func ExecuteTool(toolCall ToolCall) (string, error) {
	return ExecuteToolWithPlanMode(toolCall, false)
}

func ExecuteToolWithPlanMode(toolCall ToolCall, planMode bool) (string, error) {
	// Check if tool is allowed in plan mode
	if planMode && !isToolAllowedInPlanMode(toolCall.Function.Name) {
		return fmt.Sprintf("⚠️ Plan Mode: Cannot execute '%s'. This tool is restricted in plan mode.\n\nConsider adding this operation to your execution plan:\n- %s with the specified parameters", 
			toolCall.Function.Name, toolCall.Function.Name), nil
	}
	
	switch toolCall.Function.Name {
	case "create_file":
		return executeCreateFile(toolCall.Function.Arguments)
	case "read_file":
		return executeReadFile(toolCall.Function.Arguments)
	case "update_file":
		return executeUpdateFile(toolCall.Function.Arguments)
	case "delete_file":
		return executeDeleteFile(toolCall.Function.Arguments)
	case "list_files":
		return executeListFiles(toolCall.Function.Arguments)
	default:
		return "", fmt.Errorf("unknown tool: %s", toolCall.Function.Name)
	}
}

func isToolAllowedInPlanMode(toolName string) bool {
	allowedTools := map[string]bool{
		"read_file":  true,
		"list_files": true,
	}
	return allowedTools[toolName]
}

func executeCreateFile(args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("path argument is required and must be a string")
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("content argument is required and must be a string")
	}

	if err := validatePath(path); err != nil {
		return "", err
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %v", err)
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("failed to create file: %v", err)
	}

	return fmt.Sprintf("File created successfully: %s", path), nil
}

func executeReadFile(args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("path argument is required and must be a string")
	}

	if err := validatePath(path); err != nil {
		return "", err
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %v", err)
	}

	return string(content), nil
}

func executeUpdateFile(args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("path argument is required and must be a string")
	}

	newContent, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("content argument is required and must be a string")
	}

	if err := validatePath(path); err != nil {
		return "", err
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", path)
	}

	// Read current file content
	currentContent, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read current file: %v", err)
	}

	currentContentStr := string(currentContent)
	
	// Check if content is actually different
	if currentContentStr == newContent {
		return fmt.Sprintf("No changes needed for %s (content is identical)", path), nil
	}

	// Show diff and get user confirmation
	confirmed, err := showDiffAndConfirm(currentContentStr, newContent, path)
	if err != nil {
		return "", fmt.Errorf("failed to get user confirmation: %v", err)
	}

	if !confirmed {
		return fmt.Sprintf("File update cancelled by user: %s", path), nil
	}

	// Apply the changes
	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("failed to update file: %v", err)
	}

	return fmt.Sprintf("File updated successfully: %s", path), nil
}

func executeDeleteFile(args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("path argument is required and must be a string")
	}

	if err := validatePath(path); err != nil {
		return "", err
	}

	if err := os.Remove(path); err != nil {
		return "", fmt.Errorf("failed to delete file: %v", err)
	}

	return fmt.Sprintf("File deleted successfully: %s", path), nil
}

func executeListFiles(args map[string]interface{}) (string, error) {
	path := "."
	if p, ok := args["path"].(string); ok && p != "" {
		path = p
	}

	if err := validatePath(path); err != nil {
		return "", err
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("failed to list directory: %v", err)
	}

	var files []string
	for _, entry := range entries {
		if entry.IsDir() {
			files = append(files, entry.Name()+"/")
		} else {
			files = append(files, entry.Name())
		}
	}

	if len(files) == 0 {
		return "Directory is empty", nil
	}

	return "Files in " + path + ":\n" + strings.Join(files, "\n"), nil
}

func validatePath(path string) error {
	cleanPath := filepath.Clean(path)
	
	if strings.Contains(cleanPath, "..") {
		return fmt.Errorf("path traversal not allowed: %s", path)
	}

	if filepath.IsAbs(cleanPath) {
		absPath, err := filepath.Abs(".")
		if err != nil {
			return fmt.Errorf("failed to get current directory: %v", err)
		}
		if !strings.HasPrefix(cleanPath, absPath) {
			return fmt.Errorf("access outside current directory not allowed: %s", path)
		}
	}

	return nil
}

