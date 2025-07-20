package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
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
		{
			Type: "function",
			Function: Function{
				Name:        "exec_command",
				Description: "Execute shell commands safely (whitelist of allowed commands)",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"command": map[string]interface{}{
							"type":        "string",
							"description": "Command to execute (from whitelist: find, grep, pwd, tree, wc, sort, uniq, whoami, date, which)",
						},
						"args": map[string]interface{}{
							"type":        "array",
							"description": "Command arguments as array of strings",
							"items": map[string]interface{}{
								"type": "string",
							},
						},
					},
					"required": []string{"command"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "find_files",
				Description: "Find files by name pattern using find command",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
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
					"required": []string{"pattern"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "grep_content",
				Description: "Search for patterns in file contents using grep",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
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
					"required": []string{"pattern", "files"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "get_pwd",
				Description: "Get current working directory",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
					"required":   []string{},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "tree_view",
				Description: "Show directory tree structure",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Directory path to show tree for (default: current directory)",
						},
						"depth": map[string]interface{}{
							"type":        "number",
							"description": "Maximum depth to show (default: 3)",
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
	case "exec_command":
		return executeCommand(toolCall.Function.Arguments)
	case "find_files":
		return executeFindFiles(toolCall.Function.Arguments)
	case "grep_content":
		return executeGrepContent(toolCall.Function.Arguments)
	case "get_pwd":
		return executeGetPwd(toolCall.Function.Arguments)
	case "tree_view":
		return executeTreeView(toolCall.Function.Arguments)
	default:
		return "", fmt.Errorf("unknown tool: %s", toolCall.Function.Name)
	}
}

func isToolAllowedInPlanMode(toolName string) bool {
	allowedTools := map[string]bool{
		"read_file":    true,
		"list_files":   true,
		"find_files":   true,
		"grep_content": true,
		"get_pwd":      true,
		"tree_view":    true,
		// exec_command is NOT allowed in plan mode for security
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

// Command whitelist for security
var allowedCommands = map[string]bool{
	"find":   true,
	"grep":   true,
	"pwd":    true,
	"tree":   true,
	"wc":     true,
	"sort":   true,
	"uniq":   true,
	"whoami": true,
	"date":   true,
	"which":  true,
	// Removed: "ls" (covered by list_files), "cat"/"head"/"tail" (covered by read_file)
}

func executeCommand(args map[string]interface{}) (string, error) {
	command, ok := args["command"].(string)
	if !ok {
		return "", fmt.Errorf("command argument is required and must be a string")
	}

	// Security check: command must be in whitelist
	if !allowedCommands[command] {
		return "", fmt.Errorf("command '%s' is not allowed. Allowed commands: %s", 
			command, getAvailableCommands())
	}

	// Parse arguments
	var cmdArgs []string
	if argsInterface, hasArgs := args["args"]; hasArgs {
		if argsList, ok := argsInterface.([]interface{}); ok {
			for _, arg := range argsList {
				if argStr, ok := arg.(string); ok {
					cmdArgs = append(cmdArgs, argStr)
				}
			}
		}
	}

	// Create and configure command
	cmd := exec.Command(command, cmdArgs...)
	cmd.Dir = "." // Run in current directory
	
	// Set timeout to prevent hanging
	timeout := 30 * time.Second
	
	// Execute with timeout
	done := make(chan error, 1)
	var output []byte
	var err error
	
	go func() {
		output, err = cmd.CombinedOutput()
		done <- err
	}()
	
	select {
	case err := <-done:
		if err != nil {
			return "", fmt.Errorf("command failed: %v\nOutput: %s", err, string(output))
		}
		
		// Limit output size to prevent memory issues
		outputStr := string(output)
		if len(outputStr) > 10000 {
			outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
		}
		
		return outputStr, nil
		
	case <-time.After(timeout):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		return "", fmt.Errorf("command timed out after %v", timeout)
	}
}

func executeFindFiles(args map[string]interface{}) (string, error) {
	pattern, ok := args["pattern"].(string)
	if !ok {
		return "", fmt.Errorf("pattern argument is required and must be a string")
	}

	// Default path
	searchPath := "."
	if path, hasPath := args["path"].(string); hasPath {
		if err := validatePath(path); err != nil {
			return "", err
		}
		searchPath = path
	}

	// Build find command
	findArgs := []string{searchPath, "-name", pattern}
	
	// Add type filter if specified
	if fileType, hasType := args["type"].(string); hasType {
		if fileType == "f" || fileType == "d" {
			findArgs = append(findArgs, "-type", fileType)
		}
	}

	// Execute find command
	cmd := exec.Command("find", findArgs...)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		return "", fmt.Errorf("find command failed: %v\nOutput: %s", err, string(output))
	}

	outputStr := strings.TrimSpace(string(output))
	if outputStr == "" {
		return fmt.Sprintf("No files found matching pattern '%s' in %s", pattern, searchPath), nil
	}

	// Limit output size
	if len(outputStr) > 10000 {
		outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
	}

	return fmt.Sprintf("Files found matching '%s':\n%s", pattern, outputStr), nil
}

func executeGrepContent(args map[string]interface{}) (string, error) {
	pattern, ok := args["pattern"].(string)
	if !ok {
		return "", fmt.Errorf("pattern argument is required and must be a string")
	}

	files, ok := args["files"].(string)
	if !ok {
		return "", fmt.Errorf("files argument is required and must be a string")
	}

	// Build grep command
	grepArgs := []string{pattern}
	
	// Add options if specified
	if options, hasOptions := args["options"].(string); hasOptions {
		// Parse options safely
		if strings.Contains(options, "-i") {
			grepArgs = append([]string{"-i"}, grepArgs...)
		}
		if strings.Contains(options, "-n") {
			grepArgs = append([]string{"-n"}, grepArgs...)
		}
		if strings.Contains(options, "-r") {
			grepArgs = append([]string{"-r"}, grepArgs...)
		}
		if strings.Contains(options, "-l") {
			grepArgs = append([]string{"-l"}, grepArgs...)
		}
	}
	
	// Add files pattern
	grepArgs = append(grepArgs, files)

	// Execute grep command
	cmd := exec.Command("grep", grepArgs...)
	output, err := cmd.CombinedOutput()
	
	// grep returns exit code 1 when no matches found, which is not an error
	if err != nil && cmd.ProcessState.ExitCode() != 1 {
		return "", fmt.Errorf("grep command failed: %v\nOutput: %s", err, string(output))
	}

	outputStr := strings.TrimSpace(string(output))
	if outputStr == "" {
		return fmt.Sprintf("No matches found for pattern '%s' in %s", pattern, files), nil
	}

	// Limit output size
	if len(outputStr) > 10000 {
		outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
	}

	return fmt.Sprintf("Matches for '%s':\n%s", pattern, outputStr), nil
}

func executeGetPwd(args map[string]interface{}) (string, error) {
	cmd := exec.Command("pwd")
	output, err := cmd.Output()
	
	if err != nil {
		return "", fmt.Errorf("pwd command failed: %v", err)
	}

	return fmt.Sprintf("Current directory: %s", strings.TrimSpace(string(output))), nil
}

func executeTreeView(args map[string]interface{}) (string, error) {
	// Default path
	treePath := "."
	if path, hasPath := args["path"].(string); hasPath {
		if err := validatePath(path); err != nil {
			return "", err
		}
		treePath = path
	}

	// Default depth
	depth := 3
	if depthInterface, hasDepth := args["depth"]; hasDepth {
		if depthFloat, ok := depthInterface.(float64); ok {
			depth = int(depthFloat)
		} else if depthStr, ok := depthInterface.(string); ok {
			if d, err := strconv.Atoi(depthStr); err == nil {
				depth = d
			}
		}
	}

	// Limit depth for safety
	if depth > 10 {
		depth = 10
	}

	// Try tree command first, fall back to ls if not available
	cmd := exec.Command("tree", "-L", strconv.Itoa(depth), treePath)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		// Fallback to ls -la if tree is not available
		cmd = exec.Command("ls", "-la", treePath)
		output, err = cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("tree/ls command failed: %v", err)
		}
		return fmt.Sprintf("Directory listing for %s:\n%s", treePath, string(output)), nil
	}

	outputStr := string(output)
	
	// Limit output size
	if len(outputStr) > 10000 {
		outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
	}

	return outputStr, nil
}

func getAvailableCommands() string {
	var commands []string
	for cmd := range allowedCommands {
		commands = append(commands, cmd)
	}
	return strings.Join(commands, ", ")
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

