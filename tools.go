package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"loki-code/clients"
)

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
				Description: "Read the contents of a file",
				Arguments: map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to read",
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
						"type": "array",
						"description": "Command arguments as array of strings",
						"items": map[string]interface{}{
							"type": "string",
						},
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
				Arguments: map[string]interface{}{},
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
		analyzerTool := createAnalyzerTool(projectInfo)
		baseTools = append(baseTools, analyzerTool)
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

// ExecuteToolWithPlanMode executes a tool call with optional plan mode restriction
func ExecuteToolWithPlanMode(toolCall clients.ToolCall, planMode bool) (string, error) {
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
	case "analyze_code":
		return executeAnalyzeCode(toolCall.Function.Arguments)
	case "http_request":
		return executeCurl(toolCall.Function.Arguments)
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
		"analyze_code": true,
		"http_request": true,
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

	currentContent, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read current file: %v", err)
	}

	currentContentStr := string(currentContent)

	if currentContentStr == newContent {
		return fmt.Sprintf("No changes needed for %s (content is identical)", path), nil
	}

	confirmed, err := showDiffAndConfirm(currentContentStr, newContent, path)
	if err != nil {
		return "", fmt.Errorf("failed to get user confirmation: %v", err)
	}

	if !confirmed {
		return fmt.Sprintf("File update cancelled by user: %s", path), nil
	}

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

// Utility functions for project detection
func commandExists(command string) bool {
	_, err := exec.LookPath(command)
	return err == nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func findConfigFile(candidates []string) string {
	for _, file := range candidates {
		if fileExists(file) {
			return file
		}
	}
	return ""
}

func hasPythonFiles() bool {
	pythonIndicators := []string{
		"requirements.txt", "pyproject.toml", "setup.py",
		"setup.cfg", "Pipfile", "poetry.lock",
	}

	for _, indicator := range pythonIndicators {
		if fileExists(indicator) {
			return true
		}
	}

	entries, err := os.ReadDir(".")
	if err != nil {
		return false
	}

	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".py") {
			return true
		}
	}

	return false
}

// Project detection functions
type ProjectInfo struct {
	Language    string
	HasConfig   bool
	ConfigFiles []string
	Analyzers   []AnalyzerInfo
}

type AnalyzerInfo struct {
	Name       string
	Command    string
	Args       []string
	Available  bool
	ConfigFile string
}

func detectProject() ProjectInfo {
	if fileExists("go.mod") {
		return detectGoProject()
	}

	if fileExists("package.json") {
		return detectNodeProject()
	}

	if hasPythonFiles() {
		return detectPythonProject()
	}

	return ProjectInfo{Language: "unknown"}
}

func detectGoProject() ProjectInfo {
	analyzers := []AnalyzerInfo{}

	if commandExists("golangci-lint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "golangci-lint",
			Command:    "golangci-lint",
			Args:       []string{"run"},
			Available:  true,
			ConfigFile: findConfigFile([]string{".golangci.yml", ".golangci.yaml"}),
		})
	}

	if commandExists("go") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "go vet",
			Command:   "go",
			Args:      []string{"vet", "./..."},
			Available: true,
		})
	}

	return ProjectInfo{
		Language:  "go",
		HasConfig: fileExists("go.mod"),
		Analyzers: analyzers,
	}
}

func detectPythonProject() ProjectInfo {
	analyzers := []AnalyzerInfo{}

	if commandExists("ruff") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "ruff",
			Command:    "ruff",
			Args:       []string{"check"},
			Available:  true,
			ConfigFile: findConfigFile([]string{"ruff.toml", "pyproject.toml"}),
		})
	}

	if commandExists("pylint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "pylint",
			Command:    "pylint",
			Args:       []string{},
			Available:  true,
			ConfigFile: findConfigFile([]string{".pylintrc", "pylint.ini"}),
		})
	}

	if commandExists("flake8") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "flake8",
			Command:    "flake8",
			Args:       []string{},
			Available:  true,
			ConfigFile: findConfigFile([]string{".flake8", "setup.cfg"}),
		})
	}

	if commandExists("python3") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "python3",
			Command:   "python3",
			Args:      []string{"-m", "py_compile"},
			Available: true,
		})
	} else if commandExists("python") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "python",
			Command:   "python",
			Args:      []string{"-m", "py_compile"},
			Available: true,
		})
	}

	hasConfig := fileExists("pyproject.toml") || fileExists("requirements.txt") || fileExists("setup.py")

	return ProjectInfo{
		Language:  "python",
		HasConfig: hasConfig,
		Analyzers: analyzers,
	}
}

func detectNodeProject() ProjectInfo {
	analyzers := []AnalyzerInfo{}
	isTypeScript := fileExists("tsconfig.json")

	if isTypeScript && commandExists("tsc") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "tsc",
			Command:    "tsc",
			Args:       []string{"--noEmit"},
			Available:  true,
			ConfigFile: "tsconfig.json",
		})
	}

	eslintConfig := findConfigFile([]string{
		".eslintrc.js", ".eslintrc.json", ".eslintrc.yml",
		".eslintrc.yaml", "eslint.config.js", ".eslintrc",
	})
	if eslintConfig != "" && commandExists("eslint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "eslint",
			Command:    "eslint",
			Args:       []string{},
			Available:  true,
			ConfigFile: eslintConfig,
		})
	}

	if commandExists("jshint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "jshint",
			Command:   "jshint",
			Args:      []string{},
			Available: true,
		})
	}

	if commandExists("node") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "node",
			Command:   "node",
			Args:      []string{"--check"},
			Available: true,
		})
	}

	language := "javascript"
	if isTypeScript {
		language = "typescript"
	}

	return ProjectInfo{
		Language:  language,
		HasConfig: fileExists("package.json"),
		Analyzers: analyzers,
	}
}

func getAnalyzerNames(analyzers []AnalyzerInfo) []string {
	var names []string
	for _, analyzer := range analyzers {
		if analyzer.Available {
			names = append(names, analyzer.Name)
		}
	}
	return names
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
}

func executeCommand(args map[string]interface{}) (string, error) {
	command, ok := args["command"].(string)
	if !ok {
		return "", fmt.Errorf("command argument is required and must be a string")
	}

	if !allowedCommands[command] {
		return "", fmt.Errorf("command '%s' is not allowed. Allowed commands: %s",
			command, getAvailableCommands())
	}

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

	cmd := exec.Command(command, cmdArgs...)
	cmd.Dir = "."

	timeout := 30 * time.Second

	done := make(chan error, 1)
	var output []byte
	var err error

	go func() {
		output, err = cmd.CombinedOutput()
		done <- err
	}()

	select {
	case cmdErr := <-done:
		if cmdErr != nil {
			return "", fmt.Errorf("command failed: %v\nOutput: %s", cmdErr, string(output))
		}

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

	searchPath := "."
	if path, hasPath := args["path"].(string); hasPath {
		if err := validatePath(path); err != nil {
			return "", err
		}
		searchPath = path
	}

	findArgs := []string{searchPath, "-name", pattern}

	if fileType, hasType := args["type"].(string); hasType {
		if fileType == "f" || fileType == "d" {
			findArgs = append(findArgs, "-type", fileType)
		}
	}

	cmd := exec.Command("find", findArgs...)
	output, err := cmd.CombinedOutput()

	if err != nil {
		return "", fmt.Errorf("find command failed: %v\nOutput: %s", err, string(output))
	}

	outputStr := strings.TrimSpace(string(output))
	if outputStr == "" {
		return fmt.Sprintf("No files found matching pattern '%s' in %s", pattern, searchPath), nil
	}

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

	grepArgs := []string{pattern}

	if options, hasOptions := args["options"].(string); hasOptions {
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

	grepArgs = append(grepArgs, files)

	cmd := exec.Command("grep", grepArgs...)
	output, err := cmd.CombinedOutput()

	if err != nil && cmd.ProcessState.ExitCode() != 1 {
		return "", fmt.Errorf("grep command failed: %v\nOutput: %s", err, string(output))
	}

	outputStr := strings.TrimSpace(string(output))
	if outputStr == "" {
		return fmt.Sprintf("No matches found for pattern '%s' in %s", pattern, files), nil
	}

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
	treePath := "."
	if path, hasPath := args["path"].(string); hasPath {
		if err := validatePath(path); err != nil {
			return "", err
		}
		treePath = path
	}

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

	if depth > 10 {
		depth = 10
	}

	cmd := exec.Command("tree", "-L", strconv.Itoa(depth), treePath)
	output, err := cmd.CombinedOutput()

	if err != nil {
		cmd = exec.Command("ls", "-la", treePath)
		output, err = cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("tree/ls command failed: %v", err)
		}
		return fmt.Sprintf("Directory listing for %s:\n%s", treePath, string(output)), nil
	}

	outputStr := string(output)

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

func executeAnalyzeCode(args map[string]interface{}) (string, error) {
	projectInfo := detectProject()
	if len(projectInfo.Analyzers) == 0 {
		return fmt.Sprintf("No static analyzers detected for %s projects in current directory", projectInfo.Language), nil
	}

	scope := "package"
	if s, hasScope := args["scope"].(string); hasScope {
		scope = s
	}

	var filePath string
	if fp, hasFilePath := args["file_path"].(string); hasFilePath {
		filePath = fp
	}

	if filePath != "" {
		if err := validatePath(filePath); err != nil {
			return "", err
		}
		if !fileExists(filePath) {
			return "", fmt.Errorf("file not found: %s", filePath)
		}
	}

	analyzer := selectAnalyzer(args, projectInfo.Analyzers)
	if analyzer == nil {
		return "No suitable analyzer found", nil
	}

	cmdArgs, err := buildAnalyzerArgs(*analyzer, scope, filePath, args)
	if err != nil {
		return "", err
	}

	result, err := runAnalyzer(*analyzer, cmdArgs)
	if err != nil {
		return fmt.Sprintf("Analysis failed with %s: %v\nOutput: %s", analyzer.Name, err, result), nil
	}

	return formatAnalysisResults(analyzer.Name, result, projectInfo.Language, scope, filePath), nil
}

func selectAnalyzer(args map[string]interface{}, analyzers []AnalyzerInfo) *AnalyzerInfo {
	if analyzerName, hasAnalyzer := args["analyzer"].(string); hasAnalyzer {
		for _, analyzer := range analyzers {
			if analyzer.Available && analyzer.Name == analyzerName {
				return &analyzer
			}
		}
	}

	for _, analyzer := range analyzers {
		if analyzer.Available {
			return &analyzer
		}
	}

	return nil
}

func buildAnalyzerArgs(analyzer AnalyzerInfo, scope, filePath string, args map[string]interface{}) ([]string, error) {
	cmdArgs := make([]string, len(analyzer.Args))
	copy(cmdArgs, analyzer.Args)

	shouldFix := false
	if fix, hasFix := args["fix"].(bool); hasFix {
		shouldFix = fix
	}

	switch analyzer.Name {
	case "golangci-lint":
		if shouldFix {
			cmdArgs = append(cmdArgs, "--fix")
		}
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else if scope == "package" {
			cmdArgs = append(cmdArgs, "./...")
		}

	case "go vet":
		if scope == "file" && filePath != "" {
			cmdArgs = []string{"vet", filePath}
		}

	case "ruff":
		if shouldFix {
			cmdArgs = []string{"check", "--fix"}
		}
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			cmdArgs = append(cmdArgs, ".")
		}

	case "pylint":
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			cmdArgs = append(cmdArgs, ".")
		}

	case "flake8":
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			cmdArgs = append(cmdArgs, ".")
		}

	case "tsc":
		// tsc --noEmit doesn't need file-specific args

	case "eslint":
		if shouldFix {
			cmdArgs = append(cmdArgs, "--fix")
		}
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			cmdArgs = append(cmdArgs, ".")
		}

	case "jshint":
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			cmdArgs = append(cmdArgs, ".")
		}

	case "node":
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			return nil, fmt.Errorf("node --check requires a specific file")
		}

	case "python3", "python":
		if scope == "file" && filePath != "" {
			cmdArgs = append(cmdArgs, filePath)
		} else {
			return nil, fmt.Errorf("python syntax check requires a specific file")
		}
	}

	return cmdArgs, nil
}

func runAnalyzer(analyzer AnalyzerInfo, cmdArgs []string) (string, error) {
	cmd := exec.Command(analyzer.Command, cmdArgs...)
	cmd.Dir = "."

	timeout := 60 * time.Second

	done := make(chan error, 1)
	var output []byte
	var err error

	go func() {
		output, err = cmd.CombinedOutput()
		done <- err
	}()

	select {
	case cmdErr := <-done:
		outputStr := string(output)

		if cmdErr != nil {
			if cmd.ProcessState != nil && cmd.ProcessState.ExitCode() > 2 {
				return outputStr, cmdErr
			}
		}

		if len(outputStr) > 15000 {
			outputStr = outputStr[:15000] + "\n... (output truncated at 15,000 characters)"
		}

		return outputStr, nil

	case <-time.After(timeout):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		return "", fmt.Errorf("analysis timed out after %v", timeout)
	}
}

func formatAnalysisResults(analyzerName, output, language, scope, filePath string) string {
	var result strings.Builder

	result.WriteString(fmt.Sprintf("🔍 Static Analysis Results (%s)\n", analyzerName))
	result.WriteString(fmt.Sprintf("Language: %s | Scope: %s", language, scope))
	if filePath != "" {
		result.WriteString(fmt.Sprintf(" | File: %s", filePath))
	}
	result.WriteString("\n")
	result.WriteString(strings.Repeat("=", 50) + "\n\n")

	if strings.TrimSpace(output) == "" {
		result.WriteString("✅ No issues found!\n")
	} else {
		result.WriteString(output)

		result.WriteString("\n" + strings.Repeat("-", 30) + "\n")
		switch analyzerName {
		case "golangci-lint":
			result.WriteString("💡 Tip: Use 'analyze_code' with 'fix': true to auto-fix some issues")
		case "ruff":
			result.WriteString("💡 Tip: Use 'analyze_code' with 'fix': true to auto-fix formatting and imports")
		case "eslint":
			result.WriteString("💡 Tip: Use 'analyze_code' with 'fix': true to auto-fix style issues")
		case "tsc":
			result.WriteString("💡 TypeScript type checking complete. Fix type errors to improve code safety.")
		}
	}

	return result.String()
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

func executeCurl(args map[string]interface{}) (string, error) {
	url, ok := args["url"].(string)
	if !ok {
		return "", fmt.Errorf("url argument is required and must be a string")
	}

	if err := validateURL(url); err != nil {
		return "", err
	}

	method := "GET"
	if methodArg, hasMethod := args["method"].(string); hasMethod {
		method = strings.ToUpper(methodArg)
		if !isValidHTTPMethod(method) {
			return "", fmt.Errorf("invalid HTTP method: %s", methodArg)
		}
	}

	timeout := 10
	if timeoutArg, hasTimeout := args["timeout"].(float64); hasTimeout {
		if timeoutArg > 30 {
			return "", fmt.Errorf("timeout cannot exceed 30 seconds")
		}
		if timeoutArg < 1 {
			return "", fmt.Errorf("timeout must be at least 1 second")
		}
		timeout = int(timeoutArg)
	}

	followRedirects := true
	if followArg, hasFollow := args["follow_redirects"].(bool); hasFollow {
		followRedirects = followArg
	}

	curlArgs := []string{
		"--silent",
		"--show-error",
		"--max-time", fmt.Sprintf("%d", timeout),
		"--request", method,
	}

	if followRedirects {
		curlArgs = append(curlArgs, "--location")
	}

	if headersArg, hasHeaders := args["headers"]; hasHeaders {
		if headers, ok := headersArg.(map[string]interface{}); ok {
			for key, value := range headers {
				headerStr := fmt.Sprintf("%s: %v", key, value)
				if err := validateHeader(headerStr); err != nil {
					return "", fmt.Errorf("invalid header %s: %v", key, err)
				}
				curlArgs = append(curlArgs, "--header", headerStr)
			}
		}
	}

	if dataArg, hasData := args["data"].(string); hasData && dataArg != "" {
		if method == "GET" || method == "HEAD" {
			return "", fmt.Errorf("cannot send data with %s method", method)
		}
		curlArgs = append(curlArgs, "--data", dataArg)
	}

	curlArgs = append(curlArgs, url)

	cmd := exec.Command("curl", curlArgs...)
	cmd.Dir = "."

	done := make(chan error, 1)
	var output []byte
	var err error

	go func() {
		output, err = cmd.CombinedOutput()
		done <- err
	}()

	select {
	case cmdErr := <-done:
		outputStr := string(output)

		if cmdErr != nil {
			return "", fmt.Errorf("curl command failed: %v\nOutput: %s", cmdErr, outputStr)
		}

		if len(outputStr) > 10000 {
			outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
		}

		return formatHTTPResponse(method, url, outputStr), nil

	case <-time.After(time.Duration(timeout+5) * time.Second):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		return "", fmt.Errorf("HTTP request timed out after %d seconds", timeout)
	}
}

func validateURL(url string) error {
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return fmt.Errorf("URL must start with http:// or https://")
	}

	if strings.Contains(url, "://") {
		parts := strings.SplitN(url, "://", 2)
		if len(parts) > 1 {
			hostPart := parts[1]
			if strings.Contains(hostPart, "/") {
				hostPart = strings.SplitN(hostPart, "/", 2)[0]
			}
			if strings.Contains(hostPart, ":") {
				hostPart = strings.SplitN(hostPart, ":", 2)[0]
			}

			prohibitedHosts := []string{
				"localhost", "127.0.0.1", "::1",
				"0.0.0.0", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
				"172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
				"172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
				"192.168.",
			}

			hostLower := strings.ToLower(hostPart)
			for _, prohibited := range prohibitedHosts {
				if hostLower == prohibited || strings.HasPrefix(hostLower, prohibited) {
					return fmt.Errorf("access to internal/localhost addresses is not allowed")
				}
			}
		}
	}

	return nil
}

func isValidHTTPMethod(method string) bool {
	validMethods := map[string]bool{
		"GET": true, "POST": true, "PUT": true, "DELETE": true,
		"PATCH": true, "HEAD": true, "OPTIONS": true,
	}
	return validMethods[method]
}

func validateHeader(header string) error {
	if !strings.Contains(header, ":") {
		return fmt.Errorf("header must be in 'Key: Value' format")
	}

	if strings.Contains(header, "\n") || strings.Contains(header, "\r") {
		return fmt.Errorf("headers cannot contain newline characters")
	}

	return nil
}

func formatHTTPResponse(method, url, response string) string {
	var result strings.Builder

	result.WriteString(fmt.Sprintf("🌐 HTTP %s Request\n", method))
	result.WriteString(fmt.Sprintf("URL: %s\n", url))
	result.WriteString(strings.Repeat("=", 50) + "\n\n")

	if strings.TrimSpace(response) == "" {
		result.WriteString("(Empty response)\n")
	} else {
		result.WriteString(response)
	}

	return result.String()
}
