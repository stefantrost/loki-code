package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"loki-code/clients"
)

func init() {
	// Auto-confirm in tests; interactive prompts would block.
	confirmYN = func(string) (bool, error) { return true, nil }
	confirmDiff = func(_, _, _ string) (bool, error) { return true, nil }
}

func TestValidatePath(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{"valid relative path", "src/main.go", false},
		{"valid nested path", "src/sub/dir/file.go", false},
		{"path traversal up", "../etc/passwd", true},
		{"path traversal nested", "src/../../etc/passwd", true},
		{"empty path", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePath(tt.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("validatePath(%q) error = %v, wantErr %v", tt.path, err, tt.wantErr)
			}
		})
	}
}

func TestIsToolAllowedInPlanMode(t *testing.T) {
	tests := []struct {
		name     string
		toolName string
		allowed  bool
	}{
		{"read_file is allowed", "read_file", true},
		{"list_files is allowed", "list_files", true},
		{"find_files is allowed", "find_files", true},
		{"grep_content is allowed", "grep_content", true},
		{"get_pwd is allowed", "get_pwd", true},
		{"tree_view is allowed", "tree_view", true},
		{"analyze_code is allowed", "analyze_code", true},
		{"http_request is allowed", "http_request", true},
		{"create_file is NOT allowed", "create_file", false},
		{"update_file is NOT allowed", "update_file", false},
		{"delete_file is NOT allowed", "delete_file", false},
		{"exec_command is NOT allowed", "exec_command", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isToolAllowedInPlanMode(tt.toolName)
			if got != tt.allowed {
				t.Errorf("isToolAllowedInPlanMode(%q) = %v, want %v", tt.toolName, got, tt.allowed)
			}
		})
	}
}

func TestExecuteCreateFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	path := "testfile.txt"
	content := "hello world"

	result, err := executeCreateFile(map[string]interface{}{
		"path":    path,
		"content": content,
	})

	if err != nil {
		t.Fatalf("executeCreateFile error: %v", err)
	}

	expected := "File created successfully: testfile.txt"
	if result != expected {
		t.Errorf("executeCreateFile result = %q, want %q", result, expected)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, path))
	if err != nil {
		t.Fatalf("failed to read created file: %v", err)
	}
	if string(data) != content {
		t.Errorf("file content = %q, want %q", string(data), content)
	}
}

func TestExecuteCreateFileNestedDir(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	path := "sub/dir/file.txt"
	content := "nested content"

	_, err := executeCreateFile(map[string]interface{}{
		"path":    path,
		"content": content,
	})

	if err != nil {
		t.Fatalf("executeCreateFile error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, path))
	if err != nil {
		t.Fatalf("failed to read created file: %v", err)
	}
	if string(data) != content {
		t.Errorf("file content = %q, want %q", string(data), content)
	}
}

func TestExecuteReadFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	testFile := filepath.Join(tmpDir, "test.txt")
	expectedContent := "read me content"
	os.WriteFile(testFile, []byte(expectedContent), 0644)

	result, err := executeReadFile(map[string]interface{}{
		"path": "test.txt",
	})

	if err != nil {
		t.Fatalf("executeReadFile error: %v", err)
	}
	if result != expectedContent {
		t.Errorf("executeReadFile result = %q, want %q", result, expectedContent)
	}
}

func TestExecuteReadFileNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	_, err := executeReadFile(map[string]interface{}{
		"path": "nonexistent.txt",
	})

	if err == nil {
		t.Fatal("executeReadFile expected error for nonexistent file")
	}
}

// ── read_file offset / length tests ──────────────────────────────────────────

func TestExecuteReadFile_Offset(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)
	os.WriteFile(filepath.Join(tmpDir, "f.txt"), []byte("hello world"), 0644)

	result, err := executeReadFile(map[string]interface{}{"path": "f.txt", "offset": 6})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(result, "hello") {
		t.Errorf("result should not contain 'hello' when offset=6: %q", result)
	}
	if !strings.Contains(result, "world") {
		t.Errorf("result should contain 'world': %q", result)
	}
}

func TestExecuteReadFile_Length(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)
	os.WriteFile(filepath.Join(tmpDir, "f.txt"), []byte("hello world"), 0644)

	result, err := executeReadFile(map[string]interface{}{"path": "f.txt", "length": 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("result should contain 'hello': %q", result)
	}
	if strings.Contains(result, "world") {
		t.Errorf("result should not contain 'world' when length=5: %q", result)
	}
}

func TestExecuteReadFile_OffsetAndLength(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)
	os.WriteFile(filepath.Join(tmpDir, "f.txt"), []byte("hello world"), 0644)

	result, err := executeReadFile(map[string]interface{}{"path": "f.txt", "offset": 6, "length": 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "wor") {
		t.Errorf("result should contain 'wor': %q", result)
	}
	if strings.Contains(result, "hello") || strings.Contains(result, "ld") {
		t.Errorf("result should only contain 'wor', got: %q", result)
	}
}

func TestExecuteReadFile_OffsetPastEnd(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)
	os.WriteFile(filepath.Join(tmpDir, "f.txt"), []byte("hi"), 0644)

	result, err := executeReadFile(map[string]interface{}{"path": "f.txt", "offset": 100})
	if err != nil {
		t.Fatalf("unexpected error (should return message, not error): %v", err)
	}
	if !strings.Contains(result, "past end") {
		t.Errorf("result should contain 'past end': %q", result)
	}
}

func TestExecuteListFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile(tmpDir+"file1.txt", []byte("a"), 0644)
	os.WriteFile(tmpDir+"file2.txt", []byte("b"), 0644)
	os.Mkdir(tmpDir+"subdir", 0755)

	result, err := executeListFiles(map[string]interface{}{})
	if err != nil {
		t.Fatalf("executeListFiles error: %v", err)
	}

	if len(result) == 0 {
		t.Fatal("executeListFiles returned empty result")
	}
}

func TestAllowedCommands(t *testing.T) {
	expected := []string{"find", "grep", "pwd", "tree", "wc", "sort", "uniq", "whoami", "date", "which"}
	for _, cmd := range expected {
		if !allowedCommands[cmd] {
			t.Errorf("expected command %q to be in whitelist", cmd)
		}
	}

	dangerous := []string{"rm", "curl", "wget", "bash", "sh", "python", "node"}
	for _, cmd := range dangerous {
		if allowedCommands[cmd] {
			t.Errorf("dangerous command %q should NOT be in whitelist", cmd)
		}
	}
}

func TestValidateURL(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{"valid https url", "https://example.com/api", false},
		{"valid http url", "http://example.com/api", false},
		{"missing protocol", "example.com/api", true},
		{"localhost blocked", "http://localhost:8080/api", true},
		{"127.0.0.1 blocked", "http://127.0.0.1:8080/api", true},
		{"192.168 blocked", "http://192.168.1.1/api", true},
		{"10.x blocked", "http://10.0.0.1/api", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateURL(tt.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateURL(%q) error = %v, wantErr %v", tt.url, err, tt.wantErr)
			}
		})
	}
}

func TestValidateHeader(t *testing.T) {
	tests := []struct {
		name    string
		header  string
		wantErr bool
	}{
		{"valid header", "Content-Type: application/json", false},
		{"valid header with colon in value", "Location: http://example.com:8080", false},
		{"missing colon", "Content-Type application/json", true},
		{"newline injection", "Header: value\nInjected: evil", true},
		{"carriage return injection", "Header: value\rInjected: evil", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateHeader(tt.header)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateHeader(%q) error = %v, wantErr %v", tt.header, err, tt.wantErr)
			}
		})
	}
}

func TestIsValidHTTPMethod(t *testing.T) {
	valid := []string{"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
	invalid := []string{"CONNECT", "TRACE", "FOOBAR", "", "get"}

	for _, method := range valid {
		if !isValidHTTPMethod(method) {
			t.Errorf("isValidHTTPMethod(%q) = false, want true", method)
		}
	}

	for _, method := range invalid {
		if isValidHTTPMethod(method) {
			t.Errorf("isValidHTTPMethod(%q) = true, want false", method)
		}
	}
}

func TestExecuteDeleteFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	testFile := "test_delete.txt"
	os.WriteFile(testFile, []byte("delete me"), 0644)

	result, err := executeDeleteFile(map[string]interface{}{
		"path": testFile,
	})

	if err != nil {
		t.Fatalf("executeDeleteFile error: %v", err)
	}

	expected := "File deleted successfully: test_delete.txt"
	if result != expected {
		t.Errorf("executeDeleteFile result = %q, want %q", result, expected)
	}

	if _, err := os.Stat(testFile); !os.IsNotExist(err) {
		t.Error("file should have been deleted")
	}
}

func TestExecuteDeleteFileNotFound(t *testing.T) {
	_, err := executeDeleteFile(map[string]interface{}{
		"path": "nonexistent.txt",
	})

	if err == nil {
		t.Fatal("executeDeleteFile expected error for nonexistent file")
	}
}

func TestExecuteUpdateFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	testFile := "test_update.txt"
	os.WriteFile(testFile, []byte("original content"), 0644)

	result, err := executeUpdateFile(map[string]interface{}{
		"path":    testFile,
		"content": "original content",
	})

	if err != nil {
		t.Fatalf("executeUpdateFile error: %v", err)
	}

	expected := "No changes needed for test_update.txt (content is identical)"
	if result != expected {
		t.Errorf("executeUpdateFile result = %q, want %q", result, expected)
	}

	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if string(data) != "original content" {
		t.Errorf("file content = %q, want %q", string(data), "original content")
	}
}

func TestExecuteUpdateFileNotFound(t *testing.T) {
	_, err := executeUpdateFile(map[string]interface{}{
		"path":    "nonexistent.txt",
		"content": "new content",
	})

	if err == nil {
		t.Fatal("executeUpdateFile expected error for nonexistent file")
	}
}

func TestExecuteGetPwd(t *testing.T) {
	result, err := executeGetPwd(map[string]interface{}{})

	if err != nil {
		t.Fatalf("executeGetPwd error: %v", err)
	}

	if !strings.Contains(result, "Current directory:") {
		t.Errorf("executeGetPwd result = %q, should contain 'Current directory:'", result)
	}
}

func TestExecuteCommand(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	result, err := executeCommand(map[string]interface{}{
		"command": "pwd",
	})

	if err != nil {
		t.Fatalf("executeCommand error: %v", err)
	}

	if !strings.Contains(result, tmpDir) {
		t.Errorf("executeCommand result = %q, should contain %q", result, tmpDir)
	}
}

func TestExecuteCommandNotAllowed(t *testing.T) {
	_, err := executeCommand(map[string]interface{}{
		"command": "rm",
	})

	if err == nil {
		t.Fatal("executeCommand expected error for disallowed command")
	}
}

func TestExecuteFindFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile(tmpDir+"test1.txt", []byte("a"), 0644)
	os.WriteFile(tmpDir+"test2.txt", []byte("b"), 0644)

	result, err := executeFindFiles(map[string]interface{}{
		"pattern": "*.txt",
	})

	if err != nil {
		t.Fatalf("executeFindFiles error: %v", err)
	}

	if len(result) == 0 {
		t.Fatal("executeFindFiles returned empty result")
	}
}

func TestExecuteFindFilesNoMatch(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	result, err := executeFindFiles(map[string]interface{}{
		"pattern": "*.nonexistent",
	})

	if err != nil {
		t.Fatalf("executeFindFiles error: %v", err)
	}

	if !strings.Contains(result, "No files found") {
		t.Errorf("executeFindFiles result = %q, should contain 'No files found'", result)
	}
}

func TestExecuteGrepContent(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	testFile := "test_grep.txt"
	os.WriteFile(testFile, []byte("hello world\nfoo bar"), 0644)

	result, err := executeGrepContent(map[string]interface{}{
		"pattern": "hello",
		"files":   testFile,
	})

	if err != nil {
		t.Fatalf("executeGrepContent error: %v", err)
	}

	if !strings.Contains(result, "hello") {
		t.Errorf("executeGrepContent result = %q, should contain 'hello'", result)
	}
}

func TestExecuteGrepContentNoMatch(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	testFile := "test_grep2.txt"
	os.WriteFile(testFile, []byte("hello world"), 0644)

	result, err := executeGrepContent(map[string]interface{}{
		"pattern": "xyz123",
		"files":   testFile,
	})

	if err != nil {
		t.Fatalf("executeGrepContent error: %v", err)
	}

	if !strings.Contains(result, "No matches found") {
		t.Errorf("executeGrepContent result = %q, should contain 'No matches found'", result)
	}
}

func TestExecuteTreeView(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile(tmpDir+"file1.txt", []byte("a"), 0644)
	os.Mkdir(tmpDir+"subdir", 0755)

	result, err := executeTreeView(map[string]interface{}{})

	if err != nil {
		t.Fatalf("executeTreeView error: %v", err)
	}

	if len(result) == 0 {
		t.Fatal("executeTreeView returned empty result")
	}
}

func TestExecuteDispatch_PlanModeBlocked(t *testing.T) {
	toolCall := clients.ToolCall{
		Function: clients.ToolFunc{
			Name: "create_file",
			Arguments: map[string]interface{}{
				"path":    "test.txt",
				"content": "content",
			},
		},
	}

	result, err := executeDispatch(toolCall, true)
	if err != nil {
		t.Fatalf("executeDispatch error: %v", err)
	}

	if !strings.Contains(result, "Plan Mode") {
		t.Errorf("executeDispatch result = %q, should contain 'Plan Mode'", result)
	}
}

func TestExecuteDispatch_ReadAllowedInPlanMode(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	testFile := "test_plan.txt"
	os.WriteFile(testFile, []byte("plan test"), 0644)

	toolCall := clients.ToolCall{
		Function: clients.ToolFunc{
			Name: "read_file",
			Arguments: map[string]interface{}{
				"path": testFile,
			},
		},
	}

	result, err := executeDispatch(toolCall, true)
	if err != nil {
		t.Fatalf("executeDispatch error: %v", err)
	}

	if result != "plan test" {
		t.Errorf("executeDispatch result = %q, want %q", result, "plan test")
	}
}

func TestExecuteToolUnknown(t *testing.T) {
	toolCall := clients.ToolCall{
		Function: clients.ToolFunc{
			Name: "unknown_tool",
		},
	}

	_, err := executeDispatch(toolCall, false)
	if err == nil {
		t.Fatal("executeDispatch expected error for unknown tool")
	}
}

func TestDetectProjectGo(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile("go.mod", []byte("module test"), 0644)

	projectInfo := detectProject()
	if projectInfo.Language != "go" {
		t.Errorf("project language = %q, want %q", projectInfo.Language, "go")
	}
	if !projectInfo.HasConfig {
		t.Error("project should have config")
	}
}

func TestDetectProjectNode(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile("package.json", []byte(`{"name": "test"}`), 0644)

	projectInfo := detectProject()
	if projectInfo.Language != "javascript" && projectInfo.Language != "typescript" {
		t.Errorf("project language = %q, want javascript or typescript", projectInfo.Language)
	}
}

func TestDetectProjectPython(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile("requirements.txt", []byte("flask"), 0644)

	projectInfo := detectProject()
	if projectInfo.Language != "python" {
		t.Errorf("project language = %q, want %q", projectInfo.Language, "python")
	}
}

func TestDetectProjectUnknown(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	projectInfo := detectProject()
	if projectInfo.Language != "unknown" {
		t.Errorf("project language = %q, want %q", projectInfo.Language, "unknown")
	}
}

func TestDetectTypeScriptProject(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	os.WriteFile("package.json", []byte(`{"name": "test"}`), 0644)
	os.WriteFile("tsconfig.json", []byte(`{"compilerOptions": {}}`), 0644)

	projectInfo := detectProject()
	if projectInfo.Language != "typescript" {
		t.Errorf("project language = %q, want %q", projectInfo.Language, "typescript")
	}
}
