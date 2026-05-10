package main

import (
	"os"
	"path/filepath"
	"testing"
)

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
		name      string
		toolName  string
		allowed   bool
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

	// Verify file was actually created
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

	// Create a test file
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

func TestExecuteListFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	// Create some test files
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
	// Verify command whitelist contains expected commands
	expected := []string{"find", "grep", "pwd", "tree", "wc", "sort", "uniq", "whoami", "date", "which"}
	for _, cmd := range expected {
		if !allowedCommands[cmd] {
			t.Errorf("expected command %q to be in whitelist", cmd)
		}
	}

	// Verify dangerous commands are not allowed
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
