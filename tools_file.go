package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

func executeCreateFile(args map[string]interface{}) (string, error) {
	slog.Debug("executeCreateFile called", "arguments", args)

	path, ok := args["path"].(string)
	if !ok {
		slog.Error("Invalid path argument type", "expected", "string", "got", fmt.Sprintf("%T", args["path"]))
		return "", fmt.Errorf("path argument is required and must be a string")
	}

	content, ok := args["content"].(string)
	if !ok {
		slog.Error("Invalid content argument type", "expected", "string", "got", fmt.Sprintf("%T", args["content"]))
		return "", fmt.Errorf("content argument is required and must be a string")
	}

	slog.Debug("Validating file path", "path", path)
	if err := validatePath(path); err != nil {
		slog.Error("Path validation failed", "path", path, "error", err)
		return "", err
	}

	confirmed, err := confirmDiff("", content, path)
	if err != nil {
		return "", fmt.Errorf("failed to get user confirmation: %v", err)
	}
	if !confirmed {
		return fmt.Sprintf("File creation cancelled by user: %s", path), nil
	}

	dir := filepath.Dir(path)
	slog.Debug("Creating directory", "dir", dir)
	if err := os.MkdirAll(dir, 0755); err != nil {
		slog.Error("Failed to create directory", "dir", dir, "error", err)
		return "", fmt.Errorf("failed to create directory: %v", err)
	}

	slog.Debug("Writing file", "path", path, "content_length", len(content))
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		slog.Error("Failed to write file", "path", path, "error", err)
		return "", fmt.Errorf("failed to create file: %v", err)
	}

	slog.Debug("File created successfully", "path", path)
	return fmt.Sprintf("File created successfully: %s", path), nil
}

func executeReadFile(args map[string]interface{}) (string, error) {
	slog.Debug("executeReadFile called", "arguments", args)

	path, ok := args["path"].(string)
	if !ok {
		slog.Error("Invalid path argument type", "expected", "string", "got", fmt.Sprintf("%T", args["path"]))
		return "", fmt.Errorf("path argument is required and must be a string")
	}

	slog.Debug("Validating file path", "path", path)
	if err := validatePath(path); err != nil {
		slog.Error("Path validation failed", "path", path, "error", err)
		return "", err
	}

	slog.Debug("Reading file", "path", path)
	content, err := os.ReadFile(path)
	if err != nil {
		slog.Error("Failed to read file", "path", path, "error", err)
		return "", fmt.Errorf("failed to read file: %v", err)
	}

	slog.Debug("File read successfully", "path", path, "content_length", len(content))
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

	confirmed, err := confirmDiff(currentContentStr, newContent, path)
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

	confirmed, err := confirmYN(fmt.Sprintf("Delete file %s?", path))
	if err != nil {
		return "", fmt.Errorf("failed to get user confirmation: %v", err)
	}
	if !confirmed {
		return fmt.Sprintf("File deletion cancelled by user: %s", path), nil
	}

	if err := os.Remove(path); err != nil {
		return "", fmt.Errorf("failed to delete file: %v", err)
	}

	return fmt.Sprintf("File deleted successfully: %s", path), nil
}

func executeListFiles(args map[string]interface{}) (string, error) {
	slog.Debug("executeListFiles called", "arguments", args)

	path := "."
	if p, ok := args["path"].(string); ok && p != "" {
		path = p
	}
	slog.Debug("List files target path", "path", path)

	slog.Debug("Validating directory path", "path", path)
	if err := validatePath(path); err != nil {
		slog.Error("Path validation failed", "path", path, "error", err)
		return "", err
	}

	slog.Debug("Reading directory", "path", path)
	entries, err := os.ReadDir(path)
	if err != nil {
		slog.Error("Failed to read directory", "path", path, "error", err)
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

	slog.Debug("Directory listing completed", "path", path, "file_count", len(files))

	if len(files) == 0 {
		slog.Debug("Directory is empty", "path", path)
		return "Directory is empty", nil
	}

	result := "Files in " + path + ":\n" + strings.Join(files, "\n")
	slog.Debug("List files result", "result_length", len(result))
	return result, nil
}
