package main

import (
	"fmt"
	"log/slog"
	"os/exec"
	"strings"
	"time"
)

func executeAnalyzeCode(args map[string]interface{}) (string, error) {
	slog.Debug("executeAnalyzeCode called", "arguments", args)

	projectInfo := detectProject()
	slog.Debug("Project detection completed", "language", projectInfo.Language, "has_config", projectInfo.HasConfig,
		"analyzers_count", len(projectInfo.Analyzers))

	if len(projectInfo.Analyzers) == 0 {
		slog.Warn("No static analyzers detected", "language", projectInfo.Language)
		return fmt.Sprintf("No static analyzers detected for %s projects in current directory", projectInfo.Language), nil
	}

	scope := "package"
	if s, hasScope := args["scope"].(string); hasScope {
		scope = s
		slog.Debug("Scope specified", "scope", scope)
	}

	var filePath string
	if fp, hasFilePath := args["file_path"].(string); hasFilePath {
		filePath = fp
		slog.Debug("File path specified", "file_path", filePath)
	}

	if filePath != "" {
		slog.Debug("Validating file path", "file_path", filePath)
		if err := validatePath(filePath); err != nil {
			slog.Error("Path validation failed", "file_path", filePath, "error", err)
			return "", err
		}
		if !fileExists(filePath) {
			slog.Error("File not found", "file_path", filePath)
			return "", fmt.Errorf("file not found: %s", filePath)
		}
	}

	analyzer := selectAnalyzer(args, projectInfo.Analyzers)
	if analyzer == nil {
		slog.Warn("No suitable analyzer found", "available_analyzers", getAnalyzerNames(projectInfo.Analyzers))
		return "No suitable analyzer found", nil
	}

	slog.Debug("Selected analyzer", "analyzer_name", analyzer.Name, "command", analyzer.Command,
		"args", analyzer.Args)

	cmdArgs, err := buildAnalyzerArgs(*analyzer, scope, filePath, args)
	if err != nil {
		slog.Error("Failed to build analyzer arguments", "error", err)
		return "", err
	}

	slog.Debug("Running analyzer", "analyzer", analyzer.Name, "args", cmdArgs)
	result, err := runAnalyzer(*analyzer, cmdArgs)
	if err != nil {
		slog.Error("Analysis failed", "analyzer", analyzer.Name, "error", err, "output", result)
		return fmt.Sprintf("Analysis failed with %s: %v\nOutput: %s", analyzer.Name, err, result), nil
	}

	slog.Debug("Analysis completed successfully", "analyzer", analyzer.Name, "output_length", len(result))
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
			_ = cmd.Process.Kill()
		}
		return "", fmt.Errorf("analysis timed out after %v", timeout)
	}
}

func formatAnalysisResults(analyzerName, output, language, scope, filePath string) string {
	var result strings.Builder

	fmt.Fprintf(&result, "🔍 Static Analysis Results (%s)\n", analyzerName)
	fmt.Fprintf(&result, "Language: %s | Scope: %s", language, scope)
	if filePath != "" {
		fmt.Fprintf(&result, " | File: %s", filePath)
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
