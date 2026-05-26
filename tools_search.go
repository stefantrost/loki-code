package main

import (
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
)

func executeFindFiles(args map[string]interface{}) (string, error) {
	slog.Debug("executeFindFiles called", "arguments", args)

	pattern, ok := args["pattern"].(string)
	if !ok {
		slog.Error("Invalid pattern argument type", "expected", "string", "got", fmt.Sprintf("%T", args["pattern"]))
		return "", fmt.Errorf("pattern argument is required and must be a string")
	}

	searchPath := "."
	if path, hasPath := args["path"].(string); hasPath {
		slog.Debug("Validating search path", "path", path)
		if err := validatePath(path); err != nil {
			slog.Error("Path validation failed", "path", path, "error", err)
			return "", err
		}
		searchPath = path
	}
	slog.Debug("Executing find command", "pattern", pattern, "search_path", searchPath)

	findArgs := []string{searchPath, "-name", pattern}

	if fileType, hasType := args["type"].(string); hasType {
		if fileType == "f" || fileType == "d" {
			findArgs = append(findArgs, "-type", fileType)
			slog.Debug("Adding file type filter", "type", fileType)
		}
	}

	cmd := exec.Command("find", findArgs...)
	slog.Debug("Running find command", "args", findArgs)
	output, err := cmd.CombinedOutput()

	if err != nil {
		slog.Error("Find command failed", "error", err, "output", string(output))
		return "", fmt.Errorf("find command failed: %v\nOutput: %s", err, string(output))
	}

	outputStr := strings.TrimSpace(string(output))
	slog.Debug("Find command completed", "matches_found", len(strings.Split(outputStr, "\n")))

	if outputStr == "" {
		slog.Debug("No files found matching pattern", "pattern", pattern, "search_path", searchPath)
		return fmt.Sprintf("No files found matching pattern '%s' in %s", pattern, searchPath), nil
	}

	if len(outputStr) > 10000 {
		outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
	}

	result := fmt.Sprintf("Files found matching '%s':\n%s", pattern, outputStr)
	slog.Debug("Find files completed successfully", "result_length", len(result))
	return result, nil
}

func executeGrepContent(args map[string]interface{}) (string, error) {
	slog.Debug("executeGrepContent called", "arguments", args)

	pattern, ok := args["pattern"].(string)
	if !ok {
		slog.Error("Invalid pattern argument type", "expected", "string", "got", fmt.Sprintf("%T", args["pattern"]))
		return "", fmt.Errorf("pattern argument is required and must be a string")
	}

	files, ok := args["files"].(string)
	if !ok {
		slog.Error("Invalid files argument type", "expected", "string", "got", fmt.Sprintf("%T", args["files"]))
		return "", fmt.Errorf("files argument is required and must be a string")
	}

	grepArgs := []string{pattern}
	slog.Debug("Building grep command", "pattern", pattern, "files", files)

	if options, hasOptions := args["options"].(string); hasOptions {
		slog.Debug("Processing grep options", "options", options)
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
	slog.Debug("Executing grep command", "args", grepArgs)

	cmd := exec.Command("grep", grepArgs...)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	exitCode := cmd.ProcessState.ExitCode()
	// grep exit codes: 0 = matches, 1 = no matches, 2+ = error. Some I/O errors
	// can also surface with code 1 plus a stderr message, so refuse to treat
	// exit-1-with-stderr as "no matches".
	if exitCode >= 2 || (exitCode == 1 && stderr.Len() > 0) || (runErr != nil && exitCode < 0) {
		slog.Error("Grep command failed", "error", runErr, "exit_code", exitCode, "stderr", stderr.String())
		return "", fmt.Errorf("grep command failed (exit %d): %s", exitCode, strings.TrimSpace(stderr.String()))
	}

	outputStr := strings.TrimSpace(stdout.String())
	slog.Debug("Grep command completed", "matches_found", len(strings.Split(outputStr, "\n")))

	if outputStr == "" {
		slog.Debug("No matches found", "pattern", pattern, "files", files)
		return fmt.Sprintf("No matches found for pattern '%s' in %s", pattern, files), nil
	}

	if len(outputStr) > 10000 {
		outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
	}

	result := fmt.Sprintf("Matches for '%s':\n%s", pattern, outputStr)
	slog.Debug("Grep completed successfully", "result_length", len(result))
	return result, nil
}

func executeTreeView(args map[string]interface{}) (string, error) {
	slog.Debug("executeTreeView called", "arguments", args)

	treePath := "."
	if path, hasPath := args["path"].(string); hasPath {
		slog.Debug("Validating tree path", "path", path)
		if err := validatePath(path); err != nil {
			slog.Error("Path validation failed", "path", path, "error", err)
			return "", err
		}
		treePath = path
	}

	depth := 3
	if depthInterface, hasDepth := args["depth"]; hasDepth {
		slog.Debug("Processing depth parameter", "depth_value", depthInterface)
		if depthFloat, ok := depthInterface.(float64); ok {
			depth = int(depthFloat)
		} else if depthStr, ok := depthInterface.(string); ok {
			if d, err := strconv.Atoi(depthStr); err == nil {
				depth = d
			} else {
				slog.Warn("Failed to parse depth as integer", "depth_str", depthStr, "error", err)
			}
		} else {
			slog.Warn("Depth parameter has unexpected type", "depth_type", fmt.Sprintf("%T", depthInterface))
		}
	}

	if depth > 10 {
		depth = 10
		slog.Debug("Depth capped at maximum", "depth", depth)
	}

	slog.Debug("Executing tree command", "path", treePath, "depth", depth)
	cmd := exec.Command("tree", "-L", strconv.Itoa(depth), treePath)
	output, err := cmd.CombinedOutput()

	if err != nil {
		slog.Debug("tree command failed, falling back to ls", "path", treePath, "error", err)
		cmd = exec.Command("ls", "-la", treePath)
		output, err = cmd.CombinedOutput()
		if err != nil {
			slog.Error("tree/ls command failed", "error", err)
			return "", fmt.Errorf("tree/ls command failed: %v", err)
		}
		result := fmt.Sprintf("Directory listing for %s:\n%s", treePath, string(output))
		slog.Debug("ls fallback completed", "result_length", len(result))
		return result, nil
	}

	outputStr := string(output)
	slog.Debug("tree command completed successfully", "output_length", len(outputStr))

	if len(outputStr) > 10000 {
		outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
	}

	return outputStr, nil
}
