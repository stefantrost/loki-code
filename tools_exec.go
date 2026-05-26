package main

import (
	"fmt"
	"log/slog"
	"os/exec"
	"strings"
	"time"
)

// allowedCommands is the strict whitelist for exec_command. Adding entries
// here is a deliberate security decision; the README and CLAUDE.md both
// reference this list.
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

func getAvailableCommands() string {
	commands := make([]string, 0, len(allowedCommands))
	for cmd := range allowedCommands {
		commands = append(commands, cmd)
	}
	return strings.Join(commands, ", ")
}

func executeCommand(args map[string]interface{}) (string, error) {
	slog.Debug("executeCommand called", "arguments", args)

	command, ok := args["command"].(string)
	if !ok {
		slog.Error("Invalid command argument type", "expected", "string", "got", fmt.Sprintf("%T", args["command"]))
		return "", fmt.Errorf("command argument is required and must be a string")
	}

	slog.Debug("Checking command whitelist", "command", command)
	if !allowedCommands[command] {
		slog.Warn("Command not in whitelist", "command", command, "allowed_commands", getAvailableCommands())
		return "", fmt.Errorf("command '%s' is not allowed. Allowed commands: %s",
			command, getAvailableCommands())
	}

	var cmdArgs []string
	if argsInterface, hasArgs := args["args"]; hasArgs {
		slog.Debug("Parsing command arguments", "args_type", fmt.Sprintf("%T", argsInterface))
		if argsList, ok := argsInterface.([]interface{}); ok {
			for _, arg := range argsList {
				if argStr, ok := arg.(string); ok {
					cmdArgs = append(cmdArgs, argStr)
				} else {
					slog.Warn("Skipping non-string argument", "arg_type", fmt.Sprintf("%T", arg))
				}
			}
		} else {
			slog.Warn("Args is not an array", "args_type", fmt.Sprintf("%T", argsInterface))
		}
	}
	slog.Debug("Executing command", "command", command, "args", cmdArgs)

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
		slog.Debug("Command completed", "command", command, "error", cmdErr, "output_length", len(output))
		if cmdErr != nil {
			slog.Error("Command execution failed", "command", command, "error", cmdErr, "output", string(output))
			return "", fmt.Errorf("command failed: %v\nOutput: %s", cmdErr, string(output))
		}

		outputStr := string(output)
		if len(outputStr) > 10000 {
			outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
		}

		slog.Debug("Command executed successfully", "command", command, "output_length", len(outputStr))
		return outputStr, nil

	case <-time.After(timeout):
		slog.Error("Command timed out", "command", command, "timeout", timeout)
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		return "", fmt.Errorf("command timed out after %v", timeout)
	}
}

func executeGetPwd(args map[string]interface{}) (string, error) {
	slog.Debug("executeGetPwd called", "arguments", args)
	slog.Debug("Executing pwd command")
	cmd := exec.Command("pwd")
	output, err := cmd.Output()

	if err != nil {
		slog.Error("pwd command failed", "error", err)
		return "", fmt.Errorf("pwd command failed: %v", err)
	}

	result := fmt.Sprintf("Current directory: %s", strings.TrimSpace(string(output)))
	slog.Debug("pwd command completed successfully", "directory", result)
	return result, nil
}
