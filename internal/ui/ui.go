package ui

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// UIInput is the source UI prompts read from. Defaults to stdin; tests can
// override with a bytes.Buffer / strings.Reader. Package-level so we don't
// have to thread it through every confirmation helper.
var UIInput io.Reader = os.Stdin

// ANSI color constants for terminal output
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"  // Deletions
	ColorGreen  = "\033[32m"  // Additions
	ColorCyan   = "\033[36m"  // Headers
	ColorYellow = "\033[33m"  // Separators
)

func colorize(text, color string) string {
	return color + text + ColorReset
}

// RenderDiff returns an ANSI-coloured diff string without printing to stdout.
// Used by the TUI to display diffs inside the alt-screen modal.
func RenderDiff(oldContent, newContent, filename string) string {
	return generateDiff(oldContent, newContent, filename)
}

func generateDiff(oldContent, newContent, filename string) string {
	oldLines := strings.Split(oldContent, "\n")
	newLines := strings.Split(newContent, "\n")

	var diff strings.Builder

	diff.WriteString(colorize(fmt.Sprintf("--- %s (current)", filename), ColorCyan) + "\n")
	diff.WriteString(colorize(fmt.Sprintf("+++ %s (proposed)", filename), ColorCyan) + "\n")

	maxLines := len(oldLines)
	if len(newLines) > maxLines {
		maxLines = len(newLines)
	}

	for i := 0; i < maxLines; i++ {
		oldLine := ""
		newLine := ""

		if i < len(oldLines) {
			oldLine = oldLines[i]
		}
		if i < len(newLines) {
			newLine = newLines[i]
		}

		if oldLine != newLine {
			if oldLine != "" {
				diff.WriteString(colorize(fmt.Sprintf("-%s", oldLine), ColorRed) + "\n")
			}
			if newLine != "" {
				diff.WriteString(colorize(fmt.Sprintf("+%s", newLine), ColorGreen) + "\n")
			}
		}
	}

	return diff.String()
}

// ShowDiffAndConfirm displays a diff and prompts for user confirmation.
func ShowDiffAndConfirm(oldContent, newContent, filename string) (bool, error) {
	diff := generateDiff(oldContent, newContent, filename)
	separator := colorize(strings.Repeat("=", 52), ColorYellow)

	fmt.Println("\n📝 Proposed file changes:")
	fmt.Println(separator)
	fmt.Print(diff)
	fmt.Println(separator)

	fmt.Print("Apply this change? (y/n): ")
	scanner := bufio.NewScanner(UIInput)

	if !scanner.Scan() {
		return false, fmt.Errorf("failed to read user input")
	}

	response := strings.ToLower(strings.TrimSpace(scanner.Text()))
	return response == "y" || response == "yes", nil
}

// PromptUser displays a prompt and returns the user's y/n response.
func PromptUser(prompt string) (bool, error) {
	fmt.Print(prompt + " (y/n): ")
	scanner := bufio.NewScanner(UIInput)

	if !scanner.Scan() {
		return false, fmt.Errorf("failed to read user input")
	}

	response := strings.ToLower(strings.TrimSpace(scanner.Text()))
	return response == "y" || response == "yes", nil
}
