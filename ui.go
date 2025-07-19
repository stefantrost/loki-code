package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// ANSI color constants for terminal output
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"  // Deletions
	ColorGreen  = "\033[32m"  // Additions  
	ColorCyan   = "\033[36m"  // Headers
	ColorYellow = "\033[33m"  // Separators
)

// colorize wraps text in ANSI color codes
func colorize(text, color string) string {
	return color + text + ColorReset
}

// generateDiff creates a colored unified diff between old and new content
func generateDiff(oldContent, newContent, filename string) string {
	oldLines := strings.Split(oldContent, "\n")
	newLines := strings.Split(newContent, "\n")
	
	var diff strings.Builder
	
	// Colored file headers
	diff.WriteString(colorize(fmt.Sprintf("--- %s (current)", filename), ColorCyan) + "\n")
	diff.WriteString(colorize(fmt.Sprintf("+++ %s (proposed)", filename), ColorCyan) + "\n")
	
	// Simple line-by-line diff
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
				// Color deleted lines in red
				diff.WriteString(colorize(fmt.Sprintf("-%s", oldLine), ColorRed) + "\n")
			}
			if newLine != "" {
				// Color added lines in green
				diff.WriteString(colorize(fmt.Sprintf("+%s", newLine), ColorGreen) + "\n")
			}
		}
	}
	
	return diff.String()
}

// showDiffAndConfirm displays a diff and prompts for user confirmation
func showDiffAndConfirm(oldContent, newContent, filename string) (bool, error) {
	// Generate and display diff
	diff := generateDiff(oldContent, newContent, filename)
	
	// Colored output with visual separators
	separator := colorize(strings.Repeat("=", 52), ColorYellow)
	
	fmt.Println("\n📝 Proposed file changes:")
	fmt.Println(separator)
	fmt.Print(diff)
	fmt.Println(separator)
	
	// Prompt for confirmation
	fmt.Print("Apply this change? (y/n): ")
	scanner := bufio.NewScanner(os.Stdin)
	
	if !scanner.Scan() {
		return false, fmt.Errorf("failed to read user input")
	}
	
	response := strings.ToLower(strings.TrimSpace(scanner.Text()))
	return response == "y" || response == "yes", nil
}

// promptUser displays a prompt and returns user's y/n response
func promptUser(prompt string) (bool, error) {
	fmt.Print(prompt + " (y/n): ")
	scanner := bufio.NewScanner(os.Stdin)
	
	if !scanner.Scan() {
		return false, fmt.Errorf("failed to read user input")
	}
	
	response := strings.ToLower(strings.TrimSpace(scanner.Text()))
	return response == "y" || response == "yes", nil
}