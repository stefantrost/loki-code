package view

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"loki-code/internal/ui"
)

// CLIView is the plain-terminal fallback: bufio input, fmt output.
// Zero framework dependency — used when the terminal doesn't support a TUI.
type CLIView struct {
	scanner *bufio.Scanner
}

// NewCLIView returns a CLIView reading from ui.UIInput (defaults to os.Stdin).
func NewCLIView() *CLIView {
	return &CLIView{scanner: bufio.NewScanner(ui.UIInput)}
}

func (v *CLIView) ReadInput(prompt string) (string, error) {
	fmt.Print(prompt)
	if !v.scanner.Scan() {
		if err := v.scanner.Err(); err != nil {
			return "", err
		}
		return "", io.EOF
	}
	return strings.TrimSpace(v.scanner.Text()), nil
}

// WriteToken prints each token as it arrives.
func (v *CLIView) WriteToken(token string) { fmt.Print(token) }

// CommitMessage is a no-op: tokens were already printed live via WriteToken.
func (v *CLIView) CommitMessage(_, _ string) {}

// WriteSystem prints a status or feedback line.
func (v *CLIView) WriteSystem(msg string) { fmt.Println(msg) }

func (v *CLIView) ShowDiffAndConfirm(old, newContent, filename string) (bool, error) {
	return ui.ShowDiffAndConfirm(old, newContent, filename)
}

func (v *CLIView) Confirm(prompt string) (bool, error) {
	return ui.PromptUser(prompt)
}

// BeginStream is a no-op for CLI; tokens are printed live and need no anchor.
func (v *CLIView) BeginStream() {}

// UpdateStatus is a no-op for CLI; the user runs /stats manually.
func (v *CLIView) UpdateStatus(_ Status) {}

// Run calls fn synchronously and blocks until it returns.
func (v *CLIView) Run(fn func()) error {
	fn()
	return nil
}

func (v *CLIView) Stop() {}
