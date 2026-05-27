// Package view defines the View interface that decouples the REPL loop from
// any rendering technology (CLI, TUI, Web). Each implementation satisfies the
// same contract; the agent layer imports only this interface.
package view

// Status is a snapshot of client state shown in the sidebar / status line.
type Status struct {
	Tokens    int
	MaxTokens int
	Mode      string // "execute" | "plan"
	Response  string // "verbose" | "concise"
	Task      string // "" if none
}

// View is the rendering surface the REPL loop drives.
//
// Streaming flow:
//
//	WriteToken is called once per content token during generation.
//	CommitMessage is called once after generation completes with the full text.
//	  CLI:  WriteToken prints live; CommitMessage is a no-op.
//	  TUI:  WriteToken accumulates raw text; CommitMessage glamour-renders it.
//	  Web:  WriteToken sends a token WS frame; CommitMessage sends rendered HTML.
type View interface {
	// ReadInput blocks until the user submits a line.
	// Returns io.EOF (or any error) when the session ends.
	ReadInput(prompt string) (string, error)

	// WriteToken is called once per streamed content token.
	WriteToken(token string)

	// CommitMessage is called once streaming is complete.
	// role is "assistant"; fullContent is the concatenated token stream.
	CommitMessage(role, fullContent string)

	// WriteSystem appends a system/status line (slash-command feedback,
	// tool blocks, context stats). Not glamour-rendered.
	WriteSystem(msg string)

	// ShowDiffAndConfirm shows a diff and asks for y/n confirmation.
	// Implementations should suspend/restore the TUI around the raw prompt.
	ShowDiffAndConfirm(oldContent, newContent, filename string) (bool, error)

	// Confirm shows a yes/no prompt.
	Confirm(prompt string) (bool, error)

	// BeginStream re-anchors the pre-stream position before each streaming
	// segment.  Must be called by the agent before StreamChat and by the
	// onStreamStart callback for follow-up streams after tool calls.
	//   TUI:  records preStreamIdx and resets the raw-token buffer.
	//   CLI/Web: no-op.
	BeginStream()

	// UpdateStatus refreshes the sidebar / status line.
	// Called after every response and after every state-changing slash command.
	UpdateStatus(s Status)

	// Run starts the view and calls fn (the REPL loop) in the appropriate
	// context. Blocks until the session ends.
	//   CLI:  calls fn synchronously.
	//   TUI:  starts the tea.Program; runs fn in a goroutine.
	//   Web:  starts HTTP+WS server; runs fn in a goroutine.
	Run(fn func()) error

	// Stop tears down the view (quit the tea.Program, close WS, etc.).
	// Safe to call multiple times.
	Stop()
}
