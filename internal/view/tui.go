package view

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/bubbles/textinput"  // allow: input field component
	"github.com/charmbracelet/bubbles/viewport"   // allow: scrollable chat pane
	tea "github.com/charmbracelet/bubbletea"      // allow: TUI event loop
	"github.com/charmbracelet/glamour"            // allow: markdown rendering
	"github.com/charmbracelet/lipgloss"           // allow: layout and styling

	"loki-code/internal/ui"
)

const tuiSidebarWidth = 24

// ── View messages ─────────────────────────────────────────────────────────────
// All cross-goroutine communication from TUIView methods to the tea.Program
// is funnelled through a single typed events channel, read by listenForEvents.

type tuiMsg interface{ isTUIMsg() }

type tokenMsg    string          // one streaming token
type systemMsg   string          // tool block / status / context line
type commitMsg   struct{ content string } // trigger glamour post-render
type statusMsg   Status           // sidebar refresh
type beginMsg    struct{}          // re-anchor pre-stream position
type suspendMsg  struct {          // hand terminal to raw mode for diff/confirm
	fn     func() (bool, error)
	result chan<- suspendResult
}
type quitMsg struct{} // Stop() was called from outside the program

func (tokenMsg) isTUIMsg()   {}
func (systemMsg) isTUIMsg()  {}
func (commitMsg) isTUIMsg()  {}
func (statusMsg) isTUIMsg()  {}
func (beginMsg) isTUIMsg()   {}
func (suspendMsg) isTUIMsg() {}
func (quitMsg) isTUIMsg()    {}

type suspendResult struct {
	ok  bool
	err error
}

// listenForEvents reads one event from the channel and returns it as a
// tea.Msg. Update re-queues this Cmd after every event to keep the loop live.
func listenForEvents(ch <-chan tuiMsg) tea.Cmd {
	return func() tea.Msg { return <-ch }
}

// ── TUIView ───────────────────────────────────────────────────────────────────

// TUIView implements View using Bubble Tea for a three-pane terminal UI:
//
//	┌──────────────────────────────────┬────────────┐
//	│ (scrollable chat history)        │ Tokens     │
//	│                                  │ 1,204      │
//	│ ── You: hello ────────────────── │ ────────── │
//	│                                  │ Mode       │
//	│ Assistant: Here is the plan...   │ PLAN       │
//	│ ┌─ 🔧 read_file ─────────────┐  │ ────────── │
//	│ └────────────────────────────┘  │ Task       │
//	├──────────────────────────────────┤ refactor   │
//	│ > type here_                     │            │
//	└──────────────────────────────────┴────────────┘
//
// Streaming: tokens appear live in the viewport. On CommitMessage the raw
// text is replaced with a glamour-rendered version (code blocks, bold, etc.).
// Mouse/trackpad scrolling works natively via bubbles/viewport.
type TUIView struct {
	program  *tea.Program
	events   chan tuiMsg // REPL goroutine → tea program
	inputCh  chan string // tea program → REPL goroutine (submitted lines)
	renderer *glamour.TermRenderer
	stopOnce sync.Once
}

// NewTUIView constructs the TUI. Returns an error if glamour cannot initialise
// a renderer (e.g. no colour support detected).
func NewTUIView() (*TUIView, error) {
	renderer, err := glamour.NewTermRenderer(glamour.WithAutoStyle())
	if err != nil {
		return nil, fmt.Errorf("glamour renderer: %w", err)
	}
	v := &TUIView{
		events:   make(chan tuiMsg, 64),
		inputCh:  make(chan string, 1),
		renderer: renderer,
	}
	m := newModel(v.events, v.inputCh, renderer)
	v.program = tea.NewProgram(m,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(), // enables wheel + click; viewport handles scroll natively
	)
	return v, nil
}

// ReadInput blocks until the user submits a line or the session ends.
func (v *TUIView) ReadInput(_ string) (string, error) {
	line, ok := <-v.inputCh
	if !ok {
		return "", fmt.Errorf("session ended")
	}
	return line, nil
}

// WriteToken sends a streaming token to the tea program.
func (v *TUIView) WriteToken(token string) { v.events <- tokenMsg(token) }

// CommitMessage triggers a glamour re-render of the full response.
func (v *TUIView) CommitMessage(_, fullContent string) {
	v.events <- commitMsg{content: fullContent}
}

// WriteSystem appends a system/status line (tool blocks, context stats, etc.).
func (v *TUIView) WriteSystem(msg string) { v.events <- systemMsg(msg) }

// ShowDiffAndConfirm suspends the TUI to show the diff on the raw terminal.
func (v *TUIView) ShowDiffAndConfirm(old, newContent, filename string) (bool, error) {
	result := make(chan suspendResult, 1)
	v.events <- suspendMsg{
		fn:     func() (bool, error) { return ui.ShowDiffAndConfirm(old, newContent, filename) },
		result: result,
	}
	r := <-result
	return r.ok, r.err
}

// Confirm suspends the TUI to show a y/n prompt on the raw terminal.
func (v *TUIView) Confirm(prompt string) (bool, error) {
	result := make(chan suspendResult, 1)
	v.events <- suspendMsg{
		fn:     func() (bool, error) { return ui.PromptUser(prompt) },
		result: result,
	}
	r := <-result
	return r.ok, r.err
}

// UpdateStatus refreshes the sidebar.
func (v *TUIView) UpdateStatus(s Status) { v.events <- statusMsg(s) }

// BeginStream re-anchors the pre-stream position. Called by the agent before
// each StreamChat call (and by the onStreamStart callback for follow-up
// streams after tool calls) so CommitMessage only replaces the last segment.
func (v *TUIView) BeginStream() { v.events <- beginMsg{} }

// Run starts the tea.Program and calls fn (the REPL loop) in a goroutine.
// Blocks until Stop is called or the program exits.  After program.Run
// returns (e.g. Ctrl+C key event → tea.Quit), Stop is called to unblock
// the REPL goroutine which may be waiting on ReadInput.
func (v *TUIView) Run(fn func()) error {
	go fn()
	_, err := v.program.Run()
	v.Stop() // unblock ReadInput if the program exited via Ctrl+C
	return err
}

// Stop closes the input channel (unblocking ReadInput) and quits the program.
// Safe to call multiple times.
func (v *TUIView) Stop() {
	v.stopOnce.Do(func() {
		v.events <- quitMsg{}
		close(v.inputCh)
	})
}

// ── Bubble Tea model ──────────────────────────────────────────────────────────

type tuiModel struct {
	viewport viewport.Model
	input    textinput.Model
	status   Status
	width    int
	height   int

	// chat content — all committed lines + the current stream buffer
	lines          []string        // fully rendered lines (system + committed responses)
	preStreamIdx   int             // index into lines before current stream started
	streamBuf      strings.Builder // raw tokens accumulating during a stream
	streamRendered string          // last glamour render of streamBuf (updated per token)

	events   <-chan tuiMsg
	inputCh  chan<- string
	renderer *glamour.TermRenderer

	ready        bool  // true after first WindowSizeMsg
	lastScrollMs int64 // unix-ms of last WheelUp/WheelDown; used to filter ANSI residue
}

func newModel(events <-chan tuiMsg, inputCh chan<- string, renderer *glamour.TermRenderer) *tuiModel {
	ti := textinput.New()
	ti.Placeholder = "type a message…"
	ti.Focus()
	ti.CharLimit = 4096
	ti.Width = 80

	return &tuiModel{
		input:    ti,
		events:   events,
		inputCh:  inputCh,
		renderer: renderer,
	}
}

func (m *tuiModel) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		listenForEvents(m.events),
	)
}

func (m *tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	// ── Terminal resize ──────────────────────────────────────────────────
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.Width = msg.Width - tuiSidebarWidth - 4
		vpHeight := m.height - lipgloss.Height(m.inputView())
		if !m.ready {
			m.viewport = viewport.New(msg.Width-tuiSidebarWidth, vpHeight)
			m.viewport.SetContent(m.renderContent())
			m.ready = true
		} else {
			m.viewport.Width = msg.Width - tuiSidebarWidth
			m.viewport.Height = vpHeight
		}

	// ── Keyboard ─────────────────────────────────────────────────────────
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEnter:
			text := strings.TrimSpace(m.input.Value())
			m.input.SetValue("")
			if text != "" {
				select {
				case m.inputCh <- text:
				default:
				}
			}
		case tea.KeyEsc:
			// Swallow Escape: it is never meaningful for the text input, and is
			// the leading byte of SGR mouse sequences (ESC[<Cb;Cx;CyM) that may
			// be fragmented across read boundaries on fast scrolling.
		default:
			// Discard rune bursts that arrive shortly after a wheel event and
			// consist only of bytes found in SGR mouse sequences.  Fast trackpad
			// scrolling can fragment the escape sequence so the ESC arrives first
			// (handled above) and the remaining "[<64;54;49M" arrives here as
			// typed input — which we don't want.
			if msg.Type == tea.KeyRunes &&
				time.Now().UnixMilli()-m.lastScrollMs < 80 &&
				isMouseResidueRunes(msg.Runes) {
				break
			}
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			cmds = append(cmds, cmd)
		}

	// ── Mouse (scroll forwarded to viewport) ─────────────────────────────
	case tea.MouseMsg:
		if msg.Button == tea.MouseButtonWheelUp || msg.Button == tea.MouseButtonWheelDown {
			m.lastScrollMs = time.Now().UnixMilli()
		}
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)

	// ── Events from TUIView methods ───────────────────────────────────────
	case tokenMsg:
		m.streamBuf.WriteString(string(msg))
		// Glamour-render the accumulated buffer on every token so live streaming
		// text looks identical to the final committed output (no raw **bold** etc.).
		rendered, err := m.renderer.Render(m.streamBuf.String())
		if err != nil {
			rendered = m.streamBuf.String()
		}
		m.streamRendered = rendered
		atBottom := m.viewport.AtBottom()
		m.viewport.SetContent(m.renderContent())
		if atBottom {
			m.viewport.GotoBottom()
		}
		cmds = append(cmds, listenForEvents(m.events))

	case systemMsg:
		m.lines = append(m.lines, string(msg))
		atBottom := m.viewport.AtBottom()
		m.viewport.SetContent(m.renderContent())
		if atBottom {
			m.viewport.GotoBottom()
		}
		cmds = append(cmds, listenForEvents(m.events))

	case beginMsg:
		// Re-anchor: subsequent tokenMsgs belong to a fresh stream segment.
		// If tokens accumulated before this anchor point (model text before a
		// tool call, or thinking output) they must be committed to lines now —
		// not silently discarded — so they remain visible in the chat history.
		if m.streamBuf.Len() > 0 {
			// Reuse the cached glamour render; no need to re-render.
			committed := m.streamRendered
			if committed == "" {
				committed = m.streamBuf.String()
			}
			// INSERT at m.preStreamIdx rather than replacing from it.
			// Any systemMsg lines (tool-call boxes) that arrived after the
			// anchor was set live at m.lines[m.preStreamIdx:] — they must be
			// preserved and shifted right, not overwritten.
			newLines := make([]string, 0, len(m.lines)+1)
			newLines = append(newLines, m.lines[:m.preStreamIdx]...)
			newLines = append(newLines, committed)
			newLines = append(newLines, m.lines[m.preStreamIdx:]...)
			m.lines = newLines
			m.streamBuf.Reset()
			m.streamRendered = ""
			m.viewport.SetContent(m.renderContent())
		}
		m.preStreamIdx = len(m.lines)
		cmds = append(cmds, listenForEvents(m.events))

	case commitMsg:
		// Replace everything since preStreamIdx with the glamour-rendered version.
		rendered, err := m.renderer.Render(msg.content)
		if err != nil {
			rendered = msg.content
		}
		m.lines = append(m.lines[:m.preStreamIdx], rendered)
		m.streamBuf.Reset()
		m.streamRendered = ""
		atBottom := m.viewport.AtBottom()
		m.viewport.SetContent(m.renderContent())
		if atBottom {
			m.viewport.GotoBottom()
		}
		cmds = append(cmds, listenForEvents(m.events))

	case statusMsg:
		m.status = Status(msg)
		cmds = append(cmds, listenForEvents(m.events))

	case suspendMsg:
		// Exit alt-screen, run the raw terminal function, re-enter alt-screen.
		cmds = append(cmds, tea.Sequence(
			tea.ExitAltScreen,
			func() tea.Msg {
				ok, err := msg.fn()
				msg.result <- suspendResult{ok: ok, err: err}
				return tea.EnterAltScreen
			},
			listenForEvents(m.events),
		))

	case quitMsg:
		return m, tea.Quit
	}

	return m, tea.Batch(cmds...)
}

func (m *tuiModel) View() string {
	if !m.ready {
		return "Initialising…"
	}
	left := lipgloss.JoinVertical(lipgloss.Left,
		m.viewport.View(),
		m.inputView(),
	)
	right := m.sidebarView()
	return lipgloss.JoinHorizontal(lipgloss.Top, left, right)
}

// renderContent combines committed lines with the current stream buffer.
// The stream buffer is shown via streamRendered (glamour output cached on
// every tokenMsg) so live tokens look identical to the final committed render.
func (m *tuiModel) renderContent() string {
	var sb strings.Builder
	for _, l := range m.lines {
		sb.WriteString(l)
		if !strings.HasSuffix(l, "\n") {
			sb.WriteByte('\n')
		}
	}
	if m.streamBuf.Len() > 0 {
		sb.WriteString(m.streamRendered)
	}
	return sb.String()
}

func (m *tuiModel) inputView() string {
	return lipgloss.NewStyle().
		BorderStyle(lipgloss.NormalBorder()).
		BorderTop(true).
		Width(m.width - tuiSidebarWidth).
		Render(" > " + m.input.View())
}

func (m *tuiModel) sidebarView() string {
	task := m.status.Task
	if task == "" {
		task = "(none)"
	}
	maxTask := tuiSidebarWidth - 4
	if len([]rune(task)) > maxTask {
		task = string([]rune(task)[:maxTask-1]) + "…"
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "\nTokens\n%d / %d\n\n", m.status.Tokens, m.status.MaxTokens)
	fmt.Fprintf(&sb, "────────────────\n\n")
	fmt.Fprintf(&sb, "Mode\n%s\n\n", strings.ToUpper(m.status.Mode))
	fmt.Fprintf(&sb, "────────────────\n\n")
	fmt.Fprintf(&sb, "Response\n%s\n\n", strings.ToUpper(m.status.Response))
	fmt.Fprintf(&sb, "────────────────\n\n")
	fmt.Fprintf(&sb, "Task\n%s\n", task)

	return lipgloss.NewStyle().
		Width(tuiSidebarWidth).
		Height(m.height).
		BorderStyle(lipgloss.NormalBorder()).
		BorderLeft(true).
		Render(sb.String())
}

// isMouseResidueRunes returns true when every rune in the slice is a character
// that appears in SGR mouse-event sequences (ESC[<Cb;Cx;CyM).  Used to
// discard fragmented mouse-escape bytes that arrived as keyboard input after
// fast scrolling.
func isMouseResidueRunes(runes []rune) bool {
	if len(runes) == 0 {
		return false
	}
	for _, r := range runes {
		if !strings.ContainsRune("[<>;Mm0123456789", r) {
			return false
		}
	}
	return true
}
