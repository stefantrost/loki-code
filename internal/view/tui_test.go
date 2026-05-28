package view

import (
	"strings"
	"testing"

	"github.com/charmbracelet/glamour"
	tea "github.com/charmbracelet/bubbletea"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// newTestRenderer returns a glamour renderer suitable for tests (no colours,
// no word-wrap so output is predictable).
func newTestRenderer(t *testing.T) *glamour.TermRenderer {
	t.Helper()
	r, err := glamour.NewTermRenderer(glamour.WithStandardStyle("notty"), glamour.WithWordWrap(0))
	if err != nil {
		t.Fatalf("glamour.NewTermRenderer: %v", err)
	}
	return r
}

// newTestModel builds a minimal *tuiModel wired to the supplied channels.
func newTestModel(t *testing.T, events <-chan tuiMsg, inputCh chan<- string) *tuiModel {
	t.Helper()
	return newModel(events, inputCh, newTestRenderer(t))
}

// send applies a single Update call and returns the updated *tuiModel.
// It asserts that the returned tea.Model is still a *tuiModel.
func send(t *testing.T, m *tuiModel, msg tea.Msg) *tuiModel {
	t.Helper()
	next, _ := m.Update(msg)
	tm, ok := next.(*tuiModel)
	if !ok {
		t.Fatalf("Update returned %T, want *tuiModel", next)
	}
	return tm
}

// initSize sends a WindowSizeMsg to mark the model ready and size the viewport.
func initSize(t *testing.T, m *tuiModel, w, h int) *tuiModel {
	t.Helper()
	return send(t, m, tea.WindowSizeMsg{Width: w, Height: h})
}

// ── basic message routing ─────────────────────────────────────────────────────

func TestTUIModel_TokenMsg(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, tokenMsg("hello "))
	m = send(t, m, tokenMsg("world"))

	if got := m.streamBuf.String(); got != "hello world" {
		t.Errorf("streamBuf = %q, want %q", got, "hello world")
	}
}

func TestTUIModel_SystemMsg(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, systemMsg("✓ Context cleared"))

	if len(m.lines) != 1 {
		t.Fatalf("len(lines) = %d, want 1", len(m.lines))
	}
	if m.lines[0] != "✓ Context cleared" {
		t.Errorf("lines[0] = %q, want %q", m.lines[0], "✓ Context cleared")
	}
}

func TestTUIModel_BeginMsg_ResetsBuffer(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	// Empty buffer case: beginMsg when nothing is streaming.
	m = send(t, m, systemMsg("some system line"))
	linesBefore := len(m.lines)
	m = send(t, m, beginMsg{})

	if m.streamBuf.Len() != 0 {
		t.Errorf("streamBuf.Len() = %d after beginMsg (empty buf), want 0", m.streamBuf.Len())
	}
	if m.preStreamIdx != linesBefore {
		t.Errorf("preStreamIdx = %d, want %d", m.preStreamIdx, linesBefore)
	}
}

// TestTUIModel_BeginMsg_CommitsAccumulatedTokens verifies that tokens streamed
// before a tool-call boundary (pre-tool text, thinking output, etc.) are
// committed to lines rather than silently discarded when the next beginMsg
// re-anchors the stream position.
func TestTUIModel_BeginMsg_CommitsAccumulatedTokens(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	// Simulate: model starts streaming text, then a tool call fires BeginStream.
	m = send(t, m, tokenMsg("Let me check that file…"))
	if m.streamBuf.Len() == 0 {
		t.Fatal("expected non-empty streamBuf before beginMsg")
	}

	m = send(t, m, beginMsg{})

	// streamBuf must be empty after the anchor.
	if m.streamBuf.Len() != 0 {
		t.Errorf("streamBuf.Len() = %d after beginMsg, want 0", m.streamBuf.Len())
	}
	// The pre-tool text must appear in lines (not be lost).
	combined := strings.Join(m.lines, " ")
	if !strings.Contains(combined, "Let me check that file") {
		t.Errorf("lines %v — pre-tool token content was discarded instead of committed", m.lines)
	}
	// preStreamIdx must point past the newly committed entry.
	if m.preStreamIdx != len(m.lines) {
		t.Errorf("preStreamIdx = %d, want %d (len(lines))", m.preStreamIdx, len(m.lines))
	}
}

// TestTUIModel_BeginMsg_PreservesToolBoxes verifies that system messages (tool
// call boxes) appended to m.lines between the initial BeginStream anchor and the
// follow-up BeginStream are NOT overwritten when the pre-tool stream buffer is
// committed.  The correct visual order is:
//
//	[pre-tool LLM text] → [tool box line(s)] → (follow-up stream)
func TestTUIModel_BeginMsg_PreservesToolBoxes(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	// 1. First BeginStream — anchor at 0.
	m = send(t, m, beginMsg{})
	if m.preStreamIdx != 0 {
		t.Fatalf("preStreamIdx after first beginMsg = %d, want 0", m.preStreamIdx)
	}

	// 2. Streaming text before tool call.
	m = send(t, m, tokenMsg("Let me read that file"))

	// 3. Tool box arrives as system messages (before second BeginStream).
	m = send(t, m, systemMsg("┌─ 🔧 read_file ──┐"))
	m = send(t, m, systemMsg("│ path: /some/file │"))
	m = send(t, m, systemMsg("└──────────────────┘"))

	// 4. Second BeginStream (onStreamStart callback after tool execution).
	m = send(t, m, beginMsg{})

	// Stream buffer must be flushed.
	if m.streamBuf.Len() != 0 {
		t.Errorf("streamBuf.Len() = %d after second beginMsg, want 0", m.streamBuf.Len())
	}

	// All four pieces of content must be in lines.
	combined := strings.Join(m.lines, "\n")
	for _, want := range []string{
		"Let me read that file",
		"┌─ 🔧 read_file ──┐",
		"│ path: /some/file │",
		"└──────────────────┘",
	} {
		if !strings.Contains(combined, want) {
			t.Errorf("m.lines does not contain %q — it was lost\nlines: %v", want, m.lines)
		}
	}

	// Pre-tool text must appear BEFORE the tool box in lines.
	preLoc := strings.Index(combined, "Let me read that file")
	boxLoc := strings.Index(combined, "┌─ 🔧 read_file ──┐")
	if preLoc >= boxLoc {
		t.Errorf("pre-tool text (pos %d) appears after tool box (pos %d) — wrong order", preLoc, boxLoc)
	}

	// preStreamIdx must be at the end of lines (ready for follow-up stream).
	if m.preStreamIdx != len(m.lines) {
		t.Errorf("preStreamIdx = %d, want %d (len(lines))", m.preStreamIdx, len(m.lines))
	}
}

func TestTUIModel_CommitMsg(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, tokenMsg("raw token"))
	m = send(t, m, commitMsg{content: "hello commit"})

	if m.streamBuf.Len() != 0 {
		t.Errorf("streamBuf not reset after commitMsg")
	}
	// Lines should contain the rendered content, not the raw token.
	combined := strings.Join(m.lines, " ")
	if !strings.Contains(combined, "hello commit") {
		t.Errorf("lines %v do not contain committed content", m.lines)
	}
}

func TestTUIModel_StatusMsg(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))

	m = send(t, m, statusMsg(Status{Tokens: 42, MaxTokens: 1000, Mode: "plan", Task: "refactor"}))

	if m.status.Tokens != 42 {
		t.Errorf("status.Tokens = %d, want 42", m.status.Tokens)
	}
	if m.status.Mode != "plan" {
		t.Errorf("status.Mode = %q, want %q", m.status.Mode, "plan")
	}
	if m.status.Task != "refactor" {
		t.Errorf("status.Task = %q, want %q", m.status.Task, "refactor")
	}
}

func TestTUIModel_WindowSize(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))

	if m.ready {
		t.Fatal("model should not be ready before first WindowSizeMsg")
	}
	m = initSize(t, m, 120, 50)

	if !m.ready {
		t.Error("model should be ready after WindowSizeMsg")
	}
	if m.width != 120 || m.height != 50 {
		t.Errorf("size = (%d,%d), want (120,50)", m.width, m.height)
	}
	if m.viewport.Width != 120-tuiSidebarWidth {
		t.Errorf("viewport.Width = %d, want %d", m.viewport.Width, 120-tuiSidebarWidth)
	}
}

func TestTUIModel_KeyEnter_SendsToInputCh(t *testing.T) {
	events := make(chan tuiMsg, 8)
	inputCh := make(chan string, 1)
	m := newTestModel(t, events, inputCh)
	m = initSize(t, m, 100, 40)

	// Seed the text input directly.
	m.input.SetValue("hello world")
	m = send(t, m, tea.KeyMsg{Type: tea.KeyEnter})

	select {
	case got := <-inputCh:
		if got != "hello world" {
			t.Errorf("inputCh received %q, want %q", got, "hello world")
		}
	default:
		t.Error("nothing sent to inputCh after Enter key")
	}
	// Input field should be cleared.
	if m.input.Value() != "" {
		t.Errorf("input value = %q after Enter, want empty", m.input.Value())
	}
}

// ── streaming indent ──────────────────────────────────────────────────────────

// TestTUIModel_StreamingIndent verifies that renderContent() carries the same
// 2-space left margin that glamour applies to committed paragraphs (via
// glamour's own document.margin), so live streaming text does not visually
// jump when CommitMessage fires.
func TestTUIModel_StreamingIndent(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, tokenMsg("hello world"))

	content := m.renderContent()

	// Glamour applies a 2-space left margin — streaming text must carry it too.
	if !strings.Contains(content, "  hello world") {
		t.Errorf("renderContent() = %q — streaming text missing 2-space left margin", content)
	}
	// Must NOT start at column 0 (regression guard for the "jump" bug).
	if strings.HasPrefix(content, "hello") {
		t.Errorf("renderContent() starts with %q — streaming text rendered at column 0, expected 2-space indent", content[:min(20, len(content))])
	}
}

// ── glamour live rendering ────────────────────────────────────────────────────

// TestTUIModel_GlamourRenderedDuringStream verifies that tokens are passed
// through glamour immediately on receipt. We use a horizontal rule ("---")
// because the notty test renderer transforms it to a repeated-dash line
// ("--------") — a change only possible if glamour is actually applied rather
// than the old manual-indent path.
func TestTUIModel_GlamourRenderedDuringStream(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, tokenMsg("---"))

	// streamRendered must be non-empty — glamour must have been called.
	if m.streamRendered == "" {
		t.Fatal("streamRendered is empty after tokenMsg — glamour render was never cached")
	}

	content := m.renderContent()

	// Glamour transforms "---" to a repeated-dash rule; raw content would just
	// show "  ---".  Check for the expanded form to confirm glamour ran.
	if !strings.Contains(content, "---") {
		t.Errorf("renderContent() = %q — expected rendered horizontal rule", content)
	}
	// renderContent() must equal streamRendered (plus any committed lines prefix).
	if !strings.HasSuffix(content, m.streamRendered) {
		t.Errorf("renderContent() does not end with streamRendered:\ncontent=%q\nstreamRendered=%q",
			content, m.streamRendered)
	}
}

// ── scroll preservation ───────────────────────────────────────────────────────

// TestTUIModel_ScrollPreservedDuringStreaming verifies that a user who scrolled
// up is not snapped back to the bottom when new tokens arrive.
func TestTUIModel_ScrollPreservedDuringStreaming(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = populateLinesOverflow(t, m, 100, 20)

	// Scroll up — user is no longer at the bottom.
	for range 5 {
		m = send(t, m, tea.MouseMsg{Button: tea.MouseButtonWheelUp, Action: tea.MouseActionPress})
	}
	offsetBefore := m.viewport.YOffset

	// A new token arrives while the user is scrolled up.
	m = send(t, m, tokenMsg("new streamed token"))

	// Offset must not have changed.
	if m.viewport.YOffset != offsetBefore {
		t.Errorf("YOffset changed after tokenMsg while scrolled up: before=%d after=%d",
			offsetBefore, m.viewport.YOffset)
	}
}

// TestTUIModel_AutoScrollWhenAtBottom verifies that when the user is at the
// bottom, new tokens continue to advance the viewport automatically.
func TestTUIModel_AutoScrollWhenAtBottom(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 20)

	// Send enough tokens to fill the viewport and stay at the bottom.
	for range 60 {
		m = send(t, m, tokenMsg("line of text\n"))
	}

	// Must still be at the bottom — auto-scroll should have followed.
	if !m.viewport.AtBottom() {
		t.Errorf("viewport not at bottom after streaming while at bottom: YOffset=%d", m.viewport.YOffset)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ── regression: no panic on repeated copies ───────────────────────────────────

// TestTUIModel_NoStreamBufPanic ensures that sending many successive tokenMsgs
// (which previously triggered "illegal use of non-zero Builder copied by value"
// with value receivers) no longer panics.
func TestTUIModel_NoStreamBufPanic(t *testing.T) {
	events := make(chan tuiMsg, 32)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	for range 20 {
		m = send(t, m, tokenMsg("x"))
	}
	if m.streamBuf.Len() != 20 {
		t.Errorf("streamBuf.Len() = %d, want 20", m.streamBuf.Len())
	}
}

// ── scroll tests ──────────────────────────────────────────────────────────────

// populateLinesOverflow fills the model with enough system lines to make the
// viewport content taller than its display area, enabling scroll.
func populateLinesOverflow(t *testing.T, m *tuiModel, w, h int) *tuiModel {
	t.Helper()
	m = initSize(t, m, w, h)
	// Each systemMsg adds one line. Send 3× the viewport height to ensure overflow.
	for i := range h * 3 {
		line := strings.Repeat("x", 40) + " line " + string(rune('0'+i%10))
		m = send(t, m, systemMsg(line))
	}
	// Sync viewport content.
	m.viewport.SetContent(m.renderContent())
	m.viewport.GotoBottom()
	return m
}

func TestTUIModel_MouseWheelScrollsViewport(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = populateLinesOverflow(t, m, 100, 20)

	// The viewport should be at the bottom after populateLinesOverflow.
	// Scroll up (WheelUp) and confirm YOffset decreases.
	before := m.viewport.YOffset
	m = send(t, m, tea.MouseMsg{Button: tea.MouseButtonWheelUp, Action: tea.MouseActionPress})
	after := m.viewport.YOffset

	if after >= before {
		t.Errorf("YOffset did not decrease after WheelUp: before=%d after=%d", before, after)
	}
}

// ── thinking indicator ────────────────────────────────────────────────────────

// TestTUIModel_ThinkingMsg_AccumulatesInThinkBuf verifies that thinkingMsg
// tokens accumulate in thinkBuf and phase transitions to phaseThinking.
func TestTUIModel_ThinkingMsg_AccumulatesInThinkBuf(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, beginMsg{})
	m = send(t, m, thinkingMsg("I should check the file "))
	m = send(t, m, thinkingMsg("before responding."))

	if got := m.thinkBuf.String(); got != "I should check the file before responding." {
		t.Errorf("thinkBuf = %q, want full accumulated reasoning", got)
	}
	if m.streamPhase != phaseThinking {
		t.Errorf("streamPhase = %v after thinkingMsg, want phaseThinking", m.streamPhase)
	}
	// thinkBuf content must be visible in renderContent while not yet done.
	if m.thinkDone {
		t.Error("thinkDone should be false while thinking tokens are still arriving")
	}
	content := m.renderContent()
	if !strings.Contains(content, "💭 Thinking") {
		t.Errorf("renderContent() = %q — thinking indicator not shown", content)
	}
}

// TestTUIModel_ThinkingCollapses_OnFirstToken verifies that thinkDone is set
// when the first regular token arrives after reasoning tokens.
func TestTUIModel_ThinkingCollapses_OnFirstToken(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, beginMsg{})
	m = send(t, m, thinkingMsg("reasoning text"))
	m = send(t, m, tokenMsg("response starts"))

	if !m.thinkDone {
		t.Error("thinkDone should be true after first regular token")
	}
	if m.thinkTokenCount == 0 {
		t.Error("thinkTokenCount should be set when thinking collapses")
	}
	if m.streamPhase != phaseStreaming {
		t.Errorf("streamPhase = %v after first token, want phaseStreaming", m.streamPhase)
	}
	// The live thinking block is replaced by the collapsed summary in renderContent.
	content := m.renderContent()
	if strings.Contains(content, "💭 Thinking…") {
		t.Error("renderContent() still shows live thinking block after first token — should be collapsed")
	}
}

// TestTUIModel_ThoughtSummary_SurvivesCommit verifies that after commitMsg,
// the thought summary line is present in m.lines (not lost) so it persists
// when the user scrolls up after the response is complete.
func TestTUIModel_ThoughtSummary_SurvivesCommit(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, beginMsg{})
	m = send(t, m, thinkingMsg("I thought about this carefully."))
	m = send(t, m, tokenMsg("Here is my answer."))
	m = send(t, m, commitMsg{content: "Here is my answer."})

	// After commit, thinkBuf must be cleared.
	if m.thinkBuf.Len() != 0 {
		t.Error("thinkBuf should be empty after commitMsg")
	}
	if m.streamPhase != phaseIdle {
		t.Errorf("streamPhase = %v after commitMsg, want phaseIdle", m.streamPhase)
	}

	// The thought summary must be in m.lines so it survives in history.
	combined := strings.Join(m.lines, "\n")
	if !strings.Contains(combined, "💭 Thought") {
		t.Errorf("m.lines does not contain thought summary after commit\nlines: %v", m.lines)
	}
}

// TestTUIModel_ThoughtSummary_SurvivesBeginMsg verifies that a thought summary
// is also committed to m.lines when a second beginMsg fires (tool-call path)
// before commitMsg arrives.
func TestTUIModel_ThoughtSummary_SurvivesBeginMsg(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	m = send(t, m, beginMsg{})
	m = send(t, m, thinkingMsg("I will use a tool."))
	// Tool call fires — second BeginStream before any regular tokens.
	m = send(t, m, beginMsg{})

	combined := strings.Join(m.lines, "\n")
	if !strings.Contains(combined, "💭 Thought") {
		t.Errorf("m.lines does not contain thought summary after second beginMsg\nlines: %v", m.lines)
	}
	// After re-anchor, thinkBuf should be reset.
	if m.thinkBuf.Len() != 0 {
		t.Error("thinkBuf should be empty after second beginMsg")
	}
}

// TestTUIModel_PhaseIdle_Initially checks that the model starts in phaseIdle.
func TestTUIModel_PhaseIdle_Initially(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	if m.streamPhase != phaseIdle {
		t.Errorf("initial streamPhase = %v, want phaseIdle", m.streamPhase)
	}
}

// TestTUIModel_PhaseWaiting_AfterBeginMsg verifies phaseWaiting is set on beginMsg.
func TestTUIModel_PhaseWaiting_AfterBeginMsg(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)
	m = send(t, m, beginMsg{})
	if m.streamPhase != phaseWaiting {
		t.Errorf("streamPhase = %v after beginMsg, want phaseWaiting", m.streamPhase)
	}
}

// ── confirmation modals ───────────────────────────────────────────────────────

func TestTUIModel_ShowDiffMsg_SetsPendingDiff(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	resp := make(chan bool, 1)
	m = send(t, m, showDiffMsg{old: "old", new: "new", filename: "test.go", resp: resp})

	if m.pendingDiff == nil {
		t.Fatal("pendingDiff is nil after showDiffMsg")
	}
	if !strings.Contains(m.View(), "Apply this change?") {
		t.Errorf("View() does not show diff modal after showDiffMsg")
	}
}

func TestTUIModel_ShowConfirmMsg_SetsPendingConfirm(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	resp := make(chan bool, 1)
	m = send(t, m, showConfirmMsg{prompt: "Delete this file?", resp: resp})

	if m.pendingConfirm == nil {
		t.Fatal("pendingConfirm is nil after showConfirmMsg")
	}
	if !strings.Contains(m.View(), "Delete this file?") {
		t.Errorf("View() does not show confirm modal after showConfirmMsg")
	}
}

func TestTUIModel_PendingDiff_YAccepts(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	resp := make(chan bool, 1)
	m = send(t, m, showDiffMsg{old: "", new: "new content", filename: "f.go", resp: resp})
	m = send(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})

	if m.pendingDiff != nil {
		t.Error("pendingDiff should be nil after y key")
	}
	select {
	case got := <-resp:
		if !got {
			t.Error("resp = false, want true for y key")
		}
	default:
		t.Error("nothing sent to resp channel after y key")
	}
}

func TestTUIModel_PendingDiff_NRejects(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	resp := make(chan bool, 1)
	m = send(t, m, showDiffMsg{old: "", new: "content", filename: "f.go", resp: resp})
	m = send(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})

	if m.pendingDiff != nil {
		t.Error("pendingDiff should be nil after n key")
	}
	select {
	case got := <-resp:
		if got {
			t.Error("resp = true, want false for n key")
		}
	default:
		t.Error("nothing sent to resp channel after n key")
	}
}

func TestTUIModel_PendingConfirm_YAccepts(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	resp := make(chan bool, 1)
	m = send(t, m, showConfirmMsg{prompt: "Sure?", resp: resp})
	m = send(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})

	if m.pendingConfirm != nil {
		t.Error("pendingConfirm should be nil after y key")
	}
	select {
	case got := <-resp:
		if !got {
			t.Error("resp = false, want true for y key")
		}
	default:
		t.Error("nothing sent to resp channel after y key")
	}
}

func TestTUIModel_PendingConfirm_NRejects(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 40)

	resp := make(chan bool, 1)
	m = send(t, m, showConfirmMsg{prompt: "Sure?", resp: resp})
	send(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})

	select {
	case got := <-resp:
		if got {
			t.Error("resp = true, want false for n key")
		}
	default:
		t.Error("nothing sent to resp channel after n key")
	}
}

// TestTUIModel_PendingDiff_ScrollKeysKeepModalOpen verifies that arrow keys do
// not close the diff modal (pendingDiff must remain set after Up/Down).
func TestTUIModel_PendingDiff_ScrollKeysKeepModalOpen(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = initSize(t, m, 100, 20)

	resp := make(chan bool, 1)
	// Use enough diff lines to make the viewport scrollable.
	oldContent := strings.Repeat("old line\n", 40)
	newContent := strings.Repeat("new line\n", 40)
	m = send(t, m, showDiffMsg{old: oldContent, new: newContent, filename: "f.go", resp: resp})

	m = send(t, m, tea.KeyMsg{Type: tea.KeyUp})
	if m.pendingDiff == nil {
		t.Error("pendingDiff cleared by Up key — modal should remain open")
	}

	m = send(t, m, tea.KeyMsg{Type: tea.KeyDown})
	if m.pendingDiff == nil {
		t.Error("pendingDiff cleared by Down key — modal should remain open")
	}
}

func TestTUIModel_MouseWheelDown(t *testing.T) {
	events := make(chan tuiMsg, 8)
	m := newTestModel(t, events, make(chan<- string, 1))
	m = populateLinesOverflow(t, m, 100, 20)

	// Scroll up several lines, then scroll back down.
	for range 5 {
		m = send(t, m, tea.MouseMsg{Button: tea.MouseButtonWheelUp, Action: tea.MouseActionPress})
	}
	midOffset := m.viewport.YOffset

	m = send(t, m, tea.MouseMsg{Button: tea.MouseButtonWheelDown, Action: tea.MouseActionPress})
	afterDown := m.viewport.YOffset

	if afterDown <= midOffset {
		t.Errorf("YOffset did not increase after WheelDown: mid=%d after=%d", midOffset, afterDown)
	}
}
