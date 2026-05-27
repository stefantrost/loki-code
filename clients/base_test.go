package clients

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

// newTestBase builds a minimal baseClient with no context manager or HTTP
// client — enough to exercise the writer/callback helpers and truncation.
func newTestBase() *baseClient {
	return &baseClient{
		interruptChan: make(chan struct{}, 1),
	}
}

// ── Output writer ────────────────────────────────────────────────────────────

func TestBaseClient_GetOutputWriter_Default(t *testing.T) {
	c := newTestBase()
	if got := c.getOutputWriter(); got != io.Discard {
		t.Errorf("getOutputWriter() with nil writer = %v, want io.Discard", got)
	}
}

func TestBaseClient_SetOutputWriter_RoundTrip(t *testing.T) {
	c := newTestBase()
	var buf bytes.Buffer
	c.SetOutputWriter(&buf)
	if got := c.getOutputWriter(); got != &buf {
		t.Errorf("getOutputWriter() after SetOutputWriter = %v, want buf", got)
	}
}

func TestBaseClient_SetOutputWriter_NilRestoresDiscard(t *testing.T) {
	c := newTestBase()
	var buf bytes.Buffer
	c.SetOutputWriter(&buf)
	c.SetOutputWriter(nil)
	if got := c.getOutputWriter(); got != io.Discard {
		t.Errorf("getOutputWriter() after SetOutputWriter(nil) = %v, want io.Discard", got)
	}
}

// ── System writer ────────────────────────────────────────────────────────────

func TestBaseClient_GetSystemWriter_Default(t *testing.T) {
	c := newTestBase()
	if got := c.getSystemWriter(); got != io.Discard {
		t.Errorf("getSystemWriter() with nil writer = %v, want io.Discard", got)
	}
}

func TestBaseClient_SetSystemWriter_RoundTrip(t *testing.T) {
	c := newTestBase()
	var buf bytes.Buffer
	c.SetSystemWriter(&buf)
	if got := c.getSystemWriter(); got != &buf {
		t.Errorf("getSystemWriter() after SetSystemWriter = %v, want buf", got)
	}
}

func TestBaseClient_SetSystemWriter_NilRestoresDiscard(t *testing.T) {
	c := newTestBase()
	var buf bytes.Buffer
	c.SetSystemWriter(&buf)
	c.SetSystemWriter(nil)
	if got := c.getSystemWriter(); got != io.Discard {
		t.Errorf("getSystemWriter() after SetSystemWriter(nil) = %v, want io.Discard", got)
	}
}

// ── Stream-start callback ────────────────────────────────────────────────────

func TestBaseClient_NotifyStreamStart_NilCallback(t *testing.T) {
	c := newTestBase()
	// Must not panic when no callback is registered.
	c.notifyStreamStart()
}

func TestBaseClient_NotifyStreamStart_CallsCallback(t *testing.T) {
	c := newTestBase()
	called := 0
	c.SetStreamStartCallback(func() { called++ })
	c.notifyStreamStart()
	c.notifyStreamStart()
	if called != 2 {
		t.Errorf("notifyStreamStart called callback %d times, want 2", called)
	}
}

func TestBaseClient_SetStreamStartCallback_NilClears(t *testing.T) {
	c := newTestBase()
	called := false
	c.SetStreamStartCallback(func() { called = true })
	c.SetStreamStartCallback(nil)
	c.notifyStreamStart()
	if called {
		t.Error("notifyStreamStart invoked callback after it was cleared with nil")
	}
}

// ── truncateString ───────────────────────────────────────────────────────────

func TestTruncateString_ShortString(t *testing.T) {
	if got := truncateString("hello", 10); got != "hello" {
		t.Errorf("truncateString short = %q, want %q", got, "hello")
	}
}

func TestTruncateString_ExactLength(t *testing.T) {
	if got := truncateString("hello", 5); got != "hello" {
		t.Errorf("truncateString exact = %q, want %q", got, "hello")
	}
}

func TestTruncateString_Truncated(t *testing.T) {
	got := truncateString("hello world", 5)
	if got != "hello..." {
		t.Errorf("truncateString long = %q, want %q", got, "hello...")
	}
}

// ── truncateToolResult ───────────────────────────────────────────────────────

func TestTruncateToolResult_DefaultTruncator_Short(t *testing.T) {
	c := newTestBase()
	result := "short result"
	if got := c.truncateToolResult(result, "any_tool"); got != result {
		t.Errorf("truncateToolResult short = %q, want %q", got, result)
	}
}

func TestTruncateToolResult_DefaultTruncator_Long(t *testing.T) {
	c := newTestBase()
	long := strings.Repeat("x", MaxToolResultChars+100)
	got := c.truncateToolResult(long, "any_tool")
	if len(got) >= len(long) {
		t.Errorf("truncateToolResult long: len=%d, expected < %d", len(got), len(long))
	}
	if !strings.Contains(got, "truncated") {
		t.Errorf("truncateToolResult long: expected '...truncated...' suffix, got %q", got[len(got)-60:])
	}
}

func TestTruncateToolResult_CustomTruncator(t *testing.T) {
	c := newTestBase()
	c.SetTruncator(func(result, toolName string) string {
		return "custom:" + toolName
	})
	if got := c.truncateToolResult("anything", "mytool"); got != "custom:mytool" {
		t.Errorf("truncateToolResult custom = %q, want %q", got, "custom:mytool")
	}
}
