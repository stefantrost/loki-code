package clients

import "testing"

func TestMergeToolCallDeltas_NewCall(t *testing.T) {
	existing := []ToolCall{}
	delta := []ToolCall{{ID: "a", Function: ToolFunc{Name: "read_file"}}}
	got := mergeToolCallDeltas(existing, delta)
	if len(got) != 1 || got[0].ID != "a" || got[0].Function.Name != "read_file" {
		t.Fatalf("expected single read_file call, got %+v", got)
	}
}

func TestMergeToolCallDeltas_MergesByID(t *testing.T) {
	existing := []ToolCall{{
		ID:       "a",
		Function: ToolFunc{Name: "read_file", Arguments: map[string]interface{}{"path": "x"}},
	}}
	delta := []ToolCall{{
		ID:       "a",
		Function: ToolFunc{Arguments: map[string]interface{}{"limit": 100}},
	}}
	got := mergeToolCallDeltas(existing, delta)
	if len(got) != 1 {
		t.Fatalf("expected 1 merged call, got %d", len(got))
	}
	if got[0].Function.Name != "read_file" {
		t.Errorf("name lost: %q", got[0].Function.Name)
	}
	if got[0].Function.Arguments["path"] != "x" || got[0].Function.Arguments["limit"] != 100 {
		t.Errorf("arg merge failed: %+v", got[0].Function.Arguments)
	}
}

func TestMergeToolCallDeltas_DistinctIDsKeptSeparate(t *testing.T) {
	existing := []ToolCall{{ID: "a", Function: ToolFunc{Name: "read_file"}}}
	delta := []ToolCall{{ID: "b", Function: ToolFunc{Name: "list_files"}}}
	got := mergeToolCallDeltas(existing, delta)
	if len(got) != 2 {
		t.Fatalf("expected 2 distinct calls, got %d: %+v", len(got), got)
	}
}

func TestMergeToolCallDeltas_IndexBasedContinuation(t *testing.T) {
	existing := []ToolCall{{ID: "a", Index: 0, Function: ToolFunc{Name: "read_file"}}}
	// Continuation fragment: no ID, same index as existing call.
	delta := []ToolCall{{Index: 0, Function: ToolFunc{Arguments: map[string]interface{}{"path": "/tmp"}}}}
	got := mergeToolCallDeltas(existing, delta)
	if len(got) != 1 {
		t.Fatalf("expected 1 merged call, got %d", len(got))
	}
	if got[0].Function.Arguments["path"] != "/tmp" {
		t.Errorf("index-based merge failed: %+v", got[0].Function.Arguments)
	}
}

func TestMergeToolCallDeltas_IndexBasedNewCall(t *testing.T) {
	existing := []ToolCall{{ID: "a", Index: 0, Function: ToolFunc{Name: "read_file"}}}
	// No ID, different index: should be appended as a new call.
	delta := []ToolCall{{Index: 1, Function: ToolFunc{Name: "list_files"}}}
	got := mergeToolCallDeltas(existing, delta)
	if len(got) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(got))
	}
	if got[1].Function.Name != "list_files" {
		t.Errorf("new index-based call not appended: %+v", got[1])
	}
}

func TestTruncateToolResult_ReadFile(t *testing.T) {
	long := make([]byte, 6000)
	for i := range long {
		long[i] = 'x'
	}
	got := defaultTruncate(string(long), "read_file")
	if len(got) >= len(long) {
		t.Errorf("expected truncation, got len=%d", len(got))
	}
}

func TestTruncateToolResult_ShortPassthrough(t *testing.T) {
	got := defaultTruncate("hello", "read_file")
	if got != "hello" {
		t.Errorf("short result mutated: %q", got)
	}
}
