package clients

// mergeToolCallDeltas merges incoming tool-call deltas into the accumulated
// list, identifying matches by ID. Streaming providers (notably OpenAI) split
// a single tool call across multiple chunks; without merging, duplicate or
// partial calls accumulate. Unmatched deltas are appended as new calls.
func mergeToolCallDeltas(existing []ToolCall, deltas []ToolCall) []ToolCall {
	for _, d := range deltas {
		idx := -1
		if d.ID != "" {
			for i := range existing {
				if existing[i].ID == d.ID {
					idx = i
					break
				}
			}
		}
		if idx == -1 {
			existing = append(existing, d)
			continue
		}
		if d.Type != "" {
			existing[idx].Type = d.Type
		}
		if d.Function.Name != "" {
			existing[idx].Function.Name = d.Function.Name
		}
		for k, v := range d.Function.Arguments {
			if existing[idx].Function.Arguments == nil {
				existing[idx].Function.Arguments = make(map[string]interface{})
			}
			existing[idx].Function.Arguments[k] = v
		}
	}
	return existing
}
