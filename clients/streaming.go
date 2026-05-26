package clients

// mergeToolCallDeltas merges incoming tool-call deltas into the accumulated
// list. Matches first by ID (when present), then by Index. Ollama sends
// complete calls so merging is a no-op there; this logic exists mainly as a
// defensive fallback for providers that stream partial tool calls.
func mergeToolCallDeltas(existing []ToolCall, deltas []ToolCall) []ToolCall {
	for _, d := range deltas {
		idx := -1
		if d.ID != "" {
			// Delta has an ID: match by ID only. If not found, this is a new call.
			for i := range existing {
				if existing[i].ID == d.ID {
					idx = i
					break
				}
			}
		} else {
			// No ID: this is a continuation fragment; match by Index.
			for i := range existing {
				if existing[i].Index == d.Index {
					idx = i
					break
				}
			}
		}
		if idx == -1 {
			existing = append(existing, d)
			continue
		}
		if d.ID != "" {
			existing[idx].ID = d.ID
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
