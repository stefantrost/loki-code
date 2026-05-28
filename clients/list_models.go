package clients

import (
	"fmt"
	"io"
	"os/exec"
	"strings"
)

// ListModels writes available models for the configured API type to out/errOut.
// Ollama: delegates to `ollama list`. Other providers return a not-supported error.
func ListModels(cfg ClientConfig, out, errOut io.Writer) error {
	switch strings.ToLower(cfg.APIType) {
	case "ollama":
		cmd := exec.Command("ollama", "list")
		cmd.Stdout = out
		cmd.Stderr = errOut
		return cmd.Run()
	default:
		return fmt.Errorf("model listing not supported for API type %q (current model: %s)", cfg.APIType, cfg.ModelName)
	}
}
