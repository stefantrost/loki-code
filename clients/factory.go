package clients

import (
	"fmt"
	"strings"
)

// ApplyDefaults fills in missing fields based on APIType. It's a separate step
// from validation so the validator stays a pure read.
func ApplyDefaults(config *ClientConfig) {
	switch strings.ToLower(config.APIType) {
	case "ollama":
		if config.BaseURL == "" {
			config.BaseURL = "http://localhost:11434"
		}
		if config.ModelName == "" {
			config.ModelName = "qwen3:32b"
		}
	}
}

// CreateClient creates the appropriate client based on configuration.
// Callers should run ApplyDefaults and ValidateConfig first if they want
// missing fields populated; this function does the same internally for
// backwards compatibility.
func CreateClient(config ClientConfig, ctxMgr ContextManager, toolExec ToolExecutor, toolProvider ToolSchemaProvider) (LLMClient, error) {
	ApplyDefaults(&config)
	if err := ValidateConfig(config); err != nil {
		return nil, err
	}

	switch strings.ToLower(config.APIType) {
	case "ollama":
		client := NewOllamaClient(config.BaseURL, config.ModelName, ctxMgr, toolExec, toolProvider)
		if config.Debug {
			client.SetDebug(true)
		}
		if config.Truncator != nil {
			client.SetTruncator(config.Truncator)
		}
		return client, nil

	case "openai", "openai-compatible":
		client := NewOpenAIClient(config.BaseURL, config.BearerToken, config.ModelName, ctxMgr, toolExec, toolProvider)
		if config.Debug {
			client.SetDebug(true)
		}
		if config.Truncator != nil {
			client.SetTruncator(config.Truncator)
		}
		return client, nil

	default:
		return nil, fmt.Errorf("unsupported API type: %s (supported: ollama, openai, openai-compatible)", config.APIType)
	}
}

// DetectAPIType picks an APIType from the base URL. It is a heuristic of last
// resort — callers should prefer an explicit APIType. The check is ordered to
// favor "ollama" when the URL points at the standard Ollama port, since the
// substring "openai" appearing in a private hostname is more common than the
// reverse.
func DetectAPIType(baseURL string) string {
	baseURL = strings.ToLower(baseURL)

	if strings.Contains(baseURL, ":11434") || strings.Contains(baseURL, "localhost:11434") {
		return "ollama"
	}
	if strings.Contains(baseURL, "api.openai.com") || strings.Contains(baseURL, "/chat/completions") {
		return "openai-compatible"
	}
	if strings.Contains(baseURL, "ollama") {
		return "ollama"
	}
	if strings.Contains(baseURL, "openai") {
		return "openai-compatible"
	}
	return "ollama"
}

// ValidateConfig checks the client configuration without mutating it.
// Run ApplyDefaults first if you want missing optional fields populated.
func ValidateConfig(config ClientConfig) error {
	if config.APIType == "" {
		return fmt.Errorf("API type is required")
	}

	switch strings.ToLower(config.APIType) {
	case "ollama":
		// Ollama has full defaults; nothing to require.

	case "openai", "openai-compatible":
		if config.BaseURL == "" {
			return fmt.Errorf("base URL is required for OpenAI-compatible clients")
		}
		if config.ModelName == "" {
			return fmt.Errorf("model name is required for OpenAI-compatible clients")
		}

	default:
		return fmt.Errorf("unsupported API type: %s", config.APIType)
	}

	return nil
}

// GetDefaultConfig returns a default configuration for the specified API type
func GetDefaultConfig(apiType string) ClientConfig {
	switch strings.ToLower(apiType) {
	case "ollama":
		return ClientConfig{
			APIType:   "ollama",
			BaseURL:   "http://localhost:11434",
			ModelName: "qwen3:32b",
		}

	case "openai":
		return ClientConfig{
			APIType:   "openai",
			BaseURL:   "https://api.openai.com/v1",
			ModelName: "gpt-3.5-turbo",
		}

	case "openai-compatible":
		return ClientConfig{
			APIType: "openai-compatible",
		}

	default:
		return GetDefaultConfig("ollama")
	}
}
