package clients

import (
	"fmt"
	"strings"
)

// CreateClient creates the appropriate client based on configuration
func CreateClient(config ClientConfig) (LLMClient, error) {
	switch strings.ToLower(config.APIType) {
	case "ollama":
		if config.BaseURL == "" {
			config.BaseURL = "http://localhost:11434"
		}
		if config.ModelName == "" {
			config.ModelName = "qwen3:32b"
		}
		
		client := NewOllamaClient(config.BaseURL, config.ModelName)
		if config.Debug {
			client.SetDebug(true)
		}
		return client, nil
		
	case "openai", "openai-compatible":
		if config.BaseURL == "" {
			return nil, fmt.Errorf("base URL is required for OpenAI-compatible clients")
		}
		if config.ModelName == "" {
			return nil, fmt.Errorf("model name is required for OpenAI-compatible clients")
		}
		if config.BearerToken == "" {
			return nil, fmt.Errorf("bearer token is required for OpenAI-compatible clients")
		}
		
		client := NewOpenAIClient(config.BaseURL, config.BearerToken, config.ModelName)
		if config.Debug {
			client.SetDebug(true)
		}
		return client, nil
		
	default:
		return nil, fmt.Errorf("unsupported API type: %s (supported: ollama, openai, openai-compatible)", config.APIType)
	}
}

// DetectAPIType attempts to detect the API type from the base URL
func DetectAPIType(baseURL string) string {
	baseURL = strings.ToLower(baseURL)
	
	// Check for OpenAI-compatible indicators
	if strings.Contains(baseURL, "openai") || 
	   strings.Contains(baseURL, "api.openai.com") ||
	   strings.Contains(baseURL, "localmind") ||
	   strings.Contains(baseURL, "/chat/completions") {
		return "openai-compatible"
	}
	
	// Check for Ollama indicators
	if strings.Contains(baseURL, "ollama") || 
	   strings.Contains(baseURL, ":11434") ||
	   strings.Contains(baseURL, "localhost:11434") {
		return "ollama"
	}
	
	// Default to Ollama if unable to detect
	return "ollama"
}

// ValidateConfig validates the client configuration
func ValidateConfig(config ClientConfig) error {
	if config.APIType == "" {
		return fmt.Errorf("API type is required")
	}
	
	switch strings.ToLower(config.APIType) {
	case "ollama":
		// Ollama validation
		if config.BaseURL == "" {
			config.BaseURL = "http://localhost:11434" // Set default
		}
		if config.ModelName == "" {
			config.ModelName = "qwen3:32b" // Set default
		}
		
	case "openai", "openai-compatible":
		// OpenAI validation
		if config.BaseURL == "" {
			return fmt.Errorf("base URL is required for OpenAI-compatible clients")
		}
		if config.ModelName == "" {
			return fmt.Errorf("model name is required for OpenAI-compatible clients") 
		}
		if config.BearerToken == "" {
			return fmt.Errorf("bearer token is required for OpenAI-compatible clients")
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
			// BearerToken must be provided by user
		}
		
	case "openai-compatible":
		return ClientConfig{
			APIType: "openai-compatible",
			// BaseURL, ModelName, and BearerToken must be provided by user
		}
		
	default:
		// Default to Ollama
		return GetDefaultConfig("ollama")
	}
}