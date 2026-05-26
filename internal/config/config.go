package config

import (
	"bufio"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"loki-code/clients"
)

// LoadConfig loads configuration from file and environment variables.
func LoadConfig(configFilePath string) (clients.ClientConfig, error) {
	config := clients.ClientConfig{}

	config.APIType = "ollama"
	config.BaseURL = "http://localhost:11434"
	config.ModelName = "qwen3:32b"

	if configFilePath != "" {
		if err := loadConfigFromFile(configFilePath, &config); err != nil {
			return config, fmt.Errorf("failed to load config file %s: %v", configFilePath, err)
		}
	} else {
		defaultPaths := []string{
			"llm.env",
			".env",
			filepath.Join(os.Getenv("HOME"), ".loki-code", "config.env"),
		}

		for _, path := range defaultPaths {
			if fileExists(path) {
				if err := loadConfigFromFile(path, &config); err != nil {
					slog.Warn("Failed to load config file", "file", path, "error", err)
				} else {
					fmt.Printf("✓ Loaded configuration from: %s\n", path)
					break
				}
			}
		}
	}

	loadConfigFromEnv(&config)

	if config.APIType == "" || config.APIType == "auto" {
		config.APIType = clients.DetectAPIType(config.BaseURL)
	}

	return config, nil
}

func loadConfigFromFile(filePath string, config *clients.ClientConfig) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			slog.Warn("Invalid config format", "file", filePath, "line", lineNum, "content", line)
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		if (strings.HasPrefix(value, `"`) && strings.HasSuffix(value, `"`)) ||
			(strings.HasPrefix(value, `'`) && strings.HasSuffix(value, `'`)) {
			value = value[1 : len(value)-1]
		}

		switch strings.ToUpper(key) {
		case "API_TYPE":
			config.APIType = value
		case "BASE_URL", "API_BASE_URL":
			config.BaseURL = value
		case "MODEL_NAME", "MODEL":
			config.ModelName = value
		case "BEARER_TOKEN", "API_KEY", "OPENAI_API_KEY":
			config.BearerToken = value
		case "DEBUG":
			config.Debug = strings.ToLower(value) == "true" || value == "1"
		default:
			slog.Warn("Unknown configuration key", "file", filePath, "key", key)
		}
	}

	return scanner.Err()
}

func loadConfigFromEnv(config *clients.ClientConfig) {
	if value := os.Getenv("LOKI_API_TYPE"); value != "" {
		config.APIType = value
	}
	if value := os.Getenv("LOKI_BASE_URL"); value != "" {
		config.BaseURL = value
	}
	if value := os.Getenv("LOKI_MODEL"); value != "" {
		config.ModelName = value
	}
	if value := os.Getenv("LOKI_BEARER_TOKEN"); value != "" {
		config.BearerToken = value
	}
	if value := os.Getenv("LOKI_DEBUG"); value != "" {
		config.Debug = strings.ToLower(value) == "true" || value == "1"
	}

	if value := os.Getenv("OPENAI_API_KEY"); value != "" && config.BearerToken == "" {
		config.BearerToken = value
	}
	if value := os.Getenv("OPENAI_BASE_URL"); value != "" && config.BaseURL == "http://localhost:11434" {
		config.BaseURL = value
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// PrintConfig prints the current configuration (with sensitive data masked).
func PrintConfig(config clients.ClientConfig) {
	fmt.Println("📋 Current Configuration:")
	fmt.Printf("  API Type:    %s\n", config.APIType)
	fmt.Printf("  Base URL:    %s\n", config.BaseURL)
	fmt.Printf("  Model:       %s\n", config.ModelName)

	if config.BearerToken != "" {
		masked := config.BearerToken
		if len(masked) > 8 {
			masked = masked[:4] + "..." + masked[len(masked)-4:]
		} else {
			masked = "***"
		}
		fmt.Printf("  Bearer Token: %s\n", masked)
	} else {
		fmt.Printf("  Bearer Token: (not set)\n")
	}

	fmt.Printf("  Debug:       %t\n", config.Debug)
}

// ValidateAndFixConfig validates and attempts to fix common configuration issues.
func ValidateAndFixConfig(config *clients.ClientConfig) error {
	if config.APIType == "" {
		config.APIType = clients.DetectAPIType(config.BaseURL)
		fmt.Printf("ℹ️  Auto-detected API type: %s\n", config.APIType)
	}

	if err := clients.ValidateConfig(*config); err != nil {
		return err
	}

	switch strings.ToLower(config.APIType) {
	case "openai", "openai-compatible":
		config.BaseURL = strings.TrimSuffix(config.BaseURL, "/")
		if !strings.HasPrefix(config.BaseURL, "http") {
			return fmt.Errorf("base URL must start with http:// or https://")
		}
	case "ollama":
		if config.BaseURL == "" {
			config.BaseURL = "http://localhost:11434"
		}
		if config.ModelName == "" {
			config.ModelName = "qwen3:32b"
		}
	}

	return nil
}

// CreateExampleConfig creates an example configuration file.
func CreateExampleConfig(path string) error {
	exampleContent := `# Loki Code Configuration File
# Choose your API type: ollama, openai, or openai-compatible

# For Ollama (default)
API_TYPE=ollama
BASE_URL=http://localhost:11434
MODEL_NAME=qwen3:32b

# For OpenAI
# API_TYPE=openai
# BASE_URL=https://api.openai.com/v1
# MODEL_NAME=gpt-3.5-turbo
# BEARER_TOKEN=your_openai_api_key_here

# For OpenAI-compatible APIs (e.g., LocalMind)
# API_TYPE=openai-compatible
# BASE_URL=https://YOUR_INSTANCE.localmind.io/localmind/public-chat
# MODEL_NAME=localmind-ultra
# BEARER_TOKEN=YOUR_API_KEY

# Debug mode (optional)
# DEBUG=false
`

	return os.WriteFile(path, []byte(exampleContent), 0600)
}
