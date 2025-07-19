# Loki Code - AI Coding Agent Makefile

# Variables
BINARY_NAME=loki-code
MODEL_NAME?=qwen3:32b
OLLAMA_PORT=11434
OLLAMA_URL?=http://localhost:$(OLLAMA_PORT)

# Default target
.PHONY: help
help:
	@echo "Loki Code - AI Coding Agent"
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build the application"
	@echo "  start       - Start the application (alias for run)"
	@echo "  run         - Build and run the application"
	@echo "  clean       - Remove build artifacts"
	@echo "  setup       - Setup Ollama and pull required model"
	@echo "  start-ollama - Start Ollama service"
	@echo "  stop-ollama  - Stop Ollama service"
	@echo "  check-ollama - Check if Ollama is running"
	@echo "  pull-model   - Pull the required model ($(MODEL_NAME))"
	@echo "  test-api     - Test Ollama API connection"
	@echo "  dev          - Setup, start Ollama, and run the app"
	@echo "  install      - Install binary to /usr/local/bin"
	@echo "  uninstall    - Remove binary from /usr/local/bin"
	@echo ""
	@echo "Variables (override with MODEL_NAME=... make run):"
	@echo "  MODEL_NAME   - Model to use (default: $(MODEL_NAME))"
	@echo "  OLLAMA_URL   - Ollama server URL (default: $(OLLAMA_URL))"
	@echo ""
	@echo "Examples:"
	@echo "  make run                          # Use default model"
	@echo "  MODEL_NAME=llama3:8b make run     # Use specific model"
	@echo "  ./$(BINARY_NAME) --model mistral:7b  # Direct binary usage"

# Build the application
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	@go build -o $(BINARY_NAME) .
	@echo "Build complete: $(BINARY_NAME)"

# Start the application (alias for run)
.PHONY: start
start: run

# Build and run the application
.PHONY: run
run: build
	@echo "Starting $(BINARY_NAME) with model $(MODEL_NAME)..."
	@./$(BINARY_NAME) --model $(MODEL_NAME) --url $(OLLAMA_URL)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(BINARY_NAME)
	@go clean
	@echo "Clean complete"

# Start Ollama service
.PHONY: start-ollama
start-ollama:
	@echo "Starting Ollama service..."
	@if pgrep -f "ollama serve" > /dev/null; then \
		echo "Ollama is already running"; \
	else \
		ollama serve & \
		echo "Ollama started in background"; \
		sleep 3; \
	fi

# Stop Ollama service
.PHONY: stop-ollama
stop-ollama:
	@echo "Stopping Ollama service..."
	@pkill -f "ollama serve" || echo "Ollama was not running"
	@echo "Ollama stopped"

# Check if Ollama is running
.PHONY: check-ollama
check-ollama:
	@echo "Checking Ollama status..."
	@if curl -s http://localhost:$(OLLAMA_PORT)/api/version > /dev/null 2>&1; then \
		echo "✓ Ollama is running on port $(OLLAMA_PORT)"; \
	else \
		echo "✗ Ollama is not running or not accessible on port $(OLLAMA_PORT)"; \
		exit 1; \
	fi

# Pull the required model
.PHONY: pull-model
pull-model:
	@echo "Checking if model $(MODEL_NAME) exists..."
	@if ollama list | grep -q $(MODEL_NAME); then \
		echo "✓ Model $(MODEL_NAME) already exists"; \
	else \
		echo "Pulling model: $(MODEL_NAME)..."; \
		ollama pull $(MODEL_NAME); \
		echo "Model $(MODEL_NAME) pulled successfully"; \
	fi

# Test Ollama API connection
.PHONY: test-api
test-api: check-ollama
	@echo "Testing API with model $(MODEL_NAME)..."
	@curl -s -X POST http://localhost:$(OLLAMA_PORT)/api/chat \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_NAME)", "messages": [{"role": "user", "content": "Hello, are you working?"}], "stream": false}' \
		| grep -q "response" && echo "✓ API test successful" || echo "✗ API test failed"

# Setup everything (start Ollama and pull model)
.PHONY: setup
setup: start-ollama
	@echo "Setting up Loki Code environment..."
	@sleep 5
	@$(MAKE) pull-model
	@echo "Setup complete! Run 'make run' to start the application."

# Development workflow (setup and run)
.PHONY: dev
dev: setup run

# Install binary to system
.PHONY: install
install: build
	@echo "Installing $(BINARY_NAME) to /usr/local/bin..."
	@sudo cp $(BINARY_NAME) /usr/local/bin/
	@sudo chmod +x /usr/local/bin/$(BINARY_NAME)
	@echo "$(BINARY_NAME) installed successfully"

# Uninstall binary from system
.PHONY: uninstall
uninstall:
	@echo "Removing $(BINARY_NAME) from /usr/local/bin..."
	@sudo rm -f /usr/local/bin/$(BINARY_NAME)
	@echo "$(BINARY_NAME) uninstalled successfully"

# Show project status
.PHONY: status
status:
	@echo "=== Loki Code Status ==="
	@echo -n "Binary exists: "
	@if [ -f $(BINARY_NAME) ]; then echo "✓ Yes"; else echo "✗ No (run 'make build')"; fi
	@echo -n "Ollama service: "
	@if pgrep -f "ollama serve" > /dev/null; then echo "✓ Running"; else echo "✗ Not running (run 'make start-ollama')"; fi
	@echo -n "Ollama API: "
	@if curl -s http://localhost:$(OLLAMA_PORT)/api/version > /dev/null 2>&1; then echo "✓ Accessible"; else echo "✗ Not accessible"; fi
	@echo -n "Model $(MODEL_NAME): "
	@if ollama list | grep -q $(MODEL_NAME); then echo "✓ Available"; else echo "✗ Not found (run 'make pull-model')"; fi