# Loki Code

A local coding agent using LangChain and local LLMs for intelligent code assistance and automation.

## Description

Loki Code is designed to be your local coding companion, providing AI-powered assistance without relying on external APIs. Built with modularity in mind, it can be extended with various tools and capabilities to enhance your development workflow.

## Installation

### Development Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd loki-code
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. For development with additional tools:
   ```bash
   pip install -e ".[dev]"
   ```

## Quick Start

### Basic Usage

After installation, you can use Loki Code from the command line:

```bash
# Check available LLM providers
python main.py --list-providers

# Test LLM connection (requires Ollama running)
python main.py --test-llm

# Start interactive chat mode
python main.py --chat

# Send a single prompt
python main.py --chat --prompt "Help me write a Python function"

# Show help and all options
python main.py --help
```

### Python API Usage

You can also use Loki Code programmatically:

```python
import loki_code
from loki_code.core.providers import create_llm_provider, GenerationRequest
from loki_code.config import load_config

# Load configuration
config = load_config()

# Create LLM provider
provider = create_llm_provider(config)

# Generate text
request = GenerationRequest(prompt="Hello, world!")
response = provider.generate_sync(request)
print(response.content)
```

## Project Structure

```
loki-code/
├── src/loki_code/          # Main package
│   ├── core/               # Core functionality
│   │   ├── providers/      # LLM provider abstraction system
│   │   ├── llm_client.py   # Basic LLM communication
│   │   └── llm_test.py     # LLM connection testing
│   ├── tools/              # Agent tools and utilities
│   ├── ui/                 # User interface components
│   ├── config/             # Configuration management
│   │   ├── models.py       # Pydantic configuration models
│   │   └── loader.py       # Configuration loading logic
│   └── utils/              # Helper utilities
│       └── logging.py      # Production logging system
├── tests/                  # Test suite
├── configs/                # Configuration files
│   └── default.yaml        # Default configuration
├── docs/                   # Documentation
└── pyproject.toml          # Modern Python package setup
```

## Provider System

Loki Code uses a flexible provider abstraction system that supports multiple LLM providers:

### Currently Supported
- **Ollama** - Local LLM execution with model management
  - Supports streaming responses
  - Dynamic model switching
  - Local execution for privacy

### Architecture Features
- **Provider Abstraction** - Unified interface across all providers
- **Async/Sync Support** - Both asynchronous and synchronous APIs
- **Fallback Mechanisms** - Automatic provider switching on failure
- **Task-Aware Models** - Different models for different tasks
- **Easy Extension** - Simple to add new providers

### Future Providers (Planned)
- OpenAI API (for fallback scenarios)
- Anthropic Claude (if allowed)
- Hugging Face local models
- Custom API endpoints

## Development Setup

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Code formatting:
   ```bash
   black src/ tests/
   ```

4. Type checking:
   ```bash
   mypy src/
   ```

## Requirements

- Python 3.9+
- Local LLM setup (documentation coming soon)

## License

MIT License - see LICENSE file for details.

## Contributing

This project is in early development. Contribution guidelines will be established as the project matures.