# TUI Chatbot 🤖

Minimal terminal chatbot with streaming output, TPS (tokens per second) statistics, and multi-API compatibility.

Built with [uv](https://docs.astral.sh/uv/) package management and modern Python (3.11+).

## Features

- 🐚 **Shell-like interface** - Non-fullscreen terminal UI
- ⚡ **Streaming output** - Real-time token-by-token display
- 📊 **TPS statistics** - Tokens/second stats after each response
- 🔧 **Multi-API compatibility** - Works with OpenAI, Doubao/Seed, and other OpenAI-compatible endpoints
- 🧩 **Pluggable extractor architecture** - Easy to add support for new API formats
- 🔧 **Built-in commands** - /model, /clear, /help, /quit
- 🚀 **Flexible configuration** - CLI args, env vars, or .env file

## Quick Start

```bash
# Setup
cd tui-chatbot
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your_key_here

# Run
uv run chat

# With specific API
uv run chat --api-key sk-xxx --model gpt-4
uv run chat --base-url https://api.openai.com/v1 --model gpt-3.5-turbo
```

## Architecture

### Content Extraction Layer

The chatbot uses a **protocol-based extractor pattern** to handle different API response formats:

```python
class ContentExtractor(Protocol):
    def extract(self, chunk: Any) -> Optional[str]: ...
```

**Built-in extractors:**

| Extractor | API Type | Fields Checked |
|-----------|----------|----------------|
| `StandardContentExtractor` | OpenAI, Claude, etc. | `delta.content` |
| `ReasoningContentExtractor` | Doubao/Seed/Ark | `delta.reasoning_content` → `delta.content` |
| `CompositeExtractor` | Chained fallback | Multiple extractors |

Auto-detection based on URL/model patterns:
- URLs containing: `volces`, `ark`, `doubao`, `seed` → `ReasoningContentExtractor`
- Model names containing: `doubao`, `seed` → `ReasoningContentExtractor`
- Default → `StandardContentExtractor`

### Extending for New APIs

To add support for a new API format:

```python
class MyCustomExtractor:
    def extract(self, chunk: Any) -> Optional[str]:
        # Your extraction logic
        return getattr(chunk.choices[0].delta, "my_field", None)

# Register in get_extractor() or pass directly
bot = ChatBot(config, extractor=MyCustomExtractor())
```

## Commands

| Command | Description |
|-----------|-------------|
| `/model` | List available models |
| `/model <name>` | Switch model (auto-detects extractor) |
| `/clear` | Clear conversation history |
| `/help` | Show help |
| `/quit`, `/exit` | Exit |

## Configuration

Priority: CLI args > Environment variables > .env file > Defaults

```bash
# CLI arguments
chat --base-url <url> --api-key <key> --model <model> --debug

# Environment variables
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-xxx
export OPENAI_MODEL=gpt-4
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url` | `https://api.openai.com/v1` | API base URL |
| `--api-key` | (required) | API key |
| `--model` | `gpt-3.5-turbo` | Model name |
| `--debug` | `false` | Enable debug logging |

## Tested APIs

| Provider | Base URL Pattern | Model | Status |
|----------|-----------------|-------|--------|
| OpenAI | `https://api.openai.com/v1` | gpt-3.5-turbo, gpt-4 | ✅ |
| ByteDance Ark (豆包) | `https://ark.cn-beijing.volces.com/api/v3` | doubao-seed-2-* | ✅ |
| Ollama | `http://localhost:11434/v1` | llama2, mistral, etc. | ✅ |

## Project Structure

```
tui-chatbot/
├── src/tui_chatbot/
│   ├── __init__.py
│   └── main.py              # Main application (~300 lines)
│                              # - ContentExtractor protocol
│                              # - Auto-detection logic
│                              # - ChatBot class
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Modern Python Features Used

- **Type hints** - Full typing with `Protocol`, `Optional`, `Any`
- **Dataclasses** - `ChatConfig`, `StreamStats`
- **Structural subtyping** - `ContentExtractor` protocol
- **Pattern matching ready** - Clean conditional logic for extractor selection
- **Async/await** - Proper async streaming

## Development

```bash
# Install in editable mode
uv pip install -e .

# Run with debug
uv run chat --debug

# Format
uv run ruff format src/

# Type check
uv run mypy src/
```

## License

MIT
