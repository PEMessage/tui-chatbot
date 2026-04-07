# TUI Chatbot 🤖

Minimal terminal chatbot with streaming output and TPS (tokens per second) statistics.

Built with [uv](https://docs.astral.sh/uv/) package management and [OpenAI](https://platform.openai.com/) API.

## Features

- 🐚 **Shell-like interface** - Simple non-fullscreen terminal UI
- ⚡ **Streaming output** - Real-time token-by-token display
- 📊 **TPS statistics** - Tokens/second stats after each response
- 🔧 **Built-in commands** - /model, /clear, /help, /quit
- 🚀 **Flexible configuration** - CLI args, env vars, or .env file

## Quick Start

### Setup

```bash
cd tui-chatbot

# Copy and edit environment file
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your_key_here
```

### Run

```bash
# Using uv
uv run chat

# With arguments
uv run chat --api-key sk-xxx --model gpt-4
uv run chat --base-url http://localhost:11434/v1 --model llama2
```

## Commands

| Command | Description |
|---------|-------------|
| `/model` | List available models |
| `/model <name>` | Switch to model |
| `/clear` | Clear conversation history |
| `/help` | Show help |
| `/quit`, `/exit` | Exit |

## Example Session

```
🤖 ChatBot | URL: https://api.openai.com/v1 | Model: gpt-3.5-turbo
Commands: /model, /model <name>, /clear, /help, /quit

>>> Hello!

You: Hello!
Assistant: Hello! How can I help you today?
[Stats: 10 tokens, 0.45s, 22.15 tok/s]

>>> /model
Fetching models...

Available models (current: gpt-3.5-turbo):
--------------------------------------------------
  gpt-3.5-turbo *
  gpt-4
  gpt-4-turbo
  ...
--------------------------------------------------
Use /model <name> to switch

>>> /quit
Goodbye!
```

## Configuration

Priority: CLI args > Environment variables > .env file > Defaults

```bash
# CLI arguments
chat --base-url <url> --api-key <key> --model <model>

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

## Project Structure

```
tui-chatbot/
├── src/tui_chatbot/
│   ├── __init__.py
│   └── main.py          # Main application (~300 lines)
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## License

MIT
