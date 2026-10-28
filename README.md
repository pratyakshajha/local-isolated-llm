# Wonderdesk

Wonderdesk is a private, local-first chat workspace for brainstorming and writing with a language model running on your machine. Use the Web UI, CLI, or OpenAI-compatible API; prompts and conversations are not sent to a hosted chat service.

## Features

- Chat in a browser or terminal, or connect an OpenAI-compatible client.
- Switch between local Hugging Face models and check estimated hardware fit.
- Stream responses and cancel active generation.
- Run locally or with Docker.

## Setup

Requirements: Python 3.11+, Node.js 22.12+ (the frontend also allows Node 24 and 26), and enough memory and disk space for your chosen model.

```powershell
.\scripts\setup.ps1
```

On Linux/macOS, use `bash scripts/setup.sh`. Gemma is gated: accept its terms on the [Gemma model page](https://huggingface.co/google/gemma-3-1b-it), then run `hf auth login` if you have not already saved a Hugging Face token.

## Usage

### Web UI

```powershell
.\scripts\run.ps1
```

On Linux/macOS, use `bash scripts/run.sh`. Open <http://127.0.0.1:8080/>. Choose a model from the menu; switching models starts a new conversation. Use **Stop** to cancel a response. Pass a custom config with `-Config settings.yaml` (PowerShell) or `--config settings.yaml` (Linux/macOS).

### CLI

```powershell
.\scripts\cli.ps1
```

On Linux/macOS, use `bash scripts/cli.sh`. Use `:help` for commands, `:new` for a fresh conversation, and `:exit` to quit. Use `python run_llm.py --plain` for plain-text output.

### OpenAI-compatible API

Set your client's base URL to `http://127.0.0.1:8080/v1` and choose a model from `/v1/models`. Interactive API docs are at <http://127.0.0.1:8080/docs>. See [REFERENCE.md](REFERENCE.md) for endpoints and examples.

### Docker

```powershell
.\scripts\docker.ps1 build
.\scripts\docker.ps1 run
```

On Linux/macOS, use `bash scripts/docker.sh build` and `bash scripts/docker.sh run`. The helper binds to loopback and persists the model cache. For NVIDIA GPU passthrough, use `./scripts/docker.ps1 run -Gpu` in PowerShell or `bash scripts/docker.sh run --gpu` on Linux/macOS. Docker has no built-in authentication or TLS; keep it local or protect remote access with a trusted HTTPS proxy.

## Development

Start the API and hot-reload UI with `./scripts/dev.ps1` (PowerShell) or `bash scripts/dev.sh` (Linux/macOS). Run backend and frontend tests, lint, typecheck, build, and frontend audit with `./scripts/test.ps1` or `bash scripts/test.sh`.

Tests use fake model runtimes and do not download a model. See [REFERENCE.md](REFERENCE.md) for configuration, model fit, API behavior, and operational details.
