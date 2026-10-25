# Wonderdesk

Wonderdesk is a private, local-first chat workspace for brainstorming and writing with a language model running on your machine. Use the web UI, command line, or an OpenAI-compatible API; prompts and conversations are not sent to a hosted chat service.

## Setup

Requirements: Python 3.10+, Node.js 22.12+, and enough RAM, disk space, and optional GPU memory for your chosen model. The default model is Gemma 3 1B. On first use, Transformers downloads the model and server startup waits for it to load.

```powershell
.\scripts\setup.ps1
```

On Linux/macOS, use `bash scripts/setup.sh`. It creates `.venv`, installs dependencies, and builds the UI. `requirements.txt` selects CUDA 13.0 PyTorch; the macOS helper installs the platform-compatible PyPI build. Gemma is gated: accept its terms on the [Gemma model page](https://huggingface.co/google/gemma-3-1b-it), then run `hf auth login` if you have not already saved a Hugging Face token.

## Usage

### Web UI

Start the server:

```powershell
.\scripts\run.ps1
```

On Linux/macOS use `bash scripts/run.sh`; pass server options through, e.g. `bash scripts/run.sh --config settings.yaml`. In PowerShell, use `.\scripts\run.ps1 -Config settings.yaml` for a custom config.

Open <http://127.0.0.1:8080/>. The server loads the configured model before accepting requests, so startup may take a while. Select another model from the model menu; use **Stop** to cancel an active response or **New chat** to reset the conversation.

### Command line

```powershell
.\scripts\cli.ps1
```

On Linux/macOS use `bash scripts/cli.sh`. PowerShell accepts `.\scripts\cli.ps1 -Config settings.yaml -Plain`.

Use `:help` to see commands, `:new` to start a fresh conversation, and `:exit` (or `exit`) to quit. Use `python run_llm.py --plain` for plain-text output; plain mode is also selected automatically outside an interactive terminal.

### Docker

Build the image and run it bound to loopback, with a persistent model cache:

```powershell
.\scripts\docker.ps1 build
.\scripts\docker.ps1 run
```

On Linux/macOS use `bash scripts/docker.sh build` and `bash scripts/docker.sh run`. The run helper publishes on loopback, persists the model cache, and mounts the local Hugging Face token read-only when present.

The token mount is needed for gated models such as Gemma; omit it for models that do not require Hugging Face access. The image runs as a non-root user and includes a readiness health check. It has no built-in TLS or authentication. Keep it local, or put it behind a trusted HTTPS reverse proxy before allowing remote access.

## Development

Start the API server and Vite hot-reload UI together:

```powershell
.\scripts\dev.ps1
```

On Linux/macOS use `bash scripts/dev.sh`. Both helpers accept an optional backend config file, e.g. `.\scripts\dev.ps1 -Config settings.yaml` or `bash scripts/dev.sh --config settings.yaml`.

Open <http://127.0.0.1:5173/>. Vite proxies API requests to the backend at `127.0.0.1:8080`.

Run backend and frontend tests, lint, typecheck, build, and frontend audit:

```powershell
.\scripts\test.ps1
```

On Linux/macOS use `bash scripts/test.sh`.

Tests use fake model runtimes and do not download a model. CI also audits the pinned dependency graph and scans the production container image.

## Configuration and operations

Settings are read from `user_config.yaml`. For example:

```yaml
active_model: gemma_small
runtime: transformers
max_history_turns: 20

generation:
  temperature: 0.8
  top_p: 0.95
  max_tokens: 2048

server:
  host: 127.0.0.1
  port: 8080
  allowed_hosts: [localhost, 127.0.0.1, ::1]
  max_request_bytes: 4194304
  max_message_chars: 262144
  max_queued_requests: 4
  queue_timeout_seconds: 30
```

The legacy `active_model` and `generation.max_tokens` keys remain supported; `model_id` and `generation.max_new_tokens` are also accepted. Known model aliases include `gemma_small`, `gemma`, `gemma_4`, `code_gemma`, `llama`, `llama_small`, and `qwen3-8b`. A Hugging Face model ID can also be selected directly.

The app is single-user and has no authentication. It binds to loopback by default, validates Host and browser origins, limits request/message sizes, and serializes model operations with a bounded queue. Do not expose it directly to an untrusted network. When using a proxy or a custom hostname, set `server.allowed_hosts` accordingly. `LOCAL_LLM_HOST`, `LOCAL_LLM_PORT`, and comma-separated `LOCAL_LLM_ALLOWED_HOSTS` environment variables override server settings. `LOCAL_LLM_PUBLIC_URL` overrides the browser link printed when the model is ready; set it to the externally reachable base URL without a path for custom Docker port mappings or a proxy hostname.

The Web UI keeps its conversation history in browser memory and does not persist it. API clients send the conversation messages they want the model to see with each completion request.

### OpenAI-compatible provider

Use `http://127.0.0.1:8080/v1` as the base URL in OpenCode, Codex, or another OpenAI-compatible client. Select a model ID returned by `GET /v1/models` (for example, `google/gemma-3-1b-it`). The API has no authentication; if a client requires an API key field, use a placeholder. Keep the server on loopback or protect remote access with a trusted proxy.

Supported endpoints are `GET /v1/models`, `POST /v1/models/active`, and `POST /v1/chat/completions` (including SSE streaming).

Function tools are returned as OpenAI-compatible `tool_calls` for the connected harness to execute; Wonderdesk does not execute tools itself. Include the tool definitions and tool-call results in subsequent completion requests to continue the conversation.

The server logs to the terminal and a rotating file (10 MiB, three backups); message contents are not logged. Set `LOCAL_LLM_LOG_FILE` to choose the log path. Health and metrics endpoints:

- `GET /health/live`: process liveness
- `GET /health/ready`: readiness (returns 503 until the model is loaded)
- `GET /metrics`: request/generation counts and durations, model state, and process memory

Interactive API docs are at <http://127.0.0.1:8080/docs>.
