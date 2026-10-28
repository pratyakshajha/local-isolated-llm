# Wonderdesk

Wonderdesk is a private, local-first chat workspace for brainstorming and writing with a language model running on your machine. Use the web UI, command line, or an OpenAI-compatible API; prompts and conversations are not sent to a hosted chat service.

## Setup

Requirements: Python 3.11+, Node.js 22.12+ (the frontend also allows Node 24 and 26), and enough RAM, disk space, and optional GPU memory for your chosen model. The default model is Gemma 3 1B. On first use, Transformers downloads the model and server startup waits for it to load. Each model load also checks Hugging Face for updated model files; if that check fails and a complete copy is already cached, the cached copy is used.

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

Open <http://127.0.0.1:8080/>. The server loads the configured model before accepting requests, so startup may take a while. Select another model from the model menu; switching models starts a new conversation. Use **Stop** to cancel an active response. During a model switch, the overlay shows byte progress when available; otherwise it shows an activity indicator and elapsed time. During a download, if there are no transfer updates for 30 seconds, it says “No transfer updates yet.” It also distinguishes loading the model into memory from finishing the switch.

### Command line

```powershell
.\scripts\cli.ps1
```

On Linux/macOS use `bash scripts/cli.sh`. PowerShell accepts `.\scripts\cli.ps1 -Config settings.yaml -Plain`.

Use `:help` to see commands, `:new` to start a fresh conversation, and `:exit` (or `exit` or `quit`) to quit. Use `python run_llm.py --plain` for plain-text output; plain mode is also selected automatically outside an interactive terminal.

### Docker

Build the image and run it bound to loopback, with a persistent model cache:

```powershell
.\scripts\docker.ps1 build
.\scripts\docker.ps1 run
```

On Linux/macOS use `bash scripts/docker.sh build` and `bash scripts/docker.sh run`. The run helper publishes on loopback, persists the model cache, and mounts the local Hugging Face token read-only when present.

**GPU support (NVIDIA only):** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then build and run with GPU passthrough enabled:

```powershell
.\scripts\docker.ps1 build
.\scripts\docker.ps1 run -Gpu
```

On Linux/macOS, use `bash scripts/docker.sh build` and `bash scripts/docker.sh run --gpu`. The image is built on CUDA 13.0, and AMD GPU support is not included. For CPU-only Docker, use `run` without the GPU option.

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

Settings are read from `user_config.yaml`, or from the file passed with `--config`. For example:

```yaml
active_model: gemma_small
runtime: transformers
max_history_turns: 4

generation:
  temperature: 0.8
  top_p: 0.95
  max_tokens: 256

server:
  host: 127.0.0.1
  port: 8080
  allowed_hosts: [localhost, 127.0.0.1, ::1]
  max_request_bytes: 4194304
  max_message_chars: 262144
  max_queued_requests: 4
  queue_timeout_seconds: 30
```

The legacy `active_model` and `generation.max_tokens` keys remain supported; `model_id` and `generation.max_new_tokens` are also accepted. `system_prompt` sets the instructions sent with every chat request from the CLI, the web UI, and the API; the default is “You are a helpful assistant.”

Known model aliases include `gemma_small`, `gemma`, `gemma_4`, `code_gemma`, `code_gemma_it`, `llama`, `llama_small`, and `qwen3-8b`. A Hugging Face model ID can also be selected directly.

On NVIDIA CUDA, Transformers loads models with NF4 4-bit weights directly onto the active CUDA device. This requires `bitsandbytes` and Accelerate (included in the standard installs). CPU loads the model in float32. To check live usage during a response, run `nvidia-smi -l 1`; in Windows Task Manager, view the NVIDIA GPU's **CUDA** engine rather than its **3D** engine.

### Checking model fit

Use **Hardware fit** beside the model selector in the Web UI, or run `:fit` in the CLI. In the CLI, `:fit 12 32` checks a different machine using 12 GB of GPU memory and 32 GB of RAM. The Web UI list is searchable and sortable, shows each model's estimated speed and whether its files are present in the local Hugging Face cache, and has a **Use** button to switch models. Selecting a model that is not downloaded uses the normal model-switch flow and downloads its files when needed. Cache status checks for the required config, tokenizer, and weight files (including shards); it does not test-load the model. The read-only `GET /api/model-fit` endpoint returns the model list and cache status, accepts the same optional `gpu_vram_gb` and `ram_gb` query parameters, and detects omitted values on the server host. It lists the most-downloaded text-generation models from `google`, `meta-llama`, `Qwen`, `deepseek-ai`, and `mistralai`, caches that list for an hour, and falls back to the built-in aliases if the Hub is unavailable.

Fit and speed results are estimates, not load guarantees. Memory needs use parameter count with a 20% margin and 1 GiB runtime overhead; CUDA/ROCm estimates use 2 bytes per parameter, while CPU, MPS, and XPU estimates use 4. On NVIDIA, the runtime loads 4-bit weights, so real memory use is usually below the estimate. The tokens-per-second figure is bandwidth-based, not a benchmark, and it is only shown for GPUs in the app's built-in bandwidth table. Opening the fit list does not load or switch models; only **Use** changes the active model.

The app is single-user and has no authentication. It binds to loopback by default, validates Host and browser origins, limits request/message sizes, and serializes model operations with a bounded queue. Do not expose it directly to an untrusted network. When using a proxy or a custom hostname, set `server.allowed_hosts` accordingly. `LOCAL_LLM_HOST`, `LOCAL_LLM_PORT`, and comma-separated `LOCAL_LLM_ALLOWED_HOSTS` environment variables override server settings. `LOCAL_LLM_PUBLIC_URL` overrides the browser link printed when the model is ready; set it to the externally reachable base URL without a path for custom Docker port mappings or a proxy hostname.

The Web UI keeps its conversation history in browser memory and does not persist it. API clients send the conversation messages they want the model to see with each completion request.

### OpenAI-compatible provider

Use `http://127.0.0.1:8080/v1` as the base URL in OpenCode, Codex, or another OpenAI-compatible client. Pass a model ID from `GET /v1/models` (for example, `google/gemma-3-1b-it`) or any Hugging Face model ID. If the requested model is not the active one, the server switches to it before answering, and the first request may download it. The API has no authentication; if a client requires an API key field, use a placeholder. Keep the server on loopback or protect remote access with a trusted proxy.

Supported endpoints are `GET /v1/models`, `POST /v1/models/active`, `POST /v1/chat/completions` (including SSE streaming), and `POST /v1/completions` (including SSE streaming).

Chat and text completion use separate model capabilities. `google/codegemma-1.1-7b-it` (alias `code_gemma_it`) supports chat; `google/codegemma-1.1-2b` (alias `code_gemma`) is completion-only and is hidden from the Web UI chat model selector. Other completion-only models are handled the same way: chat requests return HTTP 400 and direct clients to `/v1/completions`.

Chat example. It names a different model, so the server switches to it before answering:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"google/codegemma-1.1-7b-it","messages":[{"role":"user","content":"Write a Python hello-world function."}]}'
```

Raw text completion example (the prompt is passed directly to the model, without a chat template):

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"google/codegemma-1.1-2b","prompt":"def hello():\n    ","max_tokens":64}'
```

Function tools are returned as OpenAI-compatible `tool_calls` for the connected harness to execute; Wonderdesk does not execute tools itself. Include the tool definitions and tool-call results in subsequent completion requests to continue the conversation.

The server logs to the terminal and a rotating file (10 MiB, three backups); message contents are not logged. The file defaults to `local_isolated_llm.log` in the working directory. Set `LOCAL_LLM_LOG_FILE` to choose another path. Health and metrics endpoints:

- `GET /health/live`: process liveness
- `GET /health/ready`: readiness (returns 503 until the model is loaded)
- `GET /api/models/active/status`: model download/load phase and reported byte counts, which may remain at zero until transfer updates arrive
- `GET /metrics`: request/generation counts and durations, model state, and process memory

Interactive API docs are at <http://127.0.0.1:8080/docs>.

## Future Enhancements

- [ ] Multi-model support
- [ ] MCP server integration
