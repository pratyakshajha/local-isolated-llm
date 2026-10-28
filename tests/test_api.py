import unittest
import re
import json
from threading import Event, Thread
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.server import create_app, get_public_base_url
from app.api.routes.ui import INDEX_FILE
from app.core.config import AppSettings, ServerSettings
from app.registry.model_registry import ModelSpec
from app.registry.runtime_registry import RuntimeRegistry
from app.runtimes.base import Runtime
from app.services.model_service import ModelService


class FakeRuntime(Runtime):
    def __init__(self, model_id):
        self.model_id = model_id
        self.loaded = False
        self.prompts = []
        self.generations = []

    @property
    def is_loaded(self):
        return self.loaded

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, messages, settings, cancel_event=None):
        self.prompts.append(list(messages))
        self.generations.append(settings)
        return f"fake response to {messages[-1]['content']}"


def make_test_app(system_prompt="You are a helpful assistant."):
    runtime_registry = RuntimeRegistry()
    runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
    settings = AppSettings(
        model_id="test/model",
        runtime="fake",
        server=ServerSettings(allowed_hosts=("testserver",)),
        system_prompt=system_prompt,
    )
    model_service = ModelService(settings, runtime_registry=runtime_registry)
    return create_app(settings, model_service)


class ApiTests(unittest.TestCase):
    def test_model_status_reports_live_download_progress_during_switch(self):
        download_started = Event()
        finish_download = Event()

        class ProgressRuntime(FakeRuntime):
            def prefetch(self, progress=None):
                if self.model_id == "other/model":
                    if progress:
                        progress(12, 50)
                    download_started.set()
                    finish_download.wait(timeout=3)

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: ProgressRuntime(model.model_id))
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",)),
        )
        app = create_app(settings, ModelService(settings, runtime_registry=runtime_registry))
        responses = []

        with TestClient(app) as client:
            switch = Thread(
                target=lambda: responses.append(
                    client.post(
                        "/v1/models/active", json={"model_id": "other/model"}
                    )
                )
            )
            switch.start()
            self.assertTrue(download_started.wait(timeout=2))
            status = client.get("/api/models/active/status")
            finish_download.set()
            switch.join(timeout=3)

        self.assertFalse(switch.is_alive())
        self.assertEqual(responses[0].status_code, 200)
        self.assertEqual(
            {
                key: status.json()[key]
                for key in ("state", "model_id", "bytes_done", "bytes_total", "error")
            },
            {
                "state": "downloading",
                "model_id": "other/model",
                "bytes_done": 12,
                "bytes_total": 50,
                "error": None,
            },
        )
        self.assertIsInstance(status.json()["updated_at"], (float, int))

    def test_model_fit_reports_detected_hardware_and_model_estimates(self):
        from app.api.routes import models as models_route

        detected = {
            "backend": "cuda",
            "gpu_name": "NVIDIA RTX 4090",
            "gpu_vram_gb": 8.0,
            "ram_gb": 32.0,
            "disk_free_gb": 20.0,
            "notes": [],
        }
        catalog = [
            {
                "id": "google/gemma-3-1b-it",
                "name": "Gemma 3 1B",
                "params_billion": 1.0,
                "download_gb": 2.0,
                "downloads": 100,
            }
        ]

        with (
            patch.object(models_route, "detect_hardware", return_value=detected, create=True),
            patch.object(models_route, "get_model_catalog", return_value=(catalog, "hub"), create=True),
            patch.object(
                models_route,
                "get_downloaded_model_ids",
                return_value={"google/gemma-3-1b-it"},
                create=True,
            ),
            TestClient(make_test_app()) as client,
        ):
            response = client.get("/api/model-fit")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        body = response.json()
        self.assertEqual(body["detected"], detected)
        self.assertEqual(body["used"], {"gpu_vram_gb": 8.0, "ram_gb": 32.0})
        self.assertEqual(body["catalog_source"], "hub")
        self.assertEqual(body["best_fit"], "google/gemma-3-1b-it")
        self.assertEqual(
            body["models"],
            [
                {
                    "id": "google/gemma-3-1b-it",
                    "name": "Gemma 3 1B",
                    "params_billion": 1.0,
                    "download_gb": 2.0,
                    "need_gb": 3.47,
                    "verdict": "fits_gpu",
                    "disk_ok": True,
                    "est_tokens_per_sec": 504.0,
                    "downloaded": True,
                }
            ],
        )

    def test_cached_models_lists_ids_and_resolves_registry_metadata(self):
        from app.api.routes import models as models_route

        cached_ids = {
            "test/model",
            "google/codegemma-1.1-2b",
            "other/uncatalogued-model",
        }
        with (
            patch.object(
                models_route, "get_downloaded_model_ids", return_value=cached_ids
            ),
            patch.object(
                models_route,
                "get_model_catalog",
                side_effect=AssertionError("cache listing must not use discovery"),
            ),
            TestClient(make_test_app()) as client,
        ):
            response = client.get("/api/models/cached")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        body = response.json()
        self.assertEqual(body["active_model"], "test/model")
        models = {model["id"]: model for model in body["models"]}
        self.assertEqual(set(models), cached_ids)
        self.assertTrue(models["test/model"]["active"])
        self.assertFalse(models["google/codegemma-1.1-2b"]["supports_chat"])
        self.assertEqual(
            models["other/uncatalogued-model"]["name"],
            "other/uncatalogued-model",
        )

    def test_model_fit_manual_hardware_overrides_detected_values(self):
        from app.api.routes import models as models_route

        detected = {
            "backend": "mps",
            "gpu_name": "Apple M3 Max",
            "gpu_vram_gb": 16.0,
            "ram_gb": 16.0,
            "disk_free_gb": 20.0,
            "notes": ["MPS uses unified memory"],
        }
        catalog = [
            {
                "id": "google/gemma-3-1b-it",
                "name": "Gemma 3 1B",
                "params_billion": 1.0,
                "download_gb": 2.0,
                "downloads": 100,
            }
        ]

        with (
            patch.object(models_route, "detect_hardware", return_value=detected, create=True),
            patch.object(models_route, "get_model_catalog", return_value=(catalog, "fallback"), create=True),
            TestClient(make_test_app()) as client,
        ):
            response = client.get("/api/model-fit?gpu_vram_gb=4&ram_gb=8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        body = response.json()
        self.assertEqual(body["detected"], detected)
        self.assertEqual(body["used"], {"gpu_vram_gb": 4.0, "ram_gb": 8.0})
        self.assertEqual(body["catalog_source"], "fallback")
        self.assertEqual(body["models"][0]["verdict"], "fits_gpu")

    def test_model_fit_rejects_negative_or_non_finite_overrides(self):
        with TestClient(make_test_app()) as client:
            for parameter in ("gpu_vram_gb", "ram_gb"):
                for value in ("-1", "nan", "inf"):
                    with self.subTest(parameter=parameter, value=value):
                        response = client.get(
                            f"/api/model-fit?{parameter}={value}"
                        )
                        self.assertEqual(response.status_code, 422)

    def test_openai_coalesces_messages_for_gemma_chat_template(self):
        application = make_test_app()
        with TestClient(application) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Base rules."},
                        {"role": "developer", "content": "Be concise."},
                        {"role": "system", "content": "Extra context."},
                        {"role": "user", "content": "First question."},
                        {"role": "user", "content": "More context."},
                        {"role": "assistant", "content": "Thinking."},
                        {"role": "assistant", "content": "Still thinking."},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": '{"path":"."}',
                                    },
                                },
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "README contents",
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_2",
                            "content": "README.md",
                        },
                        {"role": "assistant", "content": "Done."},
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        prompt = application.state.model_service._runtime.prompts[-1]
        self.assertEqual(
            [message["role"] for message in prompt],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(
            prompt[0]["content"],
            "You are a helpful assistant.\n\nBase rules.\n\nBe concise.\n\nExtra context.",
        )
        self.assertEqual(prompt[1]["content"], "First question.\n\nMore context.")
        self.assertEqual(
            prompt[2]["content"],
            'Thinking.\n\nStill thinking.\n\n<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call><tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>',
        )
        self.assertEqual(
            prompt[3]["content"],
            '<tool_response id="call_1">README contents</tool_response>\n\n'
            '<tool_response id="call_2">README.md</tool_response>',
        )

    def test_chat_completion_uses_configured_system_prompt(self):
        application = make_test_app(system_prompt="Keep it brief.")
        with TestClient(application) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            application.state.model_service._runtime.prompts[-1],
            [
                {"role": "system", "content": "Keep it brief."},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_chat_completion_preserves_caller_system_messages(self):
        application = make_test_app(system_prompt="Keep it brief.")
        with TestClient(application) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Caller rules."},
                        {"role": "developer", "content": "Use bullets."},
                        {"role": "user", "content": "hello"},
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            application.state.model_service._runtime.prompts[-1],
            [
                {
                    "role": "system",
                    "content": "Keep it brief.\n\nCaller rules.\n\nUse bullets.",
                },
                {"role": "user", "content": "hello"},
            ],
        )

    def test_streaming_chat_uses_configured_system_prompt(self):
        application = make_test_app(system_prompt="Keep it brief.")
        with TestClient(application) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                list(response.iter_lines())

        self.assertEqual(
            application.state.model_service._runtime.prompts[-1],
            [
                {"role": "system", "content": "Keep it brief."},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_empty_configured_system_prompt_is_omitted(self):
        application = make_test_app(system_prompt="")
        with TestClient(application) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            application.state.model_service._runtime.prompts[-1],
            [{"role": "user", "content": "hello"}],
        )

    def test_configured_system_prompt_counts_toward_message_limit(self):
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(
                allowed_hosts=("testserver",), max_message_chars=32
            ),
            system_prompt="S" * 17,
        )
        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
        model_service = ModelService(settings, runtime_registry=runtime_registry)
        with TestClient(create_app(settings, model_service)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "U" * 16}]},
            )

        self.assertEqual(response.status_code, 413)

    def test_coalesced_messages_still_obey_message_size_limit(self):
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",), max_message_chars=32),
        )
        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
        model_service = ModelService(settings, runtime_registry=runtime_registry)
        with TestClient(create_app(settings, model_service)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "x" * 16},
                        {"role": "user", "content": "y" * 16},
                    ]
                },
            )

        self.assertEqual(response.status_code, 413)

    def test_openai_chat_completions_return_tool_calls_sync_and_stream(self):
        class ToolCallingRuntime(FakeRuntime):
            def generate(self, messages, settings, cancel_event=None):
                self.prompts.append(list(messages))
                content = messages[-1]["content"]
                if content == "unknown":
                    return (
                        '<tool_call>{"name":"not_requested","arguments":{}}</tool_call>'
                    )
                return 'Checking. <tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'

            def generate_stream(self, messages, settings, cancel_event=None):
                self.prompts.append(list(messages))
                yield 'Checking. <tool_call>{"name":"read_file",'
                yield '"arguments":{"path":"README.md"}}</tool_call>'

        runtime_registry = RuntimeRegistry()
        runtime_registry.register(
            "fake", lambda model: ToolCallingRuntime(model.model_id)
        )
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",)),
        )
        application = create_app(
            settings, ModelService(settings, runtime_registry=runtime_registry)
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        with TestClient(application) as client:
            sync = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "read README"}],
                    "tools": tools,
                },
            )
            self.assertEqual(sync.status_code, 200)
            choice = sync.json()["choices"][0]
            self.assertEqual(choice["finish_reason"], "tool_calls")
            self.assertEqual(choice["message"]["content"], "Checking.")
            call = choice["message"]["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "read_file")
            self.assertEqual(
                json.loads(call["function"]["arguments"]),
                {"path": "README.md"},
            )
            self.assertIn(
                "read_file",
                application.state.model_service._runtime.prompts[-1][0]["content"],
            )

            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "read README"}],
                    "tools": tools,
                    "stream": True,
                },
            ) as streamed:
                lines = [
                    line for line in streamed.iter_lines() if line.startswith("data: ")
                ]
            chunks = [json.loads(line[6:]) for line in lines[:-1]]
            tool_chunks = [
                chunk["choices"][0]["delta"]["tool_calls"]
                for chunk in chunks
                if "tool_calls" in chunk["choices"][0]["delta"]
            ]
            self.assertEqual(len(tool_chunks), 1)
            content_chunks = [
                chunk["choices"][0]["delta"]["content"]
                for chunk in chunks
                if "content" in chunk["choices"][0]["delta"]
            ]
            self.assertEqual(content_chunks, ["Checking."])
            self.assertEqual(tool_chunks[0][0]["function"]["name"], "read_file")
            self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual(lines[-1], "data: [DONE]")

            unknown = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "unknown"}],
                    "tools": tools,
                },
            )
            self.assertEqual(unknown.status_code, 200)
            unknown_choice = unknown.json()["choices"][0]
            self.assertEqual(unknown_choice["finish_reason"], "stop")
            self.assertNotIn("tool_calls", unknown_choice["message"])

    def test_openai_rejects_malformed_tool_history_arguments(self):
        with TestClient(make_test_app()) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "[]",
                                    },
                                }
                            ],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 422)

    def test_openai_v1_models_and_chat_completion(self):
        application = make_test_app()
        with TestClient(application) as client:
            models = client.get("/v1/models")
            self.assertEqual(models.status_code, 200)
            model_list = models.json()
            self.assertEqual(model_list["object"], "list")
            self.assertIn("test/model", [model["id"] for model in model_list["data"]])
            discovered = {model["id"]: model for model in model_list["data"]}
            self.assertFalse(discovered["google/codegemma-1.1-2b"]["supports_chat"])
            self.assertTrue(discovered["google/codegemma-1.1-7b-it"]["supports_chat"])

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test/model",
                    "messages": [
                        {"role": "developer", "content": "Be brief."},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "ping"}],
                        },
                    ],
                    "temperature": 0.3,
                    "top_p": 0.7,
                    "max_completion_tokens": 17,
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["object"], "chat.completion")
            self.assertEqual(body["model"], "test/model")
            self.assertEqual(
                body["choices"][0]["message"],
                {"role": "assistant", "content": "fake response to ping"},
            )
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
            runtime = application.state.model_service._runtime
            self.assertEqual(
                runtime.prompts[-1],
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.\n\nBe brief.",
                    },
                    {"role": "user", "content": "ping"},
                ],
            )
            self.assertEqual(runtime.generations[-1].temperature, 0.3)
            self.assertEqual(runtime.generations[-1].top_p, 0.7)
            self.assertEqual(runtime.generations[-1].max_new_tokens, 17)

            switched = client.post(
                "/v1/models/active", json={"model_id": "other/model"}
            )
            self.assertEqual(switched.status_code, 200)
            self.assertEqual(switched.json()["active_model"], "other/model")
            missing = client.post(
                "/v1/chat/completions",
                json={
                    "model": "not a model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(missing.status_code, 404)

    def test_openai_chat_completion_streams_sse(self):
        with TestClient(make_test_app()) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "stream me"}],
                    "stream": True,
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                lines = [
                    line for line in response.iter_lines() if line.startswith("data: ")
                ]

        self.assertEqual(lines[-1], "data: [DONE]")
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        self.assertEqual(
            "".join(
                chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
            ),
            "fake response to stream me",
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_openai_completions_return_text_sync_and_stream(self):
        class CompletionRuntime(FakeRuntime):
            def generate_completion(self, prompt, settings, cancel_event=None):
                self.prompts.append(prompt)
                self.generations.append(settings)
                return "  completed\n"

            def generate_completion_stream(self, prompt, settings, cancel_event=None):
                self.prompts.append(prompt)
                self.generations.append(settings)
                yield "  streamed"
                yield "\n"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register(
            "fake", lambda model: CompletionRuntime(model.model_id)
        )
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",)),
        )
        application = create_app(
            settings, ModelService(settings, runtime_registry=runtime_registry)
        )

        with TestClient(application) as client:
            sync = client.post(
                "/v1/completions",
                json={
                    "model": "test/model",
                    "prompt": "  def answer():\n",
                    "max_tokens": 23,
                },
            )
            streamed = client.post(
                "/v1/completions",
                json={
                    "model": "test/model",
                    "prompt": "raw stream prompt",
                    "stream": True,
                },
            )

        self.assertEqual(sync.status_code, 200)
        sync_body = sync.json()
        self.assertEqual(sync_body["object"], "text_completion")
        self.assertEqual(sync_body["choices"][0]["text"], "  completed\n")
        self.assertEqual(sync_body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(streamed.status_code, 200)
        self.assertIn("text/event-stream", streamed.headers["content-type"])
        lines = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        self.assertEqual(
            "".join(chunk["choices"][0]["text"] for chunk in chunks),
            "  streamed\n",
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(lines[-1], "data: [DONE]")
        runtime = application.state.model_service._runtime
        self.assertEqual(runtime.prompts, ["  def answer():\n", "raw stream prompt"])
        self.assertEqual(runtime.generations[0].max_new_tokens, 23)

    def test_openai_chat_rejects_any_model_without_chat_support_before_streaming(self):
        class CompletionOnlyRegistry:
            def resolve(self, model_name):
                if model_name == "test/completion-only":
                    return ModelSpec(model_name, "Completion only", supports_chat=False)
                return ModelSpec(model_name, model_name)

            def list_models(self):
                return [self.resolve("test/completion-only")]

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
        settings = AppSettings(
            model_id="test/chat-model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",)),
        )
        model_service = ModelService(
            settings,
            model_registry=CompletionOnlyRegistry(),
            runtime_registry=runtime_registry,
        )
        with TestClient(create_app(settings, model_service)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test/completion-only",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support chat", response.json()["detail"])
        self.assertIn("/v1/completions", response.json()["detail"])
        self.assertNotIn("text/event-stream", response.headers.get("content-type", ""))
        self.assertEqual(model_service._runtime.prompts, [])
        self.assertEqual(model_service.model_id, "test/chat-model")

    def test_openai_chat_completion_hides_stop_sequence_split_across_chunks(self):
        class StopRuntime(FakeRuntime):
            def generate_stream(self, messages, settings, cancel_event=None):
                yield "answer before ST"
                yield "OP hidden suffix"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: StopRuntime(model.model_id))
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",)),
        )
        model_service = ModelService(settings, runtime_registry=runtime_registry)
        with TestClient(create_app(settings, model_service)) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "stop test"}],
                    "stop": "STOP",
                    "stream": True,
                },
            ) as response:
                lines = [
                    line for line in response.iter_lines() if line.startswith("data: ")
                ]

        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )
        self.assertEqual(content, "answer before ")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(lines[-1], "data: [DONE]")

    def test_openai_chat_completion_enforces_message_limit(self):
        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(allowed_hosts=("testserver",), max_message_chars=32),
        )
        model_service = ModelService(settings, runtime_registry=runtime_registry)
        with TestClient(create_app(settings, model_service)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "x" * 33}],
                },
            )
        self.assertEqual(response.status_code, 413)

    def test_openai_chat_accepts_message_larger_than_previous_limit(self):
        message = "x" * 65_537
        with TestClient(make_test_app()) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": message}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            f"fake response to {message}",
        )

    def test_startup_prints_ready_ui_link_for_wildcard_bind(self):
        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(host="0.0.0.0", allowed_hosts=("testserver",)),
        )
        model_service = ModelService(settings, runtime_registry=runtime_registry)
        application = create_app(settings, model_service)

        self.assertEqual(get_public_base_url(settings), "http://127.0.0.1:8080")
        self.assertEqual(
            get_public_base_url(settings, "https://wonderdesk.example/"),
            "https://wonderdesk.example",
        )
        with self.assertLogs("app.api.server", level="INFO") as captured:
            with TestClient(application):
                pass

        output = "\n".join(captured.output)
        self.assertIn("Wonderdesk is ready", output)
        self.assertIn("Web UI: http://127.0.0.1:8080/", output)
        self.assertIn("API documentation: http://127.0.0.1:8080/docs", output)

    def test_health_models_and_rest_chat(self):
        application = make_test_app()
        with TestClient(application) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["model_loaded"])

            models = client.get("/v1/models").json()
            self.assertEqual(models["active_model"], "test/model")
            self.assertTrue(models["loaded"])
            self.assertIn("test/model", [item["id"] for item in models["data"]])

            first = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(
                first.json()["choices"][0]["message"]["content"],
                "fake response to hello",
            )
            metrics = client.get("/metrics").text
            self.assertIn('local_llm_generations_total{outcome="success"} 1', metrics)

            selected = client.post(
                "/v1/models/active", json={"model_id": "other/model"}
            )
            self.assertEqual(selected.status_code, 200)
            self.assertEqual(selected.json()["active_model"], "other/model")

        self.assertFalse(application.state.model_service.is_loaded)

    def test_legacy_chat_and_session_api_are_removed(self):
        application = make_test_app()
        self.assertFalse(hasattr(application.state, "chat_service"))
        self.assertNotIn("/chat", application.openapi()["paths"])
        self.assertNotIn("/ws/chat", application.openapi()["paths"])

        with TestClient(application) as client:
            self.assertEqual(
                client.post("/chat", json={"message": "legacy"}).status_code,
                405,
            )
            metrics = client.get("/metrics").text
            self.assertNotIn("local_llm_active_sessions", metrics)

    def test_web_ui_serves_built_sse_client(self):
        with TestClient(make_test_app()) as client:
            page = client.get("/")
            if INDEX_FILE.is_file():
                self.assertEqual(page.status_code, 200)
                self.assertIn('<div id="root"></div>', page.text)
                script_path = re.search(r'src="([^"]+\.js)"', page.text)
                self.assertIsNotNone(script_path)
                script = client.get(script_path.group(1))
                self.assertEqual(script.status_code, 200)
                self.assertIn("/v1/chat/completions", script.text)
                self.assertEqual(client.get("/app/chat").status_code, 200)
                self.assertEqual(client.get("/assets/missing.js").status_code, 404)
            else:
                self.assertEqual(page.status_code, 503)

    def test_security_limits_health_and_metrics(self):
        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: FakeRuntime(model.model_id))
        settings = AppSettings(
            model_id="test/model",
            runtime="fake",
            server=ServerSettings(
                allowed_hosts=("testserver",),
                max_request_bytes=128,
                max_message_chars=32,
            ),
        )
        model_service = ModelService(settings, runtime_registry=runtime_registry)
        with TestClient(create_app(settings, model_service)) as client:
            live = client.get("/health/live")
            self.assertEqual(live.status_code, 200)
            self.assertEqual(live.json(), {"status": "ok"})
            self.assertEqual(client.get("/health/ready").status_code, 200)
            self.assertEqual(live.headers["x-content-type-options"], "nosniff")
            self.assertIn("default-src 'self'", live.headers["content-security-policy"])
            metrics = client.get("/metrics")
            self.assertIn("local_llm_model_loaded 1", metrics.text)
            self.assertEqual(metrics.headers["cache-control"], "no-store")
            self.assertEqual(
                client.get(
                    "/health", headers={"Host": "untrusted.example"}
                ).status_code,
                400,
            )

            cross_origin = client.post(
                "/v1/models/active",
                json={"model_id": "test/model"},
                headers={"Origin": "https://attacker.example"},
            )
            self.assertEqual(cross_origin.status_code, 403)

            oversized = client.post(
                "/v1/chat/completions",
                content=b"x" * 200,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(oversized.status_code, 413)

            oversized_message = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "x" * 33}]},
            )
            self.assertEqual(oversized_message.status_code, 413)

            invalid_model_id = client.post(
                "/v1/models/active", json={"model_id": "bad/model/../../path"}
            )
            self.assertEqual(invalid_model_id.status_code, 404)


if __name__ == "__main__":
    unittest.main()
