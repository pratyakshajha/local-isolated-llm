import tempfile
import json
import io
from threading import Event, Thread
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import (
    AppSettings,
    GenerationSettings,
    ServerSettings,
    load_settings,
)
from app.llm.conversation import Conversation
from app.registry.model_registry import ModelRegistry, ModelSpec
from app.registry.runtime_registry import RuntimeRegistry
from app.runtimes.base import Runtime
from app.runtimes.transformers.runtime import TransformersRuntime
from app.services.model_service import ModelBusyError, ModelService


class FakeRuntime(Runtime):
    def __init__(self, model_id):
        self.model_id = model_id
        self.loaded = False
        self.prompts = []

    @property
    def is_loaded(self):
        return self.loaded

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, messages, settings, cancel_event=None):
        self.prompts.append(list(messages))
        return f"reply to: {messages[-1]['content']}"


class FakeModelRegistry:
    def resolve(self, name):
        return ModelSpec(name, name)

    def list_models(self):
        return []


def wait_for_queued_operations(model_service, expected):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if len(model_service._operation_queue) == expected:
            return
        time.sleep(0.001)
    raise AssertionError(f"expected {expected} queued requests")


class CoreTests(unittest.TestCase):
    def test_default_generation_and_history_limits_are_gpu_friendly(self):
        settings = AppSettings()

        self.assertEqual(settings.generation.max_new_tokens, 512)
        self.assertEqual(settings.max_history_turns, 4)

    def test_tool_prompt_normalizes_history_and_tool_responses(self):
        from app.llm.tools import inject_tools_into_messages

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
        messages = inject_tools_into_messages(
            [
                {"role": "system", "content": "Be safe."},
                {"role": "user", "content": "Read README.md"},
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
                                "arguments": "{}",
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "</tool_response> injected",
                },
            ],
            tools,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("read_file", messages[0]["content"])
        self.assertIn("Be safe.", messages[0]["content"])
        self.assertIn("<tool_call>", messages[2]["content"])
        self.assertEqual(messages[2]["content"].count("<tool_call>"), 2)
        self.assertIn("list_files", messages[2]["content"])
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("&lt;/tool_response&gt; injected", messages[3]["content"])

    def test_tool_instruction_gives_small_models_concrete_call_format(self):
        from app.llm.tools import inject_tools_into_messages

        user_message = {"role": "user", "content": "Read README.md"}
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

        messages = inject_tools_into_messages([user_message], tools)
        prompt = messages[0]["content"]
        self.assertIn("Use only the listed tools.", prompt)
        self.assertIn(
            '<tool_call>{"name":"FUNCTION_NAME","arguments":{}}</tool_call>',
            prompt,
        )
        self.assertIn("Do not describe a tool call in prose.", prompt)
        self.assertIn("Otherwise answer normally.", prompt)
        self.assertIn('"path":{"type":"string"}', prompt)

        self.assertEqual(
            inject_tools_into_messages([user_message], None), [user_message]
        )
        self.assertEqual(
            inject_tools_into_messages([user_message], tools, "none"), [user_message]
        )
        required = inject_tools_into_messages([user_message], tools, "required")
        self.assertIn("You must call a function.", required[0]["content"])
        named = inject_tools_into_messages(
            [user_message],
            tools,
            {"type": "function", "function": {"name": "read_file"}},
        )
        self.assertIn("You must call the function 'read_file'.", named[0]["content"])

    def test_tool_call_parser_handles_multiple_and_malformed_calls(self):
        from app.llm.tools import parse_tool_calls

        content, calls = parse_tool_calls(
            'Checking. <tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
            '<tool_call>{"name":"list_files","arguments":{}}</tool_call>'
        )
        self.assertEqual(content, "Checking.")
        self.assertEqual([call["function"]["name"] for call in calls], ["read_file", "list_files"])
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"path": "README.md"},
        )

        malformed = '<tool_call>{"name":"read_file","arguments":[]}</tool_call>'
        self.assertEqual(parse_tool_calls(malformed), (malformed, []))

        bare = parse_tool_calls('{"name":"read_file","arguments":{"path":"a"}}')
        self.assertIsNone(bare[0])
        self.assertEqual(len(bare[1]), 1)

    def test_queue_settings_default_load_and_validation(self):
        settings = load_settings()
        self.assertEqual(settings.server.max_queued_requests, 4)
        self.assertEqual(settings.server.queue_timeout_seconds, 30.0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "settings.yaml"
            config_path.write_text(
                "server:\n  max_queued_requests: 2\n  queue_timeout_seconds: 4.5\n",
                encoding="utf-8",
            )
            configured = load_settings(config_path)
        self.assertEqual(configured.server.max_queued_requests, 2)
        self.assertEqual(configured.server.queue_timeout_seconds, 4.5)

        for count in (-1, 65):
            with self.subTest(max_queued_requests=count):
                with self.assertRaises(ValueError):
                    ServerSettings(max_queued_requests=count)
        for timeout in (-0.1, float("inf"), 600.1):
            with self.subTest(queue_timeout_seconds=timeout):
                with self.assertRaises(ValueError):
                    ServerSettings(queue_timeout_seconds=timeout)

    def test_large_agent_request_and_generation_limits_are_supported(self):
        settings = load_settings()
        self.assertEqual(settings.server.max_request_bytes, 4_194_304)
        self.assertEqual(settings.server.max_message_chars, 262_144)
        self.assertEqual(
            GenerationSettings(max_new_tokens=32_768).max_new_tokens, 32_768
        )

        custom = AppSettings(
            generation=GenerationSettings(max_new_tokens=16_384),
            server=ServerSettings(
                max_request_bytes=8_388_608,
                max_message_chars=500_000,
            ),
        )
        self.assertEqual(custom.generation.max_new_tokens, 16_384)
        self.assertEqual(custom.server.max_request_bytes, 8_388_608)
        self.assertEqual(custom.server.max_message_chars, 500_000)

        with self.assertRaises(ValueError):
            GenerationSettings(max_new_tokens=32_769)
        with self.assertRaises(ValueError):
            ServerSettings(max_request_bytes=16_777_217)
        with self.assertRaises(ValueError):
            ServerSettings(max_message_chars=1_048_577)

    def test_loads_existing_config_and_model_alias(self):
        settings = load_settings()
        self.assertEqual(settings.server.port, 8080)
        self.assertEqual(settings.generation.max_new_tokens, 512)
        self.assertEqual(settings.max_history_turns, 4)
        resolved_model = ModelRegistry().resolve(settings.model_id)
        self.assertTrue(resolved_model.model_id)
        self.assertEqual(resolved_model.runtime, "transformers")

    def test_loads_yaml_config_and_validates_generation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "settings.yaml"
            config_path.write_text(
                "model_id: Qwen/Qwen3-8B\n"
                "generation:\n  temperature: 0\n  top_p: 0.9\n  max_new_tokens: 12\n",
                encoding="utf-8",
            )
            settings = load_settings(config_path)
        self.assertEqual(settings.model_id, "Qwen/Qwen3-8B")
        self.assertEqual(settings.generation.max_new_tokens, 12)

        with self.assertRaises(FileNotFoundError):
            load_settings(config_path="missing.yaml")

    def test_conversation_bounds_history_and_keeps_system_prompt(self):
        conversation = Conversation("system", max_history_turns=2)
        conversation.add_turn("one", "answer one")
        conversation.add_turn("two", "answer two")
        conversation.add_turn("three", "answer three")

        messages = conversation.messages_for("four")
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertEqual(
            [message["content"] for message in messages],
            ["system", "two", "answer two", "three", "answer three", "four"],
        )

    def test_model_registry_rejects_malformed_model_ids(self):
        registry = ModelRegistry()
        self.assertEqual(
            registry.resolve("google/gemma-3-1b-it").model_id,
            "google/gemma-3-1b-it",
        )
        for model_id in ("bad/name/extra", "../outside", "bad space/model"):
            with self.assertRaises(ValueError):
                registry.resolve(model_id)

    def test_model_registry_preserves_capabilities_for_registered_model_ids(self):
        registry = ModelRegistry()
        for model_name in ("code_gemma", "google/codegemma-1.1-2b"):
            with self.subTest(model_name=model_name):
                self.assertFalse(registry.resolve(model_name).supports_chat)

    def test_model_registry_includes_approximate_parameter_counts(self):
        registry = ModelRegistry()
        expected = {
            "gemma_small": 1.0,
            "gemma": 4.3,
            "gemma_4": 26.0,
            "code_gemma": 2.5,
            "code_gemma_it": 7.0,
            "llama": 3.2,
            "llama_small": 1.24,
            "qwen3-8b": 8.2,
        }

        for alias, params_billion in expected.items():
            with self.subTest(alias=alias):
                spec = registry.resolve(alias)
                self.assertTrue(hasattr(spec, "params_billion"))
                if hasattr(spec, "params_billion"):
                    self.assertEqual(spec.params_billion, params_billion)
        custom_spec = registry.resolve("example/custom-model")
        self.assertTrue(hasattr(custom_spec, "params_billion"))
        if hasattr(custom_spec, "params_billion"):
            self.assertIsNone(custom_spec.params_billion)

    def test_model_catalog_parses_and_caches_hub_models(self):
        from importlib.util import find_spec
        from urllib.parse import parse_qs, urlsplit

        self.assertIsNotNone(find_spec("app.services.model_catalog"))
        from app.services import model_catalog

        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "hf_model_catalog.json").read_text(
                encoding="utf-8"
            )
        )

        def fake_urlopen(url, timeout):
            parsed = urlsplit(url)
            if parsed.path == "/api/models":
                params = parse_qs(parsed.query)
                self.assertEqual(params["pipeline_tag"], ["text-generation"])
                self.assertEqual(params["sort"], ["downloads"])
                self.assertEqual(params["direction"], ["-1"])
                self.assertEqual(params["limit"], ["10"])
                self.assertIn("safetensors", params["expand"])
                return io.BytesIO(
                    json.dumps(
                        fixture["listing"] if params["author"] == ["google"] else []
                    ).encode()
                )
            self.assertIn("blobs=true", parsed.query)
            return io.BytesIO(json.dumps(fixture["details"]).encode())

        with (
            patch.object(model_catalog, "_cached_at", 0),
            patch.object(model_catalog, "_cached_catalog", None),
            patch.object(model_catalog, "urlopen", side_effect=fake_urlopen) as open_url,
        ):
            models, source = model_catalog.get_model_catalog()
            cached_models, cached_source = model_catalog.get_model_catalog()

        self.assertEqual(source, "hub")
        self.assertEqual(cached_source, "hub")
        self.assertEqual(models, cached_models)
        self.assertEqual(open_url.call_count, 6)
        self.assertEqual(
            models,
            [
                {
                    "id": "google/gemma-3-1b-it",
                    "name": "google/gemma-3-1b-it",
                    "params_billion": 1.0,
                    "download_gb": 2.000000008,
                    "downloads": 1234,
                }
            ],
        )

    def test_model_catalog_falls_back_to_registry_when_hub_fails(self):
        from importlib.util import find_spec

        self.assertIsNotNone(find_spec("app.services.model_catalog"))
        from app.services import model_catalog

        with (
            patch.object(model_catalog, "_cached_at", 0),
            patch.object(model_catalog, "_cached_catalog", None),
            patch.object(
                model_catalog, "urlopen", side_effect=OSError("Hub unavailable")
            ),
        ):
            models, source = model_catalog.get_model_catalog()

        self.assertEqual(source, "fallback")
        self.assertEqual(len(models), len(ModelRegistry.ALIASES))
        gemma = next(model for model in models if model["id"] == "google/gemma-3-1b-it")
        self.assertEqual(gemma["params_billion"], 1.0)
        self.assertIsNone(gemma["download_gb"])

    def test_hardware_fit_estimates_memory_and_ranks_fitting_models(self):
        from importlib.util import find_spec

        self.assertIsNotNone(find_spec("app.services.hardware_fit"))
        from app.services.hardware_fit import analyze_models, estimate_bytes

        self.assertEqual(estimate_bytes(1.0, 2), 3_473_741_824)
        self.assertEqual(estimate_bytes(1.0, 4), 5_873_741_824)
        self.assertIsNone(estimate_bytes(None, 2))

        result = analyze_models(
            [
                {
                    "id": "small",
                    "name": "Small",
                    "params_billion": 1.0,
                    "download_gb": 2.0,
                    "downloads": 10,
                },
                {
                    "id": "large",
                    "name": "Large",
                    "params_billion": 2.0,
                    "download_gb": 5.0,
                    "downloads": 100,
                },
                {
                    "id": "medium",
                    "name": "Medium",
                    "params_billion": 1.5,
                    "download_gb": 3.0,
                    "downloads": 20,
                },
                {
                    "id": "medium-popular",
                    "name": "Medium Popular",
                    "params_billion": 1.5,
                    "download_gb": 3.0,
                    "downloads": 100,
                },
                {
                    "id": "unknown",
                    "name": "Unknown",
                    "params_billion": None,
                    "download_gb": None,
                    "downloads": 0,
                },
            ],
            gpu_vram_gb=6.0,
            ram_gb=16.0,
            disk_free_gb=4.0,
            gpu_name="NVIDIA GeForce RTX 4090",
        )

        models = {model["id"]: model for model in result["models"]}
        self.assertEqual(models["small"]["verdict"], "fits_gpu")
        self.assertEqual(models["small"]["need_gb"], 3.47)
        self.assertEqual(models["small"]["est_tokens_per_sec"], 504.0)
        self.assertEqual(models["large"]["verdict"], "fits_gpu")
        self.assertFalse(models["large"]["disk_ok"])
        self.assertEqual(models["medium"]["verdict"], "fits_gpu")
        self.assertEqual(models["medium-popular"]["verdict"], "fits_gpu")
        self.assertEqual(models["unknown"]["verdict"], "unknown_size")
        self.assertIsNone(models["unknown"]["disk_ok"])
        self.assertEqual(result["best_fit"], "medium-popular")

    def test_hardware_fit_accepts_exact_gpu_memory_boundary(self):
        from importlib.util import find_spec

        self.assertIsNotNone(find_spec("app.services.hardware_fit"))
        from app.services.hardware_fit import analyze_models

        model = {
            "id": "small",
            "name": "Small",
            "params_billion": 1.0,
            "download_gb": None,
            "downloads": 1,
        }
        exact_fit = analyze_models(
            [model], gpu_vram_gb=3.473741824, ram_gb=8.0,
            disk_free_gb=10.0, gpu_name=None,
        )
        one_byte_short = analyze_models(
            [model], gpu_vram_gb=3.473741823, ram_gb=8.0,
            disk_free_gb=10.0, gpu_name=None,
        )
        self.assertEqual(exact_fit["models"][0]["verdict"], "fits_gpu")
        self.assertEqual(one_byte_short["models"][0]["verdict"], "wont_fit")

    def test_hardware_fit_estimates_speed_for_rtx_5060_laptop(self):
        from app.services.hardware_fit import analyze_models

        result = analyze_models(
            [{
                "id": "google/gemma-3-3b-it",
                "name": "Gemma 3 3B",
                "params_billion": 3.0,
                "download_gb": None,
                "downloads": 1,
            }],
            gpu_vram_gb=8.55,
            ram_gb=16.47,
            disk_free_gb=100.0,
            gpu_name="NVIDIA GeForce RTX 5060 Laptop GPU",
        )

        self.assertEqual(result["models"][0]["est_tokens_per_sec"], 64.0)

    def test_hardware_fit_uses_ram_and_omits_unknown_gpu_speed(self):
        from importlib.util import find_spec

        self.assertIsNotNone(find_spec("app.services.hardware_fit"))
        from app.services.hardware_fit import analyze_models

        result = analyze_models(
            [
                {
                    "id": "small",
                    "name": "Small",
                    "params_billion": 1.0,
                    "download_gb": None,
                    "downloads": 1,
                }
            ],
            gpu_vram_gb=0.0,
            ram_gb=8.0,
            disk_free_gb=10.0,
            gpu_name="Unlisted GPU",
        )

        model = result["models"][0]
        self.assertEqual(model["verdict"], "fits_cpu")
        self.assertIsNone(model["est_tokens_per_sec"])
        self.assertEqual(result["best_fit"], "small")

    def test_hardware_fit_uses_cpu_estimate_for_mps_runtime(self):
        from importlib.util import find_spec
        from inspect import signature

        self.assertIsNotNone(find_spec("app.services.hardware_fit"))
        from app.services.hardware_fit import analyze_models
        self.assertIn("backend", signature(analyze_models).parameters)

        model = {
            "id": "small",
            "name": "Small",
            "params_billion": 1.0,
            "download_gb": None,
            "downloads": 1,
        }
        result = analyze_models(
            [model],
            gpu_vram_gb=8.0,
            ram_gb=8.0,
            disk_free_gb=10.0,
            gpu_name="Apple M3 Max",
            backend="mps",
        )

        self.assertEqual(result["models"][0]["verdict"], "fits_cpu")
        self.assertIsNone(result["models"][0]["est_tokens_per_sec"])

    def test_detect_hardware_selects_cuda_rocm_mps_xpu_and_cpu_in_order(self):
        from app.services.hardware_fit import detect_hardware

        gib = 1_000_000_000
        ram_bytes = 16 * gib

        def fake_torch(*, cuda=False, hip=None, mps=False, xpu=False):
            properties = SimpleNamespace(name="Test GPU", total_memory=8 * gib)
            return SimpleNamespace(
                version=SimpleNamespace(hip=hip),
                cuda=SimpleNamespace(
                    is_available=lambda: cuda,
                    get_device_properties=lambda index: properties,
                ),
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: mps)
                ),
                xpu=SimpleNamespace(
                    is_available=lambda: xpu,
                    get_device_properties=lambda index: properties,
                ),
            )

        cases = (
            (fake_torch(cuda=True), "cuda", 8.0, "Test GPU"),
            (fake_torch(cuda=True, hip="6.3.0"), "rocm", 8.0, "Test GPU"),
            (fake_torch(cuda=True, mps=True, xpu=True), "cuda", 8.0, "Test GPU"),
            (fake_torch(mps=True), "mps", 16.0, None),
            (fake_torch(mps=True, xpu=True), "mps", 16.0, None),
            (fake_torch(xpu=True), "xpu", 8.0, "Test GPU"),
            (fake_torch(), "cpu", 0.0, None),
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch.dict("os.environ", {"HF_HUB_CACHE": cache_dir}),
                patch.dict(
                    "sys.modules",
                    {"psutil": SimpleNamespace(
                        virtual_memory=lambda: SimpleNamespace(total=ram_bytes)
                    )},
                ),
                patch(
                    "shutil.disk_usage",
                    return_value=SimpleNamespace(free=5 * gib),
                ) as disk_usage,
            ):
                for torch_stub, backend, vram_gb, gpu_name in cases:
                    with self.subTest(backend=backend):
                        with patch.dict("sys.modules", {"torch": torch_stub}):
                            detected = detect_hardware()

                        self.assertEqual(detected["backend"], backend)
                        self.assertEqual(detected["gpu_vram_gb"], vram_gb)
                        self.assertEqual(detected["gpu_name"], gpu_name)
                        self.assertEqual(detected["ram_gb"], 16.0)
                        self.assertEqual(detected["disk_free_gb"], 5.0)
                        self.assertIsInstance(detected["notes"], list)
                disk_usage.assert_any_call(Path(cache_dir))

    def test_detect_hardware_reports_cpu_when_torch_is_unavailable(self):
        from app.services.hardware_fit import detect_hardware

        gib = 1_000_000_000
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch.dict("os.environ", {"HF_HUB_CACHE": cache_dir}),
                patch.dict(
                    "sys.modules",
                    {
                        "torch": None,
                        "psutil": SimpleNamespace(
                            virtual_memory=lambda: SimpleNamespace(total=8 * gib)
                        ),
                    },
                ),
                patch("shutil.disk_usage", return_value=SimpleNamespace(free=2 * gib)),
            ):
                detected = detect_hardware()

        self.assertEqual(detected["backend"], "cpu")
        self.assertEqual(detected["gpu_vram_gb"], 0.0)
        self.assertEqual(detected["ram_gb"], 8.0)
        self.assertTrue(any("PyTorch" in note for note in detected["notes"]))

    def test_detect_hardware_notes_distinguish_no_cuda_from_no_gpu(self):
        from app.services.hardware_fit import detect_hardware

        gib = 1_000_000_000

        def fake_torch(*, cuda_available=False, cuda_version=None):
            return SimpleNamespace(
                version=SimpleNamespace(hip=None, cuda=cuda_version),
                cuda=SimpleNamespace(
                    is_available=lambda: cuda_available,
                    get_device_properties=lambda index: SimpleNamespace(
                        name="Test GPU", total_memory=8 * gib
                    ),
                ),
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False)
                ),
                xpu=SimpleNamespace(
                    is_available=lambda: False,
                    get_device_properties=lambda index: SimpleNamespace(
                        name="Test GPU", total_memory=8 * gib
                    ),
                ),
            )

        ram_bytes = 16 * gib
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch.dict("os.environ", {"HF_HUB_CACHE": cache_dir}),
                patch.dict(
                    "sys.modules",
                    {"psutil": SimpleNamespace(
                        virtual_memory=lambda: SimpleNamespace(total=ram_bytes)
                    )},
                ),
                patch("shutil.disk_usage", return_value=SimpleNamespace(free=5 * gib)),
            ):
                with patch.dict("sys.modules", {"torch": fake_torch()}):
                    no_cuda = detect_hardware()
                with patch.dict(
                    "sys.modules",
                    {"torch": fake_torch(cuda_version="12.1", cuda_available=False)},
                ):
                    no_gpu = detect_hardware()

        self.assertEqual(no_cuda["backend"], "cpu")
        self.assertTrue(
            any("PyTorch was installed without CUDA" in note for note in no_cuda["notes"]),
            f"Expected PyTorch-no-CUDA note, got: {no_cuda['notes']}",
        )
        self.assertEqual(no_gpu["backend"], "cpu")
        self.assertTrue(
            any("no compatible GPU" in note for note in no_gpu["notes"]),
            f"Expected no-GPU note, got: {no_gpu['notes']}",
        )

    def test_model_queues_concurrent_generations_in_fifo_order(self):
        started = Event()
        release = Event()
        completed = []

        class BlockingRuntime(FakeRuntime):
            def generate(self, messages, settings, cancel_event=None):
                content = messages[-1]["content"]
                if content == "first":
                    started.set()
                    release.wait(timeout=2)
                completed.append(content)
                return f"done:{content}"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: BlockingRuntime(model.model_id))
        settings = AppSettings(
            model_id="fake-model",
            runtime="fake",
            server=ServerSettings(max_queued_requests=2, queue_timeout_seconds=2),
        )
        model_service = ModelService(
            settings,
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        results = []

        def generate(content):
            results.append(
                model_service.generate([{"role": "user", "content": content}])
            )

        first = Thread(target=generate, args=("first",))
        second = Thread(target=generate, args=("second",))
        third = Thread(target=generate, args=("third",))
        first.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            second.start()
            wait_for_queued_operations(model_service, 1)
            third.start()
            wait_for_queued_operations(model_service, 2)
        finally:
            release.set()
            first.join(timeout=2)
            if second.ident is not None:
                second.join(timeout=2)
            if third.ident is not None:
                third.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(third.is_alive())
        self.assertEqual(completed, ["first", "second", "third"])
        self.assertEqual(results, ["done:first", "done:second", "done:third"])

    def test_model_rejects_queue_overflow_and_model_switch_does_not_wait(self):
        started = Event()
        release = Event()

        class BlockingRuntime(FakeRuntime):
            def generate(self, messages, settings, cancel_event=None):
                started.set()
                release.wait(timeout=2)
                return "done"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: BlockingRuntime(model.model_id))
        settings = AppSettings(
            model_id="fake-model",
            runtime="fake",
            server=ServerSettings(max_queued_requests=1, queue_timeout_seconds=2),
        )
        model_service = ModelService(
            settings,
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        first = Thread(
            target=model_service.generate,
            args=([{"role": "user", "content": "first"}],),
        )
        first.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            second = Thread(
                target=model_service.generate,
                args=([{"role": "user", "content": "second"}],),
            )
            second.start()
            wait_for_queued_operations(model_service, 1)
            with self.assertRaises(ModelBusyError):
                model_service.generate([{"role": "user", "content": "third"}])
            with self.assertRaises(ModelBusyError):
                model_service.switch_model("other-model")
        finally:
            release.set()
            first.join(timeout=2)
            if "second" in locals():
                second.join(timeout=2)
        self.assertFalse(first.is_alive())

    def test_model_switch_rejects_active_generation_before_prefetch(self):
        started = Event()
        release = Event()
        prefetches = []

        class BlockingRuntime(FakeRuntime):
            def prefetch(self, _progress=None):
                prefetches.append(self.model_id)

            def generate(self, messages, settings, cancel_event=None):
                started.set()
                release.wait(timeout=2)
                return "done"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: BlockingRuntime(model.model_id))
        model_service = ModelService(
            AppSettings(model_id="old/model", runtime="fake"),
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        model_service.load_model()
        prefetches.clear()
        generation = Thread(
            target=model_service.generate,
            args=([{"role": "user", "content": "active"}],),
        )
        generation.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            with self.assertRaises(ModelBusyError):
                model_service.switch_model("new/model")
        finally:
            release.set()
            generation.join(timeout=2)

        self.assertFalse(generation.is_alive())
        self.assertEqual(prefetches, [])
        self.assertEqual(model_service.model_id, "old/model")
        self.assertEqual(model_service.status["state"], "ready")

    def test_model_switch_prefetches_before_unloading_and_tracks_progress(self):
        events = []
        runtimes = {}

        class TrackingRuntime(FakeRuntime):
            def prefetch(self, progress=None):
                events.append((self.model_id, "prefetch"))
                if progress:
                    progress(60, 120)

            def load(self):
                events.append((self.model_id, "load"))
                super().load()

            def unload(self):
                events.append((self.model_id, "unload"))
                super().unload()

        def create_runtime(model):
            runtime = TrackingRuntime(model.model_id)
            runtimes[model.model_id] = runtime
            return runtime

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", create_runtime)
        model_service = ModelService(
            AppSettings(model_id="old/model", runtime="fake"),
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        model_service.load_model()
        events.clear()

        self.assertEqual(model_service.switch_model("new/model"), "new/model")

        self.assertEqual(
            events,
            [
                ("new/model", "prefetch"),
                ("old/model", "unload"),
                ("new/model", "load"),
            ],
        )
        self.assertTrue(runtimes["new/model"].is_loaded)
        self.assertEqual(
            model_service.status["state"], "ready"
        )
        self.assertEqual(model_service.status["model_id"], "new/model")
        self.assertEqual(model_service.status["bytes_done"], 60)
        self.assertEqual(model_service.status["bytes_total"], 120)

    def test_model_switch_download_failure_keeps_old_model_loaded(self):
        events = []
        runtimes = {}

        class TrackingRuntime(FakeRuntime):
            def prefetch(self, _progress=None):
                events.append((self.model_id, "prefetch"))
                if self.model_id == "new/model":
                    raise RuntimeError("download failed")

            def load(self):
                events.append((self.model_id, "load"))
                super().load()

            def unload(self):
                events.append((self.model_id, "unload"))
                super().unload()

        def create_runtime(model):
            runtime = TrackingRuntime(model.model_id)
            runtimes[model.model_id] = runtime
            return runtime

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", create_runtime)
        model_service = ModelService(
            AppSettings(model_id="old/model", runtime="fake"),
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        model_service.load_model()
        events.clear()

        with self.assertRaisesRegex(RuntimeError, "download failed"):
            model_service.switch_model("new/model")

        self.assertEqual(events, [("new/model", "prefetch")])
        self.assertTrue(runtimes["old/model"].is_loaded)
        self.assertEqual(model_service.model_id, "old/model")
        self.assertEqual(model_service.status["state"], "failed")
        self.assertEqual(model_service.status["error"], "download failed")

    def test_model_switch_load_failure_rolls_back_old_model_and_reports_failure(self):
        runtimes = {}

        class TrackingRuntime(FakeRuntime):
            def load(self):
                if self.model_id == "new/model":
                    raise RuntimeError("load failed")
                super().load()

        def create_runtime(model):
            runtime = TrackingRuntime(model.model_id)
            runtimes[model.model_id] = runtime
            return runtime

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", create_runtime)
        model_service = ModelService(
            AppSettings(model_id="old/model", runtime="fake"),
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        model_service.load_model()

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            model_service.switch_model("new/model")

        self.assertTrue(runtimes["old/model"].is_loaded)
        self.assertEqual(model_service.model_id, "old/model")
        self.assertEqual(model_service.status["state"], "failed")
        self.assertEqual(model_service.status["error"], "load failed")

    def test_model_generation_queue_times_out(self):
        started = Event()
        release = Event()

        class BlockingRuntime(FakeRuntime):
            def generate(self, messages, settings, cancel_event=None):
                started.set()
                release.wait(timeout=2)
                return "done"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: BlockingRuntime(model.model_id))
        settings = AppSettings(
            model_id="fake-model",
            runtime="fake",
            server=ServerSettings(max_queued_requests=1, queue_timeout_seconds=0.02),
        )
        model_service = ModelService(
            settings,
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        first = Thread(
            target=model_service.generate,
            args=([{"role": "user", "content": "first"}],),
        )
        first.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            with self.assertRaises(ModelBusyError):
                model_service.generate([{"role": "user", "content": "queued"}])
        finally:
            release.set()
            first.join(timeout=2)
        self.assertFalse(first.is_alive())

    def test_cancelled_stream_waiting_in_queue_releases_its_ticket(self):
        started = Event()
        release = Event()
        class BlockingRuntime(FakeRuntime):
            def generate(self, messages, settings, cancel_event=None):
                if messages[-1]["content"] == "first":
                    started.set()
                    release.wait(timeout=2)
                return "done"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda model: BlockingRuntime(model.model_id))
        settings = AppSettings(
            model_id="fake-model",
            runtime="fake",
            server=ServerSettings(max_queued_requests=1, queue_timeout_seconds=2),
        )
        model_service = ModelService(
            settings,
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        first = Thread(
            target=model_service.generate,
            args=([{"role": "user", "content": "first"}],),
        )
        first.start()
        self.assertTrue(started.wait(timeout=2))

        cancel_event = Event()
        result = []
        queued_stream = Thread(
            target=lambda: result.extend(
                model_service.generate_stream(
                    [{"role": "user", "content": "cancel me"}],
                    cancel_event=cancel_event,
                )
            )
        )
        queued_stream.start()
        wait_for_queued_operations(model_service, 1)
        cancel_event.set()
        queued_stream.join(timeout=0.5)
        was_still_waiting = queued_stream.is_alive()

        release.set()
        first.join(timeout=2)
        queued_stream.join(timeout=2)

        self.assertFalse(was_still_waiting)
        self.assertFalse(queued_stream.is_alive())
        self.assertEqual(result, [])
        self.assertEqual(len(model_service._operation_queue), 0)

    def test_model_service_stream_holds_gate_until_closed_and_honors_cancellation(self):
        class StreamingRuntime(FakeRuntime):
            def generate_stream(self, messages, settings, cancel_event=None):
                yield "first"
                if cancel_event is None or not cancel_event.is_set():
                    yield "second"

        runtime_registry = RuntimeRegistry()
        runtime_registry.register(
            "fake", lambda model: StreamingRuntime(model.model_id)
        )
        settings = AppSettings(
            model_id="fake-model",
            runtime="fake",
            server=ServerSettings(max_queued_requests=1),
        )
        model_service = ModelService(
            settings,
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        cancel_event = Event()
        stream = iter(
            model_service.generate_stream(
                [{"role": "user", "content": "hello"}],
                cancel_event=cancel_event,
            )
        )

        self.assertEqual(next(stream), "first")
        result = []
        queued = Thread(
            target=lambda: result.append(
                model_service.generate([{"role": "user", "content": "queued"}])
            )
        )
        queued.start()
        wait_for_queued_operations(model_service, 1)

        cancel_event.set()
        self.assertEqual(list(stream), [])
        queued.join(timeout=2)
        self.assertFalse(queued.is_alive())
        self.assertEqual(result, ["reply to: queued"])
        self.assertEqual(
            model_service.generate([{"role": "user", "content": "next"}]),
            "reply to: next",
        )

    def test_runtime_default_stream_yields_single_generate_response(self):
        runtime = FakeRuntime("fake-model")

        self.assertEqual(
            list(
                runtime.generate_stream(
                    [{"role": "user", "content": "hello"}],
                    AppSettings().generation,
                )
            ),
            ["reply to: hello"],
        )

    def test_default_runtime_registry_does_not_load_model_on_import(self):
        registry = RuntimeRegistry()
        runtime = registry.create(
            "transformers", ModelSpec("example/model", "example/model")
        )
        self.assertFalse(runtime.is_loaded)

    def test_transformers_runtime_prefetch_reports_bytes_and_avoids_duplicate_weights(self):
        import huggingface_hub

        downloaded = {}
        progress_updates = []

        class FakeApi:
            def model_info(self, model_id):
                self.model_id = model_id
                return SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename=name)
                        for name in (
                            "config.json",
                            "tokenizer.json",
                            "model.safetensors",
                            "pytorch_model.bin",
                        )
                    ]
                )

        def snapshot_download(model_id, **kwargs):
            downloaded.update(model_id=model_id, kwargs=kwargs)
            progress = kwargs["tqdm_class"](
                total=100,
                initial=0,
                unit="B",
                desc="Downloading bytes",
                file=io.StringIO(),
            )
            progress.update(25)
            progress.close()

        with (
            patch.object(huggingface_hub, "HfApi", FakeApi),
            patch.object(huggingface_hub, "snapshot_download", snapshot_download),
            patch(
                "app.services.model_cache.get_downloaded_model_ids",
                return_value=set(),
            ),
        ):
            TransformersRuntime("example/model").prefetch(
                lambda done, total: progress_updates.append((done, total))
            )

        self.assertEqual(downloaded["model_id"], "example/model")
        self.assertIn("model.safetensors", downloaded["kwargs"]["allow_patterns"])
        self.assertNotIn("pytorch_model.bin", downloaded["kwargs"]["allow_patterns"])
        self.assertIn("tokenizer.json", downloaded["kwargs"]["allow_patterns"])
        self.assertEqual(progress_updates[-1], (25, 100))

    def test_transformers_runtime_checks_current_hub_revision_for_cached_model(self):
        import huggingface_hub

        downloads = []

        class FakeApi:
            def model_info(self, model_id):
                self.model_id = model_id
                return SimpleNamespace(
                    sha="current-revision",
                    siblings=[
                        SimpleNamespace(rfilename=name)
                        for name in (
                            "config.json",
                            "tokenizer.json",
                            "model.safetensors",
                        )
                    ],
                )

        def snapshot_download(model_id, **kwargs):
            downloads.append((model_id, kwargs))

        with (
            patch.object(huggingface_hub, "HfApi", FakeApi),
            patch.object(huggingface_hub, "snapshot_download", snapshot_download),
            patch(
                "app.services.model_cache.get_downloaded_model_ids",
                return_value={"example/model"},
            ),
        ):
            TransformersRuntime("example/model").prefetch()

        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0][0], "example/model")

    def test_transformers_runtime_uses_complete_cache_when_hub_is_offline(self):
        import huggingface_hub
        from huggingface_hub.utils import OfflineModeIsEnabled

        api_calls = []

        class FakeApi:
            def model_info(self, _model_id):
                api_calls.append(_model_id)
                raise OfflineModeIsEnabled("offline")

        with (
            patch.object(huggingface_hub, "HfApi", FakeApi),
            patch(
                "app.services.model_cache.get_downloaded_model_ids",
                return_value={"example/model"},
            ),
        ):
            TransformersRuntime("example/model").prefetch()

        self.assertEqual(api_calls, ["example/model"])

    def test_transformers_runtime_prefetch_supports_bin_only_repositories(self):
        import huggingface_hub

        downloaded = {}

        class FakeApi:
            def model_info(self, _model_id):
                return SimpleNamespace(
                    siblings=[SimpleNamespace(rfilename="pytorch_model.bin")]
                )

        def snapshot_download(_model_id, **kwargs):
            downloaded.update(kwargs)

        with (
            patch.object(huggingface_hub, "HfApi", FakeApi),
            patch.object(huggingface_hub, "snapshot_download", snapshot_download),
            patch(
                "app.services.model_cache.get_downloaded_model_ids",
                return_value=set(),
            ),
        ):
            TransformersRuntime("example/model").prefetch()

        self.assertIn("pytorch_model.bin", downloaded["allow_patterns"])
        self.assertNotIn("model.safetensors", downloaded["allow_patterns"])

    def test_transformers_runtime_prefetch_supports_sharded_model_bin_weights(self):
        import huggingface_hub

        downloaded = {}

        class FakeApi:
            def model_info(self, _model_id):
                return SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename=name)
                        for name in (
                            "model-00001-of-00002.bin",
                            "model-00002-of-00002.bin",
                            "model.bin.index.json",
                        )
                    ]
                )

        def snapshot_download(_model_id, **kwargs):
            downloaded.update(kwargs)

        with (
            patch.object(huggingface_hub, "HfApi", FakeApi),
            patch.object(huggingface_hub, "snapshot_download", snapshot_download),
            patch(
                "app.services.model_cache.get_downloaded_model_ids",
                return_value=set(),
            ),
        ):
            TransformersRuntime("example/model").prefetch()

        self.assertIn("model-00001-of-00002.bin", downloaded["allow_patterns"])
        self.assertIn("model-00002-of-00002.bin", downloaded["allow_patterns"])
        self.assertIn("model.bin.index.json", downloaded["allow_patterns"])

    def test_transformers_runtime_quantizes_and_places_cuda_model_during_load(self):
        device = SimpleNamespace(type="cuda")
        model = SimpleNamespace(eval=lambda: None)
        loaded = {}

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def from_pretrained(model_id, **kwargs):
            loaded.update(model_id=model_id, kwargs=kwargs)
            return model

        fake_torch = SimpleNamespace(
            device=lambda _name: device,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                is_bf16_supported=lambda: False,
                current_device=lambda: 0,
            ),
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
        )
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model_id: object()),
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        )

        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            runtime = TransformersRuntime("example/model")
            runtime.load()

        self.assertEqual(loaded["model_id"], "example/model")
        self.assertEqual(loaded["kwargs"]["device_map"], {"": 0})
        quantization = loaded["kwargs"]["quantization_config"]
        self.assertEqual(quantization.kwargs["load_in_4bit"], True)
        self.assertEqual(quantization.kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(quantization.kwargs["bnb_4bit_compute_dtype"], "float16")
        self.assertIs(runtime._model, model)

    def test_transformers_runtime_streams_text_iterator_chunks(self):
        class FakeTensor:
            shape = (1, 3)

            def to(self, _device):
                return self

        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

        class FakeStreamer:
            def __init__(self, tokenizer, skip_prompt, skip_special_tokens):
                self.tokenizer = tokenizer
                self.skip_prompt = skip_prompt
                self.skip_special_tokens = skip_special_tokens

            def __iter__(self):
                return iter(["  generated", " answer  "])

        class FakeModel:
            def generate(self, **kwargs):
                self.last_kwargs = kwargs

        class FakeStoppingCriteria:
            pass

        tokenizer = FakeTokenizer()
        model = FakeModel()
        runtime = TransformersRuntime("example/model")
        runtime._tokenizer = tokenizer
        runtime._model = model
        runtime._device = "cpu"
        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        fake_transformers = SimpleNamespace(
            TextIteratorStreamer=FakeStreamer,
            StoppingCriteria=FakeStoppingCriteria,
            StoppingCriteriaList=list,
        )

        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            chunks = list(
                runtime.generate_stream(
                    [{"role": "user", "content": "hello"}],
                    AppSettings().generation,
                )
            )
            response = runtime.generate(
                [{"role": "user", "content": "hello"}],
                AppSettings().generation,
            )

        self.assertEqual(chunks, ["  generated", " answer  "])
        self.assertEqual(response, "generated answer")
        streamer = model.last_kwargs["streamer"]
        self.assertIsInstance(streamer, FakeStreamer)
        self.assertIs(streamer.tokenizer, tokenizer)
        self.assertTrue(streamer.skip_prompt)
        self.assertTrue(streamer.skip_special_tokens)

    def test_transformers_runtime_generates_raw_completion_without_chat_template(self):
        class FakeTensor:
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __init__(self):
                self.prompts = []

            def __call__(self, prompt, return_tensors):
                self.prompts.append((prompt, return_tensors))
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

        class FakeStreamer:
            def __init__(self, _tokenizer, **_kwargs):
                pass

            def __iter__(self):
                return iter(["  generated", " answer  "])

        class FakeModel:
            def generate(self, **kwargs):
                self.last_kwargs = kwargs

        class FakeStoppingCriteria:
            pass

        prompt = "  def answer():\n"
        tokenizer = FakeTokenizer()
        model = FakeModel()
        runtime = TransformersRuntime("example/model")
        runtime._tokenizer = tokenizer
        runtime._model = model
        runtime._device = "cpu"
        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        fake_transformers = SimpleNamespace(
            TextIteratorStreamer=FakeStreamer,
            StoppingCriteria=FakeStoppingCriteria,
            StoppingCriteriaList=list,
        )

        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            chunks = list(
                runtime.generate_completion_stream(prompt, AppSettings().generation)
            )
            response = runtime.generate_completion(prompt, AppSettings().generation)

        self.assertEqual(chunks, ["  generated", " answer  "])
        self.assertEqual(response, "  generated answer  ")
        self.assertEqual(tokenizer.prompts, [(prompt, "pt"), (prompt, "pt")])
        self.assertIn("input_ids", model.last_kwargs)

    def test_model_service_passes_raw_completion_to_runtime(self):
        class CompletionRuntime(FakeRuntime):
            def __init__(self, model_id):
                super().__init__(model_id)
                self.completion_calls = []

            def generate_completion(self, prompt, settings, cancel_event=None):
                self.completion_calls.append((prompt, settings, cancel_event))
                return "  generated  "

            def generate_completion_stream(self, prompt, settings, cancel_event=None):
                self.completion_calls.append((prompt, settings, cancel_event))
                yield "  generated"
                yield "  "

        runtime = CompletionRuntime("fake-model")
        runtime_registry = RuntimeRegistry()
        runtime_registry.register("fake", lambda _model: runtime)
        model_service = ModelService(
            AppSettings(model_id="fake-model", runtime="fake"),
            model_registry=FakeModelRegistry(),
            runtime_registry=runtime_registry,
        )
        prompt = "  code prompt\n"
        generation = GenerationSettings(max_new_tokens=17)
        cancel_event = Event()

        self.assertEqual(
            model_service.generate_completion(prompt, generation, cancel_event),
            "  generated  ",
        )
        self.assertEqual(
            list(
                model_service.generate_completion_stream(
                    prompt, generation, cancel_event
                )
            ),
            ["  generated", "  "],
        )
        self.assertEqual(
            runtime.completion_calls,
            [
                (prompt, generation, cancel_event),
                (prompt, generation, cancel_event),
            ],
        )


if __name__ == "__main__":
    unittest.main()
