import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console

import app.main as main_module
from app.cli.terminal import TerminalUI
from app.main import run_chat


class FakeModelService:
    model_name = "Test model"

    def __init__(self):
        self.loaded = False
        self.unloaded = False
        self.messages = []

    def load_model(self):
        self.loaded = True

    def unload_model(self):
        self.unloaded = True

    def generate_stream(self, messages, generation=None, cancel_event=None):
        self.messages.append(list(messages))
        message = messages[-1]["content"]
        if message == "bad":
            raise RuntimeError("fake generation error")
        yield f"reply to {message}"


class TerminalTests(unittest.TestCase):
    def test_rich_ui_renders_separate_banner_and_markdown_response(self):
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=60, record=True)
        ui = TerminalUI(console=console, input_fn=lambda _prompt: "question")

        ui.show_banner("Test model")
        self.assertEqual(ui.prompt(), "question")
        ui.show_response("Test model", "**formatted** assistant reply")

        rendered = console.export_text(styles=False)
        self.assertIn("Local Isolated LLM", rendered)
        self.assertIn("\nYou\n", rendered)
        self.assertIn("\nAssistant · Test model\n", rendered)
        self.assertIn("formatted", rendered)
        self.assertIn("assistant reply", rendered)
        self.assertFalse(any(char in rendered for char in "╭╮╰╯│"))
        self.assertFalse(any(set(line.strip()) == {"─"} for line in rendered.splitlines()))

    def test_ui_streams_chunks_and_returns_accumulated_response(self):
        output = []
        ui = TerminalUI(plain=True, output_fn=output.append)

        response = ui.print_assistant_stream("Test model", iter(["hello ", "world"]))

        self.assertEqual(response, "hello world")
        self.assertIn("hello ", output)
        self.assertIn("world", output)

    def test_cli_commands_errors_and_new_conversation(self):
        inputs = iter([":help", "hello", "bad", ":new", "again", ":exit"])
        prompts = []
        output = []
        model_service = FakeModelService()

        def input_fn(prompt):
            prompts.append(prompt)
            return next(inputs)

        run_chat(
            input_fn=input_fn,
            output_fn=output.append,
            plain=True,
            model_service=model_service,
        )

        self.assertTrue(model_service.loaded)
        self.assertTrue(model_service.unloaded)
        self.assertEqual(
            [messages[-1]["content"] for messages in model_service.messages],
            [
                "hello",
                "bad",
                "again",
            ],
        )
        self.assertEqual(
            [message["role"] for message in model_service.messages[1]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            [message["role"] for message in model_service.messages[2]],
            ["system", "user"],
        )
        self.assertTrue(any("Commands" in line for line in output))
        self.assertTrue(any(":fit" in line for line in output))
        self.assertTrue(any("fake generation error" in line for line in output))
        self.assertTrue(any("Started a fresh conversation" in line for line in output))
        self.assertTrue(any("reply to again" in line for line in output))
        self.assertTrue(all(prompt == "  > " for prompt in prompts))

    def test_fit_command_shows_detected_and_manual_estimates_without_loading_model(self):
        inputs = iter([":fit", ":fit 4 8", ":exit"])
        output = []
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
        model_service = FakeModelService()

        with (
            patch.object(main_module, "detect_hardware", return_value=detected, create=True),
            patch.object(main_module, "get_model_catalog", return_value=(catalog, "hub"), create=True),
        ):
            run_chat(
                input_fn=lambda _prompt: next(inputs),
                output_fn=output.append,
                plain=True,
                model_service=model_service,
            )

        rendered = "\n".join(output)
        self.assertIn("Hardware fit (hub catalog)", rendered)
        self.assertIn("Approximate estimates; not a load guarantee", rendered)
        self.assertIn("Detected: mps", rendered)
        self.assertIn("fits_cpu", rendered)
        self.assertIn("Using: VRAM 4.00 GB, RAM 8.00 GB", rendered)
        self.assertIn("fits_gpu", rendered)
        self.assertIn("Best fit: google/gemma-3-1b-it", rendered)
        self.assertEqual(model_service.messages, [])
        self.assertFalse(model_service.loaded)

    def test_fit_command_rejects_invalid_manual_values(self):
        inputs = iter([":fit -1 8", ":fit nan 8", ":fit 4", ":exit"])
        output = []
        model_service = FakeModelService()

        run_chat(
            input_fn=lambda _prompt: next(inputs),
            output_fn=output.append,
            plain=True,
            model_service=model_service,
        )

        rendered = "\n".join(output)
        self.assertEqual(rendered.count("Usage: :fit [<gpu_vram_gb> <ram_gb>]"), 3)
        self.assertFalse(model_service.loaded)
        self.assertEqual(model_service.messages, [])


if __name__ == "__main__":
    unittest.main()
