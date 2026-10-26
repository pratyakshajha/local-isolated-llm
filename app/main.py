"""Command-line chat entry point."""

import argparse
import logging
import math

from app.cli.terminal import TerminalUI
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.llm.conversation import Conversation
from app.services.hardware_fit import analyze_models, detect_hardware
from app.services.model_catalog import get_model_catalog
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


def run_chat(
    config_path=None,
    input_fn=None,
    output_fn=None,
    plain: bool = False,
    model_service=None,
) -> None:
    settings = load_settings(config_path)
    setup_logging()
    ui = TerminalUI(
        plain=plain,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    model_service = model_service or ModelService(settings)
    conversation = Conversation(settings.system_prompt, settings.max_history_turns)
    model_loaded = getattr(model_service, "is_loaded", False)

    try:
        ui.show_banner(model_service.model_name)
        while True:
            try:
                user_input = ui.prompt()
            except EOFError:
                ui.show_notice("Input closed.")
                break

            command = user_input.strip().casefold()
            if command in {":exit", "exit", "quit"}:
                break
            if command == ":help":
                ui.show_help()
                continue
            if command == ":new":
                conversation = Conversation(
                    settings.system_prompt, settings.max_history_turns
                )
                ui.show_notice("Started a fresh conversation.")
                continue
            if not command:
                continue

            command_parts = command.split()
            if command_parts[0] == ":fit":
                if len(command_parts) not in (1, 3):
                    ui.show_error("Usage: :fit [<gpu_vram_gb> <ram_gb>]")
                    continue
                try:
                    manual_values = [float(value) for value in command_parts[1:]]
                except ValueError:
                    ui.show_error("Usage: :fit [<gpu_vram_gb> <ram_gb>]")
                    continue
                if any(not math.isfinite(value) or value < 0 for value in manual_values):
                    ui.show_error("Usage: :fit [<gpu_vram_gb> <ram_gb>]")
                    continue

                detected = detect_hardware()
                catalog, catalog_source = get_model_catalog()
                used_vram = manual_values[0] if manual_values else detected["gpu_vram_gb"]
                used_ram = manual_values[1] if manual_values else detected["ram_gb"]
                backend = detected["backend"] if not manual_values else (
                    "cuda" if used_vram > 0 else "cpu"
                )
                analysis = analyze_models(
                    catalog,
                    gpu_vram_gb=used_vram,
                    ram_gb=used_ram,
                    disk_free_gb=detected["disk_free_gb"],
                    gpu_name=detected["gpu_name"],
                    backend=backend,
                )

                def format_gb(value):
                    return "n/a" if value is None else f"{value:.2f} GB"

                lines = [
                    f"Hardware fit ({catalog_source} catalog)",
                    "Approximate estimates; not a load guarantee",
                    (
                        f"Detected: {detected['backend']} · GPU {detected['gpu_name'] or 'n/a'} · "
                        f"VRAM {format_gb(detected['gpu_vram_gb'])} · "
                        f"RAM {format_gb(detected['ram_gb'])} · "
                        f"disk {format_gb(detected['disk_free_gb'])}"
                    ),
                    f"Using: VRAM {format_gb(used_vram)}, RAM {format_gb(used_ram)}",
                    f"Best fit: {analysis['best_fit'] or 'none'}",
                    "Model | Verdict | Need | Download | Est. tokens/s",
                ]
                for model in analysis["models"]:
                    verdict = model["verdict"]
                    if model["disk_ok"] is False:
                        verdict += " (disk low)"
                    speed = model["est_tokens_per_sec"]
                    lines.append(
                        f"{model['id']} | {verdict} | {format_gb(model['need_gb'])} | "
                        f"{format_gb(model['download_gb'])} | "
                        f"{f'~{speed:.1f}' if speed is not None else 'n/a'}"
                    )
                lines.extend(f"Note: {note}" for note in detected["notes"])
                ui.show_notice("\n".join(lines))
                continue

            if not model_loaded:
                try:
                    with ui.status(f"Loading {model_service.model_name}"):
                        model_service.load_model()
                    model_loaded = True
                except Exception as exc:
                    logger.exception("Unable to load the model")
                    ui.show_error(f"Could not load {model_service.model_name}: {exc}")
                    continue

            try:
                with ui.status("Generating response"):
                    response = ui.print_assistant_stream(
                        model_service.model_name,
                        model_service.generate_stream(
                            conversation.messages_for(user_input)
                        ),
                    )
            except Exception as exc:
                logger.exception("Chat response generation failed")
                ui.show_error(str(exc))
                continue
            conversation.add_turn(user_input, response)
        ui.show_goodbye()
    except KeyboardInterrupt:
        ui.show_notice("Interrupted.")
    finally:
        model_service.unload_model()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Chat with a local language model")
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use plain text instead of the Rich terminal interface",
    )
    args = parser.parse_args(argv)
    try:
        run_chat(args.config, plain=args.plain)
    except KeyboardInterrupt:
        logger.info("Chat stopped")


if __name__ == "__main__":
    main()
