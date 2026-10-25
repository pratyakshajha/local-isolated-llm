"""Command-line chat entry point."""

import argparse
import logging

from app.cli.terminal import TerminalUI
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.llm.conversation import Conversation
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

    try:
        try:
            with ui.status(f"Loading {model_service.model_name}"):
                model_service.load_model()
        except Exception as exc:
            logger.exception("Unable to load the model")
            ui.show_error(f"Could not load {model_service.model_name}: {exc}")
            return

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
