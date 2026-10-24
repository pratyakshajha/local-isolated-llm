"""Command-line chat entry point."""

import argparse
import logging

from app.cli.terminal import TerminalUI
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.services.chat_service import ChatService
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


def run_chat(
    config_path=None,
    input_fn=None,
    output_fn=None,
    plain: bool = False,
    model_service=None,
    chat_service=None,
) -> None:
    settings = load_settings(config_path)
    setup_logging()
    ui = TerminalUI(
        plain=plain,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    model_service = model_service or ModelService(settings)
    chat_service = chat_service or ChatService(
        model_service,
        system_prompt=settings.system_prompt,
        max_history_turns=settings.max_history_turns,
    )
    session_id = chat_service.start_session()

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
                chat_service.close_session(session_id)
                session_id = chat_service.start_session()
                ui.show_notice("Started a fresh conversation.")
                continue
            if not command:
                continue

            try:
                with ui.status("Generating response"):
                    response = chat_service.generate(session_id, user_input)
            except Exception as exc:
                logger.exception("Chat response generation failed")
                ui.show_error(str(exc))
                continue
            ui.show_response(model_service.model_name, response)
        ui.show_goodbye()
    except KeyboardInterrupt:
        ui.show_notice("Interrupted.")
    finally:
        chat_service.close_session(session_id)
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
