"""Terminal presentation for interactive local chats."""

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator
from textwrap import indent

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


class TerminalUI:
    def __init__(
        self,
        plain: bool = False,
        console: Console | None = None,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
    ):
        self.console = console or Console()
        self.input_fn = input_fn or input
        self.output_fn = output_fn
        self.plain = plain or output_fn is not None or not self.console.is_terminal

    def _plain_print(self, message: str) -> None:
        if self.output_fn:
            self.output_fn(message)
        else:
            self.console.print(message, markup=False)

    def show_banner(self, model_name: str) -> None:
        if self.plain:
            self._plain_print(f"Local Isolated LLM · {model_name}")
            self._plain_print("Type :help for commands.")
            return

        heading = Text("Local Isolated LLM", style="bold")
        heading.append(" · ", style="dim")
        heading.append(model_name, style="cyan")
        self.console.print(heading)
        self.console.print(Text("Type :help for commands.", style="dim"))

    def prompt(self) -> str:
        if self.plain:
            self._plain_print("")
            self._plain_print("You")
            return self.input_fn("  > ")

        self.console.print()
        self.console.print(Text("You", style="bold bright_cyan"))
        self.console.print(Text("  › ", style="bright_cyan"), end="")
        return self.input_fn("")

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        if self.plain:
            self._plain_print(f"{message}...")
            yield
            return

        status_text = Text(message, style="dim")
        with self.console.status(status_text, spinner="dots"):
            yield

    def show_response(self, model_name: str, response: str) -> None:
        assistant_text = response.strip() or "(empty response)"
        if self.plain:
            self._plain_print("")
            self._plain_print(f"Assistant · {model_name}")
            self._plain_print(indent(assistant_text, "  "))
            return

        self.console.print()
        self.console.print(Text(f"Assistant · {model_name}", style="bold magenta"))
        body = Markdown(assistant_text) if response.strip() else Text(assistant_text, style="dim")
        self.console.print(Padding(body, (0, 0, 0, 2)))

    def print_assistant_stream(self, model_name: str, chunks: Iterable[str]) -> str:
        if self.plain:
            self._plain_print("")
            self._plain_print(f"Assistant · {model_name}")
        else:
            self.console.print()
            self.console.print(Text(f"Assistant · {model_name}", style="bold magenta"))

        response = []
        for chunk in chunks:
            response.append(chunk)
            if self.output_fn:
                self.output_fn(chunk)
            else:
                self.console.print(chunk, end="", markup=False)

        text = "".join(response)
        if not text.strip():
            if self.output_fn:
                self.output_fn("(empty response)")
            else:
                self.console.print("(empty response)", style="dim", markup=False)
        elif not self.output_fn:
            self.console.print()
        return text.strip()

    def show_help(self) -> None:
        if self.plain:
            self._plain_print(
                "Commands\n"
                "  :help  Show available commands\n"
                "  :new   Start a fresh conversation\n"
                "  :exit  Quit and unload the model"
            )
            return

        self.console.print()
        self.console.print(Text("Commands", style="bold cyan"))
        self.console.print(Text("  :help  Show available commands", style="dim"))
        self.console.print(Text("  :new   Start a fresh conversation", style="dim"))
        self.console.print(Text("  :exit  Quit and unload the model", style="dim"))

    def show_notice(self, message: str) -> None:
        if self.plain:
            self._plain_print(message)
        else:
            self.console.print(Text(message, style="dim cyan"))

    def show_error(self, message: str) -> None:
        if self.plain:
            self._plain_print(f"Error\n  {message}")
        else:
            self.console.print()
            self.console.print(Text("Error", style="bold red"))
            self.console.print(Padding(Text(message), (0, 0, 0, 2)))

    def show_goodbye(self) -> None:
        self.show_notice("Goodbye.")
