"""Hugging Face Transformers implementation of the runtime interface."""

from __future__ import annotations

import logging
from threading import Event, Thread
from typing import Iterator, Sequence

from app.core.config import GenerationSettings
from app.runtimes.base import Runtime

logger = logging.getLogger(__name__)


class TransformersRuntime(Runtime):
    """Load and generate from a causal language model using Transformers.

    PyTorch and Transformers are imported only when ``load`` is called, so
    importing application modules does not load ML libraries or model files.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._device = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            supports_bfloat16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
        else:
            dtype = torch.float32

        logger.info("Loading model %s on %s", self.model_id, device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
        ).to(device)
        model.eval()

        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Loaded model %s", self.model_id)

    def unload(self) -> None:
        if not self.is_loaded:
            return

        import torch

        self._model = None
        self._tokenizer = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded model %s", self.model_id)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        settings: GenerationSettings,
        cancel_event: Event | None = None,
    ) -> str:
        return "".join(
            self.generate_stream(messages, settings, cancel_event=cancel_event)
        ).strip()

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        settings: GenerationSettings,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        if not self.is_loaded:
            self.load()

        import torch
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        encoded = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {
            name: tensor.to(self._device) for name, tensor in encoded.items()
        }
        stop_requested = Event()
        generation_args = {"max_new_tokens": settings.max_new_tokens}
        if settings.temperature > 0:
            generation_args.update(
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        else:
            generation_args["do_sample"] = False

        class CancelOnRequest(StoppingCriteria):
            def __call__(self, _input_ids, _scores, **_kwargs):
                return stop_requested.is_set() or (
                    cancel_event is not None and cancel_event.is_set()
                )

        generation_args["stopping_criteria"] = StoppingCriteriaList(
            [CancelOnRequest()]
        )
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_args["streamer"] = streamer
        errors = []

        def generate() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(**encoded, **generation_args)
            except BaseException as exc:
                errors.append(exc)
                streamer.end()

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            yield from (text for text in streamer if text)
            thread.join()
            if errors:
                raise errors[0]
        finally:
            if thread.is_alive():
                stop_requested.set()
                thread.join()
