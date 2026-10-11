"""Answer generators: an extractive reader, a local chat model, or an OpenAI-compatible API.

A generator is any callable ``(question, passages) -> answer``. The extractive reader
needs no LLM and is free to run; chat models produce fluent answers.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from sufficiency.model import Passage

INSTRUCTION = (
    "Answer the question using the numbered passages. Reply with the answer only, as "
    "briefly as possible."
)


class Generator(Protocol):
    name: str

    def __call__(self, question: str, passages: Sequence[Passage]) -> str: ...


def prompt(question: str, passages: Sequence[Passage]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{i + 1}] {p.title + ': ' if p.title else ''}{p.text}" for i, p in enumerate(passages)
    )
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]


class Extractive:
    """Best span of an extractive reader across the passages (no LLM)."""

    def __init__(self, reader: Any = None) -> None:
        from sufficiency.features import ReaderFeatures

        self.reader = reader or ReaderFeatures()
        self.name = f"extractive:{self.reader.model_id}"

    def __call__(self, question: str, passages: Sequence[Passage]) -> str:
        spans = self.reader.spans(question, passages)
        if not spans:
            return ""
        return str(max(spans, key=lambda s: s[0])[2]).strip()


class LocalChat:
    """A Hugging Face instruct model run locally with greedy decoding."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str | None = None,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_id = model_id
        self.name = f"local:{model_id}"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from sufficiency.devices import default_device

        self.device = self.device or default_device()
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        self._model.to(self.device).eval()

    def __call__(
        self, question: str, passages: Sequence[Passage]
    ) -> str:  # pragma: no cover - needs weights; exercised by the benchmarks
        import torch

        self._load()
        encoded = self._tokenizer.apply_chat_template(
            prompt(question, passages),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"].to(self.device)
        with torch.inference_mode():
            out = self._model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return str(self._tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)).strip()


def load_env(path: str | os.PathLike[str] = ".env") -> None:
    """Read ``KEY=VALUE`` lines from a local ``.env`` (existing variables win); never logged."""
    env = Path(path)
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class OpenAIChat:
    """Any OpenAI-compatible chat completions endpoint. Tracks calls and tokens."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        max_tokens: int = 64,
    ) -> None:
        if not api_key and not os.environ.get("OPENAI_API_KEY"):
            load_env()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("set OPENAI_API_KEY (environment or .env) or pass api_key")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model
        self.name = f"openai:{model}"
        self.client = client or httpx.Client(timeout=60.0)
        self.max_tokens = max_tokens
        self.calls = self.input_tokens = self.output_tokens = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
            },
        )
        if not response.is_success:
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            message = (
                error.get("message", response.reason_phrase) if isinstance(error, dict) else error
            )
            raise RuntimeError(f"{self.name} returned {response.status_code}: {str(message)[:200]}")
        body = response.json()
        usage = body.get("usage", {})
        self.calls += 1
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        return str(body["choices"][0]["message"].get("content") or "").strip()

    def __call__(self, question: str, passages: Sequence[Passage]) -> str:
        return self.complete(prompt(question, passages))
