"""Retrievers: BM25 (no dependencies), dense embeddings, and a hybrid of the two.

Every retriever indexes a list of :class:`~sufficiency.model.Passage` and returns the
top ``k`` with their scores. The recovery policy can switch between them, so they share
one small interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from sufficiency.model import Passage

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from by with about as is are was were be
    been being it its this that these those what which who whom whose when where why how
    do does did has have had not no
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def index(self, passages: Sequence[Passage]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[Passage]: ...


class BM25:
    """Okapi BM25 over passage title and text."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.passages: list[Passage] = []
        self._tf: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._lengths = np.zeros(0)

    def index(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)
        self._tf = [Counter(tokenize(f"{p.title} {p.text}")) for p in self.passages]
        self._lengths = np.array([sum(tf.values()) for tf in self._tf], dtype=float)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.passages)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        if not self.passages:
            return np.zeros(0)
        avg = self._lengths.mean() or 1.0
        out = np.zeros(len(self.passages))
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            freqs = np.array([tf.get(term, 0) for tf in self._tf], dtype=float)
            denom = freqs + self.k1 * (1 - self.b + self.b * self._lengths / avg)
            out += idf * freqs * (self.k1 + 1) / np.where(denom == 0, 1, denom)
        return out

    def search(self, query: str, k: int = 5) -> list[Passage]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [self.passages[i].with_score(float(s[i])) for i in order]


class Dense:
    """Cosine similarity of mean-pooled sentence embeddings (all-MiniLM-L6-v2 by default)."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.name = "dense"
        self.device = device
        self.batch_size = batch_size
        self.passages: list[Passage] = []
        self._matrix = np.zeros((0, 0))
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        from sufficiency.devices import default_device

        self.device = self.device or default_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            encoded = self._tokenizer(
                batch, truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, 0))

    def index(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)
        self._matrix = self.embed([f"{p.title}. {p.text}" for p in self.passages])

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not self.passages:
            return []
        s = self._matrix @ self.embed([query])[0]
        order = np.argsort(-s, kind="stable")[:k]
        return [self.passages[i].with_score(float(s[i])) for i in order]


class Hybrid:
    """Reciprocal rank fusion of several retrievers (score = sum of 1 / (60 + rank))."""

    def __init__(self, *retrievers: Retriever, depth: int = 50, constant: int = 60) -> None:
        self.retrievers = retrievers
        self.name = "+".join(r.name for r in retrievers)
        self.depth, self.constant = depth, constant

    def index(self, passages: Sequence[Passage]) -> None:
        for r in self.retrievers:
            r.index(passages)

    def search(self, query: str, k: int = 5) -> list[Passage]:
        fused: dict[str, float] = {}
        by_id: dict[str, Passage] = {}
        for r in self.retrievers:
            for rank, p in enumerate(r.search(query, self.depth), start=1):
                fused[p.id] = fused.get(p.id, 0.0) + 1.0 / (self.constant + rank)
                by_id[p.id] = p
        best = sorted(fused, key=lambda pid: -fused[pid])[:k]
        return [by_id[pid].with_score(fused[pid]) for pid in best]
