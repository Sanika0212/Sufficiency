"""Model wrappers with their networks stubbed out: no weights are downloaded."""

from collections.abc import Sequence

import numpy as np
import pytest

from sufficiency import devices
from sufficiency.features import ReaderFeatures
from sufficiency.model import Passage
from sufficiency.retrieval import Dense


def test_device_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUFFICIENCY_DEVICE", "cpu")
    assert devices.default_device() == "cpu"


def test_reader_compute_summarises_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReaderFeatures()
    spans = [(2.0, 0.8, "Ulm"), (-1.0, 0.1, "ulm "), (0.5, 0.3, "Munich")]
    monkeypatch.setattr(reader, "spans", lambda q, ps: spans)
    margin_max, margin_mean, prob, agree = reader.compute("q", [])
    assert margin_max == 2.0
    assert margin_mean == pytest.approx(0.5)
    assert prob == 0.8
    assert agree == pytest.approx(2 / 3)
    assert reader.last_answers == ["Ulm", "ulm ", "Munich"]
    monkeypatch.setattr(reader, "spans", lambda q, ps: [])
    assert reader.compute("q", []) == [-10.0, -10.0, 0.0, 0.0]


def test_dense_ranks_by_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = ["river", "city", "france"]

    def embed(texts: Sequence[str]) -> np.ndarray:
        rows = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float)
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)

    dense = Dense()
    assert dense.search("anything") == []
    monkeypatch.setattr(dense, "embed", embed)
    dense.index([Passage("a", "A river runs."), Passage("b", "A city in France.")])
    top = dense.search("which city in france", k=1)
    assert [p.id for p in top] == ["b"]
    assert top[0].score > 0.9
