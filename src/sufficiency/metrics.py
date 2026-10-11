"""Answer and calibration metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation, articles and extra spaces."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, answers: Sequence[str]) -> bool:
    pred = normalize(prediction)
    return any(pred == normalize(a) for a in answers)


def f1(prediction: str, answers: Sequence[str]) -> float:
    pred = normalize(prediction).split()
    best = 0.0
    for answer in answers:
        gold = normalize(answer).split()
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        if not pred or not gold or overlap == 0:
            continue
        precision, recall = overlap / len(pred), overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains(prediction: str, answers: Sequence[str]) -> bool:
    """The normalised answer occurs in the normalised prediction (lenient for long answers)."""
    pred = f" {normalize(prediction)} "
    return any(f" {normalize(a)} " in pred for a in answers if normalize(a))


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for value in np.unique(s):  # average ranks over ties
        tied = s == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(probs: ArrayLike, labels: ArrayLike, bins: int = 10) -> float:
    """ECE with equal-width bins: the gap between confidence and accuracy, weighted by bin size."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(error)


def risk_coverage(scores: ArrayLike, correct: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among answered items as coverage grows (answer the highest scores first)."""
    s = np.asarray(scores, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    order = np.argsort(-s, kind="mergesort")
    answered = np.arange(1, len(s) + 1)
    risk = np.cumsum(~ok[order]) / answered
    coverage = answered / len(s)
    return coverage, risk


def area_under_risk_coverage(scores: ArrayLike, correct: ArrayLike) -> float:
    """Mean selective risk over all coverage levels (lower is better)."""
    _, risk = risk_coverage(scores, correct)
    return float(risk.mean()) if len(risk) else float("nan")
