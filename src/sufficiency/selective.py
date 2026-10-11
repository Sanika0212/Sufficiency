"""Choosing when to answer, with a guaranteed error rate.

Answer when the sufficiency estimate is at least ``threshold``; otherwise abstain. Two
ways to set the threshold from a labelled calibration set, for two different promises:

* :func:`crc_threshold` (conformal risk control): the share of *all* questions that are
  answered wrongly is at most ``alpha``, in expectation over calibration sets.
* :func:`ltt_threshold` (learn then test): the error rate *among answered questions*
  is at most ``alpha`` with probability at least ``1 - delta``. This is usually the
  promise a product needs ("when it answers, it is right 90% of the time"), at the cost
  of answering less.

Both assume new questions are exchangeable with the calibration questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def crc_threshold(scores: ArrayLike, wrong: ArrayLike, alpha: float) -> float:
    """Smallest threshold whose adjusted rate of answered-and-wrong is at most ``alpha``.

    With losses bounded by 1 and non-increasing in the threshold, choosing the smallest
    ``t`` with ``n / (n + 1) * risk(t) + 1 / (n + 1) <= alpha`` gives
    ``E[loss] <= alpha`` (Angelopoulos et al., conformal risk control).
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(wrong, dtype=bool)
    n = len(s)
    candidates = np.concatenate([np.unique(s), [np.inf]])
    for t in candidates:  # ascending: answer as much as the bound allows
        risk = float(np.mean((s >= t) & w)) if n else 0.0
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return float(t)
    return math.inf


def _binomial_tail(errors: int, trials: int, rate: float) -> float:
    """P(X <= errors) for X ~ Binomial(trials, rate): the p-value against 'rate > alpha'."""
    from scipy.stats import binom

    return float(binom.cdf(errors, trials, rate))


def ltt_threshold(
    scores: ArrayLike, wrong: ArrayLike, alpha: float, delta: float = 0.1, grid: int = 100
) -> float:
    """Lowest threshold whose selective error rate is certified below ``alpha``.

    Thresholds are tested from strict to lenient (fixed-sequence testing, so no
    multiplicity correction is needed). The null at each threshold is "the error rate
    among answered questions exceeds ``alpha``", rejected with a binomial tail test at
    level ``delta``. Testing stops at the first failure.

    Thresholds that answer fewer than ``log(delta) / log(1 - alpha)`` questions are
    skipped: even with no errors they could never be certified, so testing them would
    only end the sequence early. The skip depends on the scores alone, not the labels.
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(wrong, dtype=bool)
    if len(s) == 0:
        return math.inf
    least = math.ceil(math.log(delta) / math.log(1 - alpha)) if 0 < alpha < 1 else 1
    levels = np.unique(np.quantile(s, np.linspace(1.0, 0.0, grid)))[::-1]
    chosen = math.inf
    for t in levels:
        answered = s >= t
        k = int(answered.sum())
        if k < least:
            continue
        if _binomial_tail(int((answered & w).sum()), k, alpha) > delta:
            break
        chosen = float(t)
    return chosen


@dataclass(frozen=True)
class SelectiveRule:
    threshold: float
    method: str
    alpha: float
    delta: float | None = None

    def answers(self, score: float) -> bool:
        return score >= self.threshold

    @classmethod
    def fit(
        cls,
        scores: ArrayLike,
        wrong: ArrayLike,
        alpha: float,
        *,
        method: str = "ltt",
        delta: float = 0.1,
    ) -> SelectiveRule:
        if method == "crc":
            return cls(crc_threshold(scores, wrong, alpha), "crc", alpha)
        if method == "ltt":
            return cls(ltt_threshold(scores, wrong, alpha, delta), "ltt", alpha, delta)
        raise ValueError(f"unknown method {method!r}; use 'crc' or 'ltt'")


def evaluate_rule(rule: SelectiveRule, scores: ArrayLike, wrong: ArrayLike) -> dict[str, float]:
    """Coverage, error rate among answered, and answered-and-wrong rate on held-out data."""
    s = np.asarray(scores, dtype=float)
    w = np.asarray(wrong, dtype=bool)
    answered = s >= rule.threshold
    n_answered = int(answered.sum())
    return {
        "coverage": float(answered.mean()) if len(s) else 0.0,
        "selective_error": float((answered & w).sum() / n_answered) if n_answered else 0.0,
        "answered_and_wrong": float((answered & w).mean()) if len(s) else 0.0,
    }
