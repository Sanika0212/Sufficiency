"""Choosing a recovery action: learned from logged outcomes, with cost in the objective.

For every training question each action is run once and its outcome logged (did the
new context contain everything needed?). One calibrated model per action then
predicts ``P(success | action, features of the original context)``. At run time the
policy picks the action with the highest ``P(success) - cost_weight * cost``, or
abstains when even the best action is unlikely to work.

Sweeping ``cost_weight`` traces the accuracy-cost frontier. Baselines for comparison:
always the same action, the cheapest successful action in hindsight (oracle), and a
fixed rule cascade of the kind found in hand-built "self-healing" pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

COSTS = {
    "keep": 0.0,
    "expand": 1.0,
    "feedback": 1.0,
    "rewrite": 2.0,
    "decompose": 3.0,
    "switch": 1.0,
}


@dataclass
class Decision:
    action: str  # an action name, or "abstain"
    success: float  # predicted probability of success for the chosen action
    scores: dict[str, float] = field(default_factory=dict)


class LearnedPolicy:
    def __init__(
        self,
        actions: Sequence[str] = ("keep", "expand", "feedback", "decompose", "switch"),
        *,
        costs: Mapping[str, float] | None = None,
        cost_weight: float = 0.05,
        abstain_below: float = 0.0,
    ) -> None:
        self.actions = tuple(actions)
        self.costs = {**COSTS, **(costs or {})}
        self.cost_weight = cost_weight
        self.abstain_below = abstain_below
        self.models: dict[str, Any] = {}
        self.base_rates: dict[str, float] = {}

    def fit(self, features: ArrayLike, outcomes: Mapping[str, ArrayLike]) -> LearnedPolicy:
        """``outcomes[action][i]`` is 1 if ``action`` fixed question ``i``."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = np.asarray(features, dtype=float)
        for action in self.actions:
            y = np.asarray(outcomes[action], dtype=int)
            self.base_rates[action] = float(y.mean()) if len(y) else 0.0
            minority = int(np.min(np.bincount(y, minlength=2)))
            if minority < 2:
                self.models[action] = None  # constant outcome: use the base rate
                continue
            base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model = CalibratedClassifierCV(base, method="sigmoid", cv=min(5, minority))
            self.models[action] = model.fit(x, y)
        return self

    def success(self, features: ArrayLike) -> dict[str, np.ndarray]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        out = {}
        for action in self.actions:
            model = self.models.get(action)
            out[action] = (
                np.full(len(x), self.base_rates.get(action, 0.0))
                if model is None
                else np.asarray(model.predict_proba(x)[:, 1])
            )
        return out

    def decide_matrix(self, features: ArrayLike) -> list[Decision]:
        probs = self.success(features)
        decisions = []
        for i in range(len(next(iter(probs.values())))):
            utility = {a: probs[a][i] - self.cost_weight * self.costs[a] for a in self.actions}
            best = max(utility, key=lambda a: utility[a])
            p = float(probs[best][i])
            action = best if p >= self.abstain_below else "abstain"
            decisions.append(Decision(action, p, {a: float(probs[a][i]) for a in self.actions}))
        return decisions

    def decide(self, features: ArrayLike) -> Decision:
        return self.decide_matrix(features)[0]


def oracle(outcomes: Mapping[str, ArrayLike], costs: Mapping[str, float] = COSTS) -> list[str]:
    """Cheapest action that succeeded for each question in hindsight ("abstain" if none)."""
    names = sorted(outcomes, key=lambda a: costs[a])
    n = len(np.asarray(next(iter(outcomes.values()))))
    out = []
    for i in range(n):
        out.append(next((a for a in names if np.asarray(outcomes[a])[i]), "abstain"))
    return out


def rule_cascade(confidence: float, low: float = 0.4, high: float = 0.7) -> str:
    """A fixed-threshold cascade like those in hand-built pipelines (the baseline)."""
    if confidence >= high:
        return "keep"
    if confidence >= low:
        return "feedback"
    return "expand"


def evaluate_choices(
    choices: Sequence[str], outcomes: Mapping[str, ArrayLike], costs: Mapping[str, float] = COSTS
) -> dict[str, float]:
    """Success rate, abstention rate and mean cost of a list of chosen actions."""
    n = len(choices)
    if n == 0:
        return {"success": 0.0, "abstain": 0.0, "cost": 0.0}
    success = sum(bool(np.asarray(outcomes[c])[i]) for i, c in enumerate(choices) if c != "abstain")
    cost = sum(costs.get(c, 0.0) for c in choices)
    return {
        "success": success / n,
        "abstain": sum(c == "abstain" for c in choices) / n,
        "cost": cost / n,
    }
