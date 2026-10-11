"""Core data types: passages, questions and the trace of one controlled answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    title: str = ""
    score: float = 0.0  # retrieval score, higher is better
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def with_score(self, score: float) -> Passage:
        return Passage(self.id, self.text, self.title, score, self.metadata)


@dataclass
class Example:
    """A question with its passages and, when known, the answer and answerability.

    ``answerable`` is ``False`` when the passages are known not to contain the answer
    (SQuAD v2 unanswerable questions, MuSiQue unanswerable variants, or a context with
    its gold passages removed).
    """

    id: str
    question: str
    passages: list[Passage]
    answers: list[str] = field(default_factory=list)
    answerable: bool | None = None
    supporting: list[str] = field(default_factory=list)  # ids of gold passages
    sub_questions: list[str] = field(default_factory=list)
    source: str = ""


@dataclass
class Step:
    """One decision of the controller: estimate, recover, answer or abstain."""

    kind: str
    detail: str
    sufficiency: float | None = None
    cost: float = 0.0


@dataclass
class Result:
    question: str
    answer: str | None
    abstained: bool
    sufficiency: float
    passages: list[Passage]
    steps: list[Step] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.steps)
