"""Features that predict whether the retrieved passages are enough to answer a question.

Three groups, each usable alone:

- :class:`RetrievalFeatures`: how decisively the retriever ranked the passages (top
  score share, gap to the runner-up, entropy of the score distribution).
- :class:`LexicalFeatures`: how much of the question's content the passages cover, and
  cues for multi-hop questions.
- :class:`ReaderFeatures`: an extractive reader trained with unanswerable questions
  (SQuAD 2.0) scores the best answer span against the "no answer" option. This is the
  strongest single signal for single-hop questions and runs locally.

A :class:`Featurizer` concatenates the groups into one vector with stable names.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from sufficiency.model import Passage
from sufficiency.retrieval import tokenize

WH_WORDS = ("what", "who", "when", "where", "which", "how", "why")
_CAPITALIZED = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")


class FeatureGroup(Protocol):
    names: tuple[str, ...]

    def compute(self, question: str, passages: Sequence[Passage]) -> list[float]: ...


class RetrievalFeatures:
    names: tuple[str, ...] = ("ret_top_share", "ret_gap", "ret_entropy", "ret_count")

    def compute(self, question: str, passages: Sequence[Passage]) -> list[float]:
        scores = np.array([max(p.score, 0.0) for p in passages], dtype=float)
        if len(scores) == 0 or scores.sum() <= 0:
            return [0.0, 0.0, 1.0, float(len(scores))]
        share = scores / scores.sum()
        top = np.sort(scores)[::-1]
        gap = (top[0] - top[1]) / top[0] if len(top) > 1 and top[0] > 0 else 1.0
        entropy = float(-(share[share > 0] * np.log(share[share > 0])).sum())
        norm = entropy / math.log(len(scores)) if len(scores) > 1 else 0.0
        return [float(np.max(share)), float(gap), norm, float(len(scores))]


class LexicalFeatures:
    names = (
        "lex_cover_top1",
        "lex_cover_all",
        "lex_question_terms",
        "lex_entities",
        "lex_multihop_cue",
        *(f"lex_wh_{w}" for w in WH_WORDS),
    )

    def compute(self, question: str, passages: Sequence[Passage]) -> list[float]:
        terms = set(tokenize(question))
        texts = [set(tokenize(f"{p.title} {p.text}")) for p in passages]
        union = set().union(*texts) if texts else set()
        top1 = texts[0] if texts else set()
        n = len(terms) or 1
        first = question.strip().split(" ", 1)[0].lower()
        multihop = float(
            bool(re.search(r"\b(of the|whose|that|which)\b", question.lower()))
            and len(_CAPITALIZED.findall(question)) >= 2
        )
        return [
            len(terms & top1) / n,
            len(terms & union) / n,
            float(len(terms)),
            float(len(_CAPITALIZED.findall(question))),
            multihop,
            *(float(first == w) for w in WH_WORDS),
        ]


class ReaderFeatures:
    """Best-span vs no-answer margin of an extractive SQuAD 2.0 reader, over the passages."""

    names: tuple[str, ...] = (
        "reader_margin_max",
        "reader_margin_mean",
        "reader_span_prob",
        "reader_agree",
    )

    def __init__(
        self,
        model_id: str = "deepset/roberta-base-squad2",
        *,
        device: str | None = None,
        max_passages: int = 5,
        max_length: int = 384,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_passages = max_passages
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self.last_answers: list[str] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        from sufficiency.devices import default_device

        self.device = self.device or default_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_id)
        self._model.to(self.device).eval()

    def spans(self, question: str, passages: Sequence[Passage]) -> list[tuple[float, float, str]]:
        """``(margin, span probability, answer text)`` per passage."""
        import torch

        self._load()
        chosen = list(passages[: self.max_passages])
        if not chosen:
            return []
        encoded = self._tokenizer(
            [question] * len(chosen),
            [p.text for p in chosen],
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        sequence_ids = [encoded.sequence_ids(i) for i in range(len(chosen))]
        with torch.inference_mode():
            out = self._model(**encoded.to(self.device))
        start = out.start_logits.float().cpu().numpy()
        end = out.end_logits.float().cpu().numpy()
        results = []
        for i, passage in enumerate(chosen):
            context = np.array([j for j, s in enumerate(sequence_ids[i]) if s == 1])
            null = start[i, 0] + end[i, 0]
            if len(context) == 0:
                results.append((-10.0, 0.0, ""))
                continue
            pair = start[i, context][:, None] + end[i, context][None, :]
            gap = context[None, :] - context[:, None]
            pair = np.where((gap >= 0) & (gap < 30), pair, -np.inf)  # end after start, short
            a_idx, b_idx = np.unravel_index(int(np.argmax(pair)), pair.shape)
            best = float(pair[a_idx, b_idx])
            best_span = (int(context[a_idx]), int(context[b_idx]))
            s_probs = np.exp(start[i] - start[i].max())
            e_probs = np.exp(end[i] - end[i].max())
            prob = float(
                s_probs[best_span[0]] / s_probs.sum() * e_probs[best_span[1]] / e_probs.sum()
            )
            a_char = int(offsets[i, best_span[0], 0])
            b_char = int(offsets[i, best_span[1], 1])
            results.append((float(best - null), prob, passage.text[a_char:b_char]))
        return results

    def compute(self, question: str, passages: Sequence[Passage]) -> list[float]:
        spans = self.spans(question, passages)
        self.last_answers = [s for _, _, s in spans]
        if not spans:
            return [-10.0, -10.0, 0.0, 0.0]
        margins = [m for m, _, _ in spans]
        best = int(np.argmax(margins))
        answers = [s.lower().strip() for _, _, s in spans if s.strip()]
        agree = answers.count(spans[best][2].lower().strip()) / len(answers) if answers else 0.0
        return [max(margins), float(np.mean(margins)), spans[best][1], agree]


class Featurizer:
    def __init__(self, *groups: FeatureGroup) -> None:
        self.groups = groups or (RetrievalFeatures(), LexicalFeatures())
        self.names = tuple(name for g in self.groups for name in g.names)

    def __call__(self, question: str, passages: Sequence[Passage]) -> np.ndarray:
        values: list[float] = []
        for group in self.groups:
            values.extend(group.compute(question, passages))
        return np.asarray(values, dtype=float)

    def matrix(self, items: Sequence[tuple[str, Sequence[Passage]]]) -> np.ndarray:
        return (
            np.stack([self(q, ps) for q, ps in items]) if items else np.zeros((0, len(self.names)))
        )
