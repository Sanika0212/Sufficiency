import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from sufficiency.controller import Controller
from sufficiency.estimator import SufficiencyEstimator
from sufficiency.features import Featurizer, LexicalFeatures
from sufficiency.generate import Extractive, OpenAIChat, load_env, prompt
from sufficiency.model import Passage
from sufficiency.policy import LearnedPolicy
from sufficiency.retrieval import BM25
from sufficiency.selective import SelectiveRule

CORPUS = [
    Passage("einstein", "Albert Einstein was born in Ulm in 1879.", "Albert Einstein"),
    Passage("ulm", "Ulm is a city on the Danube.", "Ulm"),
    Passage("paris", "Paris is the capital of France.", "Paris"),
]


class Coverage(SufficiencyEstimator):
    """Pretend estimator: sufficiency = share of question terms covered by the passages."""

    def __init__(self) -> None:
        super().__init__(Featurizer(LexicalFeatures()))

    def predict(self, question, passages):  # type: ignore[no-untyped-def]
        return float(self.featurizer(question, passages)[1])


class Echo:
    name = "echo"

    def __call__(self, question, passages):  # type: ignore[no-untyped-def]
        return passages[0].title


class AlwaysExpand(LearnedPolicy):
    def decide(self, features):  # type: ignore[no-untyped-def]
        from sufficiency.policy import Decision

        return Decision("expand", 0.9)


class GiveUp(LearnedPolicy):
    def decide(self, features):  # type: ignore[no-untyped-def]
        from sufficiency.policy import Decision

        return Decision("abstain", 0.1)


def retriever() -> BM25:
    r = BM25()
    r.index(CORPUS)
    return r


def test_controller_answers_recovers_and_abstains() -> None:
    rule = SelectiveRule(0.99, "ltt", 0.1)
    answered = Controller(retriever(), Coverage(), rule, Echo(), k=1).answer(
        "Where was Einstein born?"
    )
    assert not answered.abstained
    assert answered.answer == "Albert Einstein"
    assert [s.kind for s in answered.steps] == ["retrieve", "answer"]

    question = "Which river is Ulm on, where Einstein was born?"
    recovered = Controller(
        retriever(), Coverage(), rule, Echo(), policy=AlwaysExpand(), k=1
    ).answer(question)
    kinds = [s.kind for s in recovered.steps]
    assert kinds[:2] == ["retrieve", "expand"]
    assert recovered.cost >= 2.0

    gave_up = Controller(retriever(), Coverage(), rule, Echo(), policy=GiveUp(), k=1).answer(
        question
    )
    assert gave_up.abstained
    assert gave_up.answer is None
    assert [s.kind for s in gave_up.steps] == ["retrieve", "decide", "abstain"]


def test_prompt_and_extractive_generator() -> None:
    messages = prompt("Q?", CORPUS[:2])
    assert messages[1]["content"].startswith("Passages:\n[1] Albert Einstein: ")

    class Spans:
        model_id = "fake"

        def spans(self, question, passages):  # type: ignore[no-untyped-def]
            return [(1.0, 0.5, " Ulm "), (2.0, 0.6, " Danube ")] if passages else []

    gen = Extractive(Spans())
    assert gen("q", CORPUS) == "Danube"
    assert gen("q", []) == ""
    assert gen.name == "extractive:fake"


def test_openai_chat_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if seen[-1]["messages"][1]["content"].endswith("fail"):
            return httpx.Response(429, json={"error": {"message": "no credit"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": " Ulm "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chat = OpenAIChat(api_key="sk-test", client=client)
    assert chat("Where?", CORPUS[:1]) == "Ulm"
    assert (chat.calls, chat.input_tokens, chat.output_tokens) == (1, 10, 1)
    with pytest.raises(RuntimeError, match="429: no credit"):
        chat("fail", [])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIChat()
    (tmp_path / ".env").write_text("# keys\nOPENAI_API_KEY='sk-from-file'\nnot a pair\n")
    assert OpenAIChat(client=client).api_key == "sk-from-file"
    load_env(tmp_path / "missing.env")
    assert isinstance(np.zeros(1), np.ndarray)
