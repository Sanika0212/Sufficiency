import numpy as np
import pytest

from sufficiency.actions import ACTIONS, Context, _merge, split_question
from sufficiency.model import Passage
from sufficiency.policy import (
    COSTS,
    LearnedPolicy,
    evaluate_choices,
    oracle,
    rule_cascade,
)
from sufficiency.retrieval import BM25

CORPUS = [
    Passage("einstein", "Albert Einstein was born in Ulm in 1879.", "Albert Einstein"),
    Passage("ulm", "Ulm is a city on the Danube in Baden-Württemberg.", "Ulm"),
    Passage("danube", "The Danube flows into the Black Sea.", "Danube"),
    Passage("paris", "Paris is the capital of France.", "Paris"),
    Passage("cheese", "Cheese is made from milk.", "Cheese"),
]


def context(question: str, k: int = 2, **kwargs) -> Context:  # type: ignore[no-untyped-def]
    bm25 = BM25()
    bm25.index(CORPUS)
    return Context(question, bm25.search(question, k), bm25, k, **kwargs)


def test_actions_change_the_context_and_report_cost() -> None:
    ctx = context("Which river flows through the city where Einstein was born?")
    assert ACTIONS["keep"](ctx).cost == 0
    expanded = ACTIONS["expand"](ctx)
    assert len(expanded.passages) == 4
    assert expanded.cost == 1
    fb = ACTIONS["feedback"](ctx)
    assert fb.cost == 1
    assert "expanded the query" in fb.detail
    assert ACTIONS["rewrite"](ctx).detail.startswith("expanded")  # no rewriter: falls back
    rw = ACTIONS["rewrite"](context("Einstein birthplace river", rewriter=lambda q: "Danube Ulm"))
    assert rw.cost == 2
    assert "ulm" in {p.id for p in rw.passages}  # "Ulm ... on the Danube" matches both terms
    dec = ACTIONS["decompose"](
        context("Where was Einstein born, and which river flows through Ulm?")
    )
    assert {"einstein", "ulm"} <= {p.id for p in dec.passages}
    llm = ACTIONS["decompose"](context("q", decomposer=lambda q: ["Einstein born", "Ulm river"]))
    assert llm.cost == 3
    other = BM25()
    other.index(CORPUS[::-1])
    assert ACTIONS["switch"](context("France capital", alternates=[other])).cost == 1
    assert ACTIONS["switch"](context("France capital")).detail.startswith("retrieved 4")


def test_merge_and_split_helpers() -> None:
    a, b, c = (Passage(x, x) for x in "abc")
    assert [p.id for p in _merge([a, b], [b, c], k=5)] == ["a", "b", "c"]
    assert [p.id for p in _merge([a, b], [c], k=2)] == ["a", "c"]
    assert split_question("Who wrote it, and when was it published?") == [
        "Who wrote it",
        "when was it published",
    ]
    assert split_question("Who wrote Hamlet?") == ["Who wrote Hamlet?"]


def test_learned_policy_beats_fixed_actions_on_synthetic_logs() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=(n, 3))
    outcomes = {
        "keep": (x[:, 0] > 0.5).astype(int),  # already enough when feature 0 is high
        "expand": (x[:, 1] > 0.0).astype(int),
        "decompose": (x[:, 2] > -0.5).astype(int),
        "feedback": np.zeros(n, dtype=int),
        "hop": np.zeros(n, dtype=int),
        "switch": np.zeros(n, dtype=int),
    }
    policy = LearnedPolicy(cost_weight=0.02).fit(
        x[:1000], {a: o[:1000] for a, o in outcomes.items()}
    )
    test = {a: o[1000:] for a, o in outcomes.items()}
    learned = evaluate_choices([d.action for d in policy.decide_matrix(x[1000:])], test)
    for action in ("keep", "expand", "decompose"):
        assert learned["success"] > evaluate_choices([action] * 500, test)["success"]
    best = oracle(test)
    assert evaluate_choices(best, test)["success"] >= learned["success"]
    assert policy.models["feedback"] is None  # constant outcome uses the base rate
    cautious = LearnedPolicy(abstain_below=0.99).fit(
        x[:1000], {a: o[:1000] for a, o in outcomes.items()}
    )
    assert cautious.decide(x[1000]).action in {"abstain", *cautious.actions}
    assert set(policy.decide(x[0]).scores) == set(policy.actions)


def test_baselines_and_evaluation() -> None:
    assert rule_cascade(0.9) == "keep"
    assert rule_cascade(0.5) == "feedback"
    assert rule_cascade(0.1) == "expand"
    outcomes = {"keep": [0, 1], "expand": [1, 1]}
    assert oracle(outcomes, {"keep": 0.0, "expand": 1.0}) == ["expand", "keep"]
    assert oracle({"keep": [0]}, {"keep": 0.0}) == ["abstain"]
    stats = evaluate_choices(["keep", "abstain"], outcomes, COSTS)
    assert stats == {"success": 0.0, "abstain": 0.5, "cost": 0.0}
    assert evaluate_choices([], outcomes) == {"success": 0.0, "abstain": 0.0, "cost": 0.0}
    assert COSTS["decompose"] == pytest.approx(3.0)


def test_decompose_fills_placeholders_with_earlier_answers() -> None:
    def answer(query: str, passages: list[Passage]) -> str:
        return "Ulm" if passages and passages[0].id == "einstein" else "Danube"

    ctx = context(
        "Which river flows through the city where Einstein was born?",
        k=1,
        decomposer=lambda q: ["Einstein >> place of birth", "Which river flows through #1?"],
        answerer=answer,
    )
    out = ACTIONS["decompose"](ctx)
    assert {"einstein", "ulm"} <= {p.id for p in out.passages}
    assert "bridges: Ulm" in out.detail
    assert out.cost == 1 + 2 + 2  # decomposition, two retrievals, two answers


def test_hop_follows_names_from_the_top_passages() -> None:
    ctx = context("Which river flows through the city where Einstein was born?", k=1)
    out = ACTIONS["hop"](ctx)
    assert "ulm" in {p.id for p in out.passages}  # "Ulm" is named in Einstein's passage
    assert out.cost >= 1
    assert out.detail.startswith("followed")
    bare = Context("q", [Passage("x", "no names here")], ctx.retriever, 1)
    assert ACTIONS["hop"](bare).detail.startswith("expanded")  # nothing to follow
