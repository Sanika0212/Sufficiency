"""Benchmark 3: when the context is not enough, which fix works, and at what cost?

Open retrieval over a corpus pooled from the paragraphs of many MuSiQue questions (2 to
4 hops). Each question starts from BM25's top ``k``. Every recovery action is run once
per question, and its outcome logged: success means every supporting paragraph's text
is in the new context.

Policies compared on held-out questions:

* each fixed action applied to every question (``keep`` is no correction);
* a rule cascade on a hand-weighted confidence score;
* Sufficiency's learned policy at several cost weights (accuracy against cost);
* the oracle: the cheapest action that worked, in hindsight.

``decompose-gold`` uses MuSiQue's annotated sub-questions. It is an upper bound for
decomposition, not something a deployed system has.

    sufficiency bench recover --questions 1000 --k 5
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sufficiency.actions import ACTIONS, Context
from sufficiency.bench.estimate import hand_weighted
from sufficiency.data import musique
from sufficiency.features import Featurizer, LexicalFeatures, ReaderFeatures, RetrievalFeatures
from sufficiency.metrics import normalize
from sufficiency.model import Example, Passage
from sufficiency.policy import COSTS, LearnedPolicy, evaluate_choices, oracle, rule_cascade
from sufficiency.retrieval import BM25, Dense

ACTION_NAMES = ("keep", "expand", "feedback", "decompose", "switch", "decompose-gold")


def key(passage: Passage) -> str:
    return normalize(passage.text)[:300]


def covers(example: Example, passages: Sequence[Passage], gold: dict[str, set[str]]) -> bool:
    have = {key(p) for p in passages}
    return gold[example.id] <= have


def gold_steps(example: Example) -> Callable[[str], list[str]]:
    """A decomposer that returns MuSiQue's annotated sub-questions."""

    def steps(question: str) -> list[str]:
        return list(example.sub_questions) or [question]

    return steps


def run(args: argparse.Namespace) -> dict[str, Any]:
    examples = musique("validation", limit=args.questions, seed=args.seed)
    corpus = [p for ex in examples for p in ex.passages]
    gold = {ex.id: {key(p) for p in ex.passages if p.id in set(ex.supporting)} for ex in examples}
    print(f"{len(examples)} questions, corpus of {len(corpus)} paragraphs", flush=True)
    bm25 = BM25()
    bm25.index(corpus)
    dense = Dense()
    dense.index(corpus)
    featurizer = Featurizer(RetrievalFeatures(), LexicalFeatures(), ReaderFeatures())
    costs = {**COSTS, "decompose-gold": 2.0}
    feats, hand = [], []
    outcomes: dict[str, list[int]] = {a: [] for a in ACTION_NAMES}
    spent: dict[str, list[float]] = {a: [] for a in ACTION_NAMES}
    sizes: dict[str, list[int]] = {a: [] for a in ACTION_NAMES}
    for n, ex in enumerate(examples):
        start = bm25.search(ex.question, args.k)
        feats.append(featurizer(ex.question, start))
        hand.append(hand_weighted(ex.question, start))
        for action in ACTION_NAMES:
            if action == "decompose-gold":
                ctx = Context(ex.question, start, bm25, args.k, decomposer=gold_steps(ex))
                result = ACTIONS["decompose"](ctx)
            else:
                ctx = Context(ex.question, start, bm25, args.k, alternates=[dense])
                result = ACTIONS[action](ctx)
            outcomes[action].append(int(covers(ex, result.passages, gold)))
            spent[action].append(result.cost)
            sizes[action].append(len(result.passages))
        if (n + 1) % 100 == 0:
            print(f"    {n + 1}/{len(examples)}", flush=True)
    x = np.stack(feats)
    half = len(examples) // 2
    test = {a: np.asarray(o[half:]) for a, o in outcomes.items()}
    deployable = [a for a in ACTION_NAMES if a != "decompose-gold"]

    def summarize(policy: str, choices: Sequence[str]) -> dict[str, Any]:
        stats: dict[str, Any] = {"policy": policy, **evaluate_choices(choices, test, costs)}
        idx = [(i, c) for i, c in enumerate(choices) if c != "abstain"]
        # measured cost (retrieval and LLM calls actually made) and final context size
        stats["cost"] = sum(spent[c][half + i] for i, c in idx) / max(len(choices), 1)
        stats["passages"] = sum(sizes[c][half + i] for i, c in idx) / max(len(idx), 1)
        stats["mix"] = {a: choices.count(a) / len(choices) for a in sorted(set(choices))}
        return stats

    n_test = len(test["keep"])
    rows = [summarize(f"always {a}", [a] * n_test) for a in ACTION_NAMES]
    cascade = [rule_cascade(h, low=args.rule_low, high=args.rule_high) for h in hand[half:]]
    rows.append(summarize("rule cascade (hand-weighted)", cascade))
    for weight in (0.0, 0.02, 0.05, 0.1, 0.2):
        policy = LearnedPolicy(deployable, costs=costs, cost_weight=weight)
        policy.fit(x[:half], {a: outcomes[a][:half] for a in deployable})
        choices = [d.action for d in policy.decide_matrix(x[half:])]
        rows.append(summarize(f"learned (cost weight {weight})", choices))
    best = oracle({a: test[a] for a in deployable}, costs)
    rows.append(summarize("oracle (deployable actions)", best))
    return {
        "questions": len(examples),
        "test_questions": len(test["keep"]),
        "corpus": len(corpus),
        "k": args.k,
        "base_rates": {a: float(np.mean(o)) for a, o in outcomes.items()},
        "rows": rows,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"{summary['test_questions']} held-out MuSiQue questions; corpus of "
        f"{summary['corpus']} paragraphs; top {summary['k']} from BM25.",
        "",
        "| Policy | Context sufficient | Abstained | Calls per question | Passages |",
        "|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['policy']} | {r['success']:.1%} | {r['abstain']:.1%} | {r['cost']:.2f} | "
            f"{r['passages']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="sufficiency bench recover", description=__doc__)
    parser.add_argument("--questions", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rule-low", type=float, default=0.4)
    parser.add_argument("--rule-high", type=float, default=0.7)
    parser.add_argument("--out", default="runs/recover")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = run(args)
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    table = markdown(summary)
    (out / "results.md").write_text(table)
    print(table)
    return 0
