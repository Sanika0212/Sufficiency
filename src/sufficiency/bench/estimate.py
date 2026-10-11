"""Benchmark 1: does the context suffice? Discrimination and calibration of each scorer.

For each dataset, every question's passages are re-ranked with BM25 and the top ``k`` are
kept, as a retriever would deliver them. The label is *sufficient*: the question is
answerable and every supporting passage made it into the top ``k`` (for SQuAD v2,
simply "answerable").

Scorers compared:

* ``estimator``: Sufficiency's calibrated model (retrieval, lexical and reader features);
* ``estimator (no reader)``: the same without the reader model;
* ``hand-weighted``: a fixed-weight blend of top score, score drop-off, coherence between
  passages and question coverage, with hand-set weights (the common design);
* ``reader margin``: the extractive reader's span-vs-no-answer margin alone;
* ``retrieval score``: the top passage's retrieval score share alone.

The estimators are trained on a training split and evaluated on held-out questions.
Reported: AUROC, expected calibration error, Brier score. Hand-set scores are not
probabilities, so their ECE is reported only after mapping to [0, 1].

    sufficiency bench estimate --datasets squad_v2 musique hotpotqa --train 1500 --test 1000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sufficiency.data import LOADERS
from sufficiency.estimator import SufficiencyEstimator
from sufficiency.features import Featurizer, LexicalFeatures, ReaderFeatures, RetrievalFeatures
from sufficiency.metrics import auroc, expected_calibration_error
from sufficiency.model import Example, Passage
from sufficiency.retrieval import BM25, tokenize


def top_k(example: Example, k: int) -> list[Passage]:
    if len(example.passages) <= 1:
        return [p.with_score(1.0) for p in example.passages]
    index = BM25()
    index.index(example.passages)
    return index.search(example.question, k)


def sufficient(example: Example, passages: Sequence[Passage]) -> bool:
    if not example.answerable:
        return False
    ids = {p.id for p in passages}
    return all(s in ids for s in example.supporting)


def hand_weighted(question: str, passages: Sequence[Passage]) -> float:
    """Fixed-weight confidence: top score, drop-off, inter-passage coherence, coverage."""
    if not passages:
        return 0.0
    scores = np.array([p.score for p in passages], dtype=float)
    top = scores.max() / (scores.max() + 1.0)  # squash an unbounded retrieval score
    dropoff = (scores[0] - scores[1]) / scores[0] if len(scores) > 1 and scores[0] > 0 else 1.0
    sets = [set(tokenize(p.text)) for p in passages]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    coherence = np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs]) if pairs else 1.0
    terms = set(tokenize(question))
    coverage = len(terms & set().union(*sets)) / max(len(terms), 1)
    return float(0.25 * top + 0.20 * dropoff + 0.25 * coherence + 0.30 * coverage)


def rows(examples: Sequence[Example], k: int, featurizer: Featurizer) -> dict[str, Any]:
    feats, labels, hand = [], [], []
    for n, ex in enumerate(examples):
        passages = top_k(ex, k)
        feats.append(featurizer(ex.question, passages))
        labels.append(sufficient(ex, passages))
        hand.append(hand_weighted(ex.question, passages))
        if (n + 1) % 250 == 0:
            print(f"    featurised {n + 1}/{len(examples)}", flush=True)
    return {"x": np.stack(feats), "y": np.asarray(labels), "hand": np.asarray(hand)}


def scores_report(name: str, probs: np.ndarray, y: np.ndarray, calibrated: bool) -> dict[str, Any]:
    clipped = np.clip(probs, 0, 1)
    return {
        "scorer": name,
        "auroc": auroc(probs, y),
        "ece": expected_calibration_error(clipped, y) if calibrated else None,
        "brier": float(np.mean((clipped - y) ** 2)) if calibrated else None,
    }


def run_dataset(
    name: str, train_n: int, test_n: int, k: int, seed: int, out: Path
) -> dict[str, Any]:
    print(f"== {name}", flush=True)
    featurizer = Featurizer(RetrievalFeatures(), LexicalFeatures(), ReaderFeatures())
    split = "train" if name in ("squad_v2", "musique") else "validation"
    loader = LOADERS[name]
    if split == "train":
        train_ex = loader("train", limit=train_n, seed=seed)
        test_ex = loader("validation", limit=test_n, seed=seed)
    else:
        pool = loader("validation", limit=train_n + test_n, seed=seed)
        train_ex, test_ex = pool[:train_n], pool[train_n:]
    cache = out / f"features-{name}.npz"
    if cache.exists():
        data = np.load(cache)
        train = {k_: data[f"train_{k_}"] for k_ in ("x", "y", "hand")}
        test = {k_: data[f"test_{k_}"] for k_ in ("x", "y", "hand")}
    else:
        train, test = rows(train_ex, k, featurizer), rows(test_ex, k, featurizer)
        np.savez(
            cache,
            **{f"train_{a}": b for a, b in train.items()},
            **{f"test_{a}": b for a, b in test.items()},
        )
    names = featurizer.names
    no_reader = [i for i, n in enumerate(names) if not n.startswith("reader_")]
    full = SufficiencyEstimator(featurizer).fit(train["x"], train["y"])
    lean = SufficiencyEstimator(Featurizer(RetrievalFeatures(), LexicalFeatures()))
    lean.fit(train["x"][:, no_reader], train["y"])
    y = test["y"]
    reader_col = names.index("reader_margin_max")
    share_col = names.index("ret_top_share")
    results = [
        scores_report("estimator", full.predict_matrix(test["x"]), y, True),
        scores_report(
            "estimator (no reader)", lean.predict_matrix(test["x"][:, no_reader]), y, True
        ),
        scores_report("hand-weighted", test["hand"], y, True),
        scores_report("reader margin", test["x"][:, reader_col], y, False),
        scores_report("retrieval score", test["x"][:, share_col], y, False),
    ]
    summary = {
        "dataset": name,
        "train": len(train["y"]),
        "test": len(y),
        "sufficient_share": float(y.mean()),
        "results": results,
        "coefficients": full.coefficients(),
    }
    full.save(out / f"estimator-{name}.joblib")
    for r in results:
        ece = "n/a" if r["ece"] is None else f"{r['ece']:.3f}"
        print(f"  {r['scorer']:24s} AUROC {r['auroc']:.3f}  ECE {ece}", flush=True)
    return summary


def markdown(summaries: Sequence[dict[str, Any]]) -> str:
    lines = ["| Dataset | Scorer | AUROC | ECE | Brier |", "|---|---|---|---|---|"]
    for s in summaries:
        for r in s["results"]:
            ece = "n/a" if r["ece"] is None else f"{r['ece']:.3f}"
            brier = "n/a" if r["brier"] is None else f"{r['brier']:.3f}"
            lines.append(f"| {s['dataset']} | {r['scorer']} | {r['auroc']:.3f} | {ece} | {brier} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="sufficiency bench estimate", description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["squad_v2", "musique", "hotpotqa"])
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/estimate")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_dataset(d, args.train, args.test, args.k, args.seed, out) for d in args.datasets
    ]
    (out / "results.json").write_text(json.dumps(summaries, indent=1))
    table = markdown(summaries)
    (out / "results.md").write_text(table)
    print(table)
    return 0
