"""Benchmark 2: answer or abstain, with a guaranteed error rate.

Uses the same held-out questions as ``estimate`` (and the estimators it saved). Each
question is answered from its top ``k`` passages. The answer counts as correct only if
the question is answerable from those passages and the answer matches. For a
question the passages cannot answer, any answer is wrong; abstaining is the right
behaviour.

The test questions are split in half: the first half calibrates each selection rule,
and the second half measures it. For each scorer (the calibrated estimator, the
hand-weighted blend, the reader margin) the benchmark reports:

* the area under the risk-coverage curve (lower is better);
* for each target error rate ``alpha``, the learn-then-test threshold's achieved error
  among answered questions and its coverage on the held-out half;
* the same for conformal risk control (answered-and-wrong rate).

    sufficiency bench selective --datasets squad_v2 musique hotpotqa --generator extractive
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sufficiency.bench.estimate import sufficient, top_k
from sufficiency.data import LOADERS
from sufficiency.estimator import SufficiencyEstimator
from sufficiency.features import Featurizer, LexicalFeatures, ReaderFeatures, RetrievalFeatures
from sufficiency.generate import Extractive, LocalChat, OpenAIChat
from sufficiency.metrics import area_under_risk_coverage, contains, f1
from sufficiency.selective import SelectiveRule, evaluate_rule

ALPHAS = (0.1, 0.2, 0.3, 0.5)


def correct(answer: str, answers: Sequence[str], answerable_here: bool) -> bool:
    if not answerable_here or not answer.strip():
        return False
    return contains(answer, answers) or f1(answer, answers) >= 0.5


def make_generator(spec: str) -> Any:
    if spec == "extractive":
        return Extractive()
    if spec.startswith("openai:"):
        return OpenAIChat(spec.split(":", 1)[1])
    return LocalChat(spec.removeprefix("local:"))


def run_dataset(name: str, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    print(f"== {name}", flush=True)
    est_dir = Path(args.estimators)
    featurizer = Featurizer(RetrievalFeatures(), LexicalFeatures(), ReaderFeatures())
    estimator = SufficiencyEstimator(featurizer).load(est_dir / f"estimator-{name}.joblib")
    cached = np.load(est_dir / f"features-{name}.npz")
    x, hand = cached["test_x"], cached["test_hand"]
    examples = LOADERS[name]("validation", limit=None, seed=args.seed)
    # HotpotQA has no labelled train split here, so estimate trained on the first
    # `train` validation questions and tested on the next ones.
    start = args.train if name == "hotpotqa" else 0
    examples = examples[start : start + len(x)]
    answers_file = out / f"answers-{name}-{args.generator.replace('/', '_').replace(':', '_')}.json"
    partial = answers_file.with_suffix(".partial.json")
    if answers_file.exists():
        generated = json.loads(answers_file.read_text())
    else:  # resumable: answers are saved every 100 questions
        generated = json.loads(partial.read_text()) if partial.exists() else []
        generator = make_generator(args.generator) if len(generated) < len(examples) else None
        for n, ex in enumerate(examples[len(generated) :], start=len(generated)):
            assert generator is not None
            generated.append(generator(ex.question, top_k(ex, args.k)))
            if (n + 1) % 100 == 0:
                partial.write_text(json.dumps(generated))
                print(f"    answered {n + 1}/{len(examples)}", flush=True)
        answers_file.write_text(json.dumps(generated))
        partial.unlink(missing_ok=True)
    ok = np.array(
        [
            correct(a, ex.answers, sufficient(ex, top_k(ex, args.k)))
            for a, ex in zip(generated, examples, strict=True)
        ]
    )
    wrong = ~ok
    scorers = {
        "estimator": estimator.predict_matrix(x),
        "hand-weighted": hand,
        "reader margin": x[:, featurizer.names.index("reader_margin_max")],
    }
    half = len(ok) // 2
    np.savez(out / f"scores-{name}.npz", ok=ok, half=half, **scorers)
    rows = []
    for scorer, s in scorers.items():
        row: dict[str, Any] = {
            "scorer": scorer,
            "aurc": area_under_risk_coverage(s[half:], ok[half:]),
        }
        for alpha in ALPHAS:
            for method in ("ltt", "crc"):
                rule = SelectiveRule.fit(s[:half], wrong[:half], alpha, method=method, delta=0.1)
                stats = evaluate_rule(rule, s[half:], wrong[half:])
                row[f"{method}@{alpha}"] = stats
        rows.append(row)
        ltt = row["ltt@0.2"]
        print(
            f"  {scorer:14s} AURC {row['aurc']:.3f}  LTT@0.2 error {ltt['selective_error']:.3f} "
            f"coverage {ltt['coverage']:.3f}",
            flush=True,
        )
    return {
        "dataset": name,
        "generator": args.generator,
        "questions": len(ok),
        "accuracy_if_always_answering": float(ok.mean()),
        "rows": rows,
    }


def markdown(summaries: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Scorer | AURC | LTT 20%: error | coverage | LTT 50%: error | coverage | "
        "CRC 10%: answered and wrong | coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        for r in s["rows"]:
            a, b, c = r["ltt@0.2"], r["ltt@0.5"], r["crc@0.1"]
            lines.append(
                f"| {s['dataset']} | {r['scorer']} | {r['aurc']:.3f} | "
                f"{a['selective_error']:.1%} | {a['coverage']:.1%} | "
                f"{b['selective_error']:.1%} | {b['coverage']:.1%} | "
                f"{c['answered_and_wrong']:.1%} | {c['coverage']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="sufficiency bench selective", description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["squad_v2", "musique", "hotpotqa"])
    parser.add_argument("--generator", default="extractive")
    parser.add_argument("--estimators", default="runs/estimate")
    parser.add_argument("--train", type=int, default=1500, help="as passed to estimate")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/selective")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [run_dataset(d, args, out) for d in args.datasets]
    (out / "results.json").write_text(json.dumps(summaries, indent=1))
    table = markdown(summaries)
    (out / "results.md").write_text(table)
    print(table)
    return 0
