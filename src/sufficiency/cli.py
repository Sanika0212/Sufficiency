"""Command line: ``sufficiency score``, ``sufficiency ask`` and ``sufficiency bench``.

    sufficiency score --estimator runs/estimate/estimator-musique.joblib \\
        --question "Who founded the company that makes the Model S?" --passages passages.json
    sufficiency ask --corpus corpus.jsonl --estimator runs/estimate/estimator-musique.joblib \\
        --threshold 0.6 --question "..."
    sufficiency bench estimate --datasets musique --train 1500 --test 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from sufficiency import __version__
from sufficiency.model import Passage

if TYPE_CHECKING:
    from sufficiency.estimator import SufficiencyEstimator


def read_passages(path: str | Path) -> list[Passage]:
    """Passages from a JSON list or JSON lines: strings, or objects with ``text``."""
    raw = Path(path).read_text()
    stripped = raw.lstrip()
    items = (
        json.loads(raw)
        if stripped.startswith("[")
        else [json.loads(line) for line in raw.splitlines() if line.strip()]
    )
    passages = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        passages.append(
            Passage(
                id=str(item.get("id", i)),
                text=item["text"],
                title=item.get("title", ""),
                score=float(item.get("score", 0.0)),
            )
        )
    return passages


def _estimator(path: str) -> SufficiencyEstimator:
    from sufficiency.estimator import SufficiencyEstimator
    from sufficiency.features import (
        Featurizer,
        LexicalFeatures,
        ReaderFeatures,
        RetrievalFeatures,
    )

    featurizer = Featurizer(RetrievalFeatures(), LexicalFeatures(), ReaderFeatures())
    return SufficiencyEstimator(featurizer).load(path)


def score(args: argparse.Namespace) -> int:
    from sufficiency.retrieval import BM25

    estimator = _estimator(args.estimator)
    passages = read_passages(args.passages)
    if not any(p.score for p in passages):  # no retrieval scores given: rank with BM25
        index = BM25()
        index.index(passages)
        passages = index.search(args.question, len(passages))
    p = estimator.predict(args.question, passages[: args.k])
    print(json.dumps({"question": args.question, "sufficiency": round(p, 4)}))
    return 0


def ask(args: argparse.Namespace) -> int:
    from sufficiency.controller import Controller
    from sufficiency.generate import Extractive, LocalChat, OpenAIChat
    from sufficiency.retrieval import BM25
    from sufficiency.selective import SelectiveRule

    retriever = BM25()
    retriever.index(read_passages(args.corpus))
    generator = (
        Extractive()
        if args.generator == "extractive"
        else OpenAIChat(args.generator.split(":", 1)[1])
        if args.generator.startswith("openai:")
        else LocalChat(args.generator.removeprefix("local:"))
    )
    controller = Controller(
        retriever,
        _estimator(args.estimator),
        SelectiveRule(args.threshold, "fixed", 0.0),
        generator,
        k=args.k,
    )
    result = controller.answer(args.question)
    for step in result.steps:
        level = "" if step.sufficiency is None else f"sufficiency {step.sufficiency:.2f}  "
        print(f"{step.kind:9s} {level}{step.detail}")
    print()
    print("not enough information" if result.abstained else result.answer)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["bench"]:
        from sufficiency.bench import run

        if len(argv) < 2:
            print("usage: sufficiency bench {estimate,selective,recover} [options]")
            return 2
        return run(argv[1], argv[2:])
    parser = argparse.ArgumentParser(prog="sufficiency", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"sufficiency {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="estimate whether some passages are enough to answer")
    s.add_argument("--estimator", required=True)
    s.add_argument("--question", required=True)
    s.add_argument("--passages", required=True, help="JSON list or JSON lines")
    s.add_argument("--k", type=int, default=5)
    s.set_defaults(func=score)
    a = sub.add_parser("ask", help="retrieve, judge sufficiency, answer or abstain")
    a.add_argument("--corpus", required=True, help="JSON list or JSON lines of passages")
    a.add_argument("--estimator", required=True)
    a.add_argument("--question", required=True)
    a.add_argument("--threshold", type=float, default=0.5)
    a.add_argument("--generator", default="extractive")
    a.add_argument("--k", type=int, default=5)
    a.set_defaults(func=ask)
    sub.add_parser("bench", help="run a benchmark suite: estimate, selective or recover")
    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
