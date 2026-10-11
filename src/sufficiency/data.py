"""Question-answering datasets with answerability labels (needs the ``bench`` extra).

=============  ===============================================================  ==========
dataset        what makes a context insufficient                                 hops
=============  ===============================================================  ==========
SQuAD v2       crowd-written questions the paragraph does not answer            1
MuSiQue        unanswerable variants with a supporting paragraph replaced       2 to 4
HotpotQA       gold paragraphs removed from the ten-paragraph distractor set   2
=============  ===============================================================  ==========

Files are downloaded once from the Hugging Face parquet exports into
``$SUFFICIENCY_CACHE`` (default ``~/.cache/sufficiency``). They are used for evaluation
only and are not redistributed.
"""

from __future__ import annotations

import os
import random
import urllib.request
from pathlib import Path
from typing import Any

from sufficiency.model import Example, Passage

SOURCES = {
    (
        "squad_v2",
        "validation",
    ): "rajpurkar/squad_v2/resolve/refs%2Fconvert%2Fparquet/squad_v2/validation/0000.parquet",
    (
        "squad_v2",
        "train",
    ): "rajpurkar/squad_v2/resolve/refs%2Fconvert%2Fparquet/squad_v2/train/0000.parquet",
    (
        "musique",
        "validation",
    ): "dgslibisey/MuSiQue/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    (
        "musique",
        "train",
    ): "dgslibisey/MuSiQue/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    (
        "hotpotqa",
        "validation",
    ): "hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet",
}


def cache_dir() -> Path:
    return Path(os.environ.get("SUFFICIENCY_CACHE", Path.home() / ".cache" / "sufficiency"))


def _frame(name: str, split: str) -> Any:
    import pandas as pd

    path = cache_dir() / f"{name}-{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/{SOURCES[(name, split)]}", path
        )
    return pd.read_parquet(path)


def squad_v2(split: str = "validation", limit: int | None = None, seed: int = 0) -> list[Example]:
    frame = _frame("squad_v2", split)
    frame = frame.sample(frac=1.0, random_state=seed)
    out = []
    for row in frame.itertuples():
        answers = list(dict.fromkeys(row.answers["text"]))
        out.append(
            Example(
                id=str(row.id),
                question=str(row.question),
                passages=[Passage(f"{row.id}-ctx", str(row.context), str(row.title))],
                answers=answers,
                answerable=bool(answers),
                supporting=[f"{row.id}-ctx"] if answers else [],
                source="squad_v2",
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def musique(split: str = "validation", limit: int | None = None, seed: int = 0) -> list[Example]:
    frame = _frame("musique", split).sample(frac=1.0, random_state=seed)
    out = []
    for row in frame.itertuples():
        passages = [
            Passage(f"{row.id}-{p['idx']}", str(p["paragraph_text"]), str(p["title"]))
            for p in row.paragraphs
        ]
        supporting = [f"{row.id}-{p['idx']}" for p in row.paragraphs if p["is_supporting"]]
        answers = [str(row.answer), *[str(a) for a in row.answer_aliases]]
        out.append(
            Example(
                id=str(row.id),
                question=str(row.question),
                passages=passages,
                answers=list(dict.fromkeys(a for a in answers if a)),
                answerable=bool(row.answerable),
                supporting=supporting,
                sub_questions=[str(d["question"]) for d in row.question_decomposition],
                source="musique",
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def hotpotqa(
    split: str = "validation", limit: int | None = None, seed: int = 0, drop_gold: float = 0.5
) -> list[Example]:
    """HotpotQA distractor questions; a ``drop_gold`` share have their gold paragraphs removed."""
    frame = _frame("hotpotqa", split).sample(frac=1.0, random_state=seed)
    rng = random.Random(seed)
    out = []
    for row in frame.itertuples():
        titles = list(row.context["title"])
        gold = set(row.supporting_facts["title"])
        passages = [
            Passage(f"{row.id}-{i}", " ".join(s).strip(), t)
            for i, (t, s) in enumerate(zip(titles, row.context["sentences"], strict=True))
        ]
        removed = rng.random() < drop_gold
        if removed:
            passages = [p for p in passages if p.title not in gold]
        out.append(
            Example(
                id=str(row.id),
                question=str(row.question),
                passages=passages,
                answers=[str(row.answer)],
                answerable=not removed,
                supporting=[p.id for p in passages if p.title in gold],
                source="hotpotqa",
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


LOADERS = {"squad_v2": squad_v2, "musique": musique, "hotpotqa": hotpotqa}
