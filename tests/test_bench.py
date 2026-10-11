"""The benchmark suites end to end on toy data, with a scripted reader (no downloads)."""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from sufficiency import data
from sufficiency.bench import estimate, recover, run, selective
from sufficiency.model import Example, Passage
from sufficiency.retrieval import BM25


class ScriptedReader:
    """Margin is high when a passage states the asked-about capital, low otherwise."""

    names = ("reader_margin_max", "reader_margin_mean", "reader_span_prob", "reader_agree")
    model_id = "scripted"

    def spans(self, question: str, passages: Sequence[Passage]) -> list[tuple[float, float, str]]:
        """The last word of each passage, as if it were the answer."""
        return [(1.0, 0.5, p.text.rstrip(".").rsplit(" ", 1)[-1]) for p in passages]

    def compute(self, question: str, passages: Sequence[Passage]) -> list[float]:
        place = question.rstrip("?").split()[-1]
        found = any(f"capital of {place} is" in p.text for p in passages)
        noise = random.Random(question).uniform(-1, 1)
        margin = (3.0 if found else -3.0) + noise
        return [margin, margin / 2, 0.9 if found else 0.1, 1.0]


class Answerer:
    name = "scripted"

    def __call__(self, question: str, passages: Sequence[Passage]) -> str:
        place = question.rstrip("?").split()[-1]
        for p in passages:
            if f"capital of {place} is" in p.text:
                return p.text.rsplit(" ", 1)[-1].rstrip(".")
        return "unknown"


def toy(split: str = "validation", limit: int | None = None, seed: int = 0) -> list[Example]:
    rng = random.Random(f"{split}-{seed}")
    out = []
    for i in range(limit or 120):
        place, city = f"Land{i}", f"City{i}"
        gold = Passage(f"{i}-gold", f"The capital of {place} is {city}.", place)
        others = [
            Passage(f"{i}-d{j}", f"{place} has rivers and hills number {j}.", place)
            for j in range(3)
        ]
        answerable = rng.random() < 0.5
        passages = [gold, *others] if answerable else others
        out.append(
            Example(
                id=str(i),
                question=f"What is the capital of {place}?",
                passages=passages,
                answers=[city],
                answerable=answerable,
                supporting=[gold.id] if answerable else [],
                source="toy",
            )
        )
    return out


def multihop(split: str = "validation", limit: int | None = None, seed: int = 0) -> list[Example]:
    out = []
    for i in range(limit or 80):
        person, town = f"Person{i}", f"Town{i}"
        first = Passage(f"{i}-a", f"{person} was born in {town}.", person)
        second = Passage(f"{i}-b", f"{town} lies on the river River{i}.", town)
        noise = Passage(f"{i}-c", f"{person} wrote books about {town} history.", person)
        out.append(
            Example(
                id=str(i),
                question=f"Which river flows through the town where {person} was born?",
                passages=[first, second, noise],
                answers=[f"River{i}"],
                answerable=True,
                supporting=[first.id, second.id],
                sub_questions=[f"Where was {person} born?", f"Which river flows through {town}?"],
                source="toy",
            )
        )
    return out


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (estimate, selective, recover):
        monkeypatch.setattr(module, "ReaderFeatures", ScriptedReader)
    monkeypatch.setitem(estimate.LOADERS, "toy", toy)
    monkeypatch.setattr(selective, "make_generator", lambda spec: Answerer())


def test_hand_weighted_and_sufficient() -> None:
    ex = toy(limit=4)[0]
    passages = estimate.top_k(ex, 2)
    assert len(passages) == 2
    assert 0.0 <= estimate.hand_weighted(ex.question, passages) <= 1.0
    assert estimate.hand_weighted(ex.question, []) == 0.0
    assert estimate.sufficient(ex, ex.passages) == ex.answerable
    single = Example("s", "q", [Passage("p", "text")], ["a"], True, ["p"])
    assert estimate.top_k(single, 5)[0].score == 1.0


def test_estimate_selective_pipeline(scripted: None, tmp_path: Path) -> None:
    out = tmp_path / "estimate"
    args = ["--datasets", "toy", "--train", "80", "--test", "80", "--out", str(out)]
    assert run("estimate", args) == 0
    summary = json.loads((out / "results.json").read_text())[0]
    scores = {r["scorer"]: r["auroc"] for r in summary["results"]}
    assert scores["estimator"] > 0.9  # the scripted reader separates the classes
    assert "| toy | estimator |" in (out / "results.md").read_text()
    assert run("estimate", args) == 0  # second run reads the cached features

    sel = tmp_path / "selective"
    argv = ["--datasets", "toy", "--estimators", str(out), "--train", "80", "--out", str(sel)]
    assert selective.main(argv) == 0
    result = json.loads((sel / "results.json").read_text())[0]
    assert result["questions"] == 80
    assert {r["scorer"] for r in result["rows"]} == {"estimator", "hand-weighted", "reader margin"}
    assert selective.main(argv) == 0  # cached answers


def test_selective_correctness_rule() -> None:
    assert selective.correct("City3", ["City3"], True)
    assert not selective.correct("City3", ["City3"], False)
    assert not selective.correct("  ", ["City3"], True)


def test_make_generator_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert selective.make_generator("openai:gpt-4.1-mini").name == "openai:gpt-4.1-mini"
    assert selective.make_generator("local:tiny/model").name == "local:tiny/model"
    assert selective.make_generator("extractive").name.startswith("extractive:")


def test_recover_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class SecondBM25(BM25):
        name = "dense"

    monkeypatch.setattr(recover, "musique", multihop)
    monkeypatch.setattr(recover, "Dense", SecondBM25)
    monkeypatch.setattr(recover, "ReaderFeatures", ScriptedReader)

    class Planner:
        name = "planner"

        def complete(self, messages, max_new_tokens=None):  # type: ignore[no-untyped-def]
            person = messages[-1]["content"].split("where ")[1].split(" was")[0]
            return f"1. Where was {person} born?\n2. Which river flows through #1?"

    monkeypatch.setattr(selective, "make_generator", lambda spec: Planner())
    out = tmp_path / "recover"
    assert recover.main(["--questions", "80", "--k", "2", "--out", str(out)]) == 0
    assert list(out.glob("steps-*.json"))  # decompositions are cached
    assert recover.main(["--questions", "80", "--k", "2", "--out", str(out)]) == 0
    plain = tmp_path / "plain"
    assert recover.main(["--questions", "40", "--decomposer", "none", "--out", str(plain)]) == 0
    summary = json.loads((out / "results.json").read_text())
    rows = {r["policy"]: r for r in summary["rows"]}
    assert rows["always keep"]["cost"] == 0.0
    assert rows["oracle (deployable actions)"]["success"] >= rows["always keep"]["success"]
    assert rows["always decompose-gold"]["success"] >= rows["always keep"]["success"]
    assert rows["always decompose-gold"]["recall"] >= rows["always keep"]["recall"]
    assert 0.0 <= rows["always hop"]["recall"] <= 1.0
    assert rows["always decompose-llm"]["success"] >= rows["always keep"]["success"]
    assert "| always keep |" in (out / "results.md").read_text()


def test_loaders_parse_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {
        "squad_v2": pd.DataFrame(
            {
                "id": ["a", "b"],
                "title": ["T", "T"],
                "context": ["Paris is in France.", "Rome is in Italy."],
                "question": ["Where is Paris?", "Who is Bob?"],
                "answers": [{"text": ["France", "France"]}, {"text": []}],
            }
        ),
        "musique": pd.DataFrame(
            {
                "id": ["m"],
                "question": ["Q?"],
                "paragraphs": [
                    [
                        {"idx": 0, "title": "A", "paragraph_text": "x", "is_supporting": True},
                        {"idx": 1, "title": "B", "paragraph_text": "y", "is_supporting": False},
                    ]
                ],
                "answer": ["x"],
                "answer_aliases": [["ex"]],
                "answerable": [True],
                "question_decomposition": [[{"question": "q1"}, {"question": "q2"}]],
            }
        ),
        "hotpotqa": pd.DataFrame(
            {
                "id": ["h1", "h2"],
                "question": ["Q1?", "Q2?"],
                "answer": ["a", "b"],
                "context": [
                    {"title": ["G", "D"], "sentences": [["Gold one."], ["Other."]]},
                    {"title": ["G", "D"], "sentences": [["Gold two."], ["Other."]]},
                ],
                "supporting_facts": [{"title": ["G"]}, {"title": ["G"]}],
            }
        ),
    }
    monkeypatch.setattr(data, "_frame", lambda name, split: frames[name])
    squad = data.squad_v2(seed=0)
    assert sorted(e.answerable for e in squad) == [False, True]
    assert data.squad_v2(limit=1)[0].source == "squad_v2"
    m = data.musique()[0]
    assert m.supporting == ["m-0"]
    assert m.sub_questions == ["q1", "q2"]
    assert m.answers == ["x", "ex"]
    assert len(data.musique(limit=1)) == 1
    kept = data.hotpotqa(drop_gold=0.0)
    assert all(e.answerable and e.supporting for e in kept)
    dropped = data.hotpotqa(drop_gold=1.0, limit=1)
    assert not dropped[0].answerable
    assert dropped[0].supporting == []
    assert all(p.title != "G" for p in dropped[0].passages)


def test_cache_dir_and_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUFFICIENCY_CACHE", str(tmp_path))
    assert data.cache_dir() == tmp_path

    def fake_download(url: str, path: Path) -> None:
        assert url.startswith("https://huggingface.co/datasets/")
        pd.DataFrame({"x": [1]}).to_parquet(path)

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake_download)
    assert list(data._frame("squad_v2", "validation")["x"]) == [1]
    assert (tmp_path / "squad_v2-validation.parquet").exists()
