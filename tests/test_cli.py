import json
from pathlib import Path
from typing import Any

import pytest

from sufficiency import cli


class FakeEstimator:
    def predict(self, question: str, passages: Any) -> float:
        return 0.9 if any("Paris" in p.text for p in passages) else 0.1


def test_read_passages_json_and_jsonl(tmp_path: Path) -> None:
    listed = tmp_path / "a.json"
    listed.write_text(json.dumps(["plain text", {"text": "with title", "title": "T", "id": "x"}]))
    lines = tmp_path / "b.jsonl"
    lines.write_text('{"text": "one"}\n\n{"text": "two", "score": 2}\n')
    a = cli.read_passages(listed)
    b = cli.read_passages(lines)
    assert [p.id for p in a] == ["0", "x"]
    assert a[1].title == "T"
    assert [p.text for p in b] == ["one", "two"]
    assert b[1].score == 2.0


def test_score_ranks_and_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_estimator", lambda path: FakeEstimator())
    passages = tmp_path / "p.json"
    passages.write_text(json.dumps(["Paris is the capital of France.", "Rome is in Italy."]))
    code = cli.main(
        [
            "score",
            "--estimator",
            "m",
            "--question",
            "capital of France",
            "--passages",
            str(passages),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["sufficiency"] == 0.9


def test_bench_without_suite_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["bench"]) == 2
    assert "usage" in capsys.readouterr().out
