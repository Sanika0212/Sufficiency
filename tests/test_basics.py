import numpy as np
import pytest

from sufficiency.metrics import (
    area_under_risk_coverage,
    auroc,
    contains,
    exact_match,
    expected_calibration_error,
    f1,
    normalize,
    risk_coverage,
)
from sufficiency.model import Passage, Result, Step
from sufficiency.retrieval import BM25, Hybrid, tokenize

PASSAGES = [
    Passage("p0", "The Eiffel Tower is 330 metres tall.", "Eiffel Tower"),
    Passage("p1", "Paris is the capital of France and its largest city.", "Paris"),
    Passage("p2", "The Louvre is the most visited museum in the world.", "Louvre"),
]


def test_answer_metrics() -> None:
    assert normalize("The  Eiffel-Tower!") == "eiffeltower"
    assert exact_match("the Paris", ["Paris"])
    assert not exact_match("Paris, France", ["Paris"])
    assert f1("Paris France", ["Paris"]) == pytest.approx(2 / 3)
    assert f1("nothing", ["Paris"]) == 0.0
    assert contains("It is Paris, France.", ["paris"])
    assert not contains("Parisian food", ["paris"])


def test_ranking_and_calibration_metrics() -> None:
    assert auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert np.isnan(auroc([1.0], [True]))
    perfect = expected_calibration_error([1.0, 0.0, 1.0], [1, 0, 1])
    assert perfect == 0.0
    off = expected_calibration_error([0.9] * 10, [1] * 5 + [0] * 5)
    assert off == pytest.approx(0.4)
    coverage, risk = risk_coverage([0.9, 0.8, 0.1], [True, False, False])
    assert coverage.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert risk.tolist() == pytest.approx([0.0, 0.5, 2 / 3])
    assert area_under_risk_coverage([0.9, 0.8, 0.1], [True, False, False]) == pytest.approx(7 / 18)


def test_bm25_and_hybrid_rank_the_right_passage() -> None:
    assert tokenize("What is the height of the tower?") == ["height", "tower"]
    bm25 = BM25()
    bm25.index(PASSAGES)
    top = bm25.search("How tall is the Eiffel Tower?", k=2)
    assert top[0].id == "p0"
    assert top[0].score > top[1].score
    assert BM25().search("anything") == []
    assert BM25().scores("x").shape == (0,)

    class Fixed:
        name = "fixed"

        def index(self, passages):  # type: ignore[no-untyped-def]
            self.passages = list(passages)

        def search(self, query, k=5):  # type: ignore[no-untyped-def]
            return [self.passages[2], self.passages[0]][:k]

    hybrid = Hybrid(bm25, Fixed())
    hybrid.index(PASSAGES)
    fused = hybrid.search("Eiffel Tower height", k=2)
    assert {p.id for p in fused} == {"p0", "p2"}
    assert fused[0].id == "p0"  # first for one retriever, second for the other
    assert hybrid.name == "bm25+fixed"


def test_result_cost() -> None:
    result = Result(
        "q", None, True, 0.1, [], [Step("expand", "k=10", cost=1.0), Step("abstain", "")]
    )
    assert result.cost == 1.0
    assert PASSAGES[0].with_score(2.0).score == 2.0
