import numpy as np
import pytest

from sufficiency.features import Featurizer, LexicalFeatures, ReaderFeatures, RetrievalFeatures
from sufficiency.model import Passage

PASSAGES = [
    Passage("a", "The Eiffel Tower is 330 metres tall.", "Eiffel Tower", score=9.0),
    Passage("b", "Paris is the capital of France.", "Paris", score=3.0),
    Passage("c", "Cheese is made from milk.", "Cheese", score=1.0),
]


def test_retrieval_features() -> None:
    share, gap, entropy, count = RetrievalFeatures().compute("q", PASSAGES)
    assert share == pytest.approx(9 / 13)
    assert gap == pytest.approx(6 / 9)
    assert 0 < entropy < 1
    assert count == 3
    assert RetrievalFeatures().compute("q", []) == [0.0, 0.0, 1.0, 0.0]
    single = RetrievalFeatures().compute("q", PASSAGES[:1])
    assert single[1] == 1.0
    assert single[2] == 0.0


def test_lexical_features() -> None:
    lex = LexicalFeatures()
    values = dict(
        zip(lex.names, lex.compute("How tall is the Eiffel Tower?", PASSAGES), strict=True)
    )
    assert values["lex_cover_top1"] == 1.0
    assert values["lex_question_terms"] == 3.0  # tall, eiffel, tower
    assert values["lex_wh_how"] == 1.0
    assert values["lex_multihop_cue"] == 0.0
    multi = dict(
        zip(
            lex.names,
            lex.compute("Who directed the film that Tom Hanks made with Meg Ryan?", []),
            strict=True,
        )
    )
    assert multi["lex_multihop_cue"] == 1.0
    assert multi["lex_cover_all"] == 0.0


class FakeReader(ReaderFeatures):
    def spans(self, question, passages):  # type: ignore[no-untyped-def]
        return [(3.0, 0.8, "330 metres"), (-2.0, 0.1, "Paris"), (1.0, 0.5, "330 metres")][
            : len(passages)
        ]


def test_reader_features_and_featurizer() -> None:
    reader = FakeReader()
    margin_max, margin_mean, prob, agree = reader.compute("How tall?", PASSAGES)
    assert margin_max == 3.0
    assert margin_mean == pytest.approx(2 / 3)
    assert prob == 0.8
    assert agree == pytest.approx(2 / 3)
    assert reader.last_answers[0] == "330 metres"
    assert FakeReader().compute("q", []) == [-10.0, -10.0, 0.0, 0.0]
    feats = Featurizer(RetrievalFeatures(), reader)
    vector = feats("How tall?", PASSAGES)
    assert vector.shape == (len(feats.names),)
    assert feats.names[:2] == ("ret_top_share", "ret_gap")
    matrix = feats.matrix([("How tall?", PASSAGES), ("Where?", PASSAGES[1:])])
    assert matrix.shape == (2, 8)
    assert Featurizer().matrix([]).shape == (0, len(Featurizer().names))
    assert isinstance(Featurizer()("q", PASSAGES), np.ndarray)
