import math
from pathlib import Path

import numpy as np
import pytest

from sufficiency.estimator import SufficiencyEstimator
from sufficiency.features import Featurizer, RetrievalFeatures
from sufficiency.model import Passage
from sufficiency.selective import (
    SelectiveRule,
    crc_threshold,
    evaluate_rule,
    ltt_threshold,
)


def synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.5
    x = np.column_stack(
        [y + rng.normal(0, 0.8, n), rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)]
    )
    return x, y


def test_estimator_is_discriminative_and_calibrated(tmp_path: Path) -> None:
    x, y = synthetic(3000, 0)
    est = SufficiencyEstimator(Featurizer(RetrievalFeatures())).fit(x[:2000], y[:2000])
    rep = est.report(x[2000:], y[2000:])
    assert rep.auroc > 0.75
    assert rep.ece < 0.06
    assert 0.0 < rep.brier < 0.25
    coefs = est.coefficients()
    assert set(coefs) == set(RetrievalFeatures.names)
    assert abs(coefs["ret_top_share"]) > abs(coefs["ret_gap"])  # only feature 0 is informative
    path = est.save(tmp_path / "est.joblib")
    again = SufficiencyEstimator(Featurizer(RetrievalFeatures())).load(path)
    np.testing.assert_allclose(again.predict_matrix(x[:5]), est.predict_matrix(x[:5]))
    passages = [Passage("a", "t", score=3.0), Passage("b", "t", score=1.0)]
    assert 0.0 <= est.predict("q", passages) <= 1.0
    with pytest.raises(ValueError, match="different features"):
        SufficiencyEstimator().load(path)
    with pytest.raises(RuntimeError, match="fit the estimator"):
        SufficiencyEstimator().predict_matrix(x[:1])
    assert SufficiencyEstimator().coefficients() == {}


def test_crc_controls_answered_and_wrong_rate_on_average() -> None:
    rng = np.random.default_rng(1)
    rates = []
    for _ in range(200):
        s = rng.random(400)
        wrong = rng.random(400) > s  # higher score, less often wrong
        t = crc_threshold(s[:200], wrong[:200], alpha=0.1)
        rates.append(np.mean((s[200:] >= t) & wrong[200:]))
    assert np.mean(rates) <= 0.1 + 0.005
    assert crc_threshold([0.5], [True], alpha=0.01) == math.inf
    assert crc_threshold([], [], alpha=0.5) == math.inf  # no data: never answer


def test_ltt_certifies_selective_error_with_high_probability() -> None:
    rng = np.random.default_rng(2)
    violations = 0
    trials = 100
    for _ in range(trials):
        s = rng.random(2000)
        wrong = rng.random(2000) > s
        t = ltt_threshold(s[:1000], wrong[:1000], alpha=0.2, delta=0.1)
        answered = s[1000:] >= t
        if answered.any() and wrong[1000:][answered].mean() > 0.2 + 0.03:
            violations += 1
    assert violations <= 0.1 * trials
    assert ltt_threshold([], [], 0.1) == math.inf
    assert ltt_threshold([0.9, 0.8], [True, True], 0.1) == math.inf


def test_selective_rule_fit_and_evaluate() -> None:
    s = np.linspace(0, 1, 101)
    wrong = s < 0.5
    rule = SelectiveRule.fit(s, wrong, alpha=0.1, method="crc")
    assert rule.method == "crc"
    assert rule.answers(0.99)
    stats = evaluate_rule(rule, s, wrong)
    assert stats["answered_and_wrong"] <= 0.1
    ltt = SelectiveRule.fit(s, wrong, alpha=0.1, method="ltt", delta=0.1)
    assert ltt.delta == 0.1
    assert evaluate_rule(ltt, s, wrong)["selective_error"] <= 0.1
    empty = evaluate_rule(SelectiveRule(math.inf, "crc", 0.1), [], [])
    assert empty == {"coverage": 0.0, "selective_error": 0.0, "answered_and_wrong": 0.0}
    with pytest.raises(ValueError, match="unknown method"):
        SelectiveRule.fit(s, wrong, 0.1, method="magic")


def test_ltt_handles_unbounded_scores_with_few_top_questions() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(5, 1, 300), rng.normal(-5, 1, 300)])  # margins
    wrong = np.concatenate([rng.random(300) < 0.05, rng.random(300) < 0.9])
    t = ltt_threshold(scores, wrong, alpha=0.2, delta=0.1)
    answered = scores >= t
    assert answered.mean() > 0.3  # the strict end has single questions; it must not stop there
    assert wrong[answered].mean() <= 0.2


@pytest.mark.parametrize("calibration", ["isotonic", "sigmoid"])
def test_json_export_matches_the_fitted_model(tmp_path: Path, calibration: str) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)
    featurizer = Featurizer(RetrievalFeatures())
    est = SufficiencyEstimator(featurizer, calibration=calibration).fit(x, y)
    path = est.save_json(tmp_path / "e.json")
    loaded = SufficiencyEstimator(featurizer).load_json(path)
    assert np.allclose(loaded.predict_matrix(x), est.predict_matrix(x))
    with pytest.raises(ValueError, match="different features"):
        SufficiencyEstimator(Featurizer()).load_json(path)
    with pytest.raises(RuntimeError):
        SufficiencyEstimator(featurizer).to_dict()


def test_pretrained_estimators_ship_with_the_package() -> None:
    for name in ("squad_v2", "musique", "hotpotqa"):
        est = SufficiencyEstimator.pretrained(name)
        p = est.predict_matrix(np.zeros(len(est.featurizer.names)))
        assert 0.0 <= float(p[0]) <= 1.0
