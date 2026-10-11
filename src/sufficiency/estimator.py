"""A calibrated estimate of P(the passages are enough to answer the question).

Logistic regression over standardised features, wrapped in isotonic calibration with
cross-validation, so that an estimate of 0.8 means that about 80% of such cases really
are answerable. Calibration is reported (expected calibration error, Brier score)
rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from sufficiency.features import Featurizer
from sufficiency.metrics import auroc, expected_calibration_error
from sufficiency.model import Passage


@dataclass(frozen=True)
class Report:
    n: int
    positives: float
    auroc: float
    ece: float
    brier: float


class SufficiencyEstimator:
    def __init__(
        self,
        featurizer: Featurizer | None = None,
        *,
        regularization: float = 1.0,
        calibration: str = "isotonic",
        folds: int = 5,
    ) -> None:
        self.featurizer = featurizer or Featurizer()
        self.regularization = regularization
        self.calibration = calibration
        self.folds = folds
        self.model: Any = None
        self.params: dict[str, Any] | None = None  # numpy-only form, from JSON

    def _pipeline(self) -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.regularization, max_iter=2000, class_weight="balanced"),
        )

    def fit(self, features: ArrayLike, labels: ArrayLike) -> SufficiencyEstimator:
        from sklearn.calibration import CalibratedClassifierCV

        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        folds = max(2, min(self.folds, int(np.min(np.bincount(y, minlength=2)))))
        self.model = CalibratedClassifierCV(self._pipeline(), method=self.calibration, cv=folds)
        self.model.fit(x, y)
        return self

    def predict_matrix(self, features: ArrayLike) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if self.model is not None:
            return np.asarray(self.model.predict_proba(x)[:, 1])
        if self.params is not None:
            return _predict_params(self.params, x)
        raise RuntimeError("fit the estimator (or load one) before predicting")

    def predict(self, question: str, passages: Sequence[Passage]) -> float:
        return float(self.predict_matrix(self.featurizer(question, passages))[0])

    def report(self, features: ArrayLike, labels: ArrayLike) -> Report:
        p = self.predict_matrix(features)
        y = np.asarray(labels, dtype=float)
        return Report(
            n=len(y),
            positives=float(y.mean()) if len(y) else float("nan"),
            auroc=auroc(p, y.astype(bool)),
            ece=expected_calibration_error(p, y),
            brier=float(np.mean((p - y) ** 2)),
        )

    def coefficients(self) -> dict[str, float]:
        """Standardised logistic coefficients, averaged over the calibration folds."""
        if self.model is None:
            return {}
        coefs = [
            c.estimator.named_steps["logisticregression"].coef_[0]
            for c in self.model.calibrated_classifiers_
        ]
        mean = np.mean(coefs, axis=0)
        return dict(zip(self.featurizer.names, (float(v) for v in mean), strict=False))

    def to_dict(self) -> dict[str, Any]:
        """Every number the fitted model needs, as plain JSON (no pickle)."""
        if self.model is None:
            raise RuntimeError("fit the estimator before exporting it")
        folds = []
        for c in self.model.calibrated_classifiers_:
            scaler = c.estimator.named_steps["standardscaler"]
            lr = c.estimator.named_steps["logisticregression"]
            cal = c.calibrators[0]
            fold: dict[str, Any] = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": lr.coef_[0].tolist(),
                "intercept": float(lr.intercept_[0]),
            }
            if self.calibration == "isotonic":
                fold["x"] = cal.X_thresholds_.tolist()
                fold["y"] = cal.y_thresholds_.tolist()
            else:
                fold["a"], fold["b"] = float(cal.a_), float(cal.b_)
            folds.append(fold)
        return {
            "names": list(self.featurizer.names),
            "calibration": self.calibration,
            "folds": folds,
        }

    def from_dict(self, data: dict[str, Any]) -> SufficiencyEstimator:
        if tuple(data["names"]) != self.featurizer.names:
            raise ValueError("the saved model was trained on different features")
        self.model, self.params = None, data
        return self

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict()) + "\n")
        return out

    def load_json(self, path: str | Path) -> SufficiencyEstimator:
        return self.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def pretrained(cls, name: str = "hotpotqa") -> SufficiencyEstimator:
        """An estimator fitted by ``sufficiency bench estimate`` and shipped with the package:
        ``squad_v2`` (single paragraph), ``musique`` (2 to 4 hops) or ``hotpotqa`` (2 hops,
        half the questions without their gold paragraphs)."""
        from importlib import resources

        from sufficiency.features import (
            Featurizer,
            LexicalFeatures,
            ReaderFeatures,
            RetrievalFeatures,
        )

        text = resources.files("sufficiency").joinpath(f"data/estimator-{name}.json").read_text()
        featurizer = Featurizer(RetrievalFeatures(), LexicalFeatures(), ReaderFeatures())
        return cls(featurizer).from_dict(json.loads(text))

    def save(self, path: str | Path) -> Path:
        """Pickle the fitted model (load only files you created yourself)."""
        import joblib

        out = Path(path)
        joblib.dump({"model": self.model, "names": self.featurizer.names}, out)
        return out

    def load(self, path: str | Path) -> SufficiencyEstimator:
        import joblib

        data = joblib.load(Path(path))
        if tuple(data["names"]) != self.featurizer.names:
            raise ValueError("the saved model was trained on different features")
        self.model = data["model"]
        return self


def _predict_params(params: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Average of the calibrated folds, exactly as scikit-learn's calibrated classifier."""
    out = np.zeros(len(x))
    for fold in params["folds"]:
        z = ((x - np.asarray(fold["mean"])) / np.asarray(fold["scale"])) @ np.asarray(
            fold["coef"]
        ) + fold["intercept"]
        if params["calibration"] == "isotonic":
            xs, ys = np.asarray(fold["x"]), np.asarray(fold["y"])
            out += np.interp(np.clip(z, xs[0], xs[-1]), xs, ys)
        else:
            out += 1.0 / (1.0 + np.exp(fold["a"] * z + fold["b"]))
    return out / len(params["folds"])
