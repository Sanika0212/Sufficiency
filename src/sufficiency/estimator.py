"""A calibrated estimate of P(the passages are enough to answer the question).

Logistic regression over standardised features, wrapped in isotonic calibration with
cross-validation, so that an estimate of 0.8 means that about 80% of such cases really
are answerable. Calibration is reported (expected calibration error, Brier score)
rather than assumed.
"""

from __future__ import annotations

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
        folds = max(2, min(self.folds, int(np.bincount(y, minlength=2).min())))
        self.model = CalibratedClassifierCV(self._pipeline(), method=self.calibration, cv=folds)
        self.model.fit(x, y)
        return self

    def predict_matrix(self, features: ArrayLike) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit the estimator (or load one) before predicting")
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.asarray(self.model.predict_proba(x)[:, 1])

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
