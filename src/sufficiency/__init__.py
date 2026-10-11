"""Sufficiency: know when retrieved context is enough to answer."""

from importlib.metadata import PackageNotFoundError, version

from sufficiency.controller import Controller
from sufficiency.estimator import SufficiencyEstimator
from sufficiency.model import Passage, Result
from sufficiency.policy import LearnedPolicy
from sufficiency.retrieval import BM25, Dense, Hybrid
from sufficiency.selective import SelectiveRule

try:
    __version__ = version("sufficiency")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "BM25",
    "Controller",
    "Dense",
    "Hybrid",
    "LearnedPolicy",
    "Passage",
    "Result",
    "SelectiveRule",
    "SufficiencyEstimator",
    "__version__",
]
