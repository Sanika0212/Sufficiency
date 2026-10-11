"""Pick the torch device: ``SUFFICIENCY_DEVICE`` if set, else CUDA, else Apple MPS, else CPU."""

from __future__ import annotations

import os


def default_device() -> str:
    override = os.environ.get("SUFFICIENCY_DEVICE")
    if override:
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
