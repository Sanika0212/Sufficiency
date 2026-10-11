"""Sufficiency: know when retrieved context is enough to answer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sufficiency")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
