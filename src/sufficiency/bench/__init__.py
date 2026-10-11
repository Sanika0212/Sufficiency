"""Benchmarks: ``estimate`` (does the context suffice?), ``selective`` (answer or
abstain with a guarantee) and ``recover`` (which fix works, at what cost).

Run with ``sufficiency bench SUITE [options]``; each suite documents its options. They
need the ``bench`` and ``models`` extras.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def run(suite: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(f"sufficiency.bench.{suite}")
    code: int = module.main(list(argv))
    return code
