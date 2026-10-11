# Contributing

Thanks for your interest in Sufficiency.

## Development setup

```bash
git clone https://github.com/Sanika0212/Sufficiency.git
cd Sufficiency
uv sync --extra models --extra bench --extra viz
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy
uv run coverage run -m pytest && uv run coverage report --fail-under=85
```

## Ground rules

1. **Tests never download weights or data, and never call an API.** Stub the reader,
   the embedder and the data frames (see `tests/test_bench.py`), and use
   `httpx.MockTransport` for API clients.
2. **Numbers come from runs.** Every number in the README must be reproducible with
   `sufficiency bench` and `scripts/make_figures.py`, and
   [docs/benchmarks.md](docs/benchmarks.md) must list the exact commands.
3. **Probabilities must be calibrated, and guarantees must hold.** A new feature or
   estimator is reported with its calibration error next to its AUROC, and a new
   selection rule must state the guarantee it gives and be checked on held-out data.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
   `docs:`, `test:`, `chore:`, `ci:`).

## License of contributions

By contributing, you agree that your contributions are licensed under the Apache
License 2.0, the license of this project.
