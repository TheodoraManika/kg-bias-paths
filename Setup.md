# fair-kg-lp

Fairness-aware knowledge graph link prediction experiments.

## Reproduce the environment

This project uses [uv](https://docs.astral.sh/uv/) and pins the interpreter in
[`.python-version`](./.python-version). Install the pinned Python version, then
sync the committed lockfile:

```bash
uv python install
uv sync --locked
uv venv
```

The project metadata configures the PyTorch CUDA 12.8 index and the matching
PyG wheel links. CPU-only tests do not require a CUDA device.

## Run checks

```bash
uv run pytest
uv run ruff check .
```

Experiment configuration belongs in `configs/`; scripts should consume config
files rather than embedding experiment settings in code. Data and run outputs
are intentionally ignored by Git and should be backed up separately.

