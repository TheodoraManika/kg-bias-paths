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


## Build the data

The data steps run through the [`Makefile`](./Makefile), in the right order. Download Wikidata5M
by hand first (links and checksums are in [`data/MANIFEST.txt`](./data/MANIFEST.txt)).

```bash
make            # list the targets
make all        # edges -> graph, and people -> covariates; skips steps whose output exists
make graph      # one step (its inputs are built first)
make -n all     # only print what would run
make clean      # delete the built files in data/processed (raw data is kept) to rebuild them
```

| Step | Script | Output |
|---|---|---|
| `edges` | `scripts/01_extract_wikidata5m.py` | `edges_*_v1.npy`, ID maps |
| `graph` | `scripts/02c_build_graph.py` | `graph_v1_train_nosens_inv.pt` |
| `people` | `scripts/02_build_people_table.py` | `people_v2.parquet` |
| `covariates` | `scripts/03_compute_covariates.py` | `covariates_v1.parquet` |

`make attributes` (script 00, about 15 minutes, internet) and `make occupations` (script 02a,
internet) are not part of `make all`. Their results are already in place or committed.
