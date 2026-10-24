# Environment and Reproducibility with uv

Oct 21, 2026 · @Theodora Manika
Every result in the report must be re-runnable by either of us, on the 5060 Ti or on Colab, months later. uv gives us one locked environment; seeds, run logs and data checksums do the rest.
Why this matters
A result is reproducible when the same code, data and settings give the same numbers, up to small random variation, on any of our machines.
Four things must be pinned for that:
What
How we pin it
Software (Python and package versions)
uv: .python-version, pyproject.toml, uv.lock
Code
Git commit hash stored with every run
Data (raw files, caches, splits)
Checksums (SHA-256) and versioned file names
Randomness
Fixed seeds, logged with every run
Fairness results are especially sensitive: a bias gap of a few points can appear or disappear with a different split or seed, so we must be able to show exactly what produced each number.
1. Set up the project with uv
uv manages the Python version, the virtual environment and the exact package versions from one place.
	•	Create the project (on Windows, do this inside WSL2):
uv init fair-kg-lp cd fair-kg-lp uv python pin 3.14
uv python pin writes .python-version, so both of us get the same Python.
	•	Add the packages:
uv add numpy pandas pyarrow scipy statsmodels scikit-learn uv add matplotlib seaborn tqdm pyyaml uv add pykeen optuna wandb uv add --dev ruff pytest pre-commit ipykernel
	•	Run everything through uv:
uv run python scripts/prepare_data.py uv run pytest
uv run checks that the environment matches the lock file before running, so nobody accidentally runs with different versions.
	•	Commit pyproject.toml, uv.lock and .python-version to Git. Never commit .venv/.
2. PyTorch (CUDA 12.8) and PyTorch Geometric through uv
The RTX 5060 Ti is a Blackwell GPU, so it needs PyTorch 2.7 or newer built for CUDA 12.8; the default PyPI build may not be the CUDA build we want, so we point uv at PyTorch's own package index.
Add to pyproject.toml:
[[tool.uv.index]] name = "pytorch-cu128" url = "https://download.pytorch.org/whl/cu128" explicit = true  [tool.uv.sources] torch = [{ index = "pytorch-cu128" }]
explicit = true means only torch comes from that index; everything else still comes from PyPI.
Then:
uv add "torch>=2.7" uv add torch_geometric
Neighbour sampling needs one extra package. PyG's NeighborLoader uses pyg-lib (or torch-sparse), which are compiled per PyTorch + CUDA version and hosted on PyG's own wheel page:
	•	Open https://data.pyg.org/whl/ and check which torch-X.Y.0+cu128 folders exist.
	•	If your PyTorch version is listed, add the matching link to pyproject.toml:
[tool.uv] find-links = ["https://data.pyg.org/whl/torch-2.7.0+cu128.html"]
then run uv add pyg-lib.
	•	If it is not listed, pin torch to the newest version that is listed (still ≥ 2.7), e.g. uv add "torch==2.7.*".
The exact version numbers above are examples; use the ones the PyG page actually lists when you set up. uv's PyTorch integration guide and the PyG installation page explain the details (links from memory; check them).
3. Test the GPU before writing project code
Spend 30 minutes proving the stack works end to end; a broken install found in week 3 costs days.
Save as scripts/check_env.py and run uv run python scripts/check_env.py:
import torch, torch_geometric from torch_geometric.nn import RGCNConv from torch_geometric.loader import NeighborLoader from torch_geometric.data import Data  print("torch", torch.__version__, "| CUDA build", torch.version.cuda) print("GPU available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0)) print("PyG", torch_geometric.__version__)  # 1. plain GPU maths x = torch.randn(2000, 2000, device="cuda") print("matmul ok:", (x @ x).sum().item() != 0)  # 2. R-GCN layer on a tiny random graph n, r = 1000, 5 edge_index = torch.randint(0, n, (2, 5000)) edge_type = torch.randint(0, r, (5000,)) data = Data(x=torch.randn(n, 32), edge_index=edge_index, edge_type=edge_type) conv = RGCNConv(32, 32, num_relations=r, num_bases=2).cuda() out = conv(data.x.cuda(), data.edge_index.cuda(), data.edge_type.cuda()) print("R-GCN ok:", out.shape)  # 3. neighbour sampling (needs pyg-lib or torch-sparse) loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=64) batch = next(iter(loader)) print("NeighborLoader ok:", batch.num_nodes, "nodes in first batch")  # 4. mixed precision with torch.autocast("cuda", dtype=torch.bfloat16):     y = x @ x print("bf16 ok:", y.dtype)
Typical errors and fixes
Error
Meaning
Fix
no kernel image is available
PyTorch build too old for Blackwell
Reinstall from the cu128 index, torch ≥ 2.7
torch.cuda.is_available() is False
Driver missing or too old (or not visible in WSL2)
Update the NVIDIA Windows driver; check nvidia-smi inside WSL2
NeighborLoader asks for pyg-lib or torch-sparse
Sampling extension missing
Install pyg-lib from the matching PyG wheel page
Import error mentioning a torch version mismatch
Extension built for another torch version
Match the find-links URL to your installed torch version
4. Freeze and share the environment
uv.lock is the single source of truth: whoever runs the code installs exactly what it lists.
Your teammate (or a new machine)
git clone <repo-url> && cd fair-kg-lp uv sync --locked uv run python scripts/check_env.py
--locked stops with an error if uv.lock is out of date, instead of silently resolving new versions.
Changing a package (deliberately)
	•	Add: uv add <package>; upgrade one: uv lock --upgrade-package <package> then uv sync.
	•	Commit the pyproject.toml + uv.lock change on its own, with a message saying why (e.g. "Upgrade torch to 2.8 for faster bf16").
	•	Rerun check_env.py and one smoke test after every change.
	•	Freeze versions before the final experiments: no upgrades during weeks 10–16.
Google Colab
Colab ships its own Python and PyTorch, so we install our locked versions on top:
# locally, once per lock change uv export --format requirements-txt --no-hashes > requirements-colab.txt

# first Colab cell !pip install -q uv !uv pip install --system -r requirements-colab.txt
Check Colab's Python version first (!python --version); if it differs from our pin, note it in the run log. Colab GPUs (e.g. T4) also work with the CUDA 12.8 PyTorch build.
5. Seeds and determinism
Seeds make runs repeatable, but GNNs on a GPU are never perfectly deterministic, so we fix what we can and measure the rest with multiple seeds.
One function, called at the start of every script
import os, random import numpy as np import torch  def set_seed(seed: int, strict: bool = False) -> None:     random.seed(seed)     np.random.seed(seed)     torch.manual_seed(seed)     torch.cuda.manual_seed_all(seed)     if strict:  # debugging only: slower, and some ops will raise errors         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"         torch.use_deterministic_algorithms(True)         torch.backends.cudnn.benchmark = False
What else needs a seed
	•	Data loader workers: pass a seeded torch.Generator to the loader so neighbour sampling and shuffling repeat.
	•	Negative sampling: use the same generator, not global randomness.
	•	Splits, subsampling, synthetic graph, counterfactual matching: each has its own seed, stored in the file name of its output.
What stays non-deterministic
	•	GPU scatter/sum operations used in message passing add numbers in varying order, so results can differ in the last decimals between identical runs.
	•	That is why every result uses 3–5 seeds and reports mean ± spread; a difference smaller than the seed spread is not a real difference.
Seed plan: fix the list once (e.g. 0, 1, 2, 3, 4) and use the same seeds for every method, so comparisons are paired.
6. Record every run
Each run writes one folder that contains everything needed to reproduce it, created automatically by the training script.
runs/2026-11-04_rgcn_cf-direct_seed2/   config.yaml        # every setting, including seeds and data file names   git_commit.txt     # output of `git rev-parse HEAD`, plus a warning if uncommitted changes exist   uv.lock            # copy of the lock file used   env.txt            # Python, torch, CUDA, GPU name   metrics.json       # accuracy + fairness metrics, per split   train_log.csv      # loss and validation metrics per epoch, time per epoch, peak GPU memory   checkpoint_best.pt # model chosen by validation
Rules
	•	Never run experiments from uncommitted code; the script should refuse, or at least log the diff.
	•	Settings live only in config.yaml, never edited inside the code.
	•	The same information goes to Weights & Biases (or a shared CSV), so we can search and compare runs.
	•	Every number in the report links back to a run folder name.
7. Data versioning and backups
Data files are too large for Git, so we version them with names and checksums and back them up separately.
Versioning
	•	Versioned names: people_v1_seed42_rel80.parquet; a change in settings means a new name, never an overwrite.
	•	Checksums: a data/MANIFEST.txt listing each file and its SHA-256 (sha256sum data/processed/* > data/MANIFEST.txt). Commit the manifest to Git; both of us check it before running.
	•	One build script per data step (scripts/01_extract.py, 02_sample_split.py, …). Anyone can rebuild every file from the raw download.
	•	Optional: DVC automates this if the manual manifest becomes painful.
Backups
What
Where
When
Code, configs, manifests
GitHub
Every push
Raw download
Keep the original archive; it can be re-downloaded
Once
Processed data, embeddings, splits
External drive + a cloud folder (e.g. university OneDrive or Google Drive)
After each rebuild
Final checkpoints and run folders
Same as above
After each important run
.gitignore must exclude data/, runs/, .venv/ and *.pt.
8. Repository layout and code quality
A fixed layout lets both of us find things and keeps experiments separate from reusable code.
fair-kg-lp/   pyproject.toml  uv.lock  .python-version  README.md   configs/            # one YAML per experiment   src/fairkg/     data/             # loading cached files, splits, samplers     models/           # R-GCN, CompGCN, decoders, Module A, Module B     losses/           # link prediction, counterfactual terms, adversary     eval/             # shared accuracy + fairness metrics, decomposition     utils/            # seeds, run logging   scripts/            # 01_extract.py, 02_sample_split.py, train.py, check_env.py   tests/              # pytest: metrics, synthetic-graph checks   notebooks/          # exploration only, never final results   data/  runs/        # git-ignored
Code quality tools (all run through uv)
	•	Ruff for formatting and linting: uv run ruff format . and uv run ruff check .
	•	pytest for tests: the fairness metrics on hand-made examples with known answers, and each component on the synthetic graph.
	•	pre-commit runs Ruff automatically before each commit: uv run pre-commit install once.
	•	GitHub Actions (optional): run uv sync --locked + uv run pytest on every pull request, on CPU with the synthetic graph only.
Checklist
	•	☐ uv init, Python pinned, packages added; pyproject.toml, uv.lock, .python-version committed
	•	☐ PyTorch ≥ 2.7 from the cu128 index; PyG + pyg-lib matching the torch version
	•	☐ check_env.py passes on the 5060 Ti (GPU, R-GCN, NeighborLoader, bf16)
	•	☐ Teammate runs uv sync --locked and check_env.py successfully
	•	☐ Colab requirements exported and tested once
	•	☐ set_seed used in every script; fixed seed list agreed
	•	☐ Run folders created automatically with config, commit, lock, environment and metrics
	•	☐ Data manifest with checksums committed; backups set up
	•	☐ Repository layout, Ruff, pytest and pre-commit in place
	•	☐ Package versions frozen before the final experiments
