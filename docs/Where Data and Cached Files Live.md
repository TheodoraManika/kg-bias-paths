# Where Data and Cached Files Live

Oct 21, 2026 · @Theodora Manika
Code, configs and small text files go in GitHub; datasets, embeddings, caches and checkpoints go in a private Hugging Face dataset repository, with a copy on an external drive. GitHub only stores the scripts that build the data and a checksum manifest that says which data version each result used.
At a glance

Where things live · GitHub, Hugging Face, our machines, backup
GitHub tells us which data version to use; Hugging Face holds that version; only one person uploads a rebuild, and everyone downloads it.
Why datasets do not go in the Git repository
Git is built for many small text files that change line by line, not for gigabytes of binary data.
	•	Hard limits: GitHub blocks files over 100 MB and warns above 50 MB; our raw download is about 3 GB and the TransE embeddings about 1.2 GB.
	•	History never forgets: every version of a large file stays in the repository forever, so cloning becomes slow for both of us, even after the file is deleted.
	•	Binary files do not diff: Git cannot show what changed inside a .npy or .parquet file, so it adds weight without adding information.
	•	Everything can be rebuilt: our numbered scripts recreate every processed file from the raw download, so the repository only needs the scripts plus a record of which version was used.
The repository's .gitignore therefore excludes data/, runs/, *.pt, *.npy and *.parquet.
Every data artifact and where it goes
Each file built by our scripts has one home in storage and one place on disk; sizes are rough estimates to plan storage, not measurements.
Artifact
Built by
Local path
Approx. size
Stored in
Why there
Raw Wikidata5M download
Download
data/raw/
~3 GB
Not re-uploaded; original Hugging Face page + external drive
Public already; we record its version and checksum
Integer edge list, ID maps
01_extract
data/interim/
~0.3–0.5 GB
Our HF dataset repo
Saves everyone the slow parse
People table (gender, region, occupation, awards, decade)
02, 03
data/processed/
tens to hundreds of MB
Our HF dataset repo
Shared input for every method
PyG graph object
03
data/processed/
~0.5 GB
Our HF dataset repo
Loads in seconds
Train/validation/test splits
04
data/splits/
a few MB
Our HF dataset repo (also the release artifact)
Must be identical for both of us and every method
Data audit tables
05
docs/ (CSV)
KB
GitHub
Small, human-readable, cited in the report
TransE input embeddings (float16)
06
data/embeddings/
~1.2 GB
Our HF dataset repo
Takes hours to train
Synthetic graphs
08
data/synthetic/
< 50 MB
Our HF dataset repo
Fixed test bed
Evaluation cache (candidate sets, filter masks, group index)
09
data/cache/
tens of MB
Our HF dataset repo
Same evaluation for every method
Counterfactual matches
10
data/cache/
tens of MB
Our HF dataset repo
Slow to compute; must be shared
Run metrics, configs, logs
train.py
runs/<run>/
KB–MB
W&B + HF repo (runs/ folder)
Every report number traces back to a run
Model checkpoints
train.py
runs/<run>/
tens of MB each
HF repo, final models only
Needed to rerun evaluation and decomposition
Checksum manifest
sha256sum
docs/data_manifest.txt
KB
GitHub
Links each commit to an exact data version
Everything also goes to an external drive after each rebuild, as a second copy.
Storage options compared
We recommend a private Hugging Face dataset repository; DVC with a cloud drive is the main alternative. Plan limits change, so check current quotas before choosing.
Option
How it works
Good for us
Drawbacks
Hugging Face Hub dataset repo (recommended)
A separate Git-based repository for large files, private or public, with a Python download API
Handles GB-sized files; versioned commits; easy snapshot_download in scripts and Colab; can be made public at the end to release the fairness split
One more account and access token; mind HF's storage terms for private repos
DVC + cloud drive (university OneDrive/Google Drive, or S3)
Small .dvc pointer files in GitHub; the data itself in the drive
Data version tied to each Git commit automatically; uses storage we already have
Extra tool to learn; drive remotes can be slow or need authentication setup
Git LFS on GitHub
Large files replaced by pointers; content stored by GitHub
Feels like normal Git
Small free storage and bandwidth quota; every clone and Colab session uses bandwidth
GitHub Releases
Attach files (up to 2 GB each) to a tagged release
Final, frozen artifacts: e.g. the split files and manifest for the published version
Not for files that change weekly
Shared drive only
Folder in OneDrive / Google Drive, downloaded by hand
Zero setup
No versioning; easy to use the wrong file; hard to automate
Our choice: Hugging Face for working data and final models, a GitHub Release at the end for the frozen split, and an external drive as backup.
Recommended setup: private Hugging Face dataset repository
One private dataset repository holds every large file, laid out exactly like our local data/ folder.
1. Accounts and organization (once)
	•	Both create Hugging Face accounts; create an organization with the same name as on GitHub (e.g. fairkg-lab) and add each other.
	•	Each person creates an access token with write permission (Settings → Access Tokens). Never commit it; store it with huggingface-cli login (or the newer hf auth login) or in an HF_TOKEN environment variable.
2. Add the library and create the repository
uv add huggingface_hub

from huggingface_hub import create_repo create_repo("fairkg-lab/fair-kg-lp-data", repo_type="dataset", private=True)
3. Layout inside the dataset repository
fair-kg-lp-data/            (Hugging Face, private)   README.md                 # dataset card: source, licence, how it was built, commit of the build scripts   interim/                  # edge list .npy, ID maps   processed/                # people table .parquet, graph .pt   splits/                   # train/valid/test index files, per seed   embeddings/               # TransE float16 .npy   cache/                    # candidate sets, filter masks, counterfactual matches   synthetic/                # synthetic graphs   models/                   # final checkpoints only   data_manifest.txt         # same file as in GitHub docs/
4. Version with tags. After each full rebuild, tag the dataset repository (data-v1, data-v2, …) and write the tag in configs/data/version.yaml in GitHub, so code always knows which data version to load.
from huggingface_hub import create_tag create_tag("fairkg-lab/fair-kg-lp-data", tag="data-v1", repo_type="dataset")
5. Licence. Wikidata is CC0 and the Wikidata5M packaging is MIT, so sharing derived files is allowed; state both in the dataset card. Keep the repository private until the project ends, then decide whether to publish the split.
What goes in GitHub to point at the data
GitHub holds no data, only the small files that say how to build it, which version to load and how to check it.
fair-kg-lp/  (GitHub)   configs/data/version.yaml      # repo_id: fairkg-lab/fair-kg-lp-data, revision: data-v1   scripts/download_data.py       # pulls the right version into data/   scripts/upload_data.py         # pushes a rebuilt folder and tags it   scripts/verify_data.py         # checks local files against the manifest   scripts/01_... to 10_...       # rebuild everything from the raw download   docs/data_manifest.txt         # SHA-256 of every data file   docs/datasheet.md              # what the data is and is not for   docs/data_audit/*.csv          # small result tables   .gitignore                     # data/, runs/, *.pt, *.npy, *.parquet, .env
configs/data/version.yaml
repo_id: fairkg-lab/fair-kg-lp-data revision: data-v1          # HF tag; change only when the data is rebuilt local_dir: data
scripts/download_data.py
import yaml from huggingface_hub import snapshot_download  cfg = yaml.safe_load(open("configs/data/version.yaml")) snapshot_download(     repo_id=cfg["repo_id"],     repo_type="dataset",     revision=cfg["revision"],     local_dir=cfg["local_dir"],     allow_patterns=["interim/*", "processed/*", "splits/*", "embeddings/*", "cache/*", "synthetic/*", "data_manifest.txt"], )
On Colab the same script works after !uv pip install --system huggingface_hub pyyaml and logging in with a token stored in Colab's Secrets panel.
Daily workflow
Most days you only download and verify; rebuilding and uploading happens only when a data step changes, and only one person does it.
Getting the data (new machine, teammate, Colab)
uv sync --locked uv run python scripts/download_data.py uv run python scripts/verify_data.py     # compares checksums with docs/data_manifest.txt
Rebuilding after a change to a data script
	•	Open an issue (label data) saying what changes and why.
	•	Run the affected numbered scripts in order; new files get a new version in their name.
	•	Regenerate the manifest: sha256sum data/interim/* data/processed/* data/splits/* data/embeddings/* data/cache/* data/synthetic/* > docs/data_manifest.txt.
	•	Upload and tag:
from huggingface_hub import upload_folder, create_tag repo = "fairkg-lab/fair-kg-lp-data" upload_folder(repo_id=repo, repo_type="dataset", folder_path="data",               ignore_patterns=["raw/*"], commit_message="Rebuild splits: min group size 200") create_tag(repo, tag="data-v2", repo_type="dataset")
	•	Change revision in configs/data/version.yaml to data-v2, commit it with the new manifest in one pull request that closes the issue.
	•	Copy the new files to the external drive.
	•	Tell your teammate to run download_data.py.
Final release (end of project)
	•	Freeze the last data tag; attach splits/ and data_manifest.txt to a GitHub Release.
	•	Decide whether to make the Hugging Face repository public (splits and code yes; per-person outputs never, see the ethics document).
Rules and checklist
Rules
	•	No data, embeddings, caches or checkpoints in Git, ever; the .gitignore enforces it and reviewers check pull requests for large files.
	•	One person rebuilds a given data step at a time, through an issue; never two parallel versions.
	•	Never overwrite a file under the same name; new settings mean a new name and a new tag.
	•	Copy docs/data_manifest.txt into data/ before each upload, so the dataset repository carries its own manifest.
	•	Every run folder records the data tag it used; every report number therefore traces to code commit + data tag + seed.
	•	Tokens and keys only in environment variables or secret stores, never in code or notebooks.
	•	Raw data stays untouched in data/raw/; all changes happen in later steps.
Setup checklist
	•	☐ Hugging Face accounts and fairkg-lab organization; write tokens stored safely
	•	☐ Private dataset repository fair-kg-lp-data created with a dataset card
	•	☐ huggingface_hub added with uv
	•	☐ version.yaml, download_data.py, upload_data.py, verify_data.py committed to GitHub
	•	☐ .gitignore covers data/, runs/, *.pt, *.npy, *.parquet, .env
	•	☐ First full build uploaded and tagged data-v1; manifest committed
	•	☐ Teammate downloads and verifies successfully; Colab tested once
	•	☐ External drive backup made
