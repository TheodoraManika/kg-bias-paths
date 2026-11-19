"""Run-folder metadata and metric logging."""

import datetime
import hashlib
import pathlib
import re
import subprocess

# run_logging.py -> utils -> fairkg -> src -> repo root
ROOT = pathlib.Path(__file__).resolve().parents[3]

START_LIST = [
    "run_id",  # number of the run, starting from 0
    "date",  # ISO format date and time of the run
    "name",
    "git_commit",
    "dirty",  # True if there are uncommitted changes in the repo
    "uv_lock_hash",
    "run_dir",
    "seed",
]

END_LIST = [
    "status",  # success, failure, timeout
    "test_mrr",
    "test_hits_10",
    "fairness_gap",
]

EMPTY = "N/A"

CSV_HEADER = START_LIST + END_LIST

LOG_FILE = ROOT / "runs" / "logs" / "experiment_log.csv"

RUNS_DIR = ROOT / "runs"

def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except Exception as e:
        print(f"Error getting git commit: {e}")
        return EMPTY


def is_dirty() -> bool | None:
    """Check if there are uncommitted changes in the git repository.

    Returns None if git could not be queried, so "unknown" is never logged as "clean".
    """
    try:
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
        return bool(status)
    except Exception as e:
        print(f"Error checking git status: {e}")
        return None


def get_uv_lock_hash() -> str:
    """Get the sha256 hash of the uv.lock file."""
    uv_lock_path = ROOT / "uv.lock"
    if uv_lock_path.exists():
        return hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    return EMPTY


def create_run_dir(name: str, seed: int) -> pathlib.Path:
    """Create runs/<date>_<name>_<seed>/ and return its absolute path.

    Raises ValueError for names with characters other than letters, digits,
    "-", "_" and ".", and FileExistsError if the folder already exists, so old
    results are never overwritten.
    Store the path in the CSV relative to ROOT: path.relative_to(ROOT).
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(f"Invalid run name {name!r}: use letters, digits, '-', '_', '.'")
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"{date}_{name}_{seed}"
    # Do not need the code because the time makes every run unique
    # if run_dir.exists():
    #     for i in range(1, 20):
    #         new_run_dir = RUNS_DIR / f"{date}_{name}_{seed}_{i}"
    #         if not new_run_dir.exists():
    #             run_dir = new_run_dir
    #             break
    #     else:
    #         raise FileExistsError(f"Run directory {run_dir} already exists and could not find a unique name.")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


if __name__ == "__main__":
    print("CSV Header:", CSV_HEADER)
    print("Git Commit:", get_git_commit())
    print("Is Dirty:", is_dirty())
    print("UV Lock Hash:", get_uv_lock_hash())

    create_run_dir("test_run", 42)
