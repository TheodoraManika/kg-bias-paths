"""Run-folder metadata and metric logging."""

import csv
import datetime
import hashlib
import json
import pathlib
import re
import subprocess

import yaml

from fairkg.utils.config import to_plain
from fairkg.utils.seed import seed_everything

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

CONFIG_FILE = "config.yaml"
META_FILE = "meta.json"
METRICS_FILE = "metrics.json"

EMPTY = "N/A"

VALID_STATUSES = ("success", "failure", "timeout")

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
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
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
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error checking git status: {e}")
        return None


def get_uv_lock_hash() -> str:
    """Get the sha256 hash of the uv.lock file."""
    uv_lock_path = ROOT / "uv.lock"
    if uv_lock_path.exists():
        return hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    return EMPTY


def write_config(config: dict, run_dir: pathlib.Path) -> None:
    """Write the resolved config to config.yaml in the run directory."""
    with open(run_dir / CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(config), f, sort_keys=False)


def next_run_id() -> int:
    """Return the number of runs already in the CSV log (0 if there is no log yet)."""
    if not LOG_FILE.exists():
        return 0
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus the header row


def write_meta(name: str, seed: int, run_dir: pathlib.Path, now: datetime.datetime) -> dict:
    """Write all start-time columns (START_LIST) to meta.json and return them."""
    meta = {
        "run_id": next_run_id(),
        "date": now.isoformat(timespec="seconds"),
        "name": name,
        "git_commit": get_git_commit(),
        "dirty": is_dirty(),  # None (null in JSON) means git could not be queried
        "uv_lock_hash": get_uv_lock_hash(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "seed": seed,
    }
    with open(run_dir / META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def write_metrics(metrics: dict, run_dir: pathlib.Path) -> None:
    """Write metrics to metrics.json (overwrites; start with {} and fill in at the end)."""
    with open(run_dir / METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(to_plain(metrics), f, indent=2)
        f.write("\n")


def create_run_dir(
    name: str, seed: int, now: datetime.datetime | None = None
) -> pathlib.Path:
    """Create runs/<date>_<name>_<seed>/ and return its absolute path.

    Pass `now` so the folder name and the "date" column share one timestamp.

    Raises ValueError for names with characters other than letters, digits,
    "-", "_" and ".", and FileExistsError if the folder already exists, so old
    results are never overwritten.
    Store the path in the CSV relative to ROOT: path.relative_to(ROOT).
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(f"Invalid run name {name!r}: use letters, digits, '-', '_', '.'")
    now = now or datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")
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


def start_run(name: str, seed: int, config: dict) -> dict:
    """Seed, create the run folder and write its files; return the meta dict.

    The returned dict has every START_LIST column (also saved as meta.json) and is
    what finish_run needs to build the CSV row.
    """
    seed = seed_everything(seed)
    now = datetime.datetime.now()
    run_dir = create_run_dir(name, seed, now)
    write_config(config, run_dir)
    meta = write_meta(name, seed, run_dir, now)
    write_metrics({}, run_dir)  # an empty metrics.json marks a run that never finished
    return meta


def finish_run(meta: dict, metrics: dict, status: str) -> None:
    """Write metrics.json and append a row to the CSV log.

    The meta dict is what start_run returned, and the metrics dict is what the
    training script produced. The status string must be one of VALID_STATUSES.
    Any missing or None value is written as EMPTY, so no cell is ever blank.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}: use one of {VALID_STATUSES}")
    write_metrics(metrics, ROOT / meta["run_dir"])  # run_dir is relative to ROOT

    values = {**meta, "status": status}
    values.update({col: metrics.get(col) for col in END_LIST if col != "status"})
    row = [EMPTY if values.get(col) is None else values[col] for col in CSV_HEADER]

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


if __name__ == "__main__":
    print("CSV Header:", CSV_HEADER)
    print("Git Commit:", get_git_commit())
    print("Is Dirty:", is_dirty())
    print("UV Lock Hash:", get_uv_lock_hash())
