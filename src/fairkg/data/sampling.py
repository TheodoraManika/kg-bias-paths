"""Stratified sampling and standardized mean-difference utilities.

Strata = gender x region x degree decile x birth decade x broad occupation.

Inputs
------
The script 02 people table and the script 03 covariates table are joined on the id
column (`prepare_people`). People with `attrs_missing` are dropped (same rule as the
script 02 audit). Each person gets ONE region label: "no_region" if none, the M49 code if
exactly one distinct region, "multiple" otherwise.

Merge rule for empty / tiny strata
----------------------------------
Counts are taken on the FULL people table. A person is placed at the finest level
whose stratum (among people not yet placed) has at least `min_stratum_size` people.
Levels, finest to coarsest (each level keeps gender and region):

    L0  gender | region | decile | decade | occupation
    L1  decade   -> 50-year era
    L2  decile   -> low / mid / high band
    L3  drop occupation
    L4  drop era
    L5  gender | region
    L6  leftover people pooled per gender as "<gender>|merged_region"

Gender and region are coarsened last because they define the fairness groups.
Strata never straddle groups, so group sizes in the sample are exactly controllable.
Empty strata simply do not exist (they are never created). Every stratum label starts
with its level ("L2:...") so the merge level of each person is auditable.

Modes
-----
proportional : same sampling fraction in every stratum.
oversample   : every gender x region group gets at least `oversample.min_group_size`
               people (capped at its size); the rest of the budget is shared
               proportionally. Inside a group, strata are sampled proportionally.

Keeping award holders
---------------------
With `keep_award_holders: true`, every person with n_awards > 0 is kept (weight 1) and
only the other people are subsampled to fill the remaining budget, using the mode above.
The group floor of the oversample mode then counts holders + non-holders. This
maximises the number of award edges (test queries) per group. `has_award` is used for
sampling only; every analysis must use `sampling_weight` and must never control for it.

Weights
-------
In ALL modes `sampling_weight` = N_cell / n_sampled_in_cell is stored per person
(the inverse inclusion probability; a cell is a stratum, split into holders and
non-holders when award holders are kept). Weighted statistics on the sample then estimate
full-population statistics. Strata that receive no draw are counted in the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MODES = ("proportional", "oversample")
UNKNOWN = "unknown"

# each level lists the feature columns used in the stratum key (see module docstring).
LEVELS: list[list[str]] = [
    ["gender", "region", "decile", "decade", "occ"],
    ["gender", "region", "decile", "era", "occ"],
    ["gender", "region", "band", "era", "occ"],
    ["gender", "region", "band", "era"],
    ["gender", "region", "band"],
    ["gender", "region"],
]


def load_config(path: str | Path = "configs/data/sampling.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg['mode']!r}")
    return cfg


# input tables
def one_region(regions) -> str:
    """Collapse a person's list of region codes into one label."""
    r = list(regions) if hasattr(regions, "__len__") else []
    if len(r) == 0:
        return "no_region"
    if len(set(r)) == 1:
        return str(r[0])
    return "multiple"


def prepare_people(people: pd.DataFrame, cov: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop attrs_missing, derive the single region label, join covariates on the id."""
    c = cfg["columns"]
    if cfg.get("inputs", {}).get("drop_attrs_missing", True) and "attrs_missing" in people:
        people = people.loc[~people["attrs_missing"]]
    people = people.copy()
    people[c["region"]] = people[c["region_list"]].map(one_region)

    wanted = [col for col in (c["degree"], c["n_awards"]) if col not in people.columns]
    out = people.merge(cov[[c["id"], *wanted]], on=c["id"], how="inner", validate="one_to_one")
    if len(out) < len(people):
        print(f"warning: {len(people) - len(out)} people have no covariates row and were dropped")
    return out.reset_index(drop=True)


# features
def build_features(people: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return the string/int features used to build strata (index = people.index)."""
    c = cfg["columns"]
    q = int(cfg["degree_bins"])

    def as_str(col: str) -> pd.Series:
        return people[col].astype("object").where(people[col].notna(), UNKNOWN).astype(str)

    # degree decile on the full table; rank(first) breaks ties so bins are equal-sized.
    decile = pd.qcut(people[c["degree"]].rank(method="first"), q=q, labels=False) + 1
    pos = (decile - 1) / q
    band = np.select([pos < 0.3, pos < 0.7], ["low", "mid"], default="high")

    decade = pd.to_numeric(people[c["birth_decade"]], errors="coerce")
    era = (decade // 50 * 50).astype("Int64").astype("string").fillna(UNKNOWN)
    decade_s = decade.astype("Int64").astype("string").fillna(UNKNOWN)

    return pd.DataFrame(
        {
            "gender": as_str(c["gender"]),
            "region": as_str(c["region"]),
            "decile": decile.astype(int).astype(str),
            "band": band,
            "decade": decade_s.astype(str),
            "era": era.astype(str),
            "occ": as_str(c["occupation"]),
        },
        index=people.index,
    )


def assign_strata(people: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Give every person a `stratum`, a fairness `group` (gender|region) and a merge `level`."""
    f = build_features(people, cfg)
    min_size = int(cfg["min_stratum_size"])
    n = len(f)
    stratum = np.empty(n, dtype=object)
    level = np.full(n, -1, dtype=int)
    todo = np.ones(n, dtype=bool)

    for lvl, cols in enumerate(LEVELS):
        if not todo.any():
            break
        sub = f.loc[todo, cols]
        key = sub[cols[0]]
        for col in cols[1:]:
            key = key + "|" + sub[col]
        counts = key.map(key.value_counts())
        ok = (counts >= min_size).to_numpy()
        pos = np.flatnonzero(todo)[ok]
        stratum[pos] = f"L{lvl}:" + key.to_numpy()[ok]
        level[pos] = lvl
        todo[pos] = False

    # L6: leftovers pooled per gender (region merged).
    last = len(LEVELS)
    pos = np.flatnonzero(todo)
    stratum[pos] = [f"L{last}:{g}|merged_region" for g in f["gender"].to_numpy()[pos]]
    level[pos] = last

    group = (f["gender"] + "|" + f["region"]).to_numpy(dtype=object)
    group[pos] = [f"{g}|merged_region" for g in f["gender"].to_numpy()[pos]]
    return pd.DataFrame(
        {"stratum": stratum, "group": group, "merge_level": level}, index=people.index
    )


# allocation
def _allocate(sizes, n: int, min_each: int = 0) -> np.ndarray:
    """Split n draws across units of given sizes, proportionally, capped at size.

    Gives each unit `min_each` first (if the budget allows), then shares the rest
    proportionally to size with largest-remainder rounding.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    n = int(min(max(n, 0), sizes.sum()))
    alloc = np.zeros(len(sizes), dtype=np.int64)
    base = np.minimum(sizes, min_each)
    if base.sum() <= n:
        alloc = base.copy()
    remaining = n - int(alloc.sum())
    while remaining > 0:
        room = sizes - alloc
        open_ = room > 0
        if not open_.any():
            break
        w = sizes * open_
        ideal = w / w.sum() * remaining
        step = np.minimum(np.floor(ideal).astype(np.int64), room)
        if step.sum() == 0:  # fewer left than units: hand out one each, largest weight first
            order = np.argsort(-(ideal + 1e-9 * w))
            order = [i for i in order if open_[i]][:remaining]
            step = np.zeros_like(alloc)
            step[order] = 1
        alloc += step
        remaining -= int(step.sum())
    return alloc


def _group_targets(group_sizes, n_total: int, floor) -> np.ndarray:
    """Per-group sample sizes: at least `floor` (scalar or per-group array, capped at the
    group size), the rest shared proportionally."""
    N = np.asarray(group_sizes, dtype=np.int64)
    if len(N) == 0:
        return N
    n_total = int(min(max(n_total, 0), N.sum()))
    floors = np.minimum(N, np.asarray(floor, dtype=np.int64))
    if (
        floors.sum() >= n_total
    ):  # budget too small for the floors: share it proportionally to floors
        return _allocate(floors, n_total)
    raised = np.zeros(len(N), dtype=bool)
    while True:
        budget = n_total - floors[raised].sum()
        rest = ~raised
        if not rest.any():
            break
        prop = N[rest] / N[rest].sum() * budget
        newly = np.zeros(len(N), dtype=bool)
        newly[np.flatnonzero(rest)[prop < floors[rest]]] = True
        if not newly.any():
            break
        raised |= newly
    out = np.zeros(len(N), dtype=np.int64)
    out[raised] = floors[raised]
    if (~raised).any():
        out[~raised] = _allocate(N[~raised], n_total - floors[raised].sum())
    return out


# sampling
def draw_sample(people: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Draw the sample. Returns (sample, report).

    sample columns: <id>, stratum, group, merge_level, [has_award], sampling_weight
    """
    c = cfg["columns"]
    id_col = c["id"]
    mode = cfg["mode"]
    min_each = int(cfg.get("min_per_stratum", 1))
    keep_awards = bool(cfg.get("keep_award_holders", False))
    rng = np.random.default_rng(int(cfg["seed"]))

    s = assign_strata(people, cfg)
    n_awards_col = c.get("n_awards")
    has_award = (people[n_awards_col] > 0).to_numpy() if n_awards_col in people else None
    if keep_awards and has_award is None:
        raise KeyError(f"keep_award_holders needs the column {n_awards_col!r} in the people table")
    holder = has_award if keep_awards else np.zeros(len(s), dtype=bool)

    n_total = min(int(cfg["sample_size"]), len(s))
    n_holders = int(holder.sum())
    if n_holders > n_total:
        raise ValueError(
            f"sample_size {n_total} is smaller than the {n_holders} award holders to keep"
        )
    budget = n_total - n_holders  # how many non-holders (or everyone, if not keeping) to draw

    pool_pos = np.flatnonzero(~holder)  # people that are subsampled
    pool = s.iloc[pool_pos]
    sizes = pool.groupby("stratum").size().sort_index()
    strata_group = pool.groupby("stratum")["group"].first().reindex(sizes.index)

    if mode == "proportional":
        alloc = _allocate(sizes.to_numpy(), budget, min_each)
    elif mode == "oversample":
        gsizes = pool.groupby("group").size().sort_index()
        holders_g = s.iloc[np.flatnonzero(holder)].groupby("group").size()
        holders_g = holders_g.reindex(gsizes.index, fill_value=0).to_numpy()
        floor = np.maximum(0, int(cfg["oversample"]["min_group_size"]) - holders_g)
        targets = _group_targets(gsizes.to_numpy(), budget, floor)
        alloc = np.zeros(len(sizes), dtype=np.int64)
        for g, t in zip(gsizes.index, targets):
            pos = np.flatnonzero(strata_group.to_numpy() == g)
            alloc[pos] = _allocate(sizes.to_numpy()[pos], t, min_each)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    n_s = pd.Series(alloc, index=sizes.index)
    # random rank inside each stratum; keep the first n_s ranks (without replacement)
    rank = pd.Series(rng.random(len(pool))).groupby(pool["stratum"].to_numpy()).rank(method="first")
    keep_pool = rank.to_numpy() <= pool["stratum"].map(n_s).to_numpy()
    keep = holder.copy()
    keep[pool_pos[keep_pool]] = True

    out = s.iloc[np.flatnonzero(keep)].copy()
    out.insert(0, id_col, people[id_col].to_numpy()[keep])
    if has_award is not None:
        out["has_award"] = has_award[keep]
    ratio = sizes / n_s.replace(0, np.nan)
    weight = out["stratum"].map(ratio).to_numpy(dtype=float, copy=True)
    weight[holder[keep]] = 1.0  # kept award holders stand only for themselves
    out["sampling_weight"] = weight

    group_tbl = pd.DataFrame({"N": s.groupby("group").size(), "n": out.groupby("group").size()})
    group_tbl = group_tbl.fillna(0).astype(int)
    group_tbl["mean_weight"] = out.groupby("group")["sampling_weight"].mean()
    if has_award is not None:
        group_tbl["award_holders_in_sample"] = out.groupby("group")["has_award"].sum().astype(int)
    report = {
        "mode": mode,
        "seed": int(cfg["seed"]),
        "keep_award_holders": keep_awards,
        "n_population": len(s),
        "n_sampled": len(out),
        "n_award_holders_kept": n_holders,
        "n_strata": len(sizes),
        "strata_without_draw": int((n_s == 0).sum()),
        "weighted_total": float(out["sampling_weight"].sum()),
        "people_per_merge_level": s["merge_level"].value_counts().sort_index().to_dict(),
        "groups": group_tbl,
    }
    return out.reset_index(drop=True), report


def output_path(cfg: dict, n: int) -> Path:
    o = cfg["output"]
    tag = "_awardkept" if cfg.get("keep_award_holders") else ""
    return Path(o["dir"]) / o["name"].format(mode=cfg["mode"], seed=cfg["seed"], n=n, tag=tag)


def load_inputs(cfg: dict) -> pd.DataFrame:
    """Read the people and covariates parquet files (only needed columns) and join them."""
    c = cfg["columns"]
    p_cols = [
        c["id"],
        c["gender"],
        c["region_list"],
        c["birth_decade"],
        c["occupation"],
        "attrs_missing",
    ]
    people = pd.read_parquet(cfg["inputs"]["people"], columns=list(dict.fromkeys(p_cols)))
    cov = pd.read_parquet(
        cfg["inputs"]["covariates"], columns=[c["id"], c["degree"], c["n_awards"]]
    )
    return prepare_people(people, cov, cfg)


def run(config_path: str | Path = "configs/data/sampling.yaml"):
    """Load, join, sample, save parquet (never overwrite). Returns (sample, report, path)."""
    cfg = load_config(config_path)
    people = load_inputs(cfg)
    sample, report = draw_sample(people, cfg)
    path = output_path(cfg, len(sample))
    if path.exists():
        raise FileExistsError(f"{path} exists; change a setting (new name) instead of overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(path, index=False)
    return sample, report, path


if __name__ == "__main__":
    smp, rep, p = run()
    print(f"saved {p}")
    for k, v in rep.items():
        print(f"{k}:\n{v}" if k == "groups" else f"{k}: {v}")
