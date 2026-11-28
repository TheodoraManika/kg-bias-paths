import numpy as np
import pandas as pd
import pytest

from fairkg.data.sampling import MODES, assign_strata, draw_sample, one_region, prepare_people

BASE_CFG = {
    "seed": 42,
    "mode": "proportional",
    "sample_size": 3000,
    "keep_award_holders": False,
    "inputs": {"drop_attrs_missing": True},
    "columns": {
        "id": "person_id",
        "gender": "gender",
        "region_list": "regions",
        "region": "region",
        "degree": "degree",
        "birth_decade": "birth_decade",
        "occupation": "occupation_group",
        "n_awards": "n_awards",
    },
    "degree_bins": 10,
    "min_stratum_size": 20,
    "min_per_stratum": 1,
    "oversample": {"min_group_size": 400},
}


def make_people(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "gender": rng.choice(["male", "female", "non-binary"], n, p=[0.8, 0.19, 0.01]),
            "region": rng.choice(
                ["north", "south", "east", "tiny"], n, p=[0.6, 0.25, 0.145, 0.005]
            ),
            "degree": rng.pareto(1.5, n) * 3 + 1,
            "birth_decade": rng.choice([1800, 1850, 1900, 1950, np.nan], n),
            "occupation_group": rng.choice(list("abcdef"), n),
        }
    )
    # added last so the columns above are unchanged: ~7% of people hold 1-3 awards
    df["n_awards"] = np.where(rng.random(n) < 0.07, rng.integers(1, 4, n), 0)
    return df


def cfg(**kw):
    return {**BASE_CFG, **kw}


def test_modes_declared():
    assert set(MODES) == {"proportional", "oversample"}


def test_every_person_has_a_stratum_and_no_tiny_strata_except_pooled():
    s = assign_strata(make_people(), cfg())
    assert s["stratum"].notna().all()
    sizes = s.groupby("stratum").size()
    pooled = sizes.index.str.startswith("L6:")
    assert (sizes[~pooled] >= 20).all()


def test_strata_do_not_straddle_groups():
    s = assign_strata(make_people(), cfg())
    assert (s.groupby("stratum")["group"].nunique() == 1).all()


@pytest.mark.parametrize("mode", MODES)
def test_weights_recover_population_size(mode):
    p = make_people()
    smp, rep = draw_sample(p, cfg(mode=mode))
    assert rep["strata_without_draw"] == 0
    assert smp["sampling_weight"].sum() == pytest.approx(len(p), rel=1e-9)
    assert smp["person_id"].is_unique


def test_proportional_size_and_roughly_constant_weight():
    p = make_people()
    smp, _ = draw_sample(p, cfg())
    assert abs(len(smp) - 3000) <= 3
    assert smp["sampling_weight"].median() == pytest.approx(len(p) / 3000, rel=0.3)


def test_oversample_meets_group_floor():
    p = make_people()
    smp, rep = draw_sample(p, cfg(mode="oversample"))
    g = rep["groups"]
    assert (g["n"] >= np.minimum(g["N"], 400) - 5).all()  # rounding slack
    w = smp.groupby("group")["sampling_weight"].mean()
    assert w["female|north"] < w["male|north"]


def test_deterministic_given_seed_and_changes_with_seed():
    p = make_people()
    a, _ = draw_sample(p, cfg())
    b, _ = draw_sample(p, cfg())
    c, _ = draw_sample(p, cfg(seed=7))
    assert a["person_id"].tolist() == b["person_id"].tolist()
    assert a["person_id"].tolist() != c["person_id"].tolist()


def test_tiny_group_is_merged_not_dropped():
    p = make_people(5000)
    p.loc[:5, "region"] = "singleton"
    s = assign_strata(p, cfg())
    assert s.loc[:5, "merge_level"].min() >= 5  # coarsened at least to gender|region or pooled
    assert len(s) == len(p)


# region rule and input preparation
def test_one_region_rule():
    assert one_region([]) == "no_region"
    assert one_region(np.array([])) == "no_region"
    assert one_region(["150"]) == "150"
    assert one_region(["150", "150"]) == "150"
    assert one_region(np.array(["150", "019"])) == "multiple"


def test_prepare_people_filters_derives_region_and_joins():
    people = pd.DataFrame(
        {
            "id": ["Q1", "Q2", "Q3", "Q4"],
            "gender": ["male", "female", "male", "female"],
            "regions": [["150"], [], ["150", "019"], ["002"]],
            "birth_decade": [1900.0, 1950.0, np.nan, 1800.0],
            "occupation_group": list("abcd"),
            "attrs_missing": [False, False, False, True],
        }
    )
    cov = pd.DataFrame(
        {"id": ["Q1", "Q2", "Q3", "Q4"], "degree": [5, 6, 7, 8], "n_awards": [0, 2, 0, 1]}
    )
    c = cfg()
    c["columns"] = {**c["columns"], "id": "id"}
    out = prepare_people(people, cov, c)
    assert out["id"].tolist() == ["Q1", "Q2", "Q3"]  # Q4 had attrs_missing
    assert out["region"].tolist() == ["150", "no_region", "multiple"]
    assert out["n_awards"].tolist() == [0, 2, 0]
    assert out["degree"].tolist() == [5, 6, 7]


# keep_award_holders
@pytest.mark.parametrize("mode", MODES)
def test_keep_award_holders_keeps_all_with_weight_one(mode):
    p = make_people()
    holders = set(p.loc[p["n_awards"] > 0, "person_id"])
    smp, rep = draw_sample(p, cfg(mode=mode, keep_award_holders=True, sample_size=6000))
    kept = smp.loc[smp["has_award"], "person_id"]
    assert set(kept) == holders
    assert (smp.loc[smp["has_award"], "sampling_weight"] == 1.0).all()
    assert (smp.loc[~smp["has_award"], "sampling_weight"] >= 1.0).all()
    assert rep["strata_without_draw"] == 0
    assert smp["sampling_weight"].sum() == pytest.approx(len(p), rel=1e-9)
    assert abs(len(smp) - 6000) <= 5


def test_keep_award_holders_raises_if_budget_too_small():
    p = make_people()
    with pytest.raises(ValueError):
        draw_sample(p, cfg(keep_award_holders=True, sample_size=100))


def test_keep_award_holders_needs_award_column():
    p = make_people().drop(columns=["n_awards"])
    with pytest.raises(KeyError):
        draw_sample(p, cfg(keep_award_holders=True))
