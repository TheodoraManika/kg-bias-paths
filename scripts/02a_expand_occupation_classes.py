"""Classify the occupations the hand mapping leaves out by following Wikidata P279 chains.

The hand mapping (configs/data/occupation_classes_hand.yaml) labels the most frequent occupations.
Every other occupation that appears in P106 is classified here: starting at the occupation,
follow "subclass of" (P279) links upward, level by level, until one of the hand-labelled
occupations is reached. Its class is used. If several hand-labelled occupations are reached at
the same level, the class with the most of them wins, and ties go to the class listed first in
the hand file. Occupations whose chain never reaches a labelled one stay unmapped ("other").

Needs the attribute download of scripts/00_download_wikidata_attributes.py. The P279 links are
fetched from QLever and cached next to it, so a rerun works offline and resumes if interrupted.

Output: configs/data/occupation_classes_p279.yaml   One entry for every classified occupation:
the hand-labelled ones (source: hand) and the P279 ones (source: p279, with the labelled
occupation it was reached from and the number of P279 steps). Load this one file for the
occupation -> class lookup.
Cache:  data/raw/wikidata_attributes/p279_<YYYY-MM-DD>.tsv   (child, parent; empty parent = none)
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from fairkg.constants import (
    ATTRIBUTES_DIR,
    DEFAULT_HOPS,
    GENERIC_ANCHORS,
    OCCUPATION_HAND_FILE,
    OCCUPATION_P279_FILE,
    ROOT,
)
from fairkg.data.occupation import (
    classify,
    fetch_chains,
    load_cache,
    person_classes,
    validate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the N most frequent unmapped occupations")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="cache file date")
    parser.add_argument("--attributes", type=Path, help="attributes TSV (default: the latest)")
    parser.add_argument(
        "--max-hops",
        type=int,
        default=DEFAULT_HOPS,
        help=f"P279 steps to follow (default {DEFAULT_HOPS})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="test the method on the hand-labelled occupations and list where it disagrees",
    )
    args = parser.parse_args()

    hand = yaml.safe_load(OCCUPATION_HAND_FILE.read_text(encoding="utf-8"))
    class_order = {c: i for i, c in enumerate(hand["classes"])}
    anchors = {q: cls for cls, members in hand["classes"].items() for q in members}

    attributes = args.attributes or max(ATTRIBUTES_DIR.glob("attributes_*.tsv"))
    print(f"reading {attributes.relative_to(ROOT)}")
    a = pd.read_csv(attributes, sep="\t", dtype=str, keep_default_na=False)
    occ = a[a.property == "P106"][["qid", "value"]].drop_duplicates()
    counts = occ.value.value_counts()
    if args.validate:
        validate(
            anchors, hand["classes"], counts, class_order, ATTRIBUTES_DIR, args.date, args.max_hops
        )
        return
    voters = {q: c for q, c in anchors.items() if q not in GENERIC_ANCHORS}
    todo = [q for q in counts.index if q not in anchors]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(counts):,} occupations, {len(anchors):,} hand-labelled, {len(todo):,} to classify")

    cache = ATTRIBUTES_DIR / f"p279_{args.date}.tsv"
    parents, done = load_cache(cache)
    fetch_chains(todo, set(voters), parents, done, cache, args.max_hops)

    result = {}
    for q in todo:
        found = classify(q, parents, voters, class_order, args.max_hops)
        if found:
            result[q] = found
    print(f"classified {len(result):,} of {len(todo):,} occupations through P279")

    humans = occ.qid.nunique()
    before = person_classes(occ, anchors, class_order)
    merged = {**{q: r["class"] for q, r in result.items()}, **anchors}  # hand entries win
    after = person_classes(occ, merged, class_order)
    print(f"humans with a P106: {humans:,}")
    print(f"  hand mapping only: {len(before):,} = {len(before) / humans:.1%} get a class")
    print(f"  hand + P279:       {len(after):,} = {len(after) / humans:.1%} get a class")

    labels_file = max(ATTRIBUTES_DIR.glob("labels_*.tsv"))
    lab = pd.read_csv(labels_file, sep="\t", dtype=str, keep_default_na=False)
    labels = dict(zip(lab.qid, lab.label, strict=True))
    labels.update(
        {q: label for members in hand["classes"].values() for q, label in members.items()}
    )
    entries = {
        q: {"label": labels.get(q, ""), "class": cls, "source": "hand"}
        for q, cls in anchors.items()
    }
    entries.update(
        {
            q: {
                "label": labels.get(q, ""),
                "class": r["class"],
                "source": "p279",
                "via": r["via"],
                "hops": r["hops"],
            }
            for q, r in result.items()
        }
    )
    doc = {
        "version": 1,
        "source": f"{attributes.name} (P106), the hand mapping and the P279 chains in {cache.name}",
        "rule_several_occupations": hand["rule_several_occupations"],
        "class_order": list(hand["classes"]),
        "method": (
            "source: hand = copied from configs/data/occupation_classes_hand.yaml, which always wins. "
            f"source: p279 = follow P279 upward, at most {args.max_hops} step(s), to the nearest "
            "hand-labelled occupation; several at the same level: most of one class wins, ties to "
            "the class listed first in class_order. Umbrella occupations (GENERIC_ANCHORS in the "
            "script) are not used. Occupations not listed here count as 'other'."
        ),
        "occupations": dict(sorted(entries.items(), key=lambda kv: -counts.get(kv[0], 0))),
    }
    OCCUPATION_P279_FILE.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=100),
        encoding="utf-8",
    )
    print(f"wrote {OCCUPATION_P279_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
