"""Classify the occupations the hand mapping leaves out by following Wikidata P279 chains.

The hand mapping (configs/data/occupation_classes.yaml) labels the most frequent occupations.
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

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES_DIR = ROOT / "data" / "raw" / "wikidata_attributes"
HAND_FILE = ROOT / "configs" / "data" / "occupation_classes_hand.yaml"
OUT_FILE = ROOT / "configs" / "data" / "occupation_classes_p279.yaml"

ENDPOINT = "https://qlever.dev/api/wikidata"
USER_AGENT = "fair-kg-lp/0.1 (research project; occupation class expansion)"
BATCH_SIZE = 20_000
MAX_RETRIES = 5

# P279 steps accepted by default. `--validate` on the hand-labelled occupations gave 84% agreement
# with the hand classes at 1 step, 73% at 2 steps and nothing useful beyond.
DEFAULT_HOPS = 1

# Umbrella occupations that sit above many unrelated ones ("author" above composer, "scientist"
# above historian). They stay in the hand mapping, but are not used to classify other occupations.
GENERIC_ANCHORS = {
    "Q901",  # scientist
    "Q483501",  # artist
    "Q3391743",  # visual artist
    "Q713200",  # performing artist
    "Q482980",  # author
    "Q36180",  # writer
    "Q2462658",  # manager
    "Q43845",  # businessperson
    "Q131524",  # entrepreneur
    "Q15253558",  # activist
    "Q212238",  # civil servant
    "Q5322166",  # designer
    "Q81096",  # engineer
    "Q1650915",  # researcher
    "Q20826540",  # scholar
    "Q3400985",  # academic
}

QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?value WHERE {{ VALUES ?item {{ {values} }} ?item wdt:P279 ?value . }}
"""


def short(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def fetch_parents(qids: list[str]) -> list[tuple[str, str]]:
    """(child, parent) P279 pairs for `qids`, from QLever."""
    values = " ".join(f"wd:{q}" for q in qids)
    body = urllib.parse.urlencode({"query": QUERY.format(values=values)}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                rows = json.load(response)["results"]["bindings"]
            return [(short(b["item"]["value"]), short(b["value"]["value"])) for b in rows]
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"  {exc}; retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def load_cache(path: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Parents per node, and the nodes already fetched (including those with no parents)."""
    parents: dict[str, set[str]] = collections.defaultdict(set)
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                done.add(row["child"])
                if row["parent"]:
                    parents[row["child"]].add(row["parent"])
    return parents, done


def fetch_chains(
    start: list[str],
    anchors: set[str],
    parents: dict[str, set[str]],
    done: set[str],
    cache: Path,
    hops: int,
) -> None:
    """Fetch P279 parents level by level, `hops` levels up, stopping chains at anchors."""
    new_file = not cache.exists()
    with cache.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writerow(["child", "parent"])
        frontier = set(start)
        seen = set(start)
        for hop in range(1, hops + 1):
            # Anchors need no parents of their own: a chain stops when it reaches one.
            todo = sorted(q for q in frontier if q not in done and q not in anchors)
            print(f"hop {hop}: {len(frontier):,} nodes, {len(todo):,} to fetch")
            for i in range(0, len(todo), BATCH_SIZE):
                batch = todo[i : i + BATCH_SIZE]
                found = fetch_parents(batch)
                for child, parent in found:
                    parents[child].add(parent)
                    writer.writerow([child, parent])
                for q in batch:
                    done.add(q)
                    if not parents.get(q):
                        writer.writerow([q, ""])  # no parents: do not ask again
                f.flush()
            frontier = {p for q in frontier if q not in anchors for p in parents.get(q, ())} - seen
            seen |= frontier
            if not frontier:
                break


def classify(
    occupation: str,
    parents: dict[str, set[str]],
    anchors: dict[str, str],
    class_order: dict[str, int],
    max_hops: int,
) -> dict | None:
    """Class of the nearest labelled occupations above `occupation`, or None."""
    level, seen = {occupation}, {occupation}
    for hop in range(1, max_hops + 1):
        reached = {p for q in level for p in parents.get(q, ())} - seen
        found = sorted(q for q in reached if q in anchors)
        if found:
            votes = collections.Counter(anchors[q] for q in found)
            best = min(votes, key=lambda c: (-votes[c], class_order[c]))
            return {"class": best, "via": [q for q in found if anchors[q] == best], "hops": hop}
        seen |= reached
        level = reached
        if not level:
            break
    return None


def person_classes(
    occ: pd.DataFrame, mapping: dict[str, str], class_order: dict[str, int]
) -> pd.Series:
    """Class per person: most occupations in one class, ties to the earlier class."""
    mapped = occ.assign(cls=occ.value.map(mapping)).dropna(subset=["cls"])
    per = mapped.groupby(["qid", "cls"]).size().reset_index(name="n")
    per["order"] = per.cls.map(class_order)
    per = per.sort_values(["qid", "n", "order"], ascending=[True, False, True])
    return per.drop_duplicates("qid").set_index("qid").cls


def validate(
    anchors: dict[str, str],
    classes: dict[str, dict[str, str]],
    counts: pd.Series,
    class_order: dict[str, int],
    directory: Path,
    date: str,
    hops: int,
) -> None:
    """Hide each hand-labelled occupation in turn and see whether P279 recovers its class."""
    labels = {q: label for members in classes.values() for q, label in members.items()}
    cache = directory / f"p279_{date}.tsv"
    parents, done = load_cache(cache)
    # No anchor stops the chains here: every labelled occupation needs its own parents.
    fetch_chains(list(anchors), set(), parents, done, cache, hops)

    outcomes = []
    for q, hand_class in anchors.items():
        voters = {a: c for a, c in anchors.items() if a != q and a not in GENERIC_ANCHORS}
        outcomes.append((q, hand_class, classify(q, parents, voters, class_order, hops)))

    agree = [o for o in outcomes if o[2] and o[2]["class"] == o[1]]
    differ = [o for o in outcomes if o[2] and o[2]["class"] != o[1]]
    none = [o for o in outcomes if not o[2]]
    n = len(outcomes)
    print(f"\n{n} hand-labelled occupations, each tested without itself:")
    print(f"  P279 gives the same class: {len(agree)} ({len(agree) / n:.1%})")
    print(f"  P279 gives a different class: {len(differ)} ({len(differ) / n:.1%})")
    print(f"  P279 reaches no labelled occupation: {len(none)} ({len(none) / n:.1%})")
    print("\nagreement by number of P279 steps:")
    for hop in sorted({o[2]["hops"] for o in outcomes if o[2]}):
        at = [o for o in outcomes if o[2] and o[2]["hops"] == hop]
        same = sum(o[2]["class"] == o[1] for o in at)
        print(f"  {hop} steps: {same}/{len(at)} = {same / len(at):.0%}")
    print("\ndisagreements (your class vs P279), most frequent occupation first:")
    for q, hand_class, found in sorted(differ, key=lambda o: -counts.get(o[0], 0)):
        via = ", ".join(labels.get(v, v) for v in found["via"])
        print(f"  {labels[q]} ({q}): hand={hand_class}  P279={found['class']} via {via}")
    print("\nno chain to a labelled occupation:")
    print("  " + ", ".join(labels[q] for q, _, _ in none))


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

    hand = yaml.safe_load(HAND_FILE.read_text(encoding="utf-8"))
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
            "source: hand = copied from configs/data/occupation_classes.yaml, which always wins. "
            f"source: p279 = follow P279 upward, at most {args.max_hops} step(s), to the nearest "
            "hand-labelled occupation; several at the same level: most of one class wins, ties to "
            "the class listed first in class_order. Umbrella occupations (GENERIC_ANCHORS in the "
            "script) are not used. Occupations not listed here count as 'other'."
        ),
        "occupations": dict(sorted(entries.items(), key=lambda kv: -counts.get(kv[0], 0))),
    }
    OUT_FILE.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=100),
        encoding="utf-8",
    )
    print(f"wrote {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
