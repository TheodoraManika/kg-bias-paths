"""Classify occupations by following Wikidata P279 ("subclass of") chains up to a labelled one.

The hand mapping (configs/data/occupation_classes_hand.yaml) labels the most frequent
occupations. Any other occupation is classified by following P279 links upward, level by level,
until a hand-labelled occupation is reached. If several are reached at the same level, the class
with the most of them wins, and ties go to the class listed first in the hand file. Occupations
whose chain never reaches a labelled one stay unmapped ("other").

The P279 links are fetched from QLever and cached in a TSV (child, parent; empty parent = none),
so a rerun works offline and resumes if interrupted. Used by scripts/02a_expand_occupation_classes.py.
"""

import collections
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from fairkg.constants import (
    GENERIC_ANCHORS,
    P279_BATCH_SIZE,
    P279_ENDPOINT,
    P279_MAX_RETRIES,
    P279_USER_AGENT,
)

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
        P279_ENDPOINT,
        data=body,
        headers={"User-Agent": P279_USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    for attempt in range(P279_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                rows = json.load(response)["results"]["bindings"]
            return [(short(b["item"]["value"]), short(b["value"]["value"])) for b in rows]
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == P279_MAX_RETRIES - 1:
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
            for i in range(0, len(todo), P279_BATCH_SIZE):
                batch = todo[i : i + P279_BATCH_SIZE]
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
