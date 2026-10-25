"""Download person attributes from Wikidata for every human in Wikidata5M.

Wikidata5M keeps only links between entities that have an English Wikipedia page, so it has
no human gender (the values Q6581097 male, Q6581072 female, ... have no page), no dates, and
only part of each person's awards and occupations. This script fetches them from Wikidata into
a separate raw file. Wikidata5M itself is never modified.

    uv run python scripts/download_wikidata_attributes.py                   # all ~1.5M humans
    uv run python scripts/download_wikidata_attributes.py --limit 5000      # quick test
    uv run python scripts/download_wikidata_attributes.py --endpoint wdqs   # official service

Two endpoints answer the same queries. QLever (University of Freiburg, the default) takes
about 15 minutes for everyone; its copy of Wikidata is rebuilt from a dump and then updated
live, and the index description is saved next to the output. The official Wikidata Query
Service (wdqs) is always current but takes 1-2 hours, because it allows only small batches.

What each property is for:

    P21  sex or gender    person attribute (sensitive group)
    P569 date of birth    person attribute (birth decade covariate)
    sitelinks             number of Wikipedia editions; popularity covariate. Every existing
                          item has one, so a person without it was deleted or merged since 2019
    P166 award received   AUDIT ONLY: what Wikidata5M dropped. Never a training target
    P106 occupation       AUDIT ONLY, as above
    P27  citizenship      AUDIT ONLY, as above

Output (tab-separated):

    data/raw/wikidata_attributes/attributes_<YYYY-MM-DD>.tsv   qid  property  value  precision
    data/raw/wikidata_attributes/labels_<YYYY-MM-DD>.tsv       qid  label
    data/raw/wikidata_attributes/source_<YYYY-MM-DD>.txt       endpoint and its data version

The attributes file is in long format, one row per person, property and value. `value` is a
Q-ID for links, an ISO timestamp for P569 and a count for sitelinks. `precision` is set for
P569 only (Wikidata's scale: 9 = year, 8 = decade, 7 = century, 11 = day). Only best-rank
statements are kept. People that no longer exist on Wikidata get one row with property
"missing". The labels file gives the English name of every linked value (gender, award,
occupation, country); it is filled after all people are downloaded.

Interrupted runs resume: people and labels already in the output files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, TextIO

# Batch sizes are measured: wdqs is fastest at 2000 people per query and refuses parallel
# queries; QLever answers 50000 people per property in about 3.5 s.
ENDPOINTS = {
    "qlever": {"url": "https://qlever.dev/api/wikidata", "batch_size": 50_000},
    "wdqs": {"url": "https://query.wikidata.org/sparql", "batch_size": 2_000},
}
# Both services ask clients to identify themselves.
USER_AGENT = "fair-kg-lp/0.1 (research project; Wikidata5M attribute join)"
# All three splits: attributes belong to people, and validation/test people need them too.
SPLITS = [
    Path(f"data/raw/wikidata5m/wikidata5m_transductive_{split}.txt")
    for split in ("train", "valid", "test")
]
OUT_DIR = Path("data/raw/wikidata_attributes")
MAX_RETRIES = 5
LINK_PROPERTIES = ["P21", "P166", "P106", "P27"]

PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
# One query per property: QLever times out on a UNION of all of them, and on wdqs separate
# queries are as fast. Each query returns ?item ?value (and ?precision for dates).
PROPERTY_QUERIES = {
    **{prop: f"?item wdt:{prop} ?value ." for prop in LINK_PROPERTIES},
    "P569": """
    ?item p:P569 ?statement .
    ?statement a wikibase:BestRank .
    ?statement psv:P569 ?node .
    ?node wikibase:timeValue ?value .
    ?node wikibase:timePrecision ?precision .""",
    "sitelinks": "?item wikibase:sitelinks ?value .",
}
QUERY = PREFIXES + "SELECT * WHERE {{ VALUES ?item {{ {values} }} {pattern} }}"
LABEL_PATTERN = '?item rdfs:label ?label . FILTER(LANG(?label) = "en")'


def load_humans(paths: list[Path]) -> list[str]:
    """Q-IDs with (P31 instance of, Q5 human) in any Wikidata5M split."""
    humans = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                head, rel, tail = line.rstrip("\n").split("\t")
                if rel == "P31" and tail == "Q5":
                    humans.add(head)
    return sorted(humans, key=lambda q: int(q[1:]))


def run_sparql(endpoint: str, pattern: str, qids: list[str]) -> list[dict]:
    values = " ".join(f"wd:{q}" for q in qids)
    body = urllib.parse.urlencode({"query": QUERY.format(values=values, pattern=pattern)}).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["results"]["bindings"]
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES - 1:
                raise
            wait = int(exc.headers.get("Retry-After", 30 * (attempt + 1)))
        except urllib.error.URLError, TimeoutError, http.client.HTTPException:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 30 * (attempt + 1)
        print(f"  retrying in {wait}s")
        time.sleep(wait)
    raise AssertionError("unreachable")


def short(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def query(endpoint: str, qids: list[str]) -> list[tuple[str, str, str, str]]:
    """Attribute rows (qid, property, value, precision) for `qids`."""
    rows = []
    for prop, pattern in PROPERTY_QUERIES.items():
        for b in run_sparql(endpoint, pattern, qids):
            value = b["value"]["value"]
            if prop in LINK_PROPERTIES:
                value = short(value)
            precision = b["precision"]["value"] if "precision" in b else ""
            rows.append((short(b["item"]["value"]), prop, value, precision))

    # Every existing item has a sitelinks count; no rows at all means deleted or merged.
    returned = {r[0] for r in rows}
    rows += [(q, "missing", "", "") for q in qids if q not in returned]
    return rows


def resumable_writer(path: Path, header: list[str], key: str) -> tuple[set[str], TextIO, Any]:
    """Keys already in `path`, and an append-mode csv writer (header written if new)."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            done = {row[key] for row in csv.DictReader(f, delimiter="\t")}
    new_file = not path.exists()
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    if new_file:
        writer.writerow(header)
    return done, f, writer


def download_attributes(endpoint: dict, humans: list[str], path: Path) -> None:
    done, f, writer = resumable_writer(path, ["qid", "property", "value", "precision"], "qid")
    todo = [q for q in humans if q not in done]
    print(f"attributes: {len(done):,} people done, {len(todo):,} to go -> {path}")
    started = time.time()
    with f:
        for start in range(0, len(todo), endpoint["batch_size"]):
            batch = todo[start : start + endpoint["batch_size"]]
            writer.writerows(query(endpoint["url"], batch))
            f.flush()
            finished = start + len(batch)
            eta = (time.time() - started) / finished * (len(todo) - finished) / 60
            print(f"  {finished:,}/{len(todo):,}  (~{eta:.0f} min left)")


def download_labels(endpoint: dict, attributes_path: Path, path: Path) -> None:
    with attributes_path.open(encoding="utf-8") as f:
        values = {
            row["value"]
            for row in csv.DictReader(f, delimiter="\t")
            if row["property"] in LINK_PROPERTIES
        }
    done, f, writer = resumable_writer(path, ["qid", "label"], "qid")
    todo = sorted(values - done)
    print(f"labels: {len(done):,} done, {len(todo):,} to go -> {path}")
    with f:
        for start in range(0, len(todo), endpoint["batch_size"]):
            chunk = todo[start : start + endpoint["batch_size"]]
            labels = {
                short(b["item"]["value"]): b["label"]["value"]
                for b in run_sparql(endpoint["url"], LABEL_PATTERN, chunk)
            }
            # Values with no English label get an empty one, so a rerun does not ask again.
            writer.writerows((q, labels.get(q, "")) for q in chunk)
            f.flush()
            print(f"  {start + len(chunk):,}/{len(todo):,}")


def write_source(name: str, path: Path) -> None:
    """Record which endpoint (and, for QLever, which Wikidata dump) the download came from."""
    url = ENDPOINTS[name]["url"]
    lines = [f"endpoint: {name} {url}", f"started: {dt.datetime.now(dt.UTC).isoformat()}"]
    if name == "qlever":
        request = urllib.request.Request(f"{url}?cmd=stats", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            lines.append(f"index: {json.load(response)['name-index']}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N humans (for testing)")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="output file date")
    parser.add_argument("--endpoint", choices=ENDPOINTS, default="qlever")
    args = parser.parse_args()
    endpoint = ENDPOINTS[args.endpoint]

    humans = load_humans(SPLITS)
    print(f"{len(humans):,} humans in Wikidata5M")
    if args.limit:
        humans = humans[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source(args.endpoint, OUT_DIR / f"source_{args.date}.txt")
    attributes_path = OUT_DIR / f"attributes_{args.date}.tsv"
    download_attributes(endpoint, humans, attributes_path)
    download_labels(endpoint, attributes_path, OUT_DIR / f"labels_{args.date}.tsv")


if __name__ == "__main__":
    main()
