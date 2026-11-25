"""Turn Wikidata5M TSV triples into integer IDs with PyKEEN's TriplesFactory.

Run from the repository root with:  PYTHONPATH=src uv run python scripts/01_extract_wikidata5m.py
"""

import json
import resource
import sys
import time

import numpy as np
from pykeen.triples import TriplesFactory

from fairkg.constants import PROCESSED_DIR, RAW_W5M_FILES, ROOT
from fairkg.data.wikidata5m import (
    edges_path,
    entity_map_path,
    relation_map_path,
)


def peak_ram_mb() -> float:
    """Peak resident memory of this process so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    start = time.perf_counter()
    splits = {
        name: np.loadtxt(path, dtype=str, delimiter="\t", comments=None, encoding="utf-8")
        for name, path in RAW_W5M_FILES.items()
    }

    # IDs come from all three files together, so they are valid for every split.
    factory = TriplesFactory.from_labeled_triples(np.concatenate(list(splits.values())))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for path, mapping in [
        (entity_map_path(), factory.entity_to_id),
        (relation_map_path(), factory.relation_to_id),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)

    # Map each split with the shared IDs. compact_id=False keeps IDs from being
    # renumbered when a split does not contain every entity. PyKEEN sorts the rows,
    # so the row order inside a split is not the order of the raw file.
    for name, triples in splits.items():
        split_factory = TriplesFactory.from_labeled_triples(
            triples,
            entity_to_id=factory.entity_to_id,
            relation_to_id=factory.relation_to_id,
            compact_id=False,
            filter_out_candidate_inverse_relations=False,
        )
        edges = split_factory.mapped_triples.numpy().astype(np.int32)  # (head, relation, tail)
        np.save(edges_path(name), edges)
        print(f"saved:     {edges_path(name).relative_to(ROOT)} {edges.shape} {edges.dtype}")

    print(f"triples:   {factory.num_triples:,}")
    print(f"entities:  {factory.num_entities:,}")
    print(f"relations: {factory.num_relations:,}")
    print(f"time:      {time.perf_counter() - start:.1f} s")
    print(f"peak RAM:  {peak_ram_mb():,.0f} MB")


if __name__ == "__main__":
    main()
