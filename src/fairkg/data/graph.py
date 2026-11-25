"""Build, save and load the PyG graph structure (edge_index, edge_type) used for message passing.

The graph holds only the train edges, without the sensitive relations, and with an inverse edge
for every edge. It is built once and loaded by every training run. The settings are part of the
file name, so a different setting writes a new file instead of overwriting an old one.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from fairkg.constants import (
    COARSE_UNTIL,
    COARSE_WIDTH,
    EARLY_UNTIL,
    GRAPH_SETTINGS,
    LAST_DECADE,
    PROCESSED_DIR,
    VERSION,
)


def graph_path(
    version: str = VERSION, directory: Path = PROCESSED_DIR, settings: str = GRAPH_SETTINGS
) -> Path:
    return directory / f"graph_{version}_{settings}.pt"


def build_graph(
    edges: np.ndarray,
    num_nodes: int,
    num_relations: int,
    drop_relation_ids: tuple[int, ...] = (),
) -> dict:
    """Turn (head, relation, tail) rows into a graph sorted by source node (CSR layout).

    Edges with a relation in `drop_relation_ids` are removed, then every remaining edge gets an
    inverse (tail -> head, relation + num_relations). Returns a dict of tensors and settings:
    edge_index [2, E] and edge_type [E] (int64, sorted by source), and rowptr [num_nodes + 1]:
    the edges of node n are positions rowptr[n]:rowptr[n + 1].
    """
    keep = ~np.isin(edges[:, 1], list(drop_relation_ids))
    kept = np.asarray(edges[keep], dtype=np.int64)
    head, relation, tail = kept[:, 0], kept[:, 1], kept[:, 2]
    if len(kept) and (
        kept.min() < 0 or relation.max() >= num_relations or kept[:, [0, 2]].max() >= num_nodes
    ):
        raise ValueError("edges contain IDs outside num_nodes / num_relations")

    source = np.concatenate([head, tail])
    target = np.concatenate([tail, head])
    edge_type = np.concatenate([relation, relation + num_relations])
    order = np.argsort(source, kind="stable")  # stable: the same input always gives the same file

    edge_index = torch.from_numpy(np.stack([source[order], target[order]]))
    edge_type = torch.from_numpy(edge_type[order])
    degree = torch.bincount(edge_index[0], minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(degree, dim=0)
    return {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "rowptr": rowptr,
        "num_nodes": num_nodes,
        "num_relations": num_relations,  # before the inverses are added
        "num_edge_types": 2 * num_relations,
        "settings": {
            "split": "train",
            "dropped_relation_ids": sorted(int(r) for r in drop_relation_ids),
            "inverse_edges": True,
            "edges_removed": int((~keep).sum()),
        },
    }


def birth_bucket_labels() -> list[str]:
    """Names of the birth buckets, in node order (the "unknown" node comes after the last one).

    Wikidata has people born thousands of years ago, so a node per decade would give hundreds of
    nodes with only a few people. The buckets are: everything before EARLY_UNTIL in one node,
    COARSE_WIDTH-year buckets up to COARSE_UNTIL, and one decade per node from there on.
    """
    labels = [f"before_{EARLY_UNTIL}"]
    labels += [
        f"{y}-{y + COARSE_WIDTH - 1}" for y in range(EARLY_UNTIL, COARSE_UNTIL, COARSE_WIDTH)
    ]
    labels += [f"{y}s" for y in range(COARSE_UNTIL, LAST_DECADE + 10, 10)]
    return labels


def birth_bucket_index(birth_decades: np.ndarray) -> np.ndarray:
    """Bucket number (an index into birth_bucket_labels()) per person; NaN -> the unknown bucket."""
    birth = np.asarray(birth_decades, dtype=np.float64)
    missing = np.isnan(birth)
    year = np.minimum(
        np.where(missing, EARLY_UNTIL, birth), LAST_DECADE
    )  # years after the last decade share it
    num_coarse = (COARSE_UNTIL - EARLY_UNTIL) // COARSE_WIDTH
    index = np.where(
        year < EARLY_UNTIL,
        0,
        np.where(
            year < COARSE_UNTIL,
            1 + (year - EARLY_UNTIL) // COARSE_WIDTH,
            1 + num_coarse + (year - COARSE_UNTIL) // 10,
        ),
    ).astype(np.int64)
    return np.where(missing, len(birth_bucket_labels()), index)


def decade_edges(
    person_node_ids: np.ndarray, birth_decades: np.ndarray, num_nodes: int, relation_id: int
) -> tuple[np.ndarray, list[str]]:
    """One (person, relation_id, bucket node) row per person.

    The bucket nodes get the IDs num_nodes, num_nodes + 1, ... in the order of
    birth_bucket_labels(). A person without a birth decade (NaN) points to one extra "unknown"
    node after the last bucket, so that having no date is explicit and not the same as having no
    edge. Returns the rows [N, 3] and the bucket labels.
    """
    person_node_ids = np.asarray(person_node_ids, dtype=np.int64)
    index = birth_bucket_index(birth_decades)
    rows = np.stack(
        [person_node_ids, np.full(len(index), relation_id, dtype=np.int64), num_nodes + index],
        axis=1,
    )
    return rows, birth_bucket_labels()


def build_graph_with_decades(
    edges: np.ndarray,
    num_nodes: int,
    num_relations: int,
    person_node_ids: np.ndarray,
    birth_decades: np.ndarray,
    drop_relation_ids: tuple[int, ...] = (),
) -> dict:
    """The graph of `build_graph` plus a "born in bucket" edge for every person.

    This is the variant (settings GRAPH_DECADE_SETTINGS, saved as its own file). Compared with the main
    graph it has:
      - len(birth_bucket_labels()) + 1 extra nodes (the buckets and one "unknown"), IDs from
        `num_nodes` on;
      - one extra relation, born_in_decade, with ID `num_relations`, so num_relations grows by 1
        and the inverse relations are shifted by one: relation + (num_relations + 1);
      - one extra edge (and its inverse) for every person in `person_node_ids`.
    `person_node_ids` and `birth_decades` are parallel arrays (node ID, decade as 1860.0 or NaN).
    The decade edges are not prediction targets, so they leak nothing from valid/test.
    """
    rows, labels = decade_edges(person_node_ids, birth_decades, num_nodes, num_relations)
    all_edges = np.concatenate([np.asarray(edges, dtype=np.int64), rows])
    graph = build_graph(
        all_edges, num_nodes + len(labels) + 1, num_relations + 1, drop_relation_ids
    )
    graph["settings"].update(
        {
            "birth_decade_edges": len(rows),
            "born_in_decade_relation_id": num_relations,
            "decade_node_start": num_nodes,
            "decade_buckets": labels,
            "unknown_decade_node": num_nodes + len(labels),
        }
    )
    return graph


def save_graph(graph: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def load_graph(
    version: str = VERSION, directory: Path = PROCESSED_DIR, settings: str = GRAPH_SETTINGS
) -> dict:
    return torch.load(graph_path(version, directory, settings), weights_only=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
