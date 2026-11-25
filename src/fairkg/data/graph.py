"""Build, save and load the PyG graph structure (edge_index, edge_type) used for message passing.

The graph holds only the train edges, without the sensitive relations, and with an inverse edge
for every edge. It is built once and loaded by every training run. The settings are part of the
file name, so a different setting writes a new file instead of overwriting an old one.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

# graph.py -> data -> fairkg -> src -> repo root
ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"

# Version of the edge arrays and ID maps (data/processed/edges_*_<version>.npy) the graph is built from.
VERSION = "v1"
# train: only train edges (no valid/test edges, so evaluation cannot leak)
# nosens: the sensitive relations are removed
# inv: every edge also has an inverse edge, with relation ID + num_relations
SETTINGS = "train_nosens_inv"
SENSITIVE_RELATIONS = ("P21", "P27")  # sex or gender, country of citizenship


def graph_path(version: str = VERSION, directory: Path = PROCESSED_DIR) -> Path:
    return directory / f"graph_{version}_{SETTINGS}.pt"


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


def save_graph(graph: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def load_graph(version: str = VERSION, directory: Path = PROCESSED_DIR) -> dict:
    return torch.load(graph_path(version, directory), weights_only=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
