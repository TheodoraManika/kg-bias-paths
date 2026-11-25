"""Build the PyG graph structure once and save it, so training runs load it in seconds.

Reads the train edges and ID maps written by scripts/01_extract_wikidata5m.py, drops the sensitive
relations, adds inverse edges, sorts by source node and saves with torch.save. Then it times loading
the file and starting a NeighborLoader.

Run from the repository root with:  PYTHONPATH=src uv run python scripts/02c_build_graph.py
The variant with birth-decade nodes (needs people_v2.parquet):  ... 02c_build_graph.py --decade
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from fairkg.constants import (
    GRAPH_DECADE_DEFAULT,
    GRAPH_DECADE_SETTINGS,
    GRAPH_SETTINGS,
    LOADER_BATCH_SIZE,
    NUM_NEIGHBORS,
    PEOPLE_FILE,
    PROCESSED_DIR,
    SENSITIVE_RELATIONS,
    VERSION,
)
from fairkg.data.graph import (
    build_graph,
    build_graph_with_decades,
    file_sha256,
    graph_path,
    load_graph,
    save_graph,
)


def main(decade: bool) -> None:
    settings = GRAPH_DECADE_SETTINGS if decade else GRAPH_SETTINGS
    with open(PROCESSED_DIR / f"relation2id_{VERSION}.json", encoding="utf-8") as f:
        relation2id = json.load(f)
    with open(PROCESSED_DIR / f"entity2id_{VERSION}.json", encoding="utf-8") as f:
        entity2id = json.load(f)
    num_nodes = len(entity2id)
    edges = np.load(PROCESSED_DIR / f"edges_train_{VERSION}.npy", mmap_mode="r")
    drop = tuple(relation2id[name] for name in SENSITIVE_RELATIONS)
    print(f"train edges: {len(edges):,}; dropping relations {dict(zip(SENSITIVE_RELATIONS, drop))}")

    start = time.perf_counter()
    if decade:
        # The variant: also a "born in decade" edge from every person to a birth-bucket node.
        # To make this the default for a plain run, set GRAPH_DECADE_DEFAULT = True in
        # src/fairkg/constants.py (the file name keeps _decade, so the graphs are never mixed up).
        people = pd.read_parquet(PROCESSED_DIR / PEOPLE_FILE, columns=["id", "birth_decade"])
        people = people[people["id"].isin(entity2id)]  # every human should be in the graph
        graph = build_graph_with_decades(
            edges,
            num_nodes,
            len(relation2id),
            people["id"].map(entity2id).to_numpy(),
            people["birth_decade"].to_numpy(),  # NaN -> the "unknown" node
            drop,
        )
        info = graph["settings"]
        print(
            f"  decade edges: {info['birth_decade_edges']:,}; buckets: {len(info['decade_buckets'])} + unknown"
        )
    else:
        graph = build_graph(edges, num_nodes, len(relation2id), drop)
    print(f"built in {time.perf_counter() - start:.1f} s")
    print(f"  edges removed (sensitive): {graph['settings']['edges_removed']:,}")
    print(f"  edges in graph (with inverses): {graph['edge_index'].shape[1]:,}")
    assert bool((graph["edge_index"][0][1:] >= graph["edge_index"][0][:-1]).all()), "not sorted"

    path = graph_path(settings=settings)
    save_graph(graph, path)
    size_mb = path.stat().st_size / 1e6
    print(f"saved {path.relative_to(PROCESSED_DIR.parents[1])} ({size_mb:,.0f} MB)")

    # What a training run does: load the file and start a NeighborLoader.
    start = time.perf_counter()
    loaded = load_graph(settings=settings)
    load_seconds = time.perf_counter() - start
    data = Data(
        edge_index=loaded["edge_index"],
        edge_type=loaded["edge_type"],
        num_nodes=loaded["num_nodes"],
    )
    generator = torch.Generator().manual_seed(0)
    seeds = torch.randint(0, num_nodes, (LOADER_BATCH_SIZE,), generator=generator)
    loader = NeighborLoader(
        data, num_neighbors=NUM_NEIGHBORS, batch_size=LOADER_BATCH_SIZE, input_nodes=seeds
    )
    batch = next(iter(loader))
    total_seconds = time.perf_counter() - start
    print(
        f"load_graph: {load_seconds:.1f} s; load + first NeighborLoader batch: {total_seconds:.1f} s"
    )
    print(f"  first batch: {batch.num_nodes:,} nodes, {batch.edge_index.shape[1]:,} edges")
    print(f"  target < 30 s: {'OK' if total_seconds < 30 else 'TOO SLOW'}")

    print("\nmanifest line:")
    print(
        f"{path.relative_to(PROCESSED_DIR.parents[1])}\tbuilt by scripts/02c_build_graph.py "
        f"from edges_train_{VERSION}.npy{' --decade' if decade else ''}\t{time.strftime('%Y-%m-%d')}\t{path.stat().st_size}\t"
        f"{file_sha256(path)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--decade",
        action=argparse.BooleanOptionalAction,
        default=GRAPH_DECADE_DEFAULT,
        help="add a birth-bucket node and a born_in_decade edge per person (saved as its own file)",
    )
    main(parser.parse_args().decade)
