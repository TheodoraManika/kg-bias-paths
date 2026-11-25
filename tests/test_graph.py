"""Graph structure tests."""

import numpy as np
import torch

from fairkg.data.graph import build_graph

# (head, relation, tail); relation 2 is "sensitive" in these tests
EDGES = np.array(
    [
        [3, 0, 1],
        [0, 1, 2],
        [1, 2, 0],  # dropped
        [2, 0, 3],
        [0, 0, 3],
    ],
    dtype=np.int32,
)


def triples(graph: dict) -> set[tuple[int, int, int]]:
    source, target = graph["edge_index"].tolist()
    return set(zip(source, graph["edge_type"].tolist(), target, strict=True))


def test_sorted_by_source_with_matching_rowptr() -> None:
    graph = build_graph(EDGES, num_nodes=4, num_relations=3, drop_relation_ids=(2,))
    source = graph["edge_index"][0]
    assert torch.all(source[1:] >= source[:-1])
    rowptr = graph["rowptr"]
    for node in range(4):
        assert torch.all(source[rowptr[node] : rowptr[node + 1]] == node)
    assert rowptr[-1] == graph["edge_index"].shape[1]


def test_sensitive_relation_is_dropped_and_inverses_added() -> None:
    graph = build_graph(EDGES, num_nodes=4, num_relations=3, drop_relation_ids=(2,))
    assert graph["edge_index"].shape == (2, 8)  # 4 kept edges, each with an inverse
    assert graph["settings"]["edges_removed"] == 1
    got = triples(graph)
    assert (1, 2, 0) not in got and (1, 5, 0) not in got and (0, 5, 1) not in got
    assert (3, 0, 1) in got  # original edge
    assert (1, 3, 3) in got  # its inverse: tail -> head, relation 0 + num_relations 3


def test_dtypes_and_no_drop() -> None:
    graph = build_graph(EDGES, num_nodes=4, num_relations=3)
    assert graph["edge_index"].dtype == torch.int64
    assert graph["edge_type"].dtype == torch.int64
    assert graph["edge_index"].shape[1] == 2 * len(EDGES)
