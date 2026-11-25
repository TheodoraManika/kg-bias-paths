"""Graph structure tests."""

import numpy as np
import torch

from fairkg.data.graph import (
    birth_bucket_index,
    birth_bucket_labels,
    build_graph,
    build_graph_with_decades,
)

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


def test_birth_buckets() -> None:
    labels = birth_bucket_labels()
    assert labels[0] == "before_1500" and labels[1] == "1500-1549" and labels[6] == "1750-1799"
    assert labels[7] == "1800s" and labels[-1] == "2020s" and len(labels) == 30
    year = np.array(
        [-4210.0, 1490.0, 1500.0, 1549.0, 1550.0, 1790.0, 1800.0, 1860.0, 2020.0, 2030.0]
    )
    got = [labels[i] for i in birth_bucket_index(year)]
    assert got == [
        "before_1500", "before_1500", "1500-1549", "1500-1549", "1550-1599", "1750-1799",
        "1800s", "1860s", "2020s", "2020s",
    ]  # fmt: skip
    assert birth_bucket_index(np.array([np.nan]))[0] == len(labels)  # the unknown bucket


def test_decade_variant_adds_bucket_nodes_relation_and_edges() -> None:
    persons = np.array([0, 1, 2, 3])
    decades = np.array([-4210.0, 1860.0, np.nan, 1860.0])
    graph = build_graph_with_decades(
        EDGES,
        num_nodes=4,
        num_relations=3,
        person_node_ids=persons,
        birth_decades=decades,
        drop_relation_ids=(2,),
    )
    info = graph["settings"]
    num_buckets = len(birth_bucket_labels())  # 30; the unknown node is one more
    assert info["decade_node_start"] == 4 and info["unknown_decade_node"] == 4 + num_buckets
    assert graph["num_nodes"] == 4 + num_buckets + 1
    assert graph["num_relations"] == 4 and info["born_in_decade_relation_id"] == 3
    node_1860s = 4 + birth_bucket_labels().index("1860s")
    got = triples(graph)
    assert (0, 3, 4) in got  # born before 1500 -> the first bucket node
    assert (1, 3, node_1860s) in got and (3, 3, node_1860s) in got
    assert (2, 3, info["unknown_decade_node"]) in got
    assert (node_1860s, 3 + 4, 1) in got  # inverse: relation 3 + new num_relations 4
    assert graph["edge_index"].shape[1] == 2 * (
        4 + 4
    )  # 4 kept edges + 4 decade edges, with inverses
    rowptr = graph["rowptr"]
    assert rowptr.shape[0] == graph["num_nodes"] + 1 and rowptr[-1] == graph["edge_index"].shape[1]
