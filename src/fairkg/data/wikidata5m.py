"""Loading interfaces for cached Wikidata5M arrays.

The cache is written by scripts/01_extract_wikidata5m.py. Every file name carries a
version, so a changed setting writes a new set of files instead of overwriting the
old one. The edges and the ID maps of one version belong together.
"""

import json
from pathlib import Path

import numpy as np

from fairkg.constants import PROCESSED_DIR, SPLITS, VERSION


def edges_path(split: str, version: str = VERSION, directory: Path = PROCESSED_DIR) -> Path:
    """Path of the int32 [E, 3] array (head, relation, tail) of one split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}: use one of {SPLITS}")
    return directory / f"edges_{split}_{version}.npy"


def entity_map_path(version: str = VERSION, directory: Path = PROCESSED_DIR) -> Path:
    return directory / f"entity2id_{version}.json"


def relation_map_path(version: str = VERSION, directory: Path = PROCESSED_DIR) -> Path:
    return directory / f"relation2id_{version}.json"


def load_edges(split: str, version: str = VERSION, directory: Path = PROCESSED_DIR) -> np.ndarray:
    """Memory-map the edges of one split: rows are read from disk only when accessed."""
    return np.load(edges_path(split, version, directory), mmap_mode="r")


def load_id_maps(
    version: str = VERSION, directory: Path = PROCESSED_DIR
) -> tuple[dict[str, int], dict[str, int]]:
    """Load (entity2id, relation2id) for one version."""
    with open(entity_map_path(version, directory), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(relation_map_path(version, directory), encoding="utf-8") as f:
        relation2id = json.load(f)
    return entity2id, relation2id
