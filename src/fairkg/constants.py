"""Constants shared by the scripts and the fairkg package: paths, versions and fixed settings.

Import from here instead of defining a path or a setting a second time. Constants that belong
to only one script (for example the sanity numbers of scripts/02_build_people_table.py) stay there.
"""

from pathlib import Path

# ---- paths ----------------------------------------------------------------------------------

# constants.py -> fairkg -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

RAW_W5M_DIR = ROOT / "data" / "raw" / "wikidata5m"
ATTRIBUTES_DIR = ROOT / "data" / "raw" / "wikidata_attributes"
PROCESSED_DIR = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
LOG_FILE = RUNS_DIR / "logs" / "experiment_log.csv"

OCCUPATION_HAND_FILE = ROOT / "configs" / "data" / "occupation_classes_hand.yaml"
OCCUPATION_P279_FILE = ROOT / "configs" / "data" / "occupation_classes_p279.yaml"

# ---- Wikidata5M and the cached arrays -------------------------------------------------------

# One raw file per official split, so train/valid/test stay separable.
RAW_W5M_FILES = {
    "train": RAW_W5M_DIR / "wikidata5m_transductive_train.txt",
    "valid": RAW_W5M_DIR / "wikidata5m_transductive_valid.txt",
    "test": RAW_W5M_DIR / "wikidata5m_transductive_test.txt",
}
SPLITS = ("train", "valid", "test")
# Version of the edge arrays and ID maps (data/processed/edges_*_<version>.npy).
VERSION = "v1"
PEOPLE_FILE = "people_v2.parquet"  # in PROCESSED_DIR; built by scripts/02_build_people_table.py

SENSITIVE_RELATIONS = ("P21", "P27")  # sex or gender, country of citizenship

# ---- graph (src/fairkg/data/graph.py, scripts/02c_build_graph.py) ---------------------------

# The settings are part of the file name, so a different setting writes a new file.
# train: only train edges (no valid/test edges, so evaluation cannot leak)
# nosens: the sensitive relations are removed
# inv: every edge also has an inverse edge, with relation ID + num_relations
GRAPH_SETTINGS = "train_nosens_inv"
# Variant: the same graph plus a "born in decade" edge from every person to a birth-bucket node.
GRAPH_DECADE_SETTINGS = "train_nosens_inv_decade"
GRAPH_DECADE_DEFAULT = False  # True: a plain run of 02c builds the decade variant

# Birth buckets: everything before EARLY_UNTIL in one node, COARSE_WIDTH-year buckets up to
# COARSE_UNTIL, then one decade per node up to LAST_DECADE (later years are counted in it).
EARLY_UNTIL = 1500
COARSE_UNTIL = 1800
COARSE_WIDTH = 50
LAST_DECADE = 2020

# Timing check of a training run: neighbours sampled per layer (two layers, as the planned
# R-GCN) and seed nodes per batch.
NUM_NEIGHBORS = [10, 10]
LOADER_BATCH_SIZE = 1024

# ---- occupation classes (src/fairkg/data/occupation.py, scripts/02a_...) --------------------

P279_ENDPOINT = "https://qlever.dev/api/wikidata"
P279_USER_AGENT = "fair-kg-lp/0.1 (research project; occupation class expansion)"
P279_BATCH_SIZE = 20_000
P279_MAX_RETRIES = 5

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
