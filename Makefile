# Run the data pipeline built so far, in order. Run from the repository root.
#
#   make              list the targets
#   make all          everything that needs no internet, in the right order
#   make graph        one step; its inputs are built first if they are missing
#   make -n all       only print what would run
#   make status       which outputs exist
#
# Every data step is a file target: if its output exists, the step is skipped. To redo a step,
# delete its output (or `make clean`), or run `make -B <target>`.
# The raw Wikidata5M files are downloaded by hand and the attribute download is its own step.
# `make all` never starts a download: a missing input stops with a message instead.

.DEFAULT_GOAL := help
# If a script fails, delete the half-written output so the next make does not skip the step.
.DELETE_ON_ERROR:
.PHONY: help all setup check test lint ci status raw-check attributes occupations \
        edges graph graph-decade people covariates clean

# The scripts import the fairkg package from src/ (it is not installed).
export PYTHONPATH := src

UV ?= uv
PY := $(UV) run python

# The attribute download used for the people table. Pinned, so the pipeline is repeatable.
ATTR_DATE := 2026-10-25

RAW_W5M     := data/raw/wikidata5m
ATTRIBUTES  := data/raw/wikidata_attributes/attributes_$(ATTR_DATE).tsv
PROCESSED   := data/processed

W5M_FILES   := $(RAW_W5M)/wikidata5m_transductive_train.txt \
               $(RAW_W5M)/wikidata5m_transductive_valid.txt \
               $(RAW_W5M)/wikidata5m_transductive_test.txt
EDGES       := $(PROCESSED)/edges_train_v1.npy
GRAPH       := $(PROCESSED)/graph_v1_train_nosens_inv.pt
GRAPH_DECADE := $(PROCESSED)/graph_v1_train_nosens_inv_decade.pt
OCC_CLASSES := configs/data/occupation_classes_p279.yaml
REGIONS     := configs/data/regions.yaml
PEOPLE      := $(PROCESSED)/people_v2.parquet
COVARIATES  := $(PROCESSED)/covariates_v1.parquet

# ---- overview -------------------------------------------------------------------------------

help: ## show this list
	@echo "Pipeline (in order; each step skips if its output exists):"
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-13s %s\n", $$1, $$2}'

all: raw-check edges graph people covariates ## raw-check, edges, graph, people, covariates (no internet)

status: ## show which outputs exist
	@for f in $(W5M_FILES) $(ATTRIBUTES) $(OCC_CLASSES) $(EDGES) $(GRAPH) $(GRAPH_DECADE) $(PEOPLE) $(COVARIATES); do \
		if [ -f "$$f" ]; then printf "  [x] %s\n" "$$f"; else printf "  [ ] %s\n" "$$f"; fi; \
	done

# ---- environment and checks -----------------------------------------------------------------

setup: ## install the locked environment (uv sync --locked)
	$(UV) sync --locked

check: ## print Python, PyTorch and CUDA (00_check_env.py)
	$(PY) scripts/00_check_env.py

test: ## run pytest
	$(UV) run pytest

lint: ## run ruff
	$(UV) run ruff check .

ci: lint test ## what the CI runs: ruff and pytest

# ---- inputs that are not built by a script --------------------------------------------------

# The Wikidata5M files are downloaded by hand: links and checksums are in data/MANIFEST.txt.
$(W5M_FILES):
	@echo "missing: $@"
	@echo "Download wikidata5m_transductive.tar.gz (link in data/MANIFEST.txt) and extract it into $(RAW_W5M)/."
	@exit 1

raw-check: $(W5M_FILES) ## check the hand-downloaded Wikidata5M files are present
	@echo "Wikidata5M raw files present"

# 00: person attributes from QLever (about 15 minutes, needs internet). Only run on request.
$(ATTRIBUTES):
	@echo "missing: $@"
	@echo "Run 'make attributes' (about 15 minutes, needs internet)."
	@exit 1

attributes: ## 00: download gender, birth date, ... from Wikidata (~15 min, internet)
	@if [ -f "$(ATTRIBUTES)" ]; then echo "already have $(ATTRIBUTES)"; \
	else $(PY) scripts/00_download_wikidata_attributes.py --date $(ATTR_DATE); fi

# 02a is optional: its output is committed, and rerunning it queries the live Wikidata, so the
# result can change.
occupations: ## 02a: rebuild the occupation classes from Wikidata P279 (optional, internet)
	$(PY) scripts/02a_expand_occupation_classes.py

# ---- data steps -----------------------------------------------------------------------------

edges: $(EDGES) ## 01: Wikidata5M triples -> integer edge arrays + ID maps
$(EDGES): $(W5M_FILES)
	$(PY) scripts/01_extract_wikidata5m.py

graph: $(GRAPH) ## 02c: PyG graph (sorted, sensitive relations dropped, inverse edges)
$(GRAPH): $(EDGES)
	$(PY) scripts/02c_build_graph.py

# Same graph plus a born-in-decade node and edge per person (needs the people table).
graph-decade: $(GRAPH_DECADE) ## 02c --decade: the graph variant with birth-decade nodes
$(GRAPH_DECADE): $(EDGES) $(PEOPLE)
	$(PY) scripts/02c_build_graph.py --decade

# Never overwrites a versioned file, so it only runs if people_v2 is missing.
people: $(PEOPLE) ## 02: one row per human (gender, birth decade, regions, occupations, awards)
$(PEOPLE): $(W5M_FILES) $(ATTRIBUTES) $(OCC_CLASSES) $(REGIONS)
	$(PY) scripts/02_build_people_table.py --attrs $(ATTRIBUTES) --out $(PEOPLE)

covariates: $(COVARIATES) ## 03: degrees, award counts, birth decade, degree deciles
$(COVARIATES): $(PEOPLE) $(W5M_FILES)
	$(PY) scripts/03_compute_covariates.py --people $(PEOPLE) --out $(COVARIATES)

# ---- clean-up -------------------------------------------------------------------------------

# Removes what `make all` and `make graph-decade` build, so the next run rebuilds it. Never
# touches data/raw/ (Wikidata5M, the attribute download) or the committed configs. Needed to redo
# `people`, because that script refuses to overwrite an existing versioned file.
clean: ## delete the built files in data/processed (raw data is kept)
	rm -f $(PROCESSED)/edges_*_v1.npy $(PROCESSED)/entity2id_v1.json $(PROCESSED)/relation2id_v1.json
	rm -f $(GRAPH) $(GRAPH_DECADE) $(PEOPLE) $(COVARIATES)
