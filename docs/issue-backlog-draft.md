# Issue backlog draft — Fair KG Link Prediction

Source: `docs/Step-by-Step Project Plan Fair KG Link Prediction.md`, following `docs/Research Habits and GitHub Workflow.md` §7–8.
Edit freely: delete issues, change owners, fix wording. Nothing has been created on GitHub yet.

## Conventions used

- **Title prefix** from the issue forms: `[Task] ` or `[Experiment] `.
- **Type label** from the forms: `task` or `experiment`. **Area label** from the guide: `data`, `model`, `fairness`, `writing`, `infra`.
- **Body**: Task = Goal / Done when / Notes. Experiment = Hypothesis / Config / Metrics to report / Result (left empty).
- **Owner** from the team split. "Teammate" stands in for a GitHub username I don't have yet.
- **Proposed new label:** `gate` (decision-gate issues: weeks 1, 6, 9 and 11).
- **Every `[Experiment]` issue** gets this pre-run checklist appended (from `docs/Faster Training and Fair Comparisons Practical Guide.md`, "Checklist before every experiment"):
  - [ ] Loads the cached data files (no raw parsing)
  - [ ] Uses the frozen split files and the shared evaluation module
  - [ ] Config file saved with the run; seed set
  - [ ] Smoke test passed (a few batches + one evaluation)
  - [ ] Passed the synthetic-graph test, if it is a new method or component
  - [ ] Mixed precision on; GPU utilisation checked in nvidia-smi
  - [ ] Checkpointing and early stopping on
  - [ ] Same tuning budget as the methods it will be compared with
  - [ ] Logged automatically: metrics, fairness metrics, training time, GPU memory
- **Comparison protocol** (guide §6) applies to every experiment that compares methods: same data, splits, input embeddings and evaluation code; same seeds and test queries (paired); 3–5 seeds, mean ± sd; a difference smaller than the seed-to-seed variation counts as "no difference".

## Milestones (proposed)

Weeks end on Wednesday, so the guide's example dates line up (Phase 2 → 18 Nov, Phase 5 → 9 Dec). Week 1 = 29 Oct – 4 Nov 2026.

| # | Milestone | Weeks | Due |
|---|-----------|-------|-----|
| M0 | Phase 0: Setup and scope | 1 | 2026-11-04 |
| M1 | Phase 1: Literature and hypotheses | 1 | 2026-11-04 |
| M2 | Phase 2: Data and splits | 2–3 | 2026-11-18 |
| M3 | Phase 3: Data audit | 3 | 2026-11-18 |
| M4 | Phase 4: Embeddings, probe, synthetic, shared code | 3–4 | 2026-11-25 |
| M5 | Phase 5: Baselines and decomposition | 5–6 | 2026-12-09 |
| MA | Version A: Counterfactual objective | 7–13 | 2027-01-27 |
| MB | Version B: Modules + objective | 7–16 | 2027-02-17 |
| MX | Optional: Semi-supervised fairness | — | none |

---

## M0 · Phase 0: Setup and scope (week 1)

### 0.1 [Task] Send proposal to professor and confirm version, deadline and novelty bar
**Labels:** task, writing, gate · **Assignee:** Theodora + Teammate · **Week:** 1
**Goal:** Written confirmation of the scope before any data work starts.
**Done when:**
- [ ] Proposal PDF (`docs/Project Proposal…`) emailed to the professor
- [ ] Reply recorded in the lab notebook: Version A or B, submission deadline, whether a new training objective counts as enough novelty
- [ ] If the scope changes, MA/MB milestone due dates are updated
**Notes:** Gate for week 1. If it fails: adjust the scope or the claimed contribution.

### 0.2 [Task] Novelty check: search Scholar and arXiv 2025–2026 for path-specific counterfactual fairness in KG link prediction
**Labels:** task, writing, gate · **Assignee:** Theodora · **Week:** 1
**Goal:** Confirm that no 2025–2026 paper already decomposes KG link-prediction bias into direct and popularity paths with a counterfactual objective.
**Done when:**
- [ ] Exact query strings and search date logged in the notebook ("path-specific counterfactual fairness" × "knowledge graph" / "link prediction"; also "fair knowledge graph embedding" 2025–2026)
- [ ] Closest 5–10 papers added to Zotero, each with a one-line note on how it differs from ours
- [ ] Verdict written: "novelty holds" or "adjust claim to …"

### 0.3 [Task] Install NVIDIA driver, PyTorch ≥ 2.7 (CUDA 12.8), PyG and PyKEEN on the RTX 5060 Ti through uv
**Labels:** task, infra · **Assignee:** Theodora · **Week:** 1
**Goal:** A GPU machine where `uv sync --locked` gives a working CUDA stack (Blackwell needs CUDA 12.8 wheels).
**Done when:**
- [ ] `nvidia-smi` shows a driver that supports CUDA 12.8 (WSL2 if on Windows)
- [ ] `pyproject.toml` has the cu128 PyTorch index and the PyG find-links; `uv.lock` committed
- [ ] `uv run python -c "import torch; print(torch.cuda.get_device_name(0))"` prints the 5060 Ti
- [ ] `torch_geometric` and `pykeen` import without errors
**Notes:** See `docs/Environment and Reproducibility with uv.md`.

### 0.4 [Task] Implement `scripts/00_check_env.py`: GPU, R-GCN forward/backward, NeighborLoader, bf16
**Labels:** task, infra · **Assignee:** Theodora · **Week:** 1
**Goal:** One command that proves a machine (5060 Ti or Colab) can run our pipeline.
**Done when:**
- [ ] The script prints torch/CUDA/PyG/PyKEEN versions, GPU name and VRAM
- [ ] Builds a random graph (10k nodes, 5 relations), runs a 2-layer `RGCNConv` with `NeighborLoader([15, 10])` forward and backward on the GPU
- [ ] Runs one step under `torch.autocast(dtype=torch.bfloat16)` and reports whether bf16 is supported
- [ ] Exits non-zero on any failure; output pasted into this issue for the 5060 Ti and for Colab

### 0.5 [Task] Configure GitHub: labels, milestones, project board, branch protection on main
**Labels:** task, infra · **Assignee:** Theodora · **Week:** 1
**Goal:** The repo setup from the workflow guide, so every task is an issue and every change a reviewed PR.
**Done when:**
- [ ] Labels `data`, `model`, `fairness`, `experiment`, `writing`, `infra`, `blocked`, `task`, `gate` exist
- [ ] Milestones M0–MX exist with due dates
- [ ] Project board with Todo → In progress → In review → Done, fields Owner/Phase/Week, auto-add for new issues
- [ ] `main` protected: PR required, 1 approval, `tests.yml` must pass
- [ ] Teammate added as a collaborator

### 0.6 [Task] Set up the shared experiment log and `src/fairkg/utils/run_logging.py`
**Labels:** task, infra · **Assignee:** Teammate · **Week:** 1
**Goal:** Every run writes a self-describing folder and a row in the shared log.
**Done when:**
- [ ] Decision recorded: Weights & Biases project or shared spreadsheet
- [ ] `run_logging.py` creates `runs/<date>_<name>_<seed>/` containing resolved config, git commit hash, dirty flag, seed, `uv.lock` hash and `metrics.json`
- [ ] `src/fairkg/utils/seed.py` seeds python/numpy/torch/CUDA in one call
- [ ] A dummy run appears in the log with all fields filled in

### 0.7 [Task] Write `docs/definitions.md`: tasks, sensitive groups and the gender-relevant relation whitelist
**Labels:** task, fairness · **Assignee:** Theodora + Teammate · **Week:** 1
**Goal:** One reference for what we predict, which groups we compare and which relations may legitimately use gender.
**Done when:**
- [ ] Tasks: P166 (award received) primary, P106 (occupation) secondary, in the form (person, r, ?)
- [ ] Gender groups: P21 values listed with QIDs (male Q6581097, female Q6581072, non-binary Q48270, trans woman Q1052281, trans man Q2449503, intersex Q1097630 — verify) and the rule for people with several values
- [ ] Whitelist of relations or award classes where gender is part of the fact (e.g. gender-specific awards such as "Best Actress"), with the rule for spotting them
- [ ] Fairness metrics defined as formulas: DP gap@k, equal-opportunity gap, amplification ratio = model disparity ÷ data disparity

### 0.8 [Task] Build the P27 → region mapping in `configs/data/regions.yaml`
**Labels:** task, data · **Assignee:** Theodora · **Week:** 1
**Goal:** A fixed, documented mapping from country of citizenship to a region group.
**Done when:**
- [ ] Scheme chosen and justified (e.g. UN M49 regions, or Global North/South) and written in `docs/definitions.md`
- [ ] Every country QID that appears as a P27 object in Wikidata5M is mapped, or listed as unmapped with a count
- [ ] Rule for several P27 values decided (e.g. multi-membership or "most frequent region")
- [ ] Historical states (USSR, Prussia, …) mapped by hand

---

## M1 · Phase 1: Literature and hypotheses (week 1)

Each issue adds one section to `research/hypotheses.md` and its references to `references.bib`.

### 1.1 [Task] Hypothesis "popularity": read Arduini et al. 2020 and Shomer et al. 2023, write paragraph and test plan
**Labels:** task, writing · **Assignee:** Theodora · **Week:** 1
**Goal:** State the mechanism by which better-connected people get better predictions, and how we will measure it.
**Done when:**
- [ ] Paragraph: claim, mechanism, evidence from both papers
- [ ] Test plan: degree stratification + Oaxaca–Blinder (Phase 5), and the result that would confirm or reject it
- [ ] Both papers in Zotero / `references.bib`

### 1.2 [Task] Hypothesis "data imbalance": read Radstok et al. 2021 and AuditLP 2025, write paragraph and test plan
**Labels:** task, writing · **Assignee:** Theodora · **Week:** 1
**Done when:**
- [ ] Paragraph on under-representation of women and Global South people and on the number of facts per person
- [ ] Test plan: Phase 3 data audit tables, with the numbers that would count as "imbalanced"
- [ ] References added

### 1.3 [Task] Hypothesis "proxies": read Fisher et al. 2020 and Du et al. 2022, write paragraph and test plan
**Labels:** task, writing · **Assignee:** Teammate · **Week:** 1
**Done when:**
- [ ] Paragraph on gender leaking through spouse, gendered occupations and names
- [ ] Test plan: probe classifier on embeddings (Phase 4), with the probe accuracy above chance that would count as leakage
- [ ] References added

### 1.4 [Task] Hypothesis "message passing / homophily": read FairGNN, FairVGNN and Fair-RGNN, write paragraph and test plan
**Labels:** task, writing · **Assignee:** Teammate · **Week:** 1
**Done when:**
- [ ] Paragraph on how aggregation spreads group information
- [ ] Test plan: depth sweep 0–3 layers (Phase 5); the expected signature is a gap that grows with depth
- [ ] References added

### 1.5 [Task] Hypothesis "missing labels": read Kraft & Usbeck 2022, write limitation paragraph
**Labels:** task, writing · **Assignee:** Theodora · **Week:** 1
**Done when:**
- [ ] Paragraph on awards that people received but Wikidata never recorded, framed as a limitation
- [ ] Optional spot-check protocol written down (e.g. 50 random low-degree women checked by hand against Wikipedia)
- [ ] Reference added

---

## M2 · Phase 2: Data and splits (weeks 2–3)

### 2.1 [Task] Download Wikidata5M and record checksums in the data manifest
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Done when:**
- [ ] Transductive split, aliases and text files saved in `data/raw/wikidata5m/` (git-ignored)
- [ ] `data/MANIFEST` (or the equivalent from `docs/Where Data and Cached Files Live.md`) lists URL, download date, size and sha256 for each file
- [ ] Line counts match the published statistics (~4.6M entities, ~20.6M triples, 822 relations)

### 2.2 [Task] Implement `scripts/01_extract_wikidata5m.py`: stream raw triples into integer edge arrays
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Goal:** Cached `int32` arrays (head, relation, tail) and ID maps without ever holding the raw text in memory.
**Done when:**
- [ ] Reads files line by line (check the format: Wikidata5M ships TSV triples, not JSONL); never calls `.read()` on a whole file
- [ ] Writes `data/processed/edges.npy` (shape [E, 3], int32) and `entity2id.json` / `relation2id.json`
- [ ] Cache filenames encode their settings (e.g. `edges_v1.npy`, `people_v1_seed42_rel80.parquet`), so changing a setting writes a new file instead of overwriting
- [ ] Peak RAM logged; runs end to end in < 30 min
- [ ] `src/fairkg/data/wikidata5m.py` loads the arrays with `mmap_mode="r"`
- [ ] Unit test on a 1,000-line fixture

### 2.3 [Task] Implement `scripts/02_build_people_table.py`: humans with P21, P27, P106, P166 and their counts
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Done when:**
- [ ] Humans = entities with (P31, Q5); stored in `people.parquet` with columns id, gender, regions, occupations, awards
- [ ] Notebook table: number of humans; coverage of P21/P27/P106/P166; counts per P21 value; counts per region
- [ ] Decision recorded: is P166 big enough to be the primary task, or do we switch to P106?

### 2.4 [Task] Fetch birth dates (P569) for selected humans, because Wikidata5M has no literals
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Goal:** A birth-decade column for stratification and covariates.
**Done when:**
- [ ] Source chosen and documented (batched Wikidata SPARQL queries, or a filtered pass over a Wikidata JSON dump), with the snapshot date
- [ ] `birth_decade` added to `people.parquet`; share missing reported by gender and region
- [ ] Raw responses cached so the step can be rerun offline

### 2.5 [Task] Map P106 occupations to ~15–25 broad occupation classes
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Goal:** A coarse occupation variable for stratification and CAF-style matching.
**Done when:**
- [ ] Mapping in `configs/data/occupation_classes.yaml` (from P279 subclass chains, or by hand for the top ~300 occupations)
- [ ] ≥ 95% of humans with a P106 get a class; the remainder go to "other"
- [ ] Rule for several occupations decided (e.g. most frequent class)

### 2.6 [Task] Select 50–100 person-related relations and map the rest to "other"
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Done when:**
- [ ] `configs/data/relations.yaml` lists the kept relations with frequencies among human-incident edges
- [ ] Kept relations cover ≥ X% of human-incident edges (X written in the issue)
- [ ] Sensitive relations (P21, P27) flagged so later steps can drop them
- [ ] Reduction applied once when saving the cache, never inside the training loop

### 2.6b [Task] Build and save the PyG graph structure once (edge_index, edge_type, CSR-sorted)
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Goal:** Training runs load a ready graph in seconds instead of rebuilding it.
**Done when:**
- [ ] After the relation reduction (2.6), edges converted to `edge_index` / `edge_type` tensors, sorted by source node (CSR layout), and saved with `torch.save` under a settings-encoded name (e.g. `graph_v1_rel80.pt`)
- [ ] Loading the graph + starting `NeighborLoader` takes < 30 s (time logged)
- [ ] Checksum in the data manifest

### 2.7 [Task] Implement `scripts/03_compute_covariates.py`: degree, award count and birth decade on the full graph
**Labels:** task, data · **Assignee:** Theodora · **Week:** 2
**Done when:**
- [ ] In-, out- and total degree computed on the **full** graph (not the sample), excluding P21/P27 edges
- [ ] Number of awards, number of facts and birth decade joined into `covariates.parquet`
- [ ] Degree deciles stored, for stratification

### 2.8 [Task] Implement stratified sampling in `src/fairkg/data/sampling.py`
**Labels:** task, data · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Strata = gender × region × degree decile × birth decade × broad occupation; empty and tiny strata handled (merge rule written down)
- [ ] Both modes available: proportional, and oversampling small groups with `sampling_weight` stored per person
- [ ] Decision on which mode to use recorded in the notebook
- [ ] Settings in `configs/data/sampling.yaml`; seed in the config

### 2.9 [Task] Compare the sample with the full graph: category shares, degree quantiles, SMD, award rates
**Labels:** task, data · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Function in `sampling.py` writes `reports/tables/sample_vs_full.csv`
- [ ] Every covariate has |SMD| < 0.1, or the exceptions are listed and explained
- [ ] Degree quantiles (10/50/90/99) and award rate by gender and by region side by side

### 2.10 [Task] Implement leakage rules in `src/fairkg/data/splits.py` and extend `tests/test_splits_no_leakage.py`
**Labels:** task, data · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Relations that are inverses or near-duplicates of P166 found automatically (e.g. > 0.8 triple overlap with P166 reversed) and listed
- [ ] P1411 (nominated for) removed for (person, award) pairs that appear in val/test
- [ ] Test fails if any val/test (h, P166, t) can be read off training through a leaking relation

### 2.11 [Task] Implement `scripts/04_sample_and_split.py`: train/val/test with group-size guarantees
**Labels:** task, data · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Split stratified so that the sample-vs-full comparison (2.9) passes for each split
- [ ] Every test person has ≥ 1 training edge (asserted)
- [ ] ≥ 200 test triples per gender × region group; regions merged where needed, and the merges logged
- [ ] Split index files frozen, with the seed in the filename (e.g. `data/splits/p166_v1_seed42/train.npy`), plus a manifest (seed, sizes per group); every method loads these same files
- [ ] Fixed development validation sample of 5,000 queries saved (`val_dev5k.npy`), identical across runs

### 2.12 [Task] (Optional) Check whether edit dates exist for a temporal split
**Labels:** task, data · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Finding recorded: do we have statement-level dates (e.g. from the Wikidata dump) for P166?
- [ ] If yes, a temporal split prototype; if no, the issue is closed with "not available"

### 2.13 [Experiment] Robustness of data disparity to sampling seed and sample size
**Labels:** experiment, data · **Assignee:** Theodora · **Week:** 3
**Hypothesis:** Bias numbers from the sample do not depend on which sample we draw.
**Config / runs:** 3 seeds × 2 sample sizes (e.g. 200k and 500k people) through 04 and 05.
**Metrics to report:** data disparity table (3.2) per run, with bootstrap 95% CIs.
**Result:** _(filled after running: do all runs fall within each other's CIs?)_
**Notes:** Depends on 2.11 and 3.2.

### 2.14 [Task] Write the "Data and sampling" report section
**Labels:** task, writing · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Covers selection, strata, the sample-vs-full table, leakage rules and minimum group sizes
- [ ] States explicitly that we claim representativeness of Wikidata, not of the real world
- [ ] Draft in Overleaf; the teammate has reviewed it

---

## M3 · Phase 3: Data audit (week 3)

### 3.1 [Task] Compute group sizes, including non-binary and other P21 values, and decide which groups to analyse
**Labels:** task, data, fairness · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] `scripts/05_data_audit.py` writes the group sizes per P21 value and per region (full graph and sample)
- [ ] Decision recorded: which groups are compared directly and which are reported only descriptively, given the ≥ 200 test-triple rule

### 3.2 [Task] Build the data disparity table: award/occupation rates, degree and facts per person by group
**Labels:** task, fairness · **Assignee:** Theodora · **Week:** 3
**Done when:**
- [ ] Award rate and per-occupation rates by gender and by region
- [ ] Degree distribution by group (median, 90th percentile, KS test) and a histogram figure
- [ ] Number of facts per person by group
- [ ] Person-level bootstrap 95% CIs on every number
- [ ] Saved as `reports/tables/data_disparity.csv`: the denominator of the amplification ratio

---

## M4 · Phase 4: Embeddings, probe, synthetic, shared code (weeks 3–4)

### 4.1 [Task] Implement `scripts/06_train_transe_inputs.py`: 128-d TransE without sensitive edges
**Labels:** task, model · **Assignee:** Theodora · **Week:** 4
**Done when:**
- [ ] PyKEEN TransE (dim 128) trained once on full Wikidata5M with P21 removed (and P27, if we study region)
- [ ] Assertion that no P21/P27 triple is in the training set
- [ ] Embeddings saved as frozen **float16** `.npy` (~1.2 GB) with sha256 in the manifest; link-prediction MRR logged as a sanity check
- [ ] Loader memory-maps the file and copies only the rows each batch needs to the GPU
- [ ] Config in `configs/baselines/transe_inputs.yaml`

### 4.2 [Task] Implement the gender probe (`src/fairkg/eval/probe.py`, `scripts/07_probe.py`)
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] Logistic regression, 5-fold CV, class-balanced; reports balanced accuracy and AUC with CIs
- [ ] Majority-class and shuffled-label baselines reported next to it
- [ ] Works on any `[N, d]` matrix, so it can be reused for R-GCN layers and the Module B content channel
- [ ] Result for the 4.1 TransE embeddings recorded

### 4.3 [Task] Decide how to handle proxy leakage: INLP projection vs. keep and report
**Labels:** task, fairness, gate · **Assignee:** Theodora + Teammate · **Week:** 4
**Done when:**
- [ ] Decision rule written down (e.g. INLP if balanced accuracy > 0.60)
- [ ] If INLP: projected embeddings saved and the probe rerun; if not: the choice is justified in the notebook

### 4.4 [Task] Add a GraphVite pretrained-embedding loader that is dev-only
**Labels:** task, infra · **Assignee:** Teammate · **Week:** 3
**Done when:**
- [ ] Loader for GraphVite Wikidata5M TransE embeddings, mapped to our entity IDs
- [ ] Config flag `embeddings.source: graphvite` sets `run.dev_only = true`; `evaluate.py` refuses to write final report tables for dev-only runs

### 4.5 [Task] Implement the synthetic graph (`src/fairkg/data/synthetic.py`, `scripts/08_make_synthetic_graph.py`)
**Labels:** task, data, fairness · **Assignee:** Teammate · **Week:** 3–4
**Goal:** A small graph with known amounts of direct bias, popularity bias and a gender proxy.
**Done when:**
- [ ] Parameters: n_people, β_direct (gender → award), β_pop (gender → degree → award), proxy strength (gender → proxy relation)
- [ ] Each mechanism can be switched on independently; ground-truth effects saved with the graph
- [ ] Runs in < 1 min on CPU; generated with a fixed seed and saved to disk once (not regenerated per run)
- [ ] `tests/test_synthetic_bias.py`: with only β_direct on, the data gap matches the planted value within tolerance (same for β_pop)

### 4.6 [Task] Implement ranking metrics in `src/fairkg/eval/ranking.py`
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] Filtered MRR, Hits@1 and Hits@10 for (person, P166, ?), with type-constrained candidates (award entities only)
- [ ] Returns per-triple ranks, so fairness metrics, the decomposition and paired per-query comparisons can reuse them
- [ ] Person vectors computed once per evaluation (one GNN pass over all query people), then all candidates scored as one batched GPU matrix multiplication, not a loop over candidates
- [ ] `tests/test_ranking.py` agrees with PyKEEN's evaluator on a toy KG

### 4.7 [Task] Implement fairness metrics and bootstrap CIs (`eval/fairness.py`, `eval/stats.py`)
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] Demographic-parity gap@k, equal-opportunity gap (on true positives) and amplification ratio, as defined in `docs/definitions.md`
- [ ] Person-level (cluster) bootstrap, 1,000 resamples, 95% CI for every metric
- [ ] Sampling weights from 2.8 respected when present
- [ ] `tests/test_fairness_metrics.py` covers hand-computed toy cases, including a zero-gap case

### 4.8 [Task] Implement `scripts/09_build_eval_cache.py`: candidate sets, filter masks, group index
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] Candidate sets: all award entities (P166) and all occupation entities (P106)
- [ ] Filter masks: per test query, the known-true answers stored as sparse index lists
- [ ] Bookkeeping: each query person's gender, region and degree bucket
- [ ] Cached to disk and loaded by `evaluate.py` in < 10 s

### 4.9 [Task] Implement a 2-layer R-GCN encoder with basis decomposition and NeighborLoader (`models/encoders.py`, `data/loaders.py`)
**Labels:** task, model · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] `RGCNConv` with basis decomposition (`num_bases: 30` default) and hidden dim 128; number of layers configurable (0–3), with 0 meaning "input embeddings only"
- [ ] `NeighborLoader` with fan-out [15, 10] over the full graph in CPU memory, `num_workers=4`, `persistent_workers=True`, `pin_memory=True`
- [ ] Forward pass under `torch.autocast("cuda", dtype=torch.bfloat16)`
- [ ] Uses the frozen inputs from 4.1 as node features
- [ ] One epoch on the sample fits in 16 GB VRAM; GPU utilisation ≥ ~80% in `nvidia-smi` (otherwise data loading is the bottleneck); time per epoch and peak memory logged
- [ ] `torch.compile` tried once; kept only if it is measurably faster

### 4.10 [Task] Implement DistMult and RotatE decoders and the link-prediction loss with negative sampling
**Labels:** task, model · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] `models/decoders.py`: DistMult and RotatE, chosen by config
- [ ] `losses/link_prediction.py`: BCE or margin loss with type-constrained corruption for P166; negatives **shared across the batch** (each positive scored against the same k negatives), k in the config
- [ ] Overfitting test: loss → ~0 on a 100-triple batch

### 4.11 [Task] Implement the config system (`utils/config.py`) and a config-driven `scripts/train.py`
**Labels:** task, infra · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] YAML configs with inheritance (base + override); the resolved config is saved to the run folder (0.6)
- [ ] Toggles for encoder, decoder, Module A, Module B and each loss term with its weight (λ_cf, μ, λ_adv, λ_ind, λ_rec)
- [ ] `uv run scripts/train.py --config configs/baselines/rgcn.yaml` trains and evaluates end to end on the synthetic graph
- [ ] `--smoke` flag: a few batches + one evaluation, finishes in < 1 min
- [ ] Validation on the fixed 5k dev sample every N epochs (N in config); early stopping after 5 checks with no improvement; best checkpoint chosen by validation
- [ ] Checkpoint saved every epoch; `--resume` continues a crashed or disconnected (Colab) run
- [ ] Full test evaluation only at the end, once per final model and seed
- [ ] Logs training time and peak GPU memory along with metrics

### 4.12 [Task] Hyperparameter search with Optuna and a fixed per-method trial budget
**Labels:** task, infra · **Assignee:** Theodora · **Week:** 4
**Goal:** Every method, including the baselines, gets the same tuning budget (Ruffinelli et al., ICLR 2020).
**Done when:**
- [ ] `scripts/tune.py --config … --n-trials N` runs Optuna with median pruning on the small stratified subset, optimising validation MRR (or a stated fairness–accuracy objective)
- [ ] Budget `tuning.n_trials` fixed in one shared config and used for every method
- [ ] The best parameters are written to a config file that the full-data runs load

### 4.13 [Task] Overnight run queue for the 5060 Ti
**Labels:** task, infra · **Assignee:** Teammate · **Week:** 4
**Done when:**
- [ ] `scripts/run_queue.py queue.txt` runs a list of `train.py` commands one after another, smoke-testing each first and skipping (and logging) failures
- [ ] A morning summary lists each run's status, time and headline metrics

---

## M5 · Phase 5: Baselines and decomposition (weeks 5–6)

### 5.1 [Task] Implement the bias decomposition (`eval/decomposition.py`, `scripts/decompose.py`)
**Labels:** task, fairness · **Assignee:** Theodora · **Week:** 5
**Goal:** Split a model's group gap into a popularity part (explained by degree and covariates) and a direct part (unexplained).
**Done when:**
- [ ] Degree stratification: gap per degree decile, plus the weighted average
- [ ] Oaxaca–Blinder on per-triple reciprocal rank with covariates log-degree, #facts, birth decade, broad occupation; outputs explained and unexplained parts with bootstrap CIs
- [ ] Synthetic check: recovers the planted β_direct and β_pop (4.5) within CI
- [ ] `decompose.py --run runs/<id>` writes `decomposition.json` into the run folder

### 5.2 [Experiment] Pilot: TransE and 2-layer R-GCN on a stratified subset, with first decomposition
**Labels:** experiment, fairness · **Assignee:** Theodora + Teammate · **Week:** 5
**Hypothesis:** The pipeline runs end to end, and early numbers show which bias path is larger.
**Config / runs:** `configs/baselines/transe.yaml`, `rgcn.yaml` on a ~50k-person subset, 1 seed.
**Metrics to report:** MRR, Hits@10, DP gap@10, EO gap, amplification ratio, Oaxaca–Blinder split.
**Result:** _(filled after running)_

### 5.3 [Experiment] Full shallow baselines: TransE and RotatE
**Labels:** experiment, model · **Assignee:** Theodora · **Week:** 5–6
**Hypothesis:** Shallow KGEs already amplify the data disparity (amplification ratio > 1).
**Config / runs:** `configs/baselines/transe.yaml`, `rotate.yaml`; 3 seeds; full split from 2.11; tuned with the 4.12 budget.
**Metrics to report:** accuracy + fairness metrics with CIs; decomposition; training time and GPU memory.
**Notes:** Save the checkpoints for reuse by 5.7.
**Result:** _(filled after running)_

### 5.4 [Experiment] Full plain R-GCN baseline
**Labels:** experiment, model · **Assignee:** Teammate · **Week:** 5–6
**Hypothesis:** R-GCN has higher accuracy but a larger popularity gap than the shallow KGEs.
**Config / runs:** `configs/baselines/rgcn.yaml` (2 layers, DistMult); 3 seeds; tuned with the 4.12 budget.
**Metrics to report:** same as 5.3, plus training time and GPU memory.
**Notes:** Save the checkpoints: 5.6 (probe) and 5.7 (decomposition) reuse them instead of retraining.
**Result:** _(filled after running)_

### 5.5 [Experiment] Depth sweep: 0, 1, 2, 3 R-GCN layers
**Labels:** experiment, fairness · **Assignee:** Teammate · **Week:** 6
**Hypothesis:** Message passing increases bias (the gap grows with depth): the homophily hypothesis (1.4).
**Config / runs:** `rgcn.yaml` with `num_layers ∈ {0,1,2,3}`; 3 seeds each.
**Metrics to report:** MRR, DP/EO gap and decomposition parts vs. depth; figure `depth_sweep.pdf`.
**Result:** _(filled after running)_

### 5.6 [Experiment] Gender probe on R-GCN representations at each depth
**Labels:** experiment, fairness · **Assignee:** Teammate · **Week:** 6
**Hypothesis:** Probe accuracy rises with depth, even though P21 is not an input edge (proxies spread through message passing).
**Config / runs:** representations from the saved 5.5 checkpoints (no retraining); probe from 4.2.
**Metrics to report:** balanced accuracy ± CI per depth vs. the input-embedding probe.
**Result:** _(filled after running)_

### 5.7 [Experiment] Decompose all baselines and compute amplification ratios against the data audit
**Labels:** experiment, fairness · **Assignee:** Theodora · **Week:** 6
**Hypothesis:** The pilot's conclusion about which path dominates holds on the full baselines.
**Config / runs:** `decompose.py` on every 5.3–5.5 run.
**Metrics to report:** one table with model, direct part, popularity part and amplification ratio (vs. `data_disparity.csv`), with CIs; explicit "pilot confirmed / contradicted" line.
**Result:** _(filled after running)_

### 5.8 [Task] Decision gate (week 6): choose the bias path to prioritise and confirm Version A or B
**Labels:** task, gate · **Assignee:** Theodora + Teammate · **Week:** 6
**Done when:**
- [ ] Gate memo in the notebook: mostly popularity → popularity term / Module A first; mostly direct → direct term / Module B first; both → both
- [ ] Version A or B confirmed with the professor, given the time left
- [ ] Issues of the unused version closed as "not planned"; MA/MB dates adjusted
**Notes:** If baselines and decomposition are not done, submit as an empirical study with existing debiasing methods.

---

## MA · Version A: Counterfactual objective (weeks 7–13)

### A.1 [Task] Implement the direct counterfactual term (P21 edge flip) in `losses/counterfactual.py`
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 7–8
**Done when:**
- [ ] For a sampled batch of people, flip only the P21 edge in their subgraph, re-encode, and penalise |s(h, P166, t) − s_cf(h, P166, t)| over candidate awards
- [ ] Weight `λ_cf` in the config; extra time per epoch logged
- [ ] Synthetic check (week-9 gate): the direct part goes down, the popularity part stays unchanged within CI
**Notes:** Requires P21 to be an input edge for this model variant; document how this fits the "no sensitive edges" TransE inputs.

### A.2 [Task] Implement CAF-style counterfactual matching in `scripts/10_match_counterfactuals.py`
**Labels:** task, fairness · **Assignee:** Theodora · **Week:** 7
**Done when:**
- [ ] For each training person: k real people of another gender, exact match on broad occupation and region, nearest neighbour on birth decade (scikit-learn `NearestNeighbors`, calipers as fallback)
- [ ] Runs once on CPU; matched IDs saved to `data/processed/matches_v1_k{k}.parquet` and reused by every run
- [ ] Balance diagnostics: SMD before/after matching for every covariate, match rate by group, and how many people are unmatched

### A.3 [Task] Implement the popularity counterfactual term using the matches
**Labels:** task, fairness · **Assignee:** Theodora · **Week:** 8–9
**Done when:**
- [ ] Loss penalising prediction differences between each person and their matched counterfactuals, weighted by `μ`
- [ ] Computed on a sampled subset of people per batch (subset size in config), not on all of them; extra time per epoch logged
- [ ] Synthetic check (week-9 gate): the popularity part goes down, the direct part stays unchanged within CI

### A.4 [Experiment] λ sweep on Wikidata: direct only (μ = 0) vs. both paths (μ > 0)
**Labels:** experiment, fairness · **Assignee:** Theodora + Teammate · **Week:** 9
**Hypothesis:** Adding the popularity term moves the accuracy–fairness Pareto front beyond what the direct term alone reaches.
**Config / runs:** `configs/version_a/cf_direct.yaml`, `cf_both.yaml`; λ_cf ∈ {0, 0.01, 0.1, 0.5, 1, 5}; μ ∈ {0, chosen value}.
**Metrics to report:** Pareto curve of MRR vs. DP gap across λ (a method is better only if its curve lies above); EO gap and decomposition per point. λ selected on validation only.
**Result:** _(filled after running)_

### A.5 [Task] Implement the adversarial filter baseline (FAN / Bose & Hamilton) in `src/fairkg/debias/`
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 10
**Done when:**
- [ ] Filter on the person embeddings with an adversary predicting gender; config `configs/debias_baselines/adversarial_filter.yaml`
- [ ] Probe accuracy after filtering reported; runs on the synthetic graph

### A.6 [Task] Implement the FairDrop baseline
**Labels:** task, fairness · **Assignee:** Theodora · **Week:** 10
**Done when:**
- [ ] Biased edge dropout that drops same-group edges more often, adapted to relational edges; `configs/debias_baselines/fairdrop.yaml`
- [ ] Runs on the synthetic graph and on the sample

### A.7 [Task] Implement the data rebalancing baseline
**Labels:** task, fairness · **Assignee:** Theodora · **Week:** 10
**Done when:**
- [ ] Reweighting or oversampling of training triples so each group contributes equally to the P166 loss; `configs/debias_baselines/rebalancing.yaml`
- [ ] Runs on the synthetic graph and on the sample

### A.8 [Task] (If time) Implement the NIFTY baseline
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 11
**Done when:**
- [ ] Counterfactual-augmentation + stability regulariser adapted to R-GCN; `configs/debias_baselines/nifty.yaml`
- [ ] Or closed with "skipped: time"

### A.9 [Task] Port the encoder and CF terms to CompGCN
**Labels:** task, model · **Assignee:** Teammate · **Week:** 10
**Done when:**
- [ ] CompGCN option in `models/encoders.py` (composition op in the config)
- [ ] Baseline CompGCN run plus one CF run on the sample both finish without errors

### A.10 [Experiment] Ablation: direct term only, popularity term only, both
**Labels:** experiment, fairness · **Assignee:** Theodora + Teammate · **Week:** 10–11
**Hypothesis:** Each term reduces mainly its own path in the decomposition.
**Config / runs:** `configs/ablations/cf_direct_only.yaml`, `cf_pop_only.yaml`, `cf_both.yaml`; R-GCN and CompGCN; 3–5 seeds.
**Metrics to report:** accuracy, fairness gaps and decomposition parts with CIs.
**Result:** _(filled after running)_

### A.11 [Experiment] Final comparison: all methods × R-GCN/CompGCN × 3–5 seeds, decomposition on every model
**Labels:** experiment, fairness · **Assignee:** Theodora + Teammate · **Week:** 11
**Hypothesis:** Our objective reduces the targeted bias path(s) at a smaller accuracy cost than the existing debiasing baselines.
**Config / runs:** no debiasing, adversarial filter, FairDrop, rebalancing, (NIFTY), CF both; optionally repeated on P106. Every method tuned with the same 4.12 trial budget; all run with the same seeds on the same test queries.
**Metrics to report:** main results table (mean ± sd over seeds, bootstrap CIs); paired per-query comparison against the no-debiasing baseline; Pareto curves per method; "which part did each method reduce" table; cost table (training time, peak GPU memory). Differences smaller than the seed-to-seed variation are reported as "no difference".
**Result:** _(filled after running)_

### A.12 [Task] Generate all report figures with `scripts/make_figures.py`
**Labels:** task, writing · **Assignee:** Theodora · **Week:** 12
**Done when:**
- [ ] Pareto curves, decomposition before/after, depth sweep, probe accuracy, ablation table and cost table (time, GPU memory); all regenerated from run folders with one command
- [ ] Figures saved as PDF in `reports/figures/`, readable in greyscale

### A.13 [Task] Write the Data, Decomposition and Results sections
**Labels:** task, writing · **Assignee:** Theodora · **Week:** 12–13

### A.14 [Task] Write the Method and Related Work sections
**Labels:** task, writing · **Assignee:** Teammate · **Week:** 12–13

### A.15 [Task] Write the Ethics and Limitations section (incl. missing labels, non-binary groups, Wikidata ≠ real world)
**Labels:** task, writing · **Assignee:** Theodora + Teammate · **Week:** 13

### A.16 [Task] Code cleanup and reproducibility check: fresh clone → `uv sync --locked` → main table
**Labels:** task, infra · **Assignee:** Teammate · **Week:** 13
**Done when:**
- [ ] README has the full command sequence 00 → make_figures
- [ ] The other person reproduces one main-table number from a fresh clone

---

## MB · Version B: Modules + objective (weeks 7–16)

Only if Version B is confirmed at the 5.8 gate. Reuses A.1–A.3, A.5–A.9 and A.12.

### B.1 [Task] Module A: degree-robust aggregation (per-relation neighbour budget, mean aggregation) in `models/degree_robust.py`
**Labels:** task, model · **Assignee:** Theodora · **Week:** 7–8
**Done when:**
- [ ] Per-relation budget for how many neighbours are sampled; mean instead of sum aggregation; both switchable in config
- [ ] Unit test: the output norm does not grow with node degree

### B.2 [Task] Module A: self-supervised low-degree completion (`losses/completion.py`)
**Labels:** task, model · **Assignee:** Theodora · **Week:** 8–9
**Done when:**
- [ ] For high-degree people, hide most edges (keep ratio in config) and train the layer to reconstruct the full-graph representation; weight `λ_rec`
- [ ] Reconstruction error by degree decile logged

### B.3 [Experiment] Module A synthetic check
**Labels:** experiment, fairness · **Assignee:** Theodora · **Week:** 9
**Hypothesis:** Module A reduces the popularity part and leaves the direct part unchanged.
**Config / runs:** `configs/version_b/module_a.yaml` on the synthetic graph; β_pop only, then both.
**Metrics to report:** decomposition parts vs. planted values. **Result:** _(week-9 gate)_

### B.4 [Task] Module B: two-channel layer (sensitive / content) with whitelist routing in `models/two_channel.py`
**Labels:** task, model · **Assignee:** Teammate · **Week:** 7–8
**Done when:**
- [ ] P21 edge feeds only the sensitive channel; P166/P106 are scored only from the content channel
- [ ] Whitelisted relations (0.7) may use both channels
- [ ] Unit test: gradients from the P166 score do not reach the sensitive channel

### B.5 [Task] Module B: adversary with gradient reversal and independence loss (`losses/adversary.py`)
**Labels:** task, fairness · **Assignee:** Teammate · **Week:** 8–9
**Done when:**
- [ ] Gradient-reversal adversary predicting gender from the content channel (`λ_adv`), plus an independence loss between channels, e.g. HSIC (`λ_ind`)
- [ ] Warm-up epochs and adversary update ratio in the config; training curves show no collapse over 3 seeds

### B.6 [Experiment] Module B check: content-channel probe near chance, direct part reduced
**Labels:** experiment, fairness · **Assignee:** Teammate · **Week:** 9
**Hypothesis:** Module B removes gender from the content channel and reduces the direct part.
**Config / runs:** `configs/version_b/module_b.yaml` on the synthetic graph and the sample.
**Metrics to report:** probe balanced accuracy on the content channel (target ≈ 0.5); decomposition. **Result:** _(week-9 gate)_

### B.7 [Task] Combine the modules with the direct and popularity CF terms; verify on the synthetic graph
**Labels:** task, fairness · **Assignee:** Theodora + Teammate · **Week:** 10–11
**Done when:**
- [ ] `configs/version_b/full.yaml` trains with both modules + both CF terms
- [ ] All eight synthetic configurations run without errors

### B.8 [Task] Fallback gate (week 11): do the modules work on Wikidata?
**Labels:** task, gate · **Assignee:** Theodora + Teammate · **Week:** 11
**Done when:**
- [ ] Decision recorded with numbers; if they fail, the Version A scope is finished (MA issues reopened)

### B.9 [Task] Integration and tuning: fix λ_ind and λ_rec on the synthetic graph, tune λ_adv and λ_cf on Wikidata
**Labels:** task, model · **Assignee:** Theodora + Teammate · **Week:** 12
**Done when:**
- [ ] λ_ind and λ_rec frozen in `full.yaml` with the synthetic runs that justify them
- [ ] Grid over λ_adv × λ_cf on validation; selection rule written down before looking at test

### B.10 [Experiment] Six configurations on Wikidata and eight on the synthetic graph
**Labels:** experiment, fairness · **Assignee:** Theodora + Teammate · **Week:** 13–14
**Hypothesis:** The full model gives the best fairness at comparable accuracy.
**Config / runs:** baseline, A only, B only, loss only, A + B, full (+ two synthetic-only configurations); R-GCN and CompGCN; 3–5 seeds, paired (same seeds, same test queries); same 4.12 tuning budget per configuration.
**Metrics to report:** main table (mean ± sd) + decomposition per configuration; cost (training time, GPU memory). **Result:** _(filled after running)_

### B.11 [Experiment] Redundancy test: is a module and its loss term on the same path both needed?
**Labels:** experiment, fairness · **Assignee:** Theodora + Teammate · **Week:** 14
**Hypothesis:** Module A and the popularity term overlap (as do Module B and the direct term); one of each pair may be enough.
**Config / runs:** A vs. A + popularity term; B vs. B + direct term.
**Metrics to report:** path-specific reduction with CIs. **Result:** _(filled after running)_

### B.12 [Task] Figures and report for Version B (per-configuration decomposition, Pareto, probe, depth sweep, redundancy)
**Labels:** task, writing · **Assignee:** Theodora + Teammate · **Week:** 15–16

---

## MX · Optional: Semi-supervised fairness

### X.1 [Task] Adversary with partially known sensitive attributes (FairGNN-style pseudo-labels)
**Labels:** task, fairness · **Assignee:** unassigned

### X.2 [Task] Fair pseudo-labelling for missing awards (per-group thresholds or caps)
**Labels:** task, fairness · **Assignee:** unassigned

### X.3 [Experiment] Naive vs. fair pseudo-labelling: does naive pseudo-labelling amplify bias?
**Labels:** experiment, fairness · **Assignee:** unassigned
