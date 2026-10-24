# Step-by-Step Project Plan: Fair KG Link Prediction

Oct 14, 2026 · 
Both versions share six weeks of setup, data work and diagnosis; the bias decomposition at week 6 decides what we build next. Version A then adds a counterfactual training objective (13 weeks total); Version B adds two modules plus that objective (16 weeks total), for a team of two.
Timeline

Project timeline · shared phases, Version A, Version B
The first six weeks are identical for both versions; the gate at week 6 picks the version and which bias path to prioritise.
Phase 0 – Setup and scope (week 1, both)
The goal is an approved scope and a machine that runs, before any data work.
	•	☐ Send the proposal to the professor; confirm which version (A or B), the deadline, and whether a new training objective counts as enough novelty
	•	☐ Final novelty check: Google Scholar and arXiv 2025–2026 for "path-specific counterfactual fairness" + "knowledge graph" / "link prediction"
	•	☐ Set up the RTX 5060 Ti: recent NVIDIA driver, PyTorch ≥ 2.7 for CUDA 12.8, PyTorch Geometric, PyKEEN (WSL2 if on Windows); test the GPU
	•	☐ Create the shared Git repository (branch per module, main always runs) and a shared experiment log (spreadsheet or Weights & Biases)
	•	d☐ Fix definitions: tasks (P166 primary, P106 secondary), groups (gender P21, region from P27), fairness metrics, whitelist of relations where gender is part of the fact
Phase 1 – Literature: what could drive bias (week 1, both)
The literature gives us a list of suspected causes; later phases test which ones matter in our data.
Hypothesis
Where it comes from
How we will test it
Popularity: well-connected people get better predictions
Arduini et al. 2020; Shomer et al. 2023
Degree stratification, Oaxaca–Blinder (Phase 5)
Data imbalance: fewer women and Global South people, fewer facts each
Radstok et al. 2021; AuditLP 2025
Data audit (Phase 3)
Proxies: spouse, gendered occupations, names leak gender
Fisher et al. 2020; Du et al. 2022
Probe classifier on embeddings (Phase 4)
Message passing spreads group information (homophily)
FairGNN; FairVGNN; Fair-RGNN
Depth sweep 0–3 layers (Phase 5)
Missing labels: real awards never recorded
Kraft & Usbeck 2022
Discussed as a limitation; optional spot-check
	•	☐ Read the core papers and write one paragraph per hypothesis
	•	☐ Turn each hypothesis into a test we will run, as in the table
Phase 2 – Data extraction, representative sampling and splits (weeks 2–3, both)
The aim is a dataset that looks like Wikidata's population of people, with splits that cannot leak answers. We claim representativeness of Wikidata, not of the real world.
Extraction
	•	☐ Download Wikidata5M and stream the JSONL line by line into edge lists (never load it whole)
	•	☐ Keep the full graph in CPU memory; train later with neighbour sampling, so no nodes are dropped
	•	☐ Select humans (P31 → Q5) with P21, P27, P106, P166; check the counts
	•	☐ Keep about 50–100 people-related relations; map the rest to "other"
	•	☐ Compute covariates on the full graph: degree, number of awards, birth decade
Representative sampling (if a subset is needed)
	•	☐ Stratify by gender, region, degree decile, birth decade and broad occupation
	•	☐ Choose proportional sampling, or oversample small groups and store sampling weights
	•	☐ Compare full graph vs. sample: category shares, degree quantiles, standardised mean difference below about 0.1, award rate by gender and region
Splits for link prediction
	•	☐ Same distribution in train, validation and test (repeat the comparison per split)
	•	☐ Every test person also has edges in training
	•	☐ Remove leaking relations (inverse or near-duplicate of P166; "nominated for" P1411)
	•	☐ At least about 200 test triples per group; merge regions if needed
	•	☐ Optional temporal split if edit dates are available
Robustness
	•	☐ Repeat sampling with several seeds and two sample sizes; bias numbers should agree within bootstrap confidence intervals
	•	☐ Write the "Data and sampling" section: selection, strata, comparison table, leakage rules, minimum group sizes
Phase 3 – Data audit: bias before any model (week 3, both)
This measures how biased Wikidata itself is, so we can later tell whether a model adds bias or only copies it (amplification ratio = model disparity ÷ data disparity). No training happens here.
	•	☐ Award and occupation rates by gender and by region
	•	☐ Degree distribution by group: are women and Global South people less connected?
	•	☐ Information per person: number of facts by group
	•	☐ Group sizes, including non-binary and other P21 values; decide which groups have enough data
	•	☐ Save these numbers as the "data disparity" table
Phase 4 – Embeddings, probe, synthetic test bed, shared code (weeks 3–4, both)
This builds the clean inputs and the shared tools every later step depends on.
Input embeddings
	•	☐ Train TransE (about 100 dimensions) on full Wikidata5M with sensitive edges removed (P21, and P27 if we study region); freeze them
	•	☐ Probe test: train a simple classifier to predict gender from the embeddings; record its accuracy
	•	☐ If gender still leaks through proxies, decide: remove the gender direction (e.g. INLP) or keep it and report it
	•	☐ Use pretrained GraphVite embeddings only for quick development, never for final results
Synthetic test bed
	•	☐ Generate a small graph with planted direct bias, planted popularity bias and a gender proxy, each of known strength
	•	☐ Every component is checked here first: it must remove the bias it targets and leave the other
Shared code
	•	☐ One evaluation module used by both of us: MRR, Hits@1/10, demographic parity gap @k, equal-opportunity gap, amplification ratio, probe accuracy, bootstrap confidence intervals
	•	☐ Base model: 2-layer R-GCN (basis decomposition, neighbour sampling e.g. [15, 10]) with a DistMult or RotatE decoder
	•	☐ Config system so modules and loss terms can be switched on and off
Phase 5 – Baselines and bias decomposition (weeks 5–6, both)
This is the diagnosis: it tells us which source of bias dominates and therefore what to build. Its results are the minimum deliverable if nothing else works.
Pilot (week 5)
	•	☐ Train a small baseline (TransE, then 2-layer R-GCN) on a stratified subset
	•	☐ Measure bias and run a first decomposition to guide design choices
Full baseline (weeks 5–6)
	•	☐ Train shallow KGE baselines (TransE, RotatE) and plain R-GCN on the full setup
	•	☐ Depth sweep: 0, 1, 2, 3 GNN layers; does message passing increase bias?
	•	☐ Decompose the bias into direct and popularity parts: degree stratification + Oaxaca–Blinder
	•	☐ Compute amplification ratios against the Phase 3 data disparity
	•	☐ Confirm the pilot's conclusions hold on the full baseline
Decision gate (end of week 6)
	•	☐ Which part dominates? Mostly popularity → prioritise the popularity term / Module A; mostly direct → prioritise the direct term / Module B; both → build both
	•	☐ Confirm Version A or B with the professor and the time left
Version A – Counterfactual objective (weeks 7–13)
Version A keeps the standard R-GCN/CompGCN and adds one new training objective. Learning type: self-supervised link prediction plus a fairness regulariser.
Weeks 7–9: the counterfactual terms
	•	☐ Direct term: for sampled people, flip only the gender edge, re-encode, penalise the difference in award scores; check on the synthetic graph
	•	☐ Matching (on CPU, once): for each person, real people of another gender with similar occupation, birth decade and region (CAF-style); report balance diagnostics
	•	☐ Popularity term: penalise prediction differences against matched counterfactuals; check on the synthetic graph
	•	☐ Run on Wikidata; sweep λ to trace the accuracy–fairness curve; μ = 0 (direct only) vs. μ > 0 (both paths)
Weeks 10–11: comparisons and robustness
	•	☐ Baselines: no debiasing, adversarial filter (FAN / Bose & Hamilton), FairDrop, data rebalancing, NIFTY if time allows
	•	☐ Ablation: direct term only, popularity term only, both
	•	☐ Port to CompGCN; 3–5 seeds with confidence intervals; optionally repeat on P106
	•	☐ Rerun the decomposition on every model: which part did each method reduce?
Weeks 12–13: analysis and writing
	•	☐ Figures: Pareto curves, decomposition before/after, depth sweep, probe accuracy
	•	☐ Report with ethics and limitations; clean code and README
Version B – Two modules + objective (weeks 7–16)
Version B adds one module per bias path inside the GNN, then trains everything with the Version A objective. Learning types: self-supervised (link prediction, Module A completion), supervised adversarial (Module B), fairness regulariser (counterfactual loss).
Weeks 7–9: the two modules (in parallel, one person each)
	•	☐ Module A, degree-robust aggregation: neighbour budget per relation, mean instead of sum aggregation
	•	☐ Module A, low-degree completion: hide most edges of high-degree people, train the layer to recover their full representation (self-supervised)
	•	☐ Check Module A on the synthetic graph: popularity part goes down, direct part unchanged
	•	☐ Module B, two-channel layer: sensitive and content channels; gender edge feeds only the sensitive channel; P166/P106 scored from the content channel; whitelisted relations may use both
	•	☐ Module B, adversary with gradient reversal + independence loss; stabilise with warm-up and adversary update ratio
	•	☐ Check Module B: probe accuracy on the content channel near chance; direct part goes down
Weeks 10–11: counterfactual terms
	•	☐ Add the direct and popularity terms from Version A; verify on synthetic data
	•	☐ Fallback point at week 11: if the modules are not working, finish with Version A's scope
Week 12: integration and tuning
	•	☐ Combine everything; fix λ_ind and λ_rec from synthetic runs; tune mainly λ_adv and λ_cf
Weeks 13–14: ablations and comparisons
	•	☐ Six configurations on Wikidata (baseline, A only, B only, loss only, A + B, full); all eight on the synthetic graph
	•	☐ Same baselines as Version A; R-GCN and CompGCN; 3–5 seeds
	•	☐ Test redundancy: is a module and its loss term on the same path both needed?
Weeks 15–16: analysis and writing
	•	☐ Figures: decomposition per configuration, Pareto curves, probe accuracy, depth sweep
	•	☐ Report, including any redundancy found; code cleanup and reproducibility
Optional extension – Semi-supervised fairness (if time allows)
Semi-supervised learning is an add-on, not the main fix for data imbalance: naive pseudo-labels from a biased model copy and can amplify the bias. Reweighting and rebalancing address imbalance more directly.
	•	☐ Partially known sensitive attributes (FairGNN-style): train the adversary on people with known gender or region, pseudo-label the rest, debias using both
	•	☐ Fair pseudo-labelling for missing awards: per-group confidence thresholds or capped pseudo-labels per group
	•	☐ Compare naive vs. fair pseudo-labelling and report whether the naive version amplifies bias
Team split, decision gates and fallbacks
We split by bias path so each of us owns one complete contribution, and build the foundation together first.
Area
Theodora
[Teammate]
Phases 0–4 (shared)
Data pipeline, sampling, covariates, data audit
Evaluation module, base GNN, synthetic graph
Phase 5
Bias decomposition
Probe tests, depth sweep
Version A
Popularity term + matching
Direct term (edge flip)
Version B modules
Module A: degree-robust aggregation
Module B: two-channel layer + adversary
Baselines
FairDrop, rebalancing
Adversarial filter, NIFTY
Writing
Data, decomposition, results
Method, related work, results

Week
Gate
If it fails
1
Professor approves scope; novelty check passes
Adjust the scope or the claimed contribution
6
Baselines + decomposition done
Submit as an empirical study with existing debiasing methods
9
Components pass the synthetic tests
Keep only the components that pass
11 (B only)
Modules work on Wikidata
Finish with Version A's scope
Working habits: merge and rerun the synthetic tests every week; log every run in the shared experiment log; schedule long runs overnight on the 5060 Ti and use Colab for light runs.
