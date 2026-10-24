# Faster Training and Fair Comparisons: Practical Guide

Oct 21, 2026 · @Theodora Manika
Most training time in this project is lost to repeated data loading, slow evaluation and runs that should never have been started, not to the GNN itself. Fix those first, then make the comparison protocol fixed so every speed-up still gives trustworthy results.
Biggest wins first

Speed-up techniques · time saved vs. effort
Do the top-left group before writing any model code; the sections below explain each technique.
1. Prepare the dataset once and cache it
Parse the 3 GB Wikidata5M file once, turn it into compact integer arrays, and never touch the raw JSON again.
	•	Stream the raw file once. Read the JSONL line by line and write out triples; never load it whole into memory.
	•	Map everything to integers. Give every entity and relation an integer ID (Q76 → 81234) and save the two dictionaries separately. Integer arrays are many times smaller and faster than strings.
	•	Save in binary formats, not CSV.
	•	Edge list as NumPy .npy (int32 is enough for 4.6M entities), shape [num_edges, 3].
	•	Tables (people, gender, region, degree, birth decade) as Parquet via pandas.
	•	Load large arrays with np.load(path, mmap_mode="r"): the file is read from disk only when needed, so loading takes seconds and uses little RAM.
	•	Build the graph structure once. Convert the edge list to PyG's format (edge_index, edge_type) and save it with torch.save. Also sort edges by source node (CSR layout), which makes neighbour sampling faster.
	•	Reduce relations once. Keep the ~50–100 people-related relations and map the rest to "other" before saving, not inside the training loop.
	•	Compute covariates once, on the full graph: degree, number of awards, birth decade, group labels. Save them as columns in the people table.
	•	Freeze the splits. Save train/validation/test indices to files with the seed in the filename. Every run and every method loads the same files.
	•	Name caches by their settings. E.g. people_v1_seed42_rel80.parquet. If a setting changes, a new file is made; old results stay reproducible.
With this in place, starting a training run should take seconds instead of minutes.
2. Cache expensive intermediate results
Anything that does not change between runs should be computed once and reused by every method.
	•	TransE input embeddings: train once (a few hours), save as float16 .npy (about 1.2 GB for 4.6M × 128), load with memory-mapping, and copy only the rows each batch needs to the GPU.
	•	Counterfactual matches (Version A popularity term): the nearest-neighbour matching runs once on CPU with scikit-learn; save each person's matched IDs to a file.
	•	Evaluation candidate sets: the list of all award entities (for P166) and occupation entities (for P106). Scoring a person against only these, instead of all 4.6M entities, is the biggest evaluation speed-up.
	•	Filter masks: for filtered evaluation, precompute for each test query which answers are already known true, and store them as sparse index lists.
	•	Group membership and fairness bookkeeping: which test people belong to which group and degree bucket; saved once, reused by every metric.
	•	Synthetic graph: generate with a fixed seed and save; do not regenerate each run.
	•	Model checkpoints: save the trained baseline. The decomposition, the probe and the depth analysis all reuse it instead of retraining.
3. Train faster on the GPU
The aim is to keep the RTX 5060 Ti busy all the time instead of waiting for data.
Keep the GPU fed
	•	Use PyG NeighborLoader with num_workers=4 (or more), persistent_workers=True and pin_memory=True, so the next batch is prepared while the current one trains.
	•	Watch nvidia-smi during training: GPU utilisation well below about 80% usually means the data loading is the bottleneck, not the model.
Do less work per step
	•	Mixed precision: wrap the forward pass in torch.autocast("cuda", dtype=torch.bfloat16). It roughly halves memory and is often much faster on recent GPUs.
	•	Small neighbour samples: e.g. 15 neighbours at hop 1 and 10 at hop 2; larger samples cost a lot and help little. Tune once on the subset.
	•	Basis decomposition in R-GCN (e.g. num_bases=30) instead of one weight matrix per relation.
	•	Moderate sizes: 128-dimensional embeddings and 2 layers are usually enough.
	•	Reuse negatives within a batch: score each positive against negatives shared by the whole batch, instead of sampling new ones per triple.
	•	Compute fairness terms on a sample: apply the counterfactual loss to a subset of people per batch, not all of them.
Stop early
	•	Early stopping on a validation metric (e.g. stop after 5 checks with no improvement) avoids paying for epochs that do not help.
	•	torch.compile(model) can speed up the model a further amount; with variable-size sampled batches the gain is smaller, so try it once and keep it only if it helps.
4. Evaluate faster
Ranking every test fact against millions of candidates is often slower than training, so evaluation needs its own speed-ups.
	•	Type-constrained ranking: for (person, award received, ?), rank only award entities, not all 4.6M entities. This is both faster and more meaningful.
	•	Score in large GPU batches: compute all candidate scores for many queries at once as one matrix multiplication, instead of looping over candidates.
	•	Compute person vectors once per evaluation: run the GNN once over all test people, store their vectors, then score.
	•	During development, validate on a fixed sample: e.g. 5,000 validation queries, the same sample every time, so numbers stay comparable.
	•	Evaluate less often: every few epochs, not every epoch.
	•	Full test evaluation only at the end, once per final model and seed. Never tune on the test set.
5. Develop small, scale late, avoid wasted runs
The cheapest run is the one you never have to repeat.
Three sizes of data
Size
What
Use it for
Typical run
Tiny
Synthetic graph, a few thousand nodes
Bugs, checking each method removes the planted bias
Seconds
Small
Stratified subset of Wikidata people
Hyperparameter search, design decisions
Minutes
Full
Whole setup with neighbour sampling
Final results only
Hours, overnight
Habits that save time
	•	Smoke test: before a long run, train for a few batches and one evaluation, to catch crashes in seconds instead of hours.
	•	Overfit one batch: a correct model can drive the loss near zero on a single batch; if it cannot, there is a bug.
	•	Config files: every setting in a YAML file, never edited inside the code; the run log stores the config.
	•	Checkpoints: save every epoch and make training resumable, so a crash or Colab disconnect costs minutes, not the whole run.
	•	Smart hyperparameter search: use Optuna with early pruning (stop bad trials early) on the small subset, then reuse the best settings on the full data.
	•	Run queue: prepare a list of runs and let them run overnight on the 5060 Ti; check results in the morning.
	•	Log everything automatically (Weights & Biases or a CSV): config, seed, metrics, training time, GPU memory.
6. Compare methods effectively and fairly
A comparison is fair when the only thing that differs between two runs is the method itself.
Keep everything else fixed
	•	Same data files, same splits, same input embeddings for every method.
	•	Same evaluation code: one shared module computes every metric.
	•	Same tuning budget: give baselines as many hyperparameter trials as your own method. Under-tuned baselines are the most common way papers exaggerate gains (Ruffinelli et al., ICLR 2020).
	•	Same training rules: same epoch limit and early-stopping rule; choose the checkpoint by validation, never by test.
Measure uncertainty
	•	Run 3–5 seeds per method and report mean ± standard deviation.
	•	Paired comparison: run every method with the same seeds and the same test queries, then compare per query. This removes much of the noise.
	•	Bootstrap confidence intervals on each metric, especially fairness metrics for small groups, which are noisy.
	•	Treat a difference smaller than the seed-to-seed variation as "no difference".
Report the right things
	•	Pareto curve: accuracy (MRR) against fairness (e.g. demographic parity gap) for each method across its fairness weight λ. A method is better only if its curve sits above the others.
	•	Decomposition per method: show which part (direct or popularity) each method reduced.
	•	Ablation table: each component on and off.
	•	Cost: training time and GPU memory per method; a faster method with the same results is a real advantage.
	•	Synthetic first: any claim that a method removes a specific bias should first be shown on the synthetic graph, where the true answer is known.
Checklist before every experiment
	•	☐ Loads the cached data files (no raw JSON parsing)
	•	☐ Uses the frozen split files and the shared evaluation module
	•	☐ Config file saved with the run; seed set
	•	☐ Smoke test passed (a few batches + one evaluation)
	•	☐ Passed the synthetic-graph test, if it is a new method or component
	•	☐ Mixed precision on; GPU utilisation checked in nvidia-smi
	•	☐ Checkpointing and early stopping on
	•	☐ Same tuning budget as the methods it will be compared with
	•	☐ Logged automatically: metrics, fairness metrics, training time, GPU memory
