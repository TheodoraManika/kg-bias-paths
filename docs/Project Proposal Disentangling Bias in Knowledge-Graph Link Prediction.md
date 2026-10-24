# Project Proposal: Disentangling Bias in Knowledge-Graph Link Prediction

Oct 9, 2026  · [Course name]
Overview
We propose to study where social bias in knowledge-graph (KG) link prediction comes from, and to remove it with a method designed for each source. Models that predict facts such as award received (P166) or occupation (P106) on Wikidata favour some genders and regions. We split that bias into two mechanisms: a direct effect (the model uses gender itself) and a popularity effect (well-connected people get better predictions, and they are disproportionately men and people from the Global North).
We offer two versions for a team of two. Version A contributes a new training objective: path-specific counterfactual fairness on a standard relational GNN. Version B adds two new architecture modules (degree-robust aggregation and a two-channel representation) and trains them with the same objective. Both share one data pipeline, evaluation framework and minimum deliverable, so either can fall back to a smaller, complete project.
Problem and motivation
KG embeddings and GNNs trained on Wikidata encode social bias: they associate occupations and awards with gender, nationality and region. These models feed search, question answering and LLM grounding, so the bias propagates downstream.
Existing work reports bias as one number and fixes it with one method, usually an adversary that removes gender from embeddings. That number mixes two mechanisms that need different fixes:
	•	Direct effect: the representation encodes gender, and gender itself shifts the score.
	•	Popularity effect: KG models predict better for high-degree entities, and degree differs between groups. Part of this is historical (women got fewer Wikipedia pages and links), so "explained by popularity" does not mean "fair".
Removing gender information leaves the popularity path untouched, and degree-aware training leaves the direct path untouched. A model can therefore look debiased on one metric while staying biased through the other path. Our project measures each path separately and builds a fix for each.
Related work and research gap
No work we found separates direct from popularity-mediated bias in KG link prediction, or applies path-specific counterfactual fairness to KG GNNs.
Work
What it does
What it leaves open for us
Fisher et al. 2020a
First measurement of social bias in KG embeddings (Wikidata, Freebase)
One aggregate score; no mechanism
Fisher et al. 2020b, EMNLP
Adversarial debiasing with a whitelist of legitimate cases
Direct path only; manual whitelist
Keidar et al. 2021
Demographic and predictive parity distance on FB15k-237
Shallow models; no popularity control
Du et al. 2022, ACL
Group and individual bias; influence functions trace triples
TransE only; no GNNs
Arduini et al. 2020
Popularity bias in KGE; FAN adversarial filter
Popularity and gender studied apart
Radstok et al. 2021
Bias comes largely from data; rebalancing helps
No model-side fix per mechanism
AuditLP 2025
Wikidata bias dataset; gender and age over 21 regions
No decomposition; ethnicity left to future work
Fair-RGNN 2025, TKDD
Multi-hop relational bias in KG GNNs
No popularity mediator
Sasaki 2025
Fairness-aware GNN benchmark on Wikidata, YAGO, DBpedia
Node classification, not link prediction
Chiappa 2019, AAAI
Path-specific counterfactual fairness
Tabular data, not graphs
NIFTY 2021; GEAR 2022; CAF 2023
Counterfactual fairness for GNNs (attribute flip; neighbour flips; real matched counterfactuals)
Node classification; no path selection
Gap. Counterfactual GNN fairness exists for node classification, and path-specific fairness exists for tabular data. Neither has been applied to multi-relational KG link prediction with graph popularity as an explicit mediator. Our final novelty check (Google Scholar, arXiv 2025–2026) is due in week 1.
Common setup (both versions)
Both versions use the same data, backbones and evaluation, built together in weeks 1–4.
Data. Wikidata5M (4.67M entities, 810 relations, about 21M triples, MIT licence). We keep the full graph in CPU memory and train with neighbour sampling, so no nodes are dropped. Tasks: P166 award received (primary) and P106 occupation (secondary). Groups: gender (P21) and region from citizenship (P27). Covariates computed on the full graph: degree, number of awards, birth decade.
Input features. TransE embeddings we train on Wikidata5M with the sensitive edges removed, then freeze. A probe classifier checks how much gender still leaks into them.
Backbones. R-GCN (main; PyTorch Geometric, basis decomposition, about 50–100 people-related relations) and CompGCN (second, to show the method generalises). Two layers, neighbour sampling, DistMult or RotatE decoder.
Metrics.
Metric
Measures
MRR, Hits@1/10
Link-prediction accuracy
Demographic parity gap @k
Group difference in how often an award appears in the top k
Equal-opportunity gap
Group difference in Hits@10 among true holders
Amplification ratio
Model disparity divided by disparity in the training data
Direct vs. popularity component
Decomposition by degree stratification and Oaxaca–Blinder
Probe accuracy
Gender recoverable from representations
Synthetic test bed. A generated graph with planted direct bias, planted popularity bias and a gender proxy, each of known strength. Every component is verified here before it touches Wikidata.
Compute. One RTX 5060 Ti (main) plus Google Colab for light runs.
Minimum deliverable (both versions). Baselines, depth sweep (0–3 GNN layers) and the bias decomposition. This is a complete empirical study on its own.
Version A: path-specific counterfactual fairness objective
Version A keeps a standard R-GCN/CompGCN and contributes a new training objective that lets us choose which bias path to remove. Estimated length: 12–14 weeks for two people.
Research questions
	•	How large is the bias in P166 and P106 predictions, and do GNNs amplify it relative to shallow embeddings and to the training data?
	•	What share of the bias is direct, and what share runs through popularity?
	•	Does a path-specific counterfactual objective reduce the chosen component, and at what accuracy cost compared with existing debiasing methods?
Method. We add a counterfactual term to the link-prediction loss:
\mathcal{L} = \mathcal{L}_{LP} + \lambda \left( \mathcal{L}_{dir} + \mu \, \mathcal{L}_{pop} \right)
	•	Direct term: for each sampled person, flip only the gender edge, re-encode, and penalise the difference between the two score distributions over candidate awards. No other part of the graph changes.
	•	Popularity term: precompute, once on CPU, real matched counterfactuals: people of another gender with similar occupation, birth decade and region, whose degree differs as it does in the data (in the spirit of CAF). Penalise prediction differences against them.
	•	Path choice: μ = 0 removes only the direct effect; μ > 0 also removes the popularity-mediated effect. λ traces the accuracy–fairness trade-off.
Experiments.
	•	Baselines: no debiasing, adversarial filter (FAN / Bose & Hamilton), FairDrop, data rebalancing, NIFTY.
	•	Ablation: direct term only, popularity term only, both.
	•	Pareto curves over λ, both backbones, 3–5 seeds, generalisation to P106.
	•	Synthetic check: each term removes the planted bias it targets and leaves the other.
Work split.

Theodora
[Teammate]
Shared (weeks 1–4)
Data pipeline, covariates
Evaluation module, base GNN, synthetic graph
Analysis
Bias decomposition
Probe tests, leakage analysis
Method
Popularity term + matching
Direct term (edge flip)
Baselines
FairDrop, rebalancing
Adversarial filter, NIFTY
Writing
Data, decomposition, results
Method, related work, results
Risks and fallbacks.
	•	Matching quality drives the popularity term. We report balance diagnostics and test sensitivity to matching choices.
	•	Unstable or weak effects: the direct term alone (μ = 0) is a complete result.
	•	Time: the minimum deliverable (baselines + decomposition) is ready by week 6.
Version B: combined architecture and objective
Version B adds one new module per bias path to the GNN and trains the whole model with the Version A objective, so each path is addressed in the representation and in the outputs. Estimated length: 15–16 weeks for two people.

Version B model · two new modules, one shared loss
Module A acts on the popularity path and Module B on the direct path inside each GNN layer; the counterfactual loss from Version A then acts on the outputs of both.
Research questions: RQ1 and RQ2 as in Version A, plus:
	•	Does each module remove the bias component it targets, and only that one?
	•	Are representation-level and output-level fixes complementary or redundant for the same path?
Modules.
	•	Module A, degree-robust aggregation (popularity path). A neighbour budget per relation, mean instead of sum aggregation, and a low-degree completion layer that adds a learned "virtual neighbour" message. The completion layer is trained self-supervised: we hide most edges of high-degree people and teach it to recover their full-information representation.
	•	Module B, two-channel layer (direct path). Each person has a sensitive channel and a content channel. The gender edge feeds only the sensitive channel; an adversary with gradient reversal and an independence loss keep gender out of the content channel. P166/P106 are scored from the content channel; whitelisted relations, where gender is part of the fact, may use both.
	•	Objective: the path-specific counterfactual loss from Version A.
\mathcal{L} = \mathcal{L}_{LP} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{ind}\mathcal{L}_{ind} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{cf}\left(\mathcal{L}_{dir} + \mu\,\mathcal{L}_{pop}\right)
We fix λ_ind and λ_rec from synthetic runs and tune mainly λ_adv and λ_cf.
Ablations.
Configuration
Module A
Module B
Counterfactual loss
Baseline
–
–
–
A only
✓
–
–
B only
–
✓
–
Loss only
–
–
✓
A + B
✓
✓
–
Full
✓
✓
✓
All eight combinations run on the synthetic graph; the six above run on Wikidata with both backbones, 3–5 seeds and the Version A baselines.
Work split: one person per bias path.

Theodora
[Teammate]
Shared (weeks 1–4)
Data pipeline, covariates
Evaluation module, base GNN, synthetic graph
Analysis
Bias decomposition
Probe tests, leakage analysis
Module
Module A: degree-robust aggregation
Module B: two-channel layer + adversary
Objective
Popularity term + matching
Direct term
Integration
Joint tuning, CompGCN port
Joint tuning, ablation runs
Risks and fallbacks.
	•	Adversarial training instability: warm-up, adversary update ratio, normalisation; drop to Module A + objective if it fails.
	•	Redundancy between a module and the loss term on the same path: the ablations test it; a redundant part is reported and removed.
	•	Many loss weights: ranges set on synthetic data before Wikidata runs.
	•	Time: fallbacks at week 6 (minimum deliverable) and week 11 (drop the modules and finish with Version A's scope).
Comparison and recommendation
We recommend Version B if the course expects a new architecture or a larger team scope, and Version A if a new training method is enough or the deadline is under 14 weeks.

Version A
Version B
Contribution
New training objective
Two new modules + the objective
Length (two people)
12–14 weeks
15–16 weeks
Main risk
Matching quality for the popularity term
Adversarial instability; module redundancy
Training runs
Moderate
Many (6 configs × 2 backbones × seeds × λ)
Individual contributions
One term each
One path each (module + term)
Falls back to
Minimum deliverable
Version A, then minimum deliverable

Project plan · both versions by week
Both versions share weeks 1–6; Version B spends three extra weeks on its two modules and can fall back to Version A's scope at week 11.
We would like your guidance on three points: whether a new training objective counts as sufficient methodological novelty, the expected scope for a two-person team, and the submission deadline.
Ethics and limitations
We report only aggregate results for groups and never predictions about named individuals.
	•	Biased ground truth. Wikidata likely under-records awards for women and for the Global South, so equal-opportunity metrics are measured against biased labels. We discuss this and, time permitting, check a sample against external sources.
	•	Gender categories. Wikidata includes non-binary and other values. We analyse every group with enough samples and describe smaller groups rather than dropping them silently.
	•	Sensitive attributes. We do not use ethnicity (P172), which is sparse and contested. Region is derived from citizenship.
	•	Snapshot. Wikidata5M reflects Wikidata around 2019; results describe that snapshot.
	•	Counterfactual assumptions. The popularity term depends on how people are matched. We make the matching criteria explicit and test their sensitivity.
	•	Fairness is a choice. Removing the popularity path is a normative decision. We report both settings (μ = 0 and μ > 0) rather than declaring one correct.
References
Entries marked * are cited from memory and still need checking.
	•	Fisher, Palfrey, Christodoulopoulos, Mittal. Measuring Social Bias in Knowledge Graph Embeddings. 2020.
	•	Fisher, Mittal, Palfrey, Christodoulopoulos. Debiasing Knowledge Graph Embeddings. EMNLP 2020.
	•	Keidar et al. Towards Automatic Bias Detection in Knowledge Graphs. Findings of EMNLP 2021.
	•	Du et al. Understanding Gender Bias in Knowledge Base Embeddings. ACL 2022.
	•	Arduini et al. Adversarial Learning for Debiasing Knowledge Graph Embeddings. MLG @ KDD 2020.
	•	Radstok, Chekol, Schäfer. Are Knowledge Graph Embedding Models Biased, or Is It the Data That They Are Trained On? Wikidata Workshop @ ISWC 2021.
	•	Social Biases in Knowledge Representations of Wikidata separates Global North from Global South (AuditLP). 2025.
	•	Kraft, Usbeck. The Lifecycle of "Facts": A Survey of Social Bias in Knowledge Graphs. AACL-IJCNLP 2022.
	•	Bose, Hamilton. Compositional Fairness Constraints for Graph Embeddings. ICML 2019.
	•	Chuang et al. Fair-RGNN: Mitigating Relational Bias on Knowledge Graphs. ACM TKDD 2025.
	•	Sasaki. Benchmarking Fairness-aware Graph Neural Networks in Knowledge Graphs. 2025.
	•	Spinelli et al. FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning. 2021.
	•	Li et al. On Dyadic Fairness: Exploring and Mitigating Bias in Graph Connections. ICLR 2021.
	•	Chiappa. Path-Specific Counterfactual Fairness. AAAI 2019.
	•	Agarwal, Lakkaraju, Zitnik. Towards a Unified Framework for Fair and Stable Graph Representation Learning (NIFTY). UAI 2021.*
	•	Ma et al. Learning Fair Node Representations with Graph Counterfactual Fairness (GEAR). WSDM 2022.
	•	Guo et al. Towards Fair Graph Neural Networks via Graph Counterfactual (CAF). CIKM 2023.
	•	Ling et al. Counterfactual Fairness on Graphs: Augmentations, Hidden Confounders, and Identifiability. TMLR 2025.
	•	Schlichtkrull et al. Modeling Relational Data with Graph Convolutional Networks (R-GCN). ESWC 2018.*
	•	Vashishth et al. Composition-based Multi-Relational Graph Convolutional Networks (CompGCN). ICLR 2020.*
	•	Liu et al. On Generalized Degree Fairness in Graph Neural Networks (DegFairGNN). AAAI 2023.*
	•	Shomer et al. Toward Degree Bias in Embedding-Based Knowledge Graph Completion. WWW 2023.*
	•	Wang et al. KEPLER / Wikidata5M dataset. TACL 2021.*
