# Beginner's Study Guide: Fair Link Prediction on Knowledge Graphs

Oct 21, 2026 · @Theodora Manika
This guide explains every idea in our project from zero, in the order you need them, with a source to study for each. Read sections 1–4 to understand the model, 5–7 to understand the fairness problem, and 8–9 to understand what we will actually do.
The project in one picture

Project pipeline · six steps, with the section that explains each
Each box names the section of this guide that explains it; step 6 is where our own contribution lives.
1. Machine learning basics
A machine learning model is a function with adjustable numbers (parameters) that we tune so its outputs match examples.
	•	Training: show the model examples, measure how wrong it is, nudge the parameters to be less wrong, repeat many times.
	•	Loss function: the number that says how wrong the model is. Training tries to make it small.
	•	Gradient descent: the method that nudges parameters in the direction that lowers the loss. Deep learning libraries compute the direction automatically.
	•	Epoch: one full pass over the training data.
	•	Train / validation / test split: train on one part, tune choices on the second, report results on the third, which the model never saw. This checks the model learned general patterns, not memorised examples.
	•	Overfitting: the model is great on training data but bad on new data, because it memorised instead of learning.
	•	Embedding: a list of numbers (a vector) that represents a thing, such as a word, a person or an award. Similar things get similar vectors. Most of our project is about learning good embeddings for people and awards.
	•	Hyperparameters: settings we choose rather than learn, such as learning rate, number of layers or embedding size.
Kinds of learning
Kind
Meaning
Where it appears in our project
Supervised
Learn from examples with human labels (input → correct answer)
The adversary that guesses gender from gender labels
Self-supervised
Create the training signal from the data itself, no human labels
Link prediction: hide known facts and learn to recover them; Module A's completion layer
Semi-supervised
Labels exist for only some examples; the rest are used too
Optional: sensitive attribute known for only some people
Adversarial
Two parts trained against each other (one tries to find something, the other tries to hide it)
Module B: the GNN hides gender, the adversary tries to find it
Regularisation
An extra term in the loss that forces a desired behaviour
The counterfactual fairness loss
Study: Dive into Deep Learning, chapters 2–5 (free online book with code); Andrew Ng's Machine Learning Specialization if you prefer video.
2. Knowledge graphs, Wikidata and benchmarks
A knowledge graph stores facts as a network: things are dots (nodes, also called entities) and facts are labelled arrows (edges, also called relations) between them.
	•	Triple: one fact written as (head, relation, tail), e.g. (Marie Curie, award received, Nobel Prize in Physics).
	•	Entity / relation: the things (people, awards, countries) and the types of links between them (award received, occupation, citizenship).
	•	Degree: how many edges an entity has. A famous person has a high degree; a lesser-known person a low one. Degree is our measure of popularity.
	•	Wikidata: a free, crowd-edited knowledge graph behind Wikipedia, with over 100 million items. Each entity has a Q-number (Q5 = human) and each relation a P-number.
	•	The P-numbers we use: P31 = instance of, P21 = sex or gender, P27 = country of citizenship, P106 = occupation, P166 = award received, P1411 = nominated for.
	•	RDF / SPARQL: the standard format for storing knowledge graphs and the query language for asking them questions. Useful for exploring Wikidata, not needed for training.
	•	Benchmark: a fixed dataset with fixed train/validation/test splits and fixed metrics, so different papers can compare results. Live Wikidata is not a benchmark because it changes every day.
	•	Wikidata5M: a fixed snapshot of Wikidata (about 4.6M entities, 810 relations, 21M triples) with official splits. It is a standard link prediction benchmark and our dataset.
Study: Wikidata's own introduction and SPARQL tutorial; Hogan et al., Knowledge Graphs (free book, kgbook.org), chapters 1–3; the Wikidata5M dataset page.
3. Link prediction and KG embeddings
Link prediction means guessing missing facts: given (Marie Curie, award received, ?), rank every award by how likely it is.
	•	KG embedding (KGE): learn a vector for every entity and every relation so that true triples get high scores and false ones low scores.
	•	TransE: treats a relation as a translation: head + relation ≈ tail. Simple and fast; we use it to make input embeddings.
	•	DistMult: scores a triple by multiplying the three vectors together. Simple, but treats every relation as symmetric (A→B scores the same as B→A).
	•	RotatE: treats a relation as a rotation in complex numbers. Handles asymmetric relations, like awards, better.
	•	Scoring function / decoder: the formula (TransE, DistMult, RotatE) that turns vectors into a score for a triple.
	•	Negative sampling: the graph only lists true facts, so we invent false ones by swapping the head or tail for a random entity, and train the model to score true facts above invented ones.
	•	MRR (mean reciprocal rank): for each test fact, find where the true answer lands in the ranking (1st, 2nd, …), take 1/rank, average. Higher is better; 1.0 is perfect.
	•	Hits@k: the share of test facts whose true answer is in the top k (e.g. Hits@10).
	•	Filtered evaluation: when ranking, ignore other answers that are also true, so the model is not punished for them.
	•	Ground truth: the held-out Wikidata triples. Warning: a fact missing from Wikidata counts as false even if it is true in reality (the open-world problem), and missing facts are more common for women and the Global South.
Study: the PyKEEN documentation, which explains each model and metric with runnable code; Stanford CS224W lecture on knowledge graph embeddings; original papers: TransE (Bordes et al. 2013), DistMult (Yang et al. 2015), RotatE (Sun et al. 2019).
4. Graph neural networks (GNNs)
A GNN builds each node's vector by collecting information from its neighbours, so a person's representation reflects their university, employer, spouse and country.
	•	Message passing: in each round, every node receives a "message" (a vector) from each neighbour, combines them (aggregation, usually sum or average) and updates its own vector.
	•	Layer: one round of message passing. Two layers mean information travels two hops (person → university → that university's other alumni).
	•	Depth sweep: training with 0, 1, 2 and 3 layers to see whether more message passing increases bias.
	•	GCN: the classic GNN; treats all edges the same.
	•	R-GCN (relational GCN): gives each relation type its own weights, so "spouse" and "educated at" are treated differently. Our main model.
	•	Basis decomposition: a trick in R-GCN that shares weights across relations, so 800+ relation types do not need 800+ separate weight matrices.
	•	CompGCN: a stronger relational GNN that learns vectors for relations too and combines them with node vectors. Our second model, to show results generalise.
	•	Encoder–decoder: the GNN (encoder) produces the vectors; a scoring function such as DistMult (decoder) uses them to rate triples.
	•	Neighbour sampling: instead of using all neighbours of all nodes at once (too big for a GPU), each training batch samples a few neighbours per node, e.g. 15 at the first hop and 10 at the second. No nodes are removed from the graph.
	•	Homophily: connected nodes tend to be similar (e.g. same gender or country). Message passing can therefore mix group information into a person's vector.
Study: A Gentle Introduction to Graph Neural Networks (interactive article, start here); Stanford CS224W lectures on GNNs; William Hamilton's free book Graph Representation Learning, chapters 5–7; PyTorch Geometric tutorials; papers: GCN (Kipf & Welling 2017), R-GCN (Schlichtkrull et al. 2018), CompGCN (Vashishth et al. 2020).
5. Bias and fairness
A model is socially biased when its predictions systematically favour or disfavour groups of people, for example recommending prestigious awards to men more often than to equally qualified women.
	•	Sensitive attribute: a characteristic we do not want to drive predictions, here gender (P21) and region (from P27).
	•	Group: the people who share a value of the sensitive attribute (e.g. women, people from the Global South).
	•	Proxy: another feature that reveals the sensitive attribute indirectly, such as spouse, the occupation "actress" or a first name. Removing the gender edge does not remove proxies.
	•	Probe: a small classifier that tries to predict gender from the model's vectors. High accuracy means gender information is still hidden inside them.
	•	Data bias vs. model bias: the data itself may be skewed (fewer women, fewer facts about them). A model can copy that skew or make it worse.
	•	Amplification ratio: model disparity divided by data disparity. Above 1 means the model makes the bias worse.
Fairness metrics
Metric
Question it answers
Needs ground truth?
Demographic parity gap @k
Do groups get an award in their top-k predictions equally often?
No
Equal-opportunity gap
Among people who truly hold the award, is the model equally good at finding it for each group?
Yes
Equalized odds
Equal true-positive and false-positive rates across groups
Yes
Per-group MRR
Is overall prediction quality the same for each group?
Yes
Study: Barocas, Hardt & Narayanan, Fairness and Machine Learning (free book), chapters 1–3; Hardt et al., Equality of Opportunity in Supervised Learning (2016); Kraft & Usbeck, The Lifecycle of "Facts" (survey of bias in knowledge graphs); Fisher et al., Measuring Social Bias in Knowledge Graph Embeddings (2020).
6. The two sources of bias and how to decompose them
One bias number can hide two different causes, and each needs a different fix.
	•	Direct effect: the model uses gender itself (or a proxy) as a signal. Two equally well-connected people still get different predictions because of gender.
	•	Popularity effect: well-connected people get better predictions because the model knows more about them; since men are better connected in Wikidata, this becomes a gender gap. Popularity here is a mediator: gender → degree → prediction.
	•	Popularity is not automatically "fair": women have fewer Wikipedia pages and links partly because of historical discrimination. Whether to remove the popularity path is a choice we report, not a fact.
	•	Bias decomposition: splitting the total gap into the direct part and the popularity part. It is statistics on a trained baseline's predictions, not a training step.
	•	Stratification: group people by degree and measure the gender gap inside each group. The gap left between equally connected people is the direct part.
	•	Oaxaca–Blinder decomposition: a regression method from economics (used for gender pay gaps) that splits a gap into a part explained by measured differences (degree) and an unexplained rest (direct effect).
	•	When: after training a normal baseline, before building the debiasing methods; then again after each method, to show which part it reduced.
Study: the Oaxaca–Blinder decomposition article for the idea, then Jann (2008), "The Blinder–Oaxaca decomposition for linear regression models", Stata Journal, for the method; Arduini et al., Adversarial Learning for Debiasing Knowledge Graph Embeddings (popularity bias in KGs); Brady Neal's free Introduction to Causal Inference course, lectures on mediation.
7. Debiasing methods
Debiasing methods change the data, the training or the model so that predictions depend less on the sensitive attribute.
	•	Data rebalancing / reweighting: change the training data, or the weight of each example, so groups are balanced. Simple and often effective.
	•	Adversarial debiasing: add a small "adversary" network that tries to guess gender from the model's vectors; train the main model to make it fail.
	•	Gradient reversal: the trick that makes adversarial training work in one pass: the adversary's learning signal is flipped before it reaches the main model.
	•	FairDrop: randomly drop more edges between same-group nodes during training, so message passing mixes less group information.
	•	Counterfactual fairness: a prediction is fair for a person if it would stay the same had their sensitive attribute been different.
	•	Counterfactual: the "what if" version of a person, e.g. the same person with the gender edge flipped.
	•	Path-specific counterfactual fairness: remove only the effect of the sensitive attribute along chosen paths (e.g. the direct path), while allowing others.
	•	Matching: finding real people who resemble each person (same occupation, decade, region) but differ in gender, to use as realistic counterfactuals instead of invented ones.
	•	Graph counterfactual methods: NIFTY (flip the node's attribute), GEAR (also neighbours'), CAF (use real matched nodes). All designed for node classification, not link prediction.
	•	Accuracy–fairness trade-off: debiasing usually costs some accuracy. We report it as a Pareto curve, not a single number.
Study: Ganin et al., Domain-Adversarial Training of Neural Networks (gradient reversal); Bose & Hamilton, Compositional Fairness Constraints for Graph Embeddings; Kusner et al., Counterfactual Fairness; Chiappa, Path-Specific Counterfactual Fairness; Spinelli et al., FairDrop; Ma et al., GEAR; Guo et al., CAF.
8. Our methods
Both versions start from the same model (TransE inputs → 2-layer R-GCN or CompGCN → DistMult decoder); Version A changes the training loss, Version B also changes the GNN layers.
Version A: path-specific counterfactual objective
	•	Direct term: for each sampled person, flip only the gender edge, recompute, and penalise any change in award scores.
	•	Popularity term: compare each person with matched real people of another gender whose connectivity differs as it does in the data, and penalise differences.
	•	μ (mu): a dial. μ = 0 removes only the direct effect; μ > 0 also removes the popularity effect.
	•	λ (lambda): how strongly fairness is weighted against accuracy; varying it traces the trade-off curve.
Version B: two modules + the same objective
	•	Module A, degree-robust aggregation (popularity path): cap the number of messages per relation, average instead of sum, and a small completion layer that fills in information for people with few connections. It learns by hiding most edges of famous people and reconstructing their full vector (self-supervised).
	•	Module B, two-channel representation (direct path): split each vector into a sensitive channel (where gender may go) and a content channel (used for award predictions). An adversary with gradient reversal keeps gender out of the content channel, like blind peer review.
	•	Whitelist: relations where gender is legitimately part of the fact may use both channels.
	•	Ablation: switching modules on and off one at a time to show which part does what.
Study: our project documents (proposal, module explainer, step-by-step plan); Dai & Wang, FairGNN, WSDM 2021 (adversarial fairness in GNNs); Liu et al., DegFairGNN, AAAI 2023 (degree fairness).
9. Doing experiments properly
Good experiments make it possible to trust that a result comes from the method, not from luck or a flawed split.
	•	Representative sample: a subset whose groups, degrees and award rates look like the full Wikidata population. We claim representativeness of Wikidata, not of the real world.
	•	Stratified sampling: sample within each combination of gender, region, degree band and decade, so every group keeps its share (or is deliberately oversampled with weights).
	•	Standardised mean difference (SMD): a number comparing two distributions; below about 0.1 counts as balanced.
	•	Data leakage: the test answer sneaks into training, e.g. through an inverse relation or "nominated for". It makes results look better than they are.
	•	Synthetic graph: a small fake graph where we plant bias of known strength. Because we know the true answer, it tests whether each method works before real data.
	•	Baseline: a standard model without our changes, to compare against.
	•	Ablation study: remove one component at a time to see what each contributes.
	•	Random seed: the starting point for randomness. Running 3–5 seeds shows whether results are stable.
	•	Bootstrap confidence interval: resample the test set many times to estimate how uncertain a metric is.
	•	Pareto curve: a plot of accuracy against fairness for different settings; better methods sit closer to the top-right.
Study: Ruffinelli et al., "You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings" (ICLR 2020) on fair comparison of KG models; Kapoor & Narayanan, Leakage and the Reproducibility Crisis in ML-based Science on leakage; Dive into Deep Learning chapter on model selection.
10. Tools
Everything runs in Python on one RTX 5060 Ti, with Google Colab as a backup.
Tool
What it is
What we use it for
Where to learn
Python 3.11+
Programming language
Everything
python.org tutorial
PyTorch ≥ 2.7 (CUDA 12.8)
Deep learning library
Models, training, GPU
PyTorch tutorials
PyTorch Geometric (PyG)
GNN library on top of PyTorch
R-GCN layer, neighbour sampling
PyG docs
PyKEEN
KG embedding library
TransE / RotatE baselines and input embeddings
PyKEEN docs
pandas, NumPy
Data tables and arrays
Data processing
pandas getting started
statsmodels
Statistics library
Regressions, Oaxaca–Blinder
statsmodels docs
scikit-learn
Classic ML library
Probe classifier, matching
scikit-learn user guide
matplotlib / seaborn
Plotting
Figures
matplotlib tutorials
Git + GitHub
Version control
Sharing code between us
GitHub skills
Weights & Biases
Experiment tracker
Logging every run
wandb docs
CUDA / WSL2
GPU driver layer / Linux on Windows
Running PyTorch on the 5060 Ti
PyTorch install selector
Google Colab
Free cloud notebooks with a GPU
Light runs, backup
colab.research.google.com
Study plan
Learn in this order: each step builds on the one before, and you can start coding after step 4.
	•	ML and PyTorch basics: Dive into Deep Learning chapters 2–5; PyTorch "Learn the Basics" tutorial. Goal: train a small neural network yourself.
	•	Knowledge graphs and Wikidata: Wikidata introduction and SPARQL tutorial; Knowledge Graphs book chapters 1–3. Goal: query people with their awards in Wikidata.
	•	KG embeddings and link prediction: PyKEEN tutorial; CS224W KG-embedding lecture; TransE and RotatE papers. Goal: train TransE in PyKEEN and read its MRR.
	•	GNNs: Gentle Introduction to GNNs; CS224W GNN lectures; PyG tutorials; R-GCN paper. Goal: run a PyG R-GCN example.
	•	Fairness basics: Fairness and ML book chapters 1–3; Hardt et al. 2016. Goal: compute demographic parity and equal opportunity on a toy dataset.
	•	Bias in KGs: Kraft & Usbeck survey; Fisher et al. 2020; Arduini et al. 2020. Goal: explain direct vs. popularity bias to your teammate.
	•	Debiasing and counterfactual fairness: Ganin et al. (gradient reversal); Bose & Hamilton; Kusner et al.; Chiappa; GEAR and CAF. Goal: explain how our objective differs from these.
	•	Decomposition: Oaxaca–Blinder article; mediation lectures in Brady Neal's course. Goal: run Oaxaca–Blinder on a toy table in statsmodels.
	•	☐ Steps 1–2 (week 1)
	•	☐ Steps 3–4 (weeks 2–3, alongside data work)
	•	☐ Steps 5–6 (week 4)
	•	☐ Steps 7–8 (weeks 5–6, before building the methods)
Sources to study
Start with the free courses and books; read papers once the basics feel familiar. Links marked * are from memory and were not re-checked while writing this guide; confirm them before citing.
Courses and free books
	•	Dive into Deep Learning* — ML and deep learning with code
	•	Stanford CS224W: Machine Learning with Graphs* — KG embeddings and GNNs (lecture videos on YouTube)
	•	Hamilton, Graph Representation Learning* — free book on graph ML
	•	A Gentle Introduction to Graph Neural Networks* — interactive article
	•	Hogan et al., Knowledge Graphs* — free book
	•	Barocas, Hardt & Narayanan, Fairness and Machine Learning* — free book
	•	Brady Neal, Introduction to Causal Inference* — free course
Documentation
	•	PyTorch tutorials, PyTorch Geometric, PyKEEN*
	•	Wikidata introduction* and SPARQL tutorial*
	•	Wikidata5M dataset
Core model papers
	•	Bordes et al. 2013, TransE (NeurIPS)*
	•	Yang et al. 2015, DistMult (ICLR)*
	•	Sun et al. 2019, RotatE*
	•	Kipf & Welling 2017, GCN*
	•	Schlichtkrull et al. 2018, R-GCN*
	•	Vashishth et al. 2020, CompGCN*
	•	Wang et al. 2021, KEPLER / Wikidata5M (TACL)*
Bias in knowledge graphs
	•	Fisher et al. 2020, Measuring Social Bias in Knowledge Graph Embeddings
	•	Kraft & Usbeck 2022, The Lifecycle of "Facts" (survey)
	•	Keidar et al. 2021, Towards Automatic Bias Detection in Knowledge Graphs
	•	Du et al. 2022, Understanding Gender Bias in Knowledge Base Embeddings
	•	Arduini et al. 2020, Adversarial Learning for Debiasing Knowledge Graph Embeddings
	•	Radstok et al. 2021, Are KG Embedding Models Biased, or Is It the Data?
	•	AuditLP: Social Biases in Knowledge Representations of Wikidata (2025)
	•	Chuang et al., Fair-RGNN: Mitigating Relational Bias on Knowledge Graphs
	•	Sasaki 2025, Benchmarking Fairness-aware GNNs in Knowledge Graphs
Fairness and debiasing methods
	•	Hardt et al. 2016, Equality of Opportunity in Supervised Learning*
	•	Ganin et al. 2016, Domain-Adversarial Training* (gradient reversal)
	•	Bose & Hamilton 2019, Compositional Fairness Constraints for Graph Embeddings
	•	Spinelli et al. 2021, FairDrop
	•	Dai & Wang 2021, FairGNN (WSDM)*
	•	Liu et al. 2023, DegFairGNN (AAAI)*
	•	Kusner et al. 2017, Counterfactual Fairness*
	•	Chiappa 2019, Path-Specific Counterfactual Fairness
	•	Agarwal et al. 2021, NIFTY (UAI)*
	•	Ma et al. 2022, GEAR
	•	Guo et al. 2023, CAF
Method and evaluation
	•	Ruffinelli et al. 2020, "You CAN Teach an Old Dog New Tricks!" (ICLR)*
	•	Kapoor & Narayanan, Leakage and the Reproducibility Crisis*
	•	Oaxaca–Blinder decomposition*; Jann 2008, Stata Journal
