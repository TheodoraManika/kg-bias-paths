# Disentangling Bias in Knowledge-Graph Link Prediction

## 1. Problem Statement

Knowledge-graph link prediction models learn to infer missing facts from existing ones, so their predictions can reproduce, and sometimes amplify, the social biases and unequal representation in the underlying data (Fisher et al., 2020; Kraft & Usbeck, 2022).

Two mechanisms are commonly proposed:

- **Encoded demographic information:** entity embeddings can absorb demographic information, either directly or through proxy facts such as spouse, occupation or name, even when the sensitive attribute is not used as an input (Fisher et al., 2020; Du et al., 2022).
- **Connectivity:** prediction quality depends on how well connected an entity is: embedding models and graph neural networks perform worse on low-degree entities (Tang et al., 2020; Shomer et al., 2023; Subramonian et al., 2024), and in graph neural networks message passing spreads information along edges, so a group's position in the graph shapes its representations. If one demographic group is less well connected, connectivity alone can produce a group disparity.

Radstok et al. (2021) further argue that much of the bias in knowledge-graph embeddings originates in the data rather than in the model.

This project studies gender disparities in predicting awards received by people (Wikidata property `P166`).

> **Central Research Question**
>
> To what extent are gender disparities in award link prediction associated with (a) gender information encoded in entity representations, and (b) differences in graph connectivity between gender groups?

We address it at two levels:

| Level | Question |
|:---|:---|
| **Data** | How unequal are gender groups in the knowledge graph itself, in representation, connectivity (degree) and recorded awards? This is measured before any model is trained, so that model disparities can later be compared with disparities already present in the data. |
| **Model** | After training a link prediction model, does its accuracy differ between gender groups, and how much of that difference remains when we compare people with similar connectivity? Can gender be recovered from the learned embeddings? |

Together these let us separate bias that the model inherits from the data from bias that comes from what the model learns, without claiming a full causal decomposition.

## 2. Dataset

We use [Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m) (Wang et al., 2021), a knowledge graph built from a 2019 snapshot of Wikidata and Wikipedia. We downloaded the official transductive split and verified these statistics against the published figures.

| Entities | Relations | Training triples |
|:---:|:---:|:---:|
| **4,594,485** | **822** | **20,614,279** |

### Missing attributes and how we added them

Wikidata5M keeps only facts between entities that have an English Wikipedia page, and it keeps no literal values such as dates. As a result:

- It contains almost no human gender: the Wikidata values for male (`Q6581097`) and female (`Q6581072`) have no Wikipedia page and were dropped, and only 366 of the 1,509,635 humans in the graph have a sex-or-gender (`P21`) fact.
- Dates of birth (`P569`) are missing entirely.

Because group membership is needed to measure any disparity, we retrieved `P21` and `P569` for all 1,509,635 humans from Wikidata via the [QLever](https://qlever.dev/wikidata) SPARQL endpoint (Wikidata dump of 10 August 2026, with live updates) and joined them on Wikidata IDs.

| Attribute | Humans with it in Wikidata5M | After adding from Wikidata |
|:---|:---:|:---:|
| Gender (`P21`) | 366 | **99.2%** |
| Date of birth (`P569`) | none | **92.9%** |

> **Note:** These attributes are stored in a separate table and are **not** added to the graph the model is trained on; they are used only to group people for evaluation, for the audit, and for probing.

We also recorded each person's current awards, occupations and citizenship to measure how much of this information Wikidata5M omits. The download script and file checksums are in the repository.

### Scope

Training on the full graph is computationally demanding for this project, so we will build a reproducible subset focused on people and their relevant relations.

| Property | Description | Role in the project |
|:---|:---|:---|
| `P166` | Award received | Prediction target |
| `P21` | Sex or gender | Defines the groups |
| `P106` | Occupation | May be used as a control variable |
| `P569` | Date of birth (birth decade) | May be used as a control variable |
| `P27` | Country of citizenship | Optional regional analysis; depends on group sizes |

Node degree in the training graph measures connectivity.

Every filtering step, the subset and the train/validation/test split will be documented and versioned with checksums. These choices are a starting point: we expect to adjust the subset, the relation set and the split as we learn what the data and our compute budget allow, and we will document each change and the reason for it.

## 3. Problem Challenges and Related Work

Existing work, collected in the repository's `research/` directory, covers:

- Social bias in knowledge-graph embeddings (Fisher et al., 2020; Du et al., 2022).
- The role of data versus model (Radstok et al., 2021).
- Degree-related bias in graph learning (Tang et al., 2020; Liu et al., 2023; Shomer et al., 2023; Subramonian et al., 2024).
- Fairness-aware and counterfactual methods (Bose & Hamilton, 2019; Chiappa, 2019; Buyl & De Bie, 2020).

These studies usually treat demographic bias and degree bias separately. Our question is how much of a gender disparity in award prediction is associated with differences in connectivity, and how much remains among similarly connected people.

### Main challenges

- **Incomplete and biased labels:** Award records in Wikidata are incomplete, and probably more so for some groups (Kraft & Usbeck, 2022). Our first comparison suggests Wikidata5M reinforces this:

  | Retained in Wikidata5M (share of current Wikidata values) | Women | Men |
  |:---|:---:|:---:|
  | Awards | 23.1% | 25.6% |
  | Occupations | 33.0% | 39.1% |

  This comparison mixes Wikidata5M's filtering with seven years of later Wikidata edits, which we will try to separate. Measured accuracy is therefore relative to labels that may themselves be biased.
- **Uneven representation:** Among humans with a recorded gender, 82.7% are male, 17.2% female and about 0.1% have another value. Small groups limit what can be measured reliably; we will report them descriptively rather than drop them silently.
- **Correlation versus causation:** Degree, occupation, birth period and gender are all related, so observed associations cannot be read as causal effects without further assumptions.
- **Attributes from two time points:** The graph reflects 2019, while gender and birth date come from 2026; we will state this limitation explicitly.

## 4. Baselines and Methodology

The plan is staged, so that each step produces a usable result even if later steps are cut.

| Stage | | Part of |
|:---:|:---|:---|
| 1 | Data audit | Minimum viable project |
| 2 | Baseline model | Minimum viable project |
| 3 | Group disparity | Minimum viable project |
| 4 | Connectivity analysis | Minimum viable project |
| 5 | Encoded gender | If time allows |

### Baselines

We use "baseline" in two senses: the models whose bias we measure, and the existing debiasing methods we compare our own methods against.

**1. Link prediction baselines (no debiasing).** These are the models whose gender gap we measure in the core study, and the reference point every debiasing method is compared with.

| Model | Type | Used in |
|:---|:---|:---|
| **TransE** (Bordes et al., 2013) | Knowledge-graph embedding: a fact is likely if *head + relation ≈ tail* | Core study (main baseline) |
| **DistMult** (Yang et al., 2015) | Knowledge-graph embedding: scores a fact by multiplying head, relation and tail | Core study; also the decoder of the R-GCN |
| **RotatE** (Sun et al., 2019) | Knowledge-graph embedding: relations as rotations | Core study, if time allows |
| **R-GCN** without any debiasing (Schlichtkrull et al., 2018) | Graph neural network with a DistMult decoder | Version A and B: the model our methods modify |

**2. Debiasing baselines (Version A and B only).** Existing methods we compare our methods against, each applied to the same R-GCN:

| Method | What it does |
|:---|:---|
| **No debiasing** | The plain R-GCN above |
| **Adversarial filter** (Bose & Hamilton, 2019) | Trains a filter on the embeddings so that an adversary cannot recover gender |
| **FairDrop** (Spinelli et al., 2021) | Randomly drops edges between people of the same group during training to reduce homophily |
| **Data rebalancing** | Reweights or resamples training facts so that gender groups contribute equally |
| **NIFTY** (Agarwal et al., 2021), if time allows | Makes representations stable when the sensitive attribute is flipped |

### Stage 1 — Data audit (no model)

- Report the size of each gender group, award rates, degree distributions and birth-period distributions.
- Report how much award and occupation information is missing per group.

This gives the data disparity against which model disparities are compared.

### Stage 2 — Baseline model

Train a standard knowledge-graph embedding model (**TransE** or **DistMult**, using PyKEEN) on the people subset, with the P21 facts **excluded** from the training graph. Evaluate award prediction with filtered mean reciprocal rank (**MRR**) and **Hits@1/3/10**:

| Metric | Purpose |
|:---|:---|
| MRR | Measures the quality of the ranking assigned to correct answers |
| Hits@1 | Measures how often the correct answer ranks first |
| Hits@3 / Hits@10 | Measure how often the correct answer appears among the top three / top ten |

### Stage 3 — Group disparity

Compute the same metrics separately for each gender group on test award triples, with **bootstrap confidence intervals** and **several random seeds**, so that differences smaller than run-to-run variation are not reported as findings.

### Stage 4 — Connectivity analysis

- Compare groups within degree bins.
- Fit a simple regression of per-triple performance on gender and degree (with occupation and birth decade as controls where available). This shows how much of the gap is associated with connectivity.
- The ratio between the model's disparity and the data's disparity indicates whether the model amplifies the bias already present in the data.

### Stage 5 — Encoded gender (if time allows)

Train a probe classifier to predict gender from frozen entity embeddings. If gender is predictable even though `P21` was removed, the model has learned it from proxy facts.

### Beyond the core: debiasing methods (Version A and Version B)

Stages 1–5 **diagnose** the bias. Version A and Version B go one step further and try to **reduce** it. Which one we attempt depends on your guidance and on the diagnosis (see Section 5).

#### The two paths we target

A graph neural network builds each person's representation from what they are connected to: university, employer, spouse, occupation, country. A gender gap in award predictions can arise along two paths:

| Path | What happens | Example |
|:---|:---|:---|
| **Direct path** | The model uses gender itself, either from the gender fact or from proxies such as spouse, "actress" or a first name. | The model learns "this person is a woman, so less likely to win this award". |
| **Connectivity path** | Well-connected people get richer representations and better predictions, and well-connected people in Wikidata are disproportionately men. | A famous scientist with 500 connections is predicted well; a lesser-known scientist with 3 connections is predicted poorly, whatever their gender. |

The diagnosis in Stages 4–5 estimates how much of the gap goes through each path. Both versions then try to remove it, path by path.

#### Shared tools for both versions

- **A relational graph neural network** (**R-GCN** with a DistMult decoder; Schlichtkrull et al., 2018), later also CompGCN, replaces the Stage 2 baseline. We first rerun Stages 3–5 on it, with 0 to 3 layers of message passing, to see whether message passing itself increases the gap.
- **A synthetic test graph:** a small artificial graph in which we plant a direct bias, a connectivity bias and a gender proxy, each of known strength. Every component is tested here first: it must remove the bias it targets and leave the other one unchanged. Only components that pass are run on Wikidata.

#### Version A: a counterfactual training objective

**Idea.** Keep the R-GCN as it is, and change only *how it is trained*. We add a fairness term to the training loss, based on path-specific counterfactual fairness (Chiappa, 2019), which asks: *would this person's award prediction change if only their gender were different?*

**Steps.**

1. **Direct term (direct path).** For a sample of people, create a copy in which only the gender fact is flipped, run both through the model, and penalise any difference in their award scores. Test on the synthetic graph.
2. **Matching (prepared once, on CPU).** For each person, find real people of another gender with a similar occupation, birth decade and region, and check that the matched groups are balanced on these variables.
3. **Popularity term (connectivity path).** Penalise differences between each person's predictions and those of their matched counterparts. Test on the synthetic graph.
4. **Training.** The model is trained on link prediction plus the two fairness terms, each with a weight. Varying the weights traces the trade-off between accuracy and fairness. We report two settings: direct term only, and both terms, since deciding which path counts as unfair is a normative choice.
5. **Comparison.** Compare against the debiasing baselines listed under *Baselines* above, and test each term alone and both together, with 3–5 random seeds.

#### Version B: two debiasing modules + the Version A objective

**Idea.** Change the *architecture* of the R-GCN, with one module per path, and then train it with the Version A objective.

**Steps.**

1. **Module A: degree-robust aggregation (connectivity path).**
   - Each person receives at most a fixed number of messages per relation, so famous people cannot hoard information.
   - Messages are averaged instead of summed, so more neighbours no longer means a "bigger" representation.
   - For people with very few connections, a small learned component fills in missing information. It is trained by hiding most connections of well-connected people and teaching it to reconstruct their full representation.
   - *Test:* on the synthetic graph, the connectivity bias goes down and the direct bias is unchanged.
2. **Module B: two-channel layer with an adversary (direct path).**
   - Each person's representation is split into a *sensitive* channel, the only place gender may go, and a *content* channel, which is the only part used to predict awards.
   - An adversary, a small classifier, tries to guess gender from the content channel; the model is trained to make it fail.
   - *Test:* a probe on the content channel guesses gender no better than chance, and the direct bias goes down.
3. **Add the Version A objective** to the model with both modules, and tune the weights.
4. **Comparison.** Six configurations on Wikidata: baseline, Module A only, Module B only, objective only, both modules, and everything together. We use the same baselines as Version A, with R-GCN and CompGCN and 3–5 seeds. This also shows whether a module and the objective term on the same path are both needed.

Each module alone closes only one path, which is why Version B combines them.

#### How we judge success (both versions)

For every model we rerun the measurements from Stages 3–5 and ask:

| Question | Measured by |
|:---|:---|
| Did the gender gap shrink? | Per-group MRR and Hits@k, with confidence intervals (Stage 3) |
| Which path did it shrink? | Connectivity analysis before and after (Stage 4) |
| Is gender still encoded? | Probe accuracy (Stage 5) |
| What did it cost? | Overall MRR and Hits@k compared with the baseline |
| Is the model still amplifying the data's bias? | Ratio of model disparity to data disparity (Stage 1) |

### Reproducibility

All experiments use fixed, versioned data splits, documented preprocessing, a locked software environment (uv) and fixed random seeds, so that every reported number can be traced to a code version, a data version and a seed.

## 5. Feasibility and Scope: Options for Discussion

We are a two-person team; the computing environment, data download and attribute retrieval are already complete. We work with a single consumer GPU (RTX 5060 Ti, 16 GB) and Google Colab, so running the full Wikidata5M graph, extensive hyperparameter searches and several architectures at once would be too ambitious.

We see three possible scopes and would value your guidance on which to aim for. They build on each other: the first six weeks are the same for all three, and each option keeps the previous one as a fallback.

| | **Core study** | **Version A** | **Version B** |
|:---|:---|:---|:---|
| **What it adds** | Diagnosis only: stages 1–4 above (stage 5 if time allows) | Core study + a **counterfactual training objective** | Version A + **two debiasing modules** inside the GNN |
| **Model** | Knowledge-graph embedding baseline (TransE or DistMult) | R-GCN, then CompGCN | R-GCN, then CompGCN |
| **Contribution** | An empirical study of how much of the gender gap is linked to connectivity and to encoded gender | A new objective with two terms: a **direct term** (flip only the gender edge and penalise the change in award scores) and a **popularity term** (compare each person with real people of another gender and similar occupation, birth decade and region) | **Module A**, degree-robust aggregation, for the connectivity path; **Module B**, a two-channel layer with an adversary that keeps gender out of the channel used for award prediction, for the direct path; both trained with the Version A objective |
| **Comparisons** | Gender groups, degree bins | No debiasing, adversarial filter (Bose & Hamilton, 2019), FairDrop, data rebalancing | Same as Version A, plus six configurations (each module alone, the objective alone, combinations) |
| **Duration** | About 6 weeks | About 13 weeks | About 16 weeks |
| **Fallback** | – | Core study (week 6) | Version A (week 11) |

**Timeline.** Weeks 1–6 are shared: data, audit, baseline and diagnosis (the core study). At week 6, the diagnosis shows which source of bias dominates, connectivity or encoded gender, and therefore which component to prioritise. Version A then builds the objective (weeks 7–13). Version B builds the two modules first (weeks 7–9), then adds the objective (weeks 10–11), and spends the remaining weeks on integration, comparisons and writing (weeks 12–16).

**Minimum viable outcome.** Whichever option we choose, the core study is completed first and stands on its own: a documented subset, a data audit, one baseline model, and a careful group and connectivity analysis with reproducible evaluation.


## 6. References

- Agarwal, C., Lakkaraju, H., & Zitnik, M. (2021). Towards a unified framework for fair and stable graph representation learning. *UAI*. [arXiv:2102.13186](https://arxiv.org/abs/2102.13186)
- Bordes, A., Usunier, N., Garcia-Durán, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- Bose, A. J., & Hamilton, W. L. (2019). Compositional fairness constraints for graph embeddings. *ICML*. [arXiv:1905.10674](https://arxiv.org/abs/1905.10674)
- Buyl, M., & De Bie, T. (2020). DeBayes: A Bayesian method for debiasing network embeddings. *ICML*. [arXiv:2002.11442](https://arxiv.org/abs/2002.11442)
- Chiappa, S. (2019). Path-specific counterfactual fairness. *AAAI*. [arXiv:1802.08139](https://arxiv.org/abs/1802.08139)
- Du, Y., Zheng, Q., Wu, Y., Lan, M., Yang, Y., & Ma, M. (2022). Understanding gender bias in knowledge base embeddings. *ACL*. [ACL Anthology](https://aclanthology.org/2022.acl-long.98/)
- Fisher, J., Palfrey, D., Christodoulopoulos, C., & Mittal, A. (2020). Measuring social bias in knowledge graph embeddings. [arXiv:1912.02761](https://arxiv.org/abs/1912.02761)
- Kraft, A., & Usbeck, R. (2022). The lifecycle of "facts": A survey of social bias in knowledge graphs. *AACL-IJCNLP*. [arXiv:2210.03353](https://arxiv.org/abs/2210.03353)
- Liu, Z., Nguyen, T.-K., & Fang, Y. (2023). On generalized degree fairness in graph neural networks. *AAAI*. [arXiv:2302.03881](https://arxiv.org/abs/2302.03881)
- Radstok, W., Chekol, M. W., & Schäfer, M. T. (2021). Are knowledge graph embedding models biased, or is it the data that they are trained on? *Wikidata Workshop @ ISWC*. [CEUR-WS Vol-2982](https://ceur-ws.org/Vol-2982/paper-5.pdf)
- Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. *ESWC*. [arXiv:1703.06103](https://arxiv.org/abs/1703.06103)
- Shomer, H., et al. (2023). Toward degree bias in embedding-based knowledge graph completion. *WWW*. [arXiv:2302.05044](https://arxiv.org/abs/2302.05044)
- Spinelli, I., Scardapane, S., Hussain, A., & Uncini, A. (2021). FairDrop: Biased edge dropout for enhancing fairness in graph representation learning. *IEEE Transactions on Artificial Intelligence*. [arXiv:2104.14210](https://arxiv.org/abs/2104.14210)
- Subramonian, A., Kang, J., & Sun, Y. (2024). Theoretical and empirical insights into the origins of degree bias in graph neural networks. [arXiv:2404.03139](https://arxiv.org/abs/2404.03139)
- Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR*. [arXiv:1902.10197](https://arxiv.org/abs/1902.10197)
- Tang, X., et al. (2020). Investigating and mitigating degree-related biases in graph convolutional networks. *CIKM*. [doi:10.1145/3340531.3411872](https://doi.org/10.1145/3340531.3411872)
- Wang, X., et al. (2021). KEPLER: A unified model for knowledge embedding and pre-trained language representation. *TACL*. [arXiv:1911.06136](https://arxiv.org/abs/1911.06136)
- Yang, B., Yih, W.-t., He, X., Gao, J., & Deng, L. (2015). Embedding entities and relations for learning and inference in knowledge bases. *ICLR*. [arXiv:1412.6575](https://arxiv.org/abs/1412.6575)

## 7. Declaration on Generative AI

The authors used the following generative AI tools in this project:

- **ChatGPT (OpenAI), Claude (Anthropic), including Claude Code in the code editor**, used for:
  - *Project guides and task planning:* drafting the guides in `docs/` (environment and reproducibility, data storage, project plan, statistical pitfalls) and breaking the plan into GitHub issues with "done when" criteria.
  - *Code:* writing and revising scripts, including the GPU environment check (`scripts/00_check_env.py`) and the Wikidata attribute download (`scripts/00_download_wikidata_attributes.py`).
  - *Setup and debugging:* fixing errors in scripts and configuration such as `.gitignore`.
  - *Data checks:* checking the downloaded Wikidata5M files against the published statistics, generating the checksum manifest, identifying the missing gender and birth-date attributes, and producing the summary statistics reported in this description.
  - *Writing:* drafting this project description and pull-request descriptions, and adding and checking web links for the references.
  - *Questions and understanding the literature:* answering the authors' questions about concepts, tools and the dataset, and explaining the methods and findings of the research papers to help the authors understand them.

The research question, the choice of dataset and methods, and all decisions about the project were made by the authors. All research papers cited in this description and collected in the `research/` folder were found by the authors through their own online searches. The authors changed and revised all AI-generated text, reviewed all AI-generated code, ran and tested the code themselves, and checked the reported numbers against the data. The authors take full responsibility for the content of this description and the repository.
