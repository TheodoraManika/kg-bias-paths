# Disentangling Bias in Knowledge-Graph Link Prediction

## **Problem statement:** 

Knowledge-graph link prediction models learn to infer missing relationships from existing facts, but their predictions may reflect social biases and unequal representation in the underlying data. **rewrite** That is because of the messaging in ... and add reference here **end rewrite** This project investigates whether gender-related disparities in predicting awards received by people are associated with gender information encoded in entity representations, differences in graph connectivity, or both **add more references** . The central research question is: To what extent are gender-related disparities in award link prediction associated with **entity  -> not sure if this is the full image something seems missing**, and **can these patterns be distinguished through controlled experiments -> NOT CLEAR ENOUNG i thin this means we will analyse the kg not do something with the models**?

## **Dataset:** 
We will use [Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m), a large knowledge graph constructed from a July 2019 Wikidata and Wikipedia snapshot, containing approximately 4.6 million entities and 20.6 million training triples. **rewrite** A problem we have found is there are a lot missing attributes such as gender and birthdate this was a problem because we would have a ground truth to compare our results so we added the missing information by quering the sparqle endpoint of [Qlever](https://qlever.dev/wikidata) **end rewrite** 

Because training on the complete dataset would be computationally demanding for this project, we will construct a reproducible, appropriately sized subsection focused on people and relevant relationships. The primary prediction task will use the Wikidata property P166 (award received); P21 (sex or gender) will identify groups for comparison, and graph degree will measure each entity’s connectivity in the training graph. P106 (occupation) may be included as contextual information if feasible. Any regional analysis using P27 (citizenship) will be optional and conducted only if group sizes support meaningful comparisons. The subset, filtering criteria, and train/validation/test split will be documented to support reproducibility. **-> we should rewrite and say we will make appropriate changes based on the way the project goes to make our life easier.**

## **Problem challenges and related work:** 
Existing research, collected in the GitHub repository’s research folder, examines social bias in knowledge-graph embeddings, data imbalance, degree-related bias in graph learning, and fairness-aware or counterfactual methods. This research motivates our question: to what extent are gender-related disparities in knowledge-graph award prediction associated with differences in entity connectivity, and how can these patterns be distinguished from other sources of bias **again we are kind of missing the gender attribute work we will do**? The main challenges are incomplete award records, uneven representation across demographic groups, and the difficulty of distinguishing correlation from causation. 

We will therefore audit a manageable dataset subset, establish a baseline model, and compare prediction performance and group-level disparities across degree groups. More complex counterfactual interventions will be considered only if time and computational resources permit.

## **Baselines and methodology:** 
We will first audit the dataset and report the representation of gender groups, award triples, and degree distributions. We will then implement a knowledge-graph embedding baseline, such as TransE or DistMult, and, if computationally feasible, a relational graph neural network (R-GCN) with a DistMult decoder. Predictive performance will be evaluated using mean reciprocal rank (MRR) and Hits@1/Hits@10. Group-level differences will be examined using clearly defined ranking-based evaluation procedures and degree-stratified comparisons. If resources permit, a probe classifier will test whether gender can be predicted from frozen entity embeddings, and controlled ablations or a limited model-depth comparison will explore the contribution of structural information. A path-specific counterfactual fairness intervention will be treated as a stretch goal rather than a requirement for project completion. We will use fixed data splits, documented preprocessing, and a small number of random seeds where feasible.

## **Feasibility:** 
The project is feasible within a 12–14-week period for a two-person team given we have already started working on some tasks. The complete Wikidata5M dataset, extensive hyperparameter searches, multiple complex architectures, and a custom counterfactual fairness objective would be too ambitious as simultaneous requirements, particularly when working with Google Colab and limited computing resources. 

The minimum viable project will therefore focus on a manageable dataset subsection, one primary baseline, a careful bias and degree analysis, and a reproducible evaluation. More complex models, additional demographic comparisons, and counterfactual interventions will be pursued only if the core experiments are completed successfully.

## **Declaration on Generative AI:** 
During the preparation of this work, the authors used Claude (Anthropic) and ChatGPT (OpenAI) to assist with drafting, editing, and organizing research materials, and Grammarly to support language and grammar revision. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the description’s content.

---

# Revised version (draft)

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

### Stretch goals

*Attempted only once stages 1–4 are complete:*

- A relational graph neural network (**R-GCN** with a DistMult decoder; Schlichtkrull et al., 2018) and a comparison across depths of message passing.
- Ablations that remove likely proxy relations.
- A path-specific counterfactual fairness intervention (Chiappa, 2019).

### Reproducibility

All experiments use fixed, versioned data splits, documented preprocessing, a locked software environment (uv) and fixed random seeds, so that every reported number can be traced to a code version, a data version and a seed.

## 5. Feasibility and Scope

The project is planned for **12–14 weeks with a two-person team**; the computing environment, data download and attribute retrieval are already complete.

Running the full Wikidata5M graph, extensive hyperparameter searches, several architectures and a custom counterfactual objective at once would be too ambitious, particularly with a single consumer GPU (RTX 5060 Ti, 16 GB) and Google Colab.

| Minimum viable project (stages 1–4) | Only if the core is finished |
|:---|:---|
| A documented subset | The probe |
| A data audit | Graph neural network models |
| One baseline model | Additional demographic comparisons |
| A careful group and connectivity analysis with reproducible evaluation | Counterfactual methods |

## 6. References

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
- Subramonian, A., Kang, J., & Sun, Y. (2024). Theoretical and empirical insights into the origins of degree bias in graph neural networks. [arXiv:2404.03139](https://arxiv.org/abs/2404.03139)
- Tang, X., et al. (2020). Investigating and mitigating degree-related biases in graph convolutional networks. *CIKM*. [doi:10.1145/3340531.3411872](https://doi.org/10.1145/3340531.3411872)
- Wang, X., et al. (2021). KEPLER: A unified model for knowledge embedding and pre-trained language representation. *TACL*. [arXiv:1911.06136](https://arxiv.org/abs/1911.06136)

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
