# Disentangling Bias in Knowledge-Graph Link Prediction

## 1. Problem Statement

Knowledge-graph link prediction models learn to infer missing relationships from existing facts. However, their predictions may reflect social biases and unequal representation in the underlying data.

This project investigates whether gender-related disparities in predicting awards received by people are associated with gender information encoded in entity representations, differences in graph connectivity, or both.

> **Central Research Question**
>
> To what extent are gender-related disparities in award link prediction associated with entity degree, and can these patterns be distinguished through controlled experiments?

## 2. Dataset

We will use **Wikidata5M**, a large knowledge graph constructed from a July 2019 Wikidata and Wikipedia snapshot. It contains approximately 4.6 million entities and 20.6 million training triples.

Because training on the complete dataset would be computationally demanding, we will construct a reproducible, appropriately sized subsection focused on people and relevant relationships.

### Relevant Wikidata properties

| Property | Description | Role in the project |
|:---|:---|:---|
| `P166` | Award received | Primary link-prediction target |
| `P21` | Sex or gender | Group identification for comparisons |
| `P106` | Occupation | Optional contextual information |
| `P27` | Country of citizenship | Optional regional analysis |

Graph degree will measure each entity's connectivity in the training graph.

The subset, filtering criteria, preprocessing procedure, and train/validation/test splits will be documented to support reproducibility. Regional analysis using `P27` will be conducted only if group sizes support meaningful comparisons.

## 3. Problem Challenges and Related Work

Existing research collected in the repository's `research/` directory examines:

- Social bias in knowledge-graph embeddings.
- Data imbalance and unequal representation.
- Degree-related bias in graph learning.
- Fairness-aware and counterfactual methods.

This work motivates our investigation into the relationship between gender-related disparities, graph connectivity, and award prediction.

### Main challenges

- **Incomplete award records:** Missing relationships may affect the model's ability to learn and evaluate award predictions.
- **Uneven representation:** Gender groups may differ substantially in size and in their representation in the graph.
- **Structural confounding:** Differences in connectivity may contribute to observed disparities, making it difficult to isolate the role of gender information.
- **Interpretation:** Observed associations do not, by themselves, establish causation.

We will address these challenges by auditing a manageable dataset subset, establishing a baseline model, and comparing prediction performance and group-level disparities across degree groups.

## 4. Baselines and Methodology

The project will follow a staged experimental design, prioritizing a reliable baseline and careful evaluation before introducing more complex methods.

### Stage 1 — Dataset audit

- Measure representation across recorded gender groups.
- Examine the availability and distribution of award triples.
- Calculate entity-degree distributions.
- Define and document dataset filtering and split procedures.

### Stage 2 — Baseline link prediction

Implement a knowledge-graph embedding model, such as **TransE** or **DistMult**, to predict missing `P166` award relationships.

If computationally feasible, we will additionally investigate a relational graph convolutional network (**R-GCN**) with a DistMult decoder.

### Stage 3 — Evaluation and bias analysis

Predictive performance will be evaluated using:

| Metric | Purpose |
|:---|:---|
| Mean Reciprocal Rank (MRR) | Measures the quality of the ranking assigned to correct answers |
| Hits@1 | Measures how often the correct answer ranks first |
| Hits@10 | Measures how often the correct answer appears among the top ten |

Group-level differences will be examined using clearly defined ranking-based evaluation procedures and degree-stratified comparisons.

The analysis will distinguish observed performance disparities from conclusions about their causes.

### Stage 4 — Optional extensions

If time and computational resources permit, we will explore:

- **Embedding probes:** Train a probe classifier to test whether gender can be predicted from frozen entity embeddings.
- **Controlled ablations:** Investigate the contribution of structural information through selected model or feature comparisons.
- **Model-depth comparison:** Compare a limited number of model configurations.
- **Counterfactual fairness:** Explore a path-specific counterfactual fairness intervention.

These extensions are stretch goals, not requirements for project completion.

### Reproducibility

Experiments will use fixed data splits, documented preprocessing, and a small number of random seeds where feasible. Dataset selection, evaluation procedures, and experimental settings will be recorded to support repeatability.

## 5. Feasibility and Scope

The project is designed to be completed over **12–14 weeks by a two-person team**, using Google Colab and limited computing resources.

The complete Wikidata5M dataset, extensive hyperparameter searches, multiple complex architectures, and a custom counterfactual fairness objective would be too ambitious as simultaneous requirements.

Accordingly, the project is divided into a minimum viable study and optional extensions.

| Core requirements | Stretch goals |
|:---|:---|
| Audit a manageable dataset subset | R-GCN with DistMult decoder |
| Train one primary baseline | Embedding probes |
| Evaluate link-prediction performance | Additional controlled ablations |
| Analyze gender-related disparities by degree | Additional demographic comparisons |
| Document preprocessing and evaluation | Path-specific counterfactual intervention |

**Minimum viable outcome:** A reproducible baseline experiment, accompanied by a careful analysis of gender-related prediction disparities and their association with entity degree.

More complex models, additional demographic comparisons, and counterfactual interventions will be pursued only if the core experiments are completed successfully.

## 6. Declaration on Generative AI

During the preparation of this work, the authors used Claude (Anthropic) and ChatGPT (OpenAI) to assist with drafting, editing, and organizing research materials, and Grammarly to support language and grammar revision. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the publication's content.
