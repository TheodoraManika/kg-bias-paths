# Disentangling Bias in Knowledge-Graph Link Prediction

## 1. Problem and Research Question

Link prediction models infer missing facts in a knowledge graph and can reproduce or amplify the biases in their data (Fisher et al., 2020; Kraft & Usbeck, 2022). A gender gap in predictions can arise along two paths:

- **Direct path:** embeddings absorb gender, directly or through proxies such as spouse, occupation or name (Fisher et al., 2020; Du et al., 2022).
- **Connectivity path:** models predict worse for low-degree entities (Tang et al., 2020; Shomer et al., 2023; Subramonian et al., 2024), so a less connected group gets worse predictions.

> **Research question:** To what extent are gender disparities in award link prediction (`P166`) associated with (a) gender information encoded in entity representations, and (b) differences in graph connectivity between gender groups?

We study this at two levels: the **data** (how unequal the graph itself is, before any model) and the **model** (whether its accuracy differs by gender, and how much of the gap remains among similarly connected people). We do not claim a full causal decomposition.

## 2. Dataset

[Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m) (Wang et al., 2021): **4,594,485 entities, 822 relations, 20,614,279 training triples** (2019 snapshot; statistics verified).

**Missing attributes.** Wikidata5M keeps only facts between entities with an English Wikipedia page and no dates, so only 366 of its 1,509,635 humans have a gender (`P21`) and none has a birth date (`P569`). We retrieved both from Wikidata via [QLever](https://qlever.dev/wikidata) (dump of 10 August 2026) and joined them on Wikidata IDs: gender is now available for **99.2%** and birth date for **92.9%** of humans. These attributes are kept in a separate table and are **not** part of the training graph; they are used only to group people for evaluation.

**Scope.** We will build a documented, versioned subset focused on people: `P166` is the target, `P21` defines the groups, node degree measures connectivity, and occupation (`P106`) and birth decade may serve as controls. We expect to adjust the subset and split as the project develops and will document each change.

## 3. Challenges

- **Biased labels:** Wikidata5M retains a smaller share of women's Wikidata awards (23.1% vs. 25.6% for men) and occupations (33.0% vs. 39.1%); this mixes dataset filtering with later Wikidata edits.
- **Uneven groups:** 82.7% male, 17.2% female, about 0.1% other; small groups are reported descriptively.
- **Correlation, not causation:** degree, occupation, birth period and gender are related.
- **Two time points:** the graph is from 2019, the attributes from 2026.

## 4. Methodology

**Core study (diagnosis), mandatory:**

1. **Data audit:** group sizes, award rates, degree distributions and missing information per gender group.
2. **Baseline model:** TransE or DistMult (PyKEEN; Bordes et al., 2013; Yang et al., 2015), trained **without** `P21` facts; evaluated with filtered MRR and Hits@1/3/10.
3. **Group disparity:** the same metrics per gender group, with bootstrap confidence intervals and several seeds.
4. **Connectivity analysis:** comparisons within degree bins and a regression of performance on gender and degree; the ratio of model to data disparity shows whether the model amplifies the data's bias.
5. **Encoded gender (if time allows):** a probe classifier predicts gender from the frozen embeddings; if it succeeds although `P21` was removed, the model learned gender from proxies.

**Debiasing (optional, Version A or B).** Both replace the baseline with an R-GCN (Schlichtkrull et al., 2018), test every component first on a synthetic graph with planted biases of known strength, and compare against no debiasing, an adversarial filter (Bose & Hamilton, 2019), FairDrop (Spinelli et al., 2021) and data rebalancing.

- **Version A, counterfactual objective:** adds a fairness term to the training loss (Chiappa, 2019) asking whether a prediction would change if only gender were different. A **direct term** flips the gender edge and penalises changes in award scores; a **popularity term** compares each person with matched people of another gender and similar occupation, birth decade and region.
- **Version B, two modules + the Version A objective:** **Module A** (degree-robust aggregation: capped and averaged messages, completion for low-degree people) targets the connectivity path; **Module B** (a two-channel layer with an adversary that keeps gender out of the channel used for prediction) targets the direct path.

Success is judged by whether the gender gap shrinks, on which path, whether gender is still encoded, and at what cost in accuracy.

## 5. Scope Options and Questions

The first six weeks (the core study) are shared; each option falls back to the previous one.

| | Core study | Version A | Version B |
|:---|:---|:---|:---|
| **Adds** | Diagnosis (stages 1–4) | + counterfactual objective | + two modules |
| **Duration** | ~6 weeks | ~13 weeks | ~16 weeks |
| **Fallback** | – | Core study (week 6) | Version A (week 11) |

Resources: a two-person team, one RTX 5060 Ti (16 GB) and Google Colab; the environment, data and attributes are already in place.

**Questions for the professor:**

1. Which scope is appropriate for a two-person team?
2. Does a new training objective (Version A) count as sufficient novelty, or are the modules of Version B expected?
3. What is the submission deadline?

## References

- Bordes, A., et al. (2013). Translating embeddings for modeling multi-relational data. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- Bose, A. J., & Hamilton, W. L. (2019). Compositional fairness constraints for graph embeddings. *ICML*. [arXiv:1905.10674](https://arxiv.org/abs/1905.10674)
- Chiappa, S. (2019). Path-specific counterfactual fairness. *AAAI*. [arXiv:1802.08139](https://arxiv.org/abs/1802.08139)
- Du, Y., et al. (2022). Understanding gender bias in knowledge base embeddings. *ACL*. [ACL Anthology](https://aclanthology.org/2022.acl-long.98/)
- Fisher, J., et al. (2020). Measuring social bias in knowledge graph embeddings. [arXiv:1912.02761](https://arxiv.org/abs/1912.02761)
- Kraft, A., & Usbeck, R. (2022). The lifecycle of "facts": A survey of social bias in knowledge graphs. *AACL-IJCNLP*. [arXiv:2210.03353](https://arxiv.org/abs/2210.03353)
- Schlichtkrull, M., et al. (2018). Modeling relational data with graph convolutional networks. *ESWC*. [arXiv:1703.06103](https://arxiv.org/abs/1703.06103)
- Shomer, H., et al. (2023). Toward degree bias in embedding-based knowledge graph completion. *WWW*. [arXiv:2302.05044](https://arxiv.org/abs/2302.05044)
- Spinelli, I., et al. (2021). FairDrop: Biased edge dropout for enhancing fairness in graph representation learning. *IEEE TAI*. [arXiv:2104.14210](https://arxiv.org/abs/2104.14210)
- Subramonian, A., Kang, J., & Sun, Y. (2024). Theoretical and empirical insights into the origins of degree bias in graph neural networks. [arXiv:2404.03139](https://arxiv.org/abs/2404.03139)
- Tang, X., et al. (2020). Investigating and mitigating degree-related biases in graph convolutional networks. *CIKM*. [doi:10.1145/3340531.3411872](https://doi.org/10.1145/3340531.3411872)
- Wang, X., et al. (2021). KEPLER: A unified model for knowledge embedding and pre-trained language representation. *TACL*. [arXiv:1911.06136](https://arxiv.org/abs/1911.06136)
- Yang, B., et al. (2015). Embedding entities and relations for learning and inference in knowledge bases. *ICLR*. [arXiv:1412.6575](https://arxiv.org/abs/1412.6575)

## Declaration on Generative AI

The authors used ChatGPT (OpenAI) and Claude (Anthropic, including Claude Code) for project guides and task planning, writing and debugging code, data checks, drafting and revising text, checking reference links, and answering questions and explaining the research papers. The research question, methods and all decisions were made by the authors, who found all cited papers through their own online searches, changed and revised all AI-generated text, reviewed and tested all code, checked the reported numbers, and take full responsibility for the content.
