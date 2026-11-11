# Disentangling Bias in Knowledge-Graph Link Prediction

## 1. Problem and Research Question

Link prediction models infer missing facts in a knowledge graph and can reproduce or amplify the biases in their data (Fisher et al., 2020; Kraft & Usbeck, 2022). A gender gap in predictions can arise along two paths:

- **Direct path:** embeddings absorb gender, directly or through proxies such as spouse, occupation or name (Fisher et al., 2020; Du et al., 2022).
- **Connectivity path:** models predict worse for low-degree entities (Tang et al., 2020; Shomer et al., 2023; Subramonian et al., 2024), so a less connected group gets worse predictions.

> **Research question:** To what extent are gender disparities in award link prediction (`P166`) associated with (a) gender information encoded in entity representations, and (b) differences in graph connectivity between gender groups?

We study this at two levels: the **data** (how unequal the graph itself is, before any model) and the **model** (whether its accuracy differs by gender, and how much of the gap remains among similarly connected people). We do not claim a full causal decomposition.

## 2. Dataset

[Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m) (Wang et al., 2021): **4,594,485 entities, 822 relations, 20,614,279 training triples** (2019 snapshot; statistics verified).

**Missing attributes.** Wikidata5M keeps only facts between entities with an English Wikipedia page and no dates, so only 366 of its 1,509,635 humans have a gender (`P21`) and none has a birth date (`P569`). We retrieved both from Wikidata via [QLever](https://qlever.dev/wikidata) (dump of 10 August 2026) and joined them on Wikidata IDs: gender is now available for **99.2%** and birth date for **92.9%** of humans. 

> **Note:** The Gender (`P21`) attribute is stored in a separate table and is **not** added to the graph the model is trained on; it is used only to group people for evaluation, for the audit, and for probing. The Date of birth (`P569`) attribute **is** used in the training.

**Subset.** Training on all of Wikidata5M (4.6M entities) is expensive, so we will work with a **subgroup of the graph** that is meant to be *representative* of Wikidata5M: it should keep the same kinds of entities (people, awards, occupations, places, and so on), the same mix of relation types, and a similar degree distribution as the full graph, so that results on the subset say something about the whole. We will check this by comparing the subset with the full graph on entity types, relation frequencies and degree distribution, and report any differences. The sampling procedure, random seed and these comparison statistics will be documented and versioned, because the subset must be rebuilt identically. Representativeness is of the *recorded* graph; it does not make the subset a complete or unbiased picture of the real world.

**Scope.** Within this subset we will focus on people: `P166` is the target, `P21` defines the groups, node degree measures connectivity, and occupation (`P106`) and birth decade may serve as controls.
## 3. Challenges

- **Biased labels:** Wikidata5M retains a smaller share of women's Wikidata awards (23.1% vs. 25.6% for men) and occupations (33.0% vs. 39.1%); this mixes dataset filtering with later Wikidata edits.
- **Uneven groups:** 82.7% male, 17.2% female, about 0.1% other; small groups are reported descriptively.
- **Correlation, not causation:** degree, occupation and birth period are related.
- **Two time points:** the graph is from 2019, the attributes from 2026.

## 4. Methodology

**Core study (diagnosis), mandatory:**

1. **Data audit:** group sizes, award rates, degree distributions and missing information per gender group.
2. **Baseline model:** TransE or DistMult (PyKEEN; Bordes et al., 2013; Yang et al., 2015), trained **without** `P21` facts, evaluated with filtered MRR and Hits@1/3/10.
3. **Group disparity:** the same metrics per gender group, with bootstrap confidence intervals and several seeds.
4. **Connectivity analysis:** comparisons within degree bins and a regression of performance on gender and degree, the ratio of model to data disparity shows whether the model amplifies the data's bias.
5. **Encoded gender (if time allows):** a probe classifier predicts gender from the frozen embeddings. If it succeeds although `P21` was removed, the model learned gender from proxies.

**Debiasing (optional, Version A or B).** Both replace the baseline with an R-GCN (Schlichtkrull et al., 2018), test every component first on a synthetic graph with planted biases of known strength, and compare against no debiasing, an adversarial filter (Bose & Hamilton, 2019), FairDrop (Spinelli et al., 2021) and data rebalancing.

- **Version A, counterfactual objective:** adds a fairness term to the training loss (Chiappa, 2019) asking whether a prediction would change if only gender were different. A **direct term** flips the gender edge and penalises changes in award scores; a **popularity term** compares each person with matched people of another gender and similar occupation, birth decade and region.
- **Version B, two modules + the Version A objective:** **Module A** (degree-robust aggregation: capped and averaged messages, completion for low-degree people) targets the connectivity path; **Module B** (a two-channel layer with an adversary that keeps gender out of the channel used for prediction) targets the direct path.

Success is judged by whether the gender gap shrinks, on which path, whether gender is still encoded, and at what cost in accuracy.

## 5. Model Formulation

Full metric definitions are in [docs/definitions.md](docs/definitions.md); this section gives the model's input and output space and the learning objective.

### 5.1 Input and output space

- $\mathcal{E}$ is the set of entities, $\mathcal{R}$ the set of relations, and a fact is a triple $(h,r,t)\in\mathcal{E}\times\mathcal{R}\times\mathcal{E}$ (head, relation, tail).
- The triples are split into $\mathcal{T}_{train}$, $\mathcal{T}_{val}$ and $\mathcal{T}_{test}$. All `P21` triples are removed from $\mathcal{T}_{train}$.
- $\mathcal{A}\subset\mathcal{E}$ is the set of candidate award entities.
- Gender is a separate map $g:\mathcal{P}\to\{M,F,\dots\}$ from the audit table. It is never an input to the model.

**Parameters.** Every entity $i$ and relation $j$ has a learned vector $\mathbf{e}_i,\mathbf{r}_j\in\mathbb{R}^d$ (an *embedding*).

**Scoring function** $f:\mathcal{E}\times\mathcal{R}\times\mathcal{E}\to\mathbb{R}$ (higher means more plausible):

$$\text{TransE: } f(h,r,t)=-\lVert \mathbf{h}+\mathbf{r}-\mathbf{t}\rVert_p \qquad\qquad \text{DistMult: } f(h,r,t)=\sum_{i=1}^{d} h_i\,r_i\,t_i$$

**Query.** The input is a pair $(p,\mathrm{P166})$ and the output is one score per candidate award:

$$\mathbf{s}\in\mathbb{R}^{|\mathcal{A}|},\qquad s_a=f(p,\mathrm{P166},a)$$

**Filtered rank** of the true award $a^*$, where $\mathcal{F}_p$ is the set of other awards already known to be true for $p$:

$$\operatorname{rank}(a^*)=1+\bigl|\{a\in\mathcal{A}\setminus\mathcal{F}_p:\ s_a>s_{a^*}\}\bigr|$$

MRR, Hits@k and the group gaps are computed from this rank (see definitions.md, Section 6).

### 5.2 Training objective

Training uses **negative sampling**: for each true triple we create fake ones by replacing the head or tail with a random entity. Call that set $\mathcal{N}(h,r,t)$. The model should score true triples higher than fake ones. Two losses are considered; the final choice will be made on validation MRR over all people (not on gender gaps) and the other will be reported as a robustness check.

**Margin ranking loss (the original TransE loss; Bordes et al., 2013).** With a margin $\gamma>0$:

$$\mathcal{L}_{margin}=\sum_{(h,r,t)\in\mathcal{T}_{train}}\ \sum_{(h',r,t')\in\mathcal{N}(h,r,t)}\max\bigl(0,\ \gamma-f(h,r,t)+f(h',r,t')\bigr)$$

In simple words: a true triple must score at least $\gamma$ higher than a fake one. If it already does, the loss is $0$ and the model learns nothing from that pair. If not, the loss equals how far short it falls. It matches rank-based evaluation (true above fake), but it treats all fakes equally, so easy fakes stop teaching the model quickly, and $\gamma$ must be tuned.

**Cross-entropy (logistic) loss.** Here $\sigma(x)=1/(1+e^{-x})$ squeezes a score into a number between $0$ and $1$, read as "the probability that this triple is true". With an offset $\gamma$ (needed for TransE, whose scores are never positive):

$$\mathcal{L}_{CE}=-\sum_{(h,r,t)\in\mathcal{T}_{train}}\Bigl[\log\sigma\bigl(f(h,r,t)+\gamma\bigr)+\sum_{(h',r,t')\in\mathcal{N}(h,r,t)}\log\Bigl(1-\sigma\bigl(f(h',r,t')+\gamma\bigr)\Bigr)\Bigr]$$

In simple words: every triple is a yes/no question. For a true triple the model is rewarded for a probability near $1$ and punished harshly for being confidently wrong; for a fake triple it is rewarded for a probability near $0$. Unlike the margin loss, it never "switches off": every example keeps teaching a little, and there is no margin to set. Its scores also behave more like probabilities, which helps when a fixed score threshold $\tau$ is needed for the demographic-parity gap (definitions.md, Section 6.3), though sampled negatives mean the probabilities would still need calibrating. The form follows the logistic loss used with RotatE (Sun et al., 2019); this should be checked against the PyKEEN implementation.

The exact loss, margin and number of negatives are set in the PyKEEN config and must match this section.

### 5.3 Analysis models

**Gender probe.** On frozen embeddings, a linear classifier predicts gender; the probe is compared against a majority-class baseline.

$$\hat g(p)=\operatorname{softmax}(W\mathbf{e}_p+b),\qquad \mathcal{L}_{probe}=-\sum_{p}\log \hat g(p)_{g(p)}$$

**Connectivity analysis (proposed plan).** These analyses run after training, on a table with one row per test query containing the person, gender, degree and rank. They describe associations; they do not show that degree causes a gap (definitions.md, Section 7).

Notation: $F_p=\mathbf{1}[g(p)=F]$ (so the comparison group is male), $x_p=\log\bigl(1+\deg(p)\bigr)$, $\operatorname{RR}_q=1/\operatorname{rank}(q)$, and $\mathbf{z}_p$ are optional controls (occupation, birth decade).

*Main analysis: regression with an interaction and splines.* Let $B_1,\dots,B_K$ be spline basis functions of $x_p$, which let the effect of degree bend instead of being a straight line:

$$\operatorname{RR}_q=\beta_0+\beta_1F_p+\sum_{k=1}^{K}\gamma_kB_k(x_p)+\sum_{k=1}^{K}\delta_kF_pB_k(x_p)+\boldsymbol{\theta}^\top\mathbf{z}_p+\varepsilon_q$$

The $\delta_k$ terms let degree affect women and men differently. The gender gap at degree level $x$ is then:

$$\Delta(x)=\beta_1+\sum_{k=1}^{K}\delta_kB_k(x)$$

If $\Delta(x)\approx0$ at every $x$, the gap is explained by connectivity; if it stays away from $0$ at equal degree, the gap is not explained by it.

*Clustering by person.* One person can have several test queries, so rows are not independent. Let $\mathbf{X}_p$ and $\hat{\boldsymbol{\varepsilon}}_p$ be the design matrix and residuals of all queries of person $p$. The cluster-robust variance of the estimates $\hat{\boldsymbol{\beta}}$ is:

$$\widehat{V}(\hat{\boldsymbol{\beta}})=(\mathbf{X}^\top\mathbf{X})^{-1}\Bigl(\sum_{p}\mathbf{X}_p^\top\hat{\boldsymbol{\varepsilon}}_p\hat{\boldsymbol{\varepsilon}}_p^\top\mathbf{X}_p\Bigr)(\mathbf{X}^\top\mathbf{X})^{-1}$$

Bootstrapping over people is an equivalent option (definitions.md, Section 9).

*Check 1: degree bins.* Split people into degree bins $b=1,\dots,B$. Within each bin, compare the groups and combine with weights $w_b$ (for example the share of women in bin $b$):

$$\Delta_b=\operatorname{MRR}_{F,b}-\operatorname{MRR}_{M,b},\qquad \Delta_{bins}=\sum_{b=1}^{B}w_b\,\Delta_b$$

*Check 2: matched comparison.* For each woman $p$, let $\mathcal{M}(p)$ be the set of men with the same degree bin, occupation and birth decade. With $\overline{\operatorname{RR}}_p$ the average reciprocal rank over a person's queries:

$$\Delta_{match}=\frac{1}{|P_F|}\sum_{p\in P_F}\Bigl[\overline{\operatorname{RR}}_p-\frac{1}{|\mathcal{M}(p)|}\sum_{p'\in\mathcal{M}(p)}\overline{\operatorname{RR}}_{p'}\Bigr]$$

Women with $\mathcal{M}(p)=\emptyset$ are dropped and counted.

*Optional headline number: gap decomposition (Oaxaca-Blinder style).* Fit the regression $\operatorname{RR}=\mathbf{x}^\top\boldsymbol{\beta}_g+\varepsilon$ separately for each group $g\in\{M,F\}$, where $\mathbf{x}$ contains an intercept and the degree terms, and let $\bar{\mathbf{x}}_g$ be the group's average of $\mathbf{x}$. Then:

$$\underbrace{\overline{\operatorname{RR}}_M-\overline{\operatorname{RR}}_F}_{\text{total gap}}=\underbrace{(\bar{\mathbf{x}}_M-\bar{\mathbf{x}}_F)^\top\boldsymbol{\beta}_M}_{\text{explained by connectivity}}+\underbrace{\bar{\mathbf{x}}_F^\top(\boldsymbol{\beta}_M-\boldsymbol{\beta}_F)}_{\text{unexplained}}$$

$$\text{Share explained}=\frac{(\bar{\mathbf{x}}_M-\bar{\mathbf{x}}_F)^\top\boldsymbol{\beta}_M}{\overline{\operatorname{RR}}_M-\overline{\operatorname{RR}}_F}$$

This is descriptive, not causal. The split depends on which group is the reference ($\boldsymbol{\beta}_M$ here), the share is unstable when the total gap is near $0$, and its uncertainty should come from a bootstrap over people.

### 5.4 Debiasing objective (Version A, schematic)

$$\mathcal{L}=\mathcal{L}_{KG}+\lambda_1\,\mathcal{L}_{direct}+\lambda_2\,\mathcal{L}_{pop}$$

$\mathcal{L}_{direct}$ penalises the change in award scores when only gender is flipped, and $\mathcal{L}_{pop}$ penalises score differences between a person and matched people of another gender. The exact terms are still to be specified.

## 6. References

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
- Sun, Z., et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR*. [arXiv:1902.10197](https://arxiv.org/abs/1902.10197)
- Tang, X., et al. (2020). Investigating and mitigating degree-related biases in graph convolutional networks. *CIKM*. [doi:10.1145/3340531.3411872](https://doi.org/10.1145/3340531.3411872)
- Wang, X., et al. (2021). KEPLER: A unified model for knowledge embedding and pre-trained language representation. *TACL*. [arXiv:1911.06136](https://arxiv.org/abs/1911.06136)
- Yang, B., et al. (2015). Embedding entities and relations for learning and inference in knowledge bases. *ICLR*. [arXiv:1412.6575](https://arxiv.org/abs/1412.6575)

## Declaration on Generative AI

The authors used ChatGPT (OpenAI) and Claude (Anthropic, including Claude Code) for project guides and task planning, data checks, drafting and revising text, and answering questions and explaining the research papers. The research question, methods and all decisions were made by the authors, who found all cited papers through their own online searches, changed and revised all AI-generated text and take full responsibility for the content.
