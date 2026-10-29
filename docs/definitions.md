# Definitions: Tasks, Sensitive Groups, and Gender-Relevant Relations

**Project:** Disentangling Bias in Knowledge-Graph Link Prediction  
**Dataset:** Wikidata5M, transductive setting  
**Status:** Working specification; update when the experimental design changes.


## 1. Purpose and Scope

This document defines the prediction tasks, gender-group assignment rules, gender-relevant facts, and fairness metrics used in the project. It is the reference for implementing data preprocessing and evaluation.

The project investigates whether gender disparities in award link prediction are associated with differences in graph connectivity and with gender information encoded in learned entity representations. It compares data-level disparities with model-level disparities without claiming a complete causal decomposition.

Gender attributes retrieved from Wikidata are stored separately from the training graph. They are used for group assignment, data auditing, evaluation, and—if feasible—embedding probes. They must not be added to the model's training graph as input facts.


## 2. Prediction Tasks

A knowledge-graph fact is represented as a triple ((h,r,t)), where (h) is the head entity, (r) is the relation, and (t) is the tail entity.

The project's prediction direction is person-to-tail prediction, represented as:

$$
(\text{person}, r, ?)
$$

### 2.1 Primary Task: Award Prediction

Given a person $p$, rank candidate award entities $a$ by the model's score for:

$$
(p,\mathrm{P166},a)
$$

- **Relation:** `P166` — award received.
- **Role:** Primary prediction target.
- **Evaluation:** Filtered mean reciprocal rank (MRR) and Hits@1, Hits@3, and Hits@10.
- **Analysis:** Compare performance across the defined gender groups, then investigate whether disparities vary with graph degree.

Only held-out target triples may be used as test positives. The evaluation protocol must document candidate entities, filtering rules, data splits, and treatment of people with multiple recorded awards.

### 2.2 Secondary Task: Occupation Prediction

The secondary task predicts occupations associated with a person:

$$
(p,\mathrm{P106},o)
$$

- **Relation:** `P106` — occupation.
- **Role:** Secondary prediction target or supporting data audit.
- **Priority:** Optional, subject to time and data availability; it must not displace the primary P166 analysis.

Occupation prediction can provide additional context about representation and gender-associated patterns. Explicitly gender-specific occupation values must be reviewed under the whitelist policy in Section 4.

### 2.3 Related Attributes and Graph Measures

| Property or measure | Meaning | Project use |
|---|---|---|
| `P21` | Sex or gender | Sensitive grouping attribute; excluded from model input |
| `P569` | Date of birth | Optional birth-decade control |
| `P27` | Country of citizenship | Optional regional analysis |
| Degree | Number of incident graph edges, under the documented definition | Connectivity analysis |

`P21` is not a prediction target in the baseline experiments. `P569` and `P27` are optional analytical controls, not required additions to the training graph.

### 2.4 Leakage and Graph Consistency

- Remove `P21` triples from the model-input graph.
- Keep externally retrieved attributes in a separate audit table.
- Ensure held-out target triples do not appear in training.
- Calculate primary degree from the training graph after the documented removals, including `P21` and held-out target edges.
- Apply the same candidate-generation, filtering, and evaluation rules across groups.
- Version and document the subset, splits, and filtering rules.


## 3. Gender groups and Assignment Rules

### 3.1 Wikidata `P21` Values

The following Wikidata item identifiers are used to recognize relevant recorded `P21` values.

| Recorded value | QID | Primary treatment |
|---|---|---|
| Male | Q6581097 | Primary binary comparison group |
| Female | Q6581072 | Primary binary comparison group |
| Non-binary | Q48270 | Separate descriptive group |
| Trans woman | Q1052281 | Separate descriptive group |
| Trans man | Q2449503 | Separate descriptive group |
| Intersex | Q1097630 | Separate recorded-value category |

These categories reflect recorded Wikidata values; they are not an exhaustive taxonomy of gender identities or sex-related classifications. In particular, intersex must not be treated as interchangeable with non-binary or automatically interpreted as a gender identity.

Wikidata records may be incomplete, change over time, or contain multiple values. A missing `P21` value means **unknown or unrecorded for this analysis**, not that a person's gender is known.

Reference: [Wikidata property P21](https://www.wikidata.org/wiki/Property:P21).

### 3.2 Primary Binary Comparison

The main comparison is between people recorded with the exact `P21` values:

- **Male:** Q6581097
- **Female:** Q6581072

For the primary analysis, use a conservative, mutually exclusive assignment rule:

1. Assign a person to the male group if their only recognized value among the project's tracked categories is Q6581097.
2. Assign a person to the female group if their only recognized value among the project's tracked categories is Q6581072.
3. If a person has multiple distinct tracked values, mark them as **multi-valued** and exclude them from the mutually exclusive primary comparison.
4. Do not automatically merge trans women into the female group or trans men into the male group for the primary QID-based comparison. Any broader grouping must be a separately documented sensitivity analysis.
5. Keep other, missing, and unrecognized values identifiable rather than silently assigning them to either primary group.

This rule is deliberately conservative. Report the number of people included in each primary group, excluded due to multiple values, and missing or carrying other values.

**ADELLE NOTE: do we want to be conservative with this approach? i have complicated opinions about treating this as sex versus gender. because they are different circumstances, i suppose it is needed. i am open to merging trans women and trans men with the main groups though, but keeping a separate identifier to signal this distinction.**

### 3.3 Multi-label Sensitivity Analysis

As a sensitivity analysis, a person may belong to every group explicitly supported by their recorded values. In this analysis, groups can overlap; group counts therefore do not necessarily sum to the total population.

For every analysis, record:

- the number of people with at least one `P21` value;
- the number in each group;
- the number with multiple recognized values;
- the number with missing or unrecognized values;
- the exact assignment rule and attribute snapshot used.

Small groups should be reported descriptively with their sample sizes and uncertainty where appropriate. Do not silently discard them or present unstable estimates as definitive findings.


## 4. Gender-relevant Relations and Values

The project distinguishes between a **relation-level policy** and an **item-level whitelist**. `P166` and `P106` identify the type of fact being predicted; the whitelist identifies particular awards or occupations whose meaning or eligibility is explicitly gender-specific.

### 4.1 Relation-level Policy

| Relation | Policy |
|---|---|
| `P21` — sex or gender | Exclude from model input. Use only in the separate audit table for group assignment and approved analyses. |
| `P166` — award received | Primary prediction target. Identify explicitly gender-specific awards and report them separately from general awards. |
| `P106` — occupation | Secondary target or audit. Identify explicitly gender-specific occupation values if this task is evaluated. |
| `P569` — date of birth | Optional control, such as birth decade, when data quality permits. |
| `P27` — country of citizenship | Optional regional analysis or control. It is not a substitute for ethnicity or gender. |
| Other relations | Do not automatically assume gender neutrality. Apply the documented baseline relation policy and any separately specified ablations. |

`P21` is never a legitimate shortcut input for the primary task. `P166` and `P106` are target relations, not sensitive-attribute inputs.

Other graph relations can indirectly encode demographic information—for example, through family connections, occupations, or affiliations. Such potential proxy relations should be investigated through clearly documented ablations if time permits; they must not be silently removed or reclassified after results are seen.

### 4.2 Item-level Whitelist 

Maintain a versioned, manually reviewed whitelist for `P166` award entities and, if P106 is evaluated, `P106` occupation entities.

An item should be flagged as explicitly gender-specific when reliable evidence indicates that its eligibility, award category, or defined role is restricted to or explicitly designated for a gender. Examples might include an award category such as “Best Actress” or an occupation category explicitly defined as gender-specific.

**Do not classify an item from its label alone.** Words such as “woman,” “man,” “female,” or “male” can help identify candidates for review, but keyword matching is not sufficient evidence. Check the item's description and, where available, authoritative information about its eligibility or definition.

For each reviewed item, record:

- Relation (`P166` or `P106`).
- Item QID and label.
- Classification: `gender-specific`, `not gender-specific`, or `uncertain`.
- Rationale and supporting source or eligibility evidence.
- Review date and reviewer.
- Evaluation treatment and reason.

Items classified as uncertain should be reviewed rather than automatically excluded or included as gender-specific.

### 4.3 How the Whitelist Affects Evaluation

- **Primary award analysis:** Report general awards separately from explicitly gender-specific awards.
- **Gender-specific award analysis:** Interpret disparities in light of documented eligibility or category definitions; a difference may reflect the award's intended eligibility rather than model unfairness.
- **Occupation analysis:** If `P106` is evaluated, document how explicitly gender-specific occupation values are handled.
- **Reproducibility:** Version the whitelist alongside the data subset and splits. Record changes and reasons; do not silently revise classifications after inspecting model results.

The whitelist does not establish that all remaining facts are free of gender bias. It is a documented rule for identifying a specific class of gender-relevant facts.


## 5. Data-level Disparity Definitions

Data-level measures describe recorded representation and labels before model evaluation. They are based on the available knowledge graph or external Wikidata attributes, not necessarily on complete real-world records.

Let $P_g$ be the set of eligible people assigned to group $g$.

### 5.1 Recorded Award Rate

Define the proportion of eligible people in group $g$ with at least one recorded `P166` fact:

$$
A_g =
\frac{
|\{p\in P_g:\exists a,\ (p,\mathrm{P166},a)\text{ is recorded}\}|
}{
|P_g|
}
$$

This measures the **recorded award rate**. It must not be interpreted as the true rate at which people receive awards, because missing or unrecorded facts may differ across groups.

**ADELLE NOTE: the rate of people with at least one recorded award to summarize that information by group... it is a plausible descriptive statistic. we can delete this if it is unneeded.**

### 5.2 Recorded Attribute Coverage

For relation $r$, define:

$$
C_{g,r} =
\frac{
|\{p\in P_g:\exists t,\ (p,r,t)\text{ is recorded}\}|
}{
|P_g|
}
$$

Specify whether the source is Wikidata5M or the later external Wikidata snapshot. Coverage and model performance are different quantities.

**ADELLE NOTE: this is a general coverage formula to measure the proportion of people in each group who have a recorded value for a relation. we can delete this if it is unneeded.**

### 5.3 Retention of Current Wikidata Values

When comparing values present in the later Wikidata snapshot with values retained in Wikidata5M, define:

$$
R_{g,r} =
\frac{
\text{eligible current values for }r\text{ also present in Wikidata5M}
}{
\text{eligible current values for }r
}
$$

Document the entity matching and value matching rules. Because Wikidata5M reflects an earlier snapshot while external Wikidata reflects a later snapshot, this is a cross-snapshot retention measure, not a pure measure of the original dataset filtering process.

**ADELLE NOTE: we can delete this this if it is unneeded, just a formal retention ratio.**

## 6. Model-level Fairness Metrics

Let $Q_g$ be the set of held-out positive test queries associated with people in group $g$. For an award query $q=(p,\mathrm{P166},a)$ let $\mathrm{rank}(a\mid p,\mathrm{P166})$ denote the rank of the true award after applying the documented filtered-ranking protocol.

The main comparisons are male versus female, with other groups reported descriptively when feasible.

### 6.1 Group-specific Hits@k

For group g:

$$
H_g@k =
\frac{1}{|Q_g|}
\sum_{q\in Q_g}
\mathbf{1}[\mathrm{rank}(q)\leq k]
$$

Hits@k is the proportion of held-out positive queries for which the true answer ranks among the top \(k\) candidates.

### 6.2 Equal-opportunity gap@k

Operationalize equal opportunity as the difference in true-positive retrieval performance among held-out positive queries:

$$
EOgap@k(g,h)=|H_g@k-H_h@k|
$$

Report the absolute gap and the signed difference $H_g@k-H_h@k$, so both disparity magnitude and direction are visible.

This is a ranking-based operational measure, not a claim that the model has access to complete real-world award outcomes. Missing facts can affect the interpretation of the metric.

### 6.3 Demographic-parity gap@k

Demographic parity conventionally compares positive prediction rates between groups. In top-$k$ ranking, simply counting recommendations is not informative if every person receives exactly $k$ candidates. Therefore, a positive-decision rule must be specified.

For this project, if a defensible score threshold $\tau$ can be selected and fixed across groups, define:

$$
D_{g}@k,\tau =
\frac{1}{|P_g|}
\sum_{p\in P_g}
\mathbf{1}
\left[
\exists a\in\mathrm{TopK}(p):
s(p,a)\geq\tau
\right]
$$

where $s(p,a)$ is the model score and $\mathrm{TopK}(p)$ is the set of the $k$ highest-ranked candidate awards for person $p$.

The demographic-parity gap is:

$$
DPgap@k,\tau(g,h)=
|D_g@k,\tau-D_h@k,\tau|
$$

This is a **project-specific, threshold-dependent** operationalization: it measures the difference in the proportion of people receiving at least one above-threshold award prediction in their top $k$. Document how $\tau$ is selected and ensure the same threshold is used for both groups.

If no defensible common threshold can be justified, report DP gap as **not evaluated** rather than presenting Hits@k disparity as demographic parity.

### 6.4 Mean Reciprocal Rank and Disparity

For group $g$:

$$
MRR_g =
\frac{1}{|Q_g|}
\sum_{q\in Q_g}
\frac{1}{\mathrm{rank}(q)}
$$

The MRR gap is:

$$
MRRgap(g,h)=|MRR_g-MRR_h|
$$

MRR and Hits@k describe ranking performance. A difference between groups is evidence of a measured performance disparity under the chosen evaluation protocol, not by itself proof of a causal mechanism or a complete normative fairness judgment.

**ADELLE NOTE: this isn't explicitly mentioned in the task. we can move it somewhere else/delete it if we need to. could be a way to summarize the gender performance gap.**


## 7. Connectivity and Degree Analysis

Degree is used to investigate whether differences in graph connectivity are associated with gender-related performance disparities.

The primary degree measure must be defined explicitly, including whether it is total, in-degree, or out-degree; how directed edges are treated; and which relations are included.

Calculate primary degree from the training graph after the documented removals, including `P21` and held-out target edges. Report:

- Degree distributions by group.
- The chosen degree bins or matching procedure.
- Group performance within comparable degree ranges.
- Where feasible, regression analyses of per-query performance against group and degree, with optional occupation and birth-decade controls.

Report sample sizes and uncertainty. Degree-adjusted associations describe differences conditional on the selected controls; they do not, by themselves, establish that degree causally produces a disparity.


## 8. Amplification Ratio

The amplification ratio is intended to compare a model-level disparity with a corresponding data-level disparity:

$$
\text{Amplification ratio} =
\frac{\text{model disparity}}{\text{data disparity}}
$$

This ratio is meaningful only when the numerator and denominator measure the **same underlying outcome**, use compatible scales, and refer to comparable populations and group definitions.

For example, a model Hits@k gap must not be divided by a difference in award prevalence or Wikidata5M retention: these measure different outcomes. Before calculating the ratio, specify the exact model and data measures and justify their comparability.

If the data-level disparity is zero or close to zero, the ratio is undefined or unstable. If the measures are incompatible, report the amplification ratio as not applicable. Always report the underlying disparities separately rather than relying on the ratio alone.


## 9. Statistical Uncertainty and Reporting

- Use multiple fixed random seeds where feasible.
- Report the number of people and held-out queries in every group.
- Report group-specific performance, absolute gaps, signed differences, and uncertainty intervals.
- Use bootstrap confidence intervals with a documented resampling unit; consider resampling people rather than individual triples when people contribute multiple queries.
- Document missing attributes, multi-valued `P21` assignments, exclusions, whitelist decisions, and the evaluation subset.
- Interpret metrics as conditional on recorded facts and the chosen evaluation protocol, not as direct measurements of real-world fairness.
- Treat small-group results cautiously and report uncertainty and sample size.


## 10. Reproducibility and Change Control

Keep versioned records of:

- Wikidata5M data version and external attribute snapshot.
- Preprocessing scripts and file checksums.
- Group assignment rules and group counts.
- The gender-specific award and occupation whitelist, including supporting evidence.
- Relation exclusions and any proxy-relation ablations.
- Entity subset, candidate set, splits, and filtering rules.
- Model configuration, environment, and random seeds.
- Evaluation code, thresholds, bootstrap procedure, and results.

Any change to these definitions must be documented with its rationale and the experiments it affects. Do not silently change group assignments, whitelist classifications, or metric definitions after reviewing results.


## 11. Geographic Classification for P27 Analysis

### 11.1 Purpose and Scope

The project may use Wikidata property `P27` — country of citizenship — as an optional attribute for regional analysis. This section defines how eligible P27 country-or-area entities are assigned to broad geographic regions.

The classification uses the United Nations Statistics Division's **UN M49 geographic regions**. The project uses only the five broad regions; UN M49 subregions and intermediate regions are outside the current scope. Subregion classification may be revisited later if the research question requires finer geographic distinctions.

This is a classification of geographic entities, not a claim about a person's cultural identity, ethnicity, political alignment, or socioeconomic position. A recorded citizenship value is not a complete description of a person's identity.

### 11.2 Chosen Classification Scheme

Use the five UN M49 broad regions:

| Broad region | M49 code |
|---|---:|
| Africa | `002` |
| Americas | `019` |
| Asia | `142` |
| Europe | `150` |
| Oceania | `009` |

The official UN M49 classification is the reference for geographic assignments:

https://unstats.un.org/unsd/methodology/M49

UN M49 describes these groupings as intended for statistical use. Their assignment does not imply a position on political affiliation, sovereignty, recognition, or territorial boundaries. This project uses broad regions because they provide a consistent geographic grouping for exploratory analysis while avoiding additional subregion-level complexity that is not currently required by the research question.

**Only the broad-region code is used in the current analysis.** Country-or-area M49 codes and subregion codes are distinct from broad-region codes and must not be substituted for them.

### 11.3 Entity Eligibility and Assignment Rules

Apply the following rules to the P27 entities considered for geographic classification:

1. **Eligible country or area:** Assign the UN M49 broad region when the entity can be matched defensibly to a country or area in the official classification.
2. **Identity verification:** Resolve entities using their Wikidata QID and supporting identity information, not the label alone. A label resembling a country name may refer to a different place, organization, language, or topic.
3. **Historical states and territories:** Review historical states and territorial entities individually. Where a defensible broad-region assignment is supported by the entity's geographic extent and historical context, record the manually reviewed decision and its rationale. Do not assume that every historical state corresponds to one current UN M49 country or area. If an entity spans multiple broad regions or cannot be assigned defensibly under this policy, leave it unmapped and explain why.
4. **Non-geographic or ineligible entities:** Do not assign a geographic region to entities outside the scope of country-of-citizenship mapping, such as people, languages, fictional places, organizations, or unrelated topics. Record them as not applicable under the review policy.
5. **No defensible assignment:** For geographic or territorial entities that cannot be assigned defensibly, retain an unmapped status and a rationale rather than inventing a region.
6. **No demographic inference:** Do not infer citizenship or geographic affiliation from a person's name, gender, occupation, or other demographic characteristics.

### 11.4 Multiple P27 Values

Preserve all underlying P27 values. If a person has multiple recorded citizenship values that map to different broad regions, use **multi-membership**: the person may be associated with each applicable region.

Do not silently choose a single “primary” region or use the most frequent region. If an analysis requires mutually exclusive regional groups, define and justify a separate selection or exclusion rule before running that analysis, and report the resulting sample sizes. The mapping itself should not discard the original P27 values.

### 11.5 Review Statuses

The geographic review register uses the following statuses:

| Status | Meaning |
|---|---|
| `approved_working_decision` | The entity has a reviewed working decision; eligible entities have a broad-region assignment. |
| `not_applicable_by_policy` | The entity is outside the scope of this geographic classification. |
| `unmapped_by_policy` | The entity is geographic or territorial, but no defensible broad-region assignment is made under the current policy. |
| `candidate_needs_validation` | The entity's identity or appropriate treatment remains unresolved and requires further review. |

A status is not interchangeable with a region assignment. In particular, `not_applicable_by_policy` and `unmapped_by_policy` are not geographic regions and must not be silently converted into one.

### 11.6 Mapping Register and Coverage Reporting

For each reviewed entity, retain its Wikidata QID, label, review status, rationale, broad region, and broad-region M49 code where applicable. The register may retain subregion-related fields for possible future work, but those fields are not required for the current broad-region analysis and must not be treated as completed classifications.

The current working-register review contains 1,284 entries:

| Review status | Count |
|---|---:|
| `approved_working_decision` | 1,028 |
| `not_applicable_by_policy` | 220 |
| `unmapped_by_policy` | 36 |
| `candidate_needs_validation` | 0 |
| **Total** | **1,284** |

These are review-register counts, not a claim that the final `configs/data/regions.yaml` artifact has already been generated or independently verified. Before Task #8 is marked complete, validate the YAML against the set of country QIDs appearing as P27 objects in the relevant Wikidata5M data. Every eligible country QID must have a broad-region mapping, or be explicitly listed as unmapped with a count and rationale. Non-applicable entities should be reported separately rather than counted as geographic mappings.

Historical-state decisions must be reviewed by hand and documented in the mapping rationale. Report the number of eligible country QIDs mapped to each broad region, the number left unmapped, the reasons for unmapped cases, and the treatment of multiple P27 values.

### 11.7 Limitations and Reporting

UN M49 broad regions are statistical geographic groupings. They do not capture all cultural, political, historical, or social distinctions and should not be interpreted as proxies for those concepts without separate justification.

Wikidata citizenship statements may be incomplete or inaccurate. Missing or unrecorded P27 values must not be interpreted as proof that a person has no citizenship or geographic affiliation.

For every regional analysis, document the classification reference/version, the entity-mapping register, the handling of multiple P27 values, the analysis population, and the number of people excluded or left without an applicable region. Interpret findings as conditional on the recorded attributes and the selected classification scheme.


## References and identifier verification

- Wikidata Property `P21`, including documented values and usage guidance: https://www.wikidata.org/wiki/Property:P21
- Male (Q6581097): https://www.wikidata.org/wiki/Q6581097
- Female (Q6581072): https://www.wikidata.org/wiki/Q6581072
- Non-binary (Q48270): https://www.wikidata.org/wiki/Q48270
- Trans woman (Q1052281): https://www.wikidata.org/wiki/Q1052281
- Trans man (Q2449503): https://www.wikidata.org/wiki/Q2449503
- Intersex (Q1097630): https://www.wikidata.org/wiki/Q1097630