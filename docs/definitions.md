''WILL UPDATE TOMORROW WITH RELEVANT FACTS''

# Definitions: Tasks, Sensitive Groups, and Gender-Relevant Relations

**Project:** Disentangling Bias in Knowledge-Graph Link Prediction  
**Dataset:** Wikidata5M, transductive setting  
**Status:** Working specification; update when the experimental design changes.

### Prediction Tasks

Knowledge-graph link prediction represents a fact as a triple \((h,r,t)\), where \(h\) is the head entity, \(r\) is the relation, and \(t\) is the tail entity. In this project, the primary prediction direction is \((\text{person},r,?)\).

| Relation | Meaning | Role |
|---|---|---|
| P166 | Award received | **Primary prediction target** |
| P106 | Occupation | Secondary prediction target or audit, depending on available time |
| P21 | Sex or gender | Sensitive group attribute; excluded from model input |
| P569 | Date of birth | Optional birth-decade control |
| P27 | Country of citizenship | Optional regional analysis/control |
| Node degree | Number of incident graph edges under the documented degree definition | Connectivity measure |

### Gender Groups and Assignment Rules

The following Wikidata item identifiers are used to recognize relevant recorded P21 values.

| Recorded value | QID | Project treatment |
|---|---|---|
| Male | Q6581097 | Primary binary comparison group |
| Female | Q6581072 | Primary binary comparison group |
| Non-binary | Q48270 | Separate descriptive group |
| Trans woman | Q1052281 | Separate descriptive group |
| Trans man | Q2449503 | Separate descriptive group |
| Intersex | Q1097630 | Separate recorded-value category; do not assume it is equivalent to a gender identity |

These identifiers are based on Wikidata's P21 documentation and item pages. The table is a project-specific list, not an exhaustive taxonomy of gender identities or sex-related classifications.

Wikidata records are incomplete, reflect editorial choices, and may contain multiple values. A missing P21 value must be treated as **unknown or unrecorded**, not as evidence of a person's gender.

### Primary binary comparison

The main comparison is between people recorded with the exact P21 value male (Q6581097) and people recorded with the exact P21 value female (Q6581072).

For the mutually exclusive primary analysis:

1. Include a person in the male group when the only recognized value among the listed categories is male.
2. Include a person in the female group when the only recognized value among the listed categories is female.
3. If a person has multiple distinct listed values, do not silently select one. Mark the person as **multi-valued** and exclude them from the mutually exclusive primary comparison.
4. Do not automatically merge trans women with the female group or trans men with the male group in the primary QID-based analysis. If a broader grouping is scientifically justified, define it as a separate, documented sensitivity analysis.
5. Treat people with other, unknown, missing, or unlisted values as separate descriptive categories where possible.

This conservative rule avoids making assumptions about how multiple Wikidata statements should be interpreted. It may reduce group sizes, so the number of included, excluded, and multi-valued people must be reported.

### Gender-relevant Relations and Values

| Relation | Policy |
|---|---|
| P21 — sex or gender | Sensitive attribute. Exclude from model inputs; retain only in the separate attribute table for grouping, auditing, and optional probing. |
| P166 — award received | Primary target. Retain for prediction, but flag explicitly gender-specific awards and analyze them separately from the primary general-award results. |
| P106 — occupation | Secondary target or audit. Flag explicitly gender-specific occupation values and report their treatment. |
| P569 — date of birth | Optional control, derived into birth periods only when available and appropriate. |
| P27 — country of citizenship | Optional control or descriptive analysis; not a substitute for ethnicity or nationality identity. |
| Other relations | No relation is automatically considered safe or gender-neutral merely because it is not P21. Use the documented preprocessing rules and item-level audit where relevant. |

P21 is never a legitimate shortcut input for the primary task. P166 and P106 are target relations, not sensitive-attribute inputs.

Other graph relations may encode demographic information indirectly—for example, through family relations, occupation, or affiliations. The baseline retains the documented graph relation set, with P21 excluded. Potential proxy-relation removals are separate ablation experiments.

## References and identifier verification

- Wikidata Property P21, including documented values and usage guidance: https://www.wikidata.org/wiki/Property:P21
- Male (Q6581097): https://www.wikidata.org/wiki/Q6581097
- Female (Q6581072): https://www.wikidata.org/wiki/Q6581072
- Non-binary (Q48270): https://www.wikidata.org/wiki/Q48270
- Trans woman (Q1052281): https://www.wikidata.org/wiki/Q1052281
- Trans man (Q2449503): https://www.wikidata.org/wiki/Q2449503
- Intersex (Q1097630): https://www.wikidata.org/wiki/Q1097630
