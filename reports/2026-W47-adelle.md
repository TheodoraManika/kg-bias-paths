## 2026-11-19 — People table and P166 decision
Question: is P166 big enough to be the primary task?
Runs: data/processed/people_v1.parquet (commit <hash from git rev-parse --short HEAD>); tables in docs/data_audit/
Thresholds: constants at the top of scripts/02_build_people_table.py 

Humans (P31=Q5): 1,509,635; attrs_missing 10,469

Coverage: P21 1,498,150 (99.2%); P27 1,306,120 (86.5%); P106 graph 807,575 (53.5%), attributes 1,440,740 (95.4%); P166 graph 98,433 (6.5%)

P21 counts: male 1,239,751; female 256,961; multi-valued 254; other value 1,184; missing 11,485

Regions (M49, before the region review): Europe 528,032; Americas 486,657; Asia 158,327; Oceania 61,359; Africa 48,244; no region 239,813; more than one region 23,005

Regions (M49, after the region review): Europe 540,376; Americas 486,657; Asia 165,515; Oceania 61,359; Africa 48,244; no region 223,785; more than one region 26,444

P166 region rule (women's award triples, need 4,000): Americas 11,972 and Europe 11,100 pass; Asia 3,111 is borderline; Oceania 1,010 and Africa 397 are far below.

Decision: P166 stays primary (conditions 1-4 pass); region analysis on P166 limited to Americas and Europe, other regions via P106

Checks: parquet reads back (1,509,635 rows, 12 columns); Marie Curie (Q7186) verified

Caveats: Asia borderline; 10,469 'missing' IDs


## 2026-11-20 — Degree, Award Count, and Birth Decade on Graph
Reads the three benchmark files once and counts in-, out- and total degree on the full graph (excluding P21 and P27). Cross-check reads 20,624,575 triples and excludes 1,151,085 that does not have valid data (P27 has 1,146,848 and P21 has 4,237). Has a runtime of 72 seconds.

Covariates are joined by id in people_v2.parquet. Everyone gets a row. A human with no graph edges gets degree 0, not missing.

degree_decile is 1 to 10, across all humans. People with the same degree always share a decile, so low-degree deciles can be larger than 10% of people. A strict 10% split would put identical people in different strata (defeats stratification).

Union of train, valid, AND test is used to create the FULL graph. If degree ever becomes a model input, recompute it on training edges only.

n_facts is the number of statemetns about the person (person as head, same exclusions as degree).

There is a small popularity gap, concentrated at the top. Women's mean degree is 6.20 against 6.73 for men. Medians are the same (5), and the 900tth percentile is 11 against 13. Most women and men are about equally connected, and men are over-represented  among the very well conneced.

Women have a larger sahre than men in deciles 1 to 5 (or example 8.0% against 6.1% in decile 1) and a smaller share in 8 to 10 (decile 9: 6.0% against 9.1%; decile 10: 8.3% against 10.0%). The columns each add up to 100%.

Degree without award edges gives almost the same picture (6.08 against 6.61), so counting award edges adds only about 0.1 to degree. This means that award edges don't drive the gap.

This alone does not tell us how much of the model's bias runs though popularity. Shows that the popularity path is real but modest in size.

## 2026-11-28 - Sampling mode decision (issue #22)

Question: how do we sample so every group has at least 200 test queries
(test fraction = 0.10)?

Data: people_v2 + covariates_v1. 1,499,166 people after dropping 10,469 with
no attributes. Strata: gender, region, degree (no award edges), birth decade,
occupation.

What we found:
1. The 200 minimum counts award edges, not people. Only about 6.6% of people
   hold an award, so a plain sample of 200,000 (13%) keeps too few. Only male
   Americas and male Europe would pass. Raising the number of people per
   group (2,000) adds only about 130 award holders per group, which is also
   not enough.
2. Five groups already fail on the full data, so no sampling can fix them:
   male Africa (95 test edges), female Africa (28), female Oceania (88),
   female "multiple" (150) and female "no_region" (92). Female Asia passes
   only if we keep 79% of its award edges. Americas, Asia and Europe pass
   for both genders.

Decision: oversample mode, keeping ALL award holders and sampling only the
rest.
Why: it is the only option where every group that can pass does pass.

Result (seed 42, 200,000 people):
- 98,311 award holders kept (weight 1) + 101,689 other people sampled
  (weight about 14 on average).
- Weights add up to 1,499,166, the full population. No stratum was left out.
- 84.8% of people are in full strata. Only 380 (0.025%) were merged into the
  coarsest levels.
- Groups with fewer than 2,000 people were raised to 2,000 (female Africa,
  Oceania, multiple). All other_value, missing and multi_valued groups are
  kept in full.
- All award edges are kept, so each group has exactly the test edges it has
  in the full data. The five failing groups stay failing and are reported
  descriptively, marked as unreliable.
- Effective sample size is 109,702 (55% of rows). Weights range from 1.0 to
  22.5.

Rules from now on:
- Every rate and comparison uses sampling_weight.
- has_award is used for sampling only. Never use it as a control in the
  decomposition.
- "multiple" and "no_region" are strata, not fairness groups.

Next: compare the sample with the full graph (SMD below 0.1), then build
the splits.