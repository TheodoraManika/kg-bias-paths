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