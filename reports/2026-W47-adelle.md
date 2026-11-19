## 2026-11-19 — People table and P166 decision
Question: is P166 big enough to be the primary task?
Runs: data/processed/people_v1.parquet (commit <hash from git rev-parse --short HEAD>); tables in docs/data_audit/
Thresholds: constants at the top of scripts/02_build_people_table.py (agreed with <teammate> before the first run: yes/no)

Humans (P31=Q5): 1,509,635; attrs_missing 10,469

Coverage: P21 1,498,150 (99.2%); P27 1,306,120 (86.5%); P106 graph 807,575 (53.5%), attributes 1,440,740 (95.4%); P166 graph 98,433 (6.5%)

P21 counts: male 1,239,751; female 256,961; multi-valued 254; other value 1,184; missing 11,485

Regions (M49, before the region review): Europe 528,032; Americas 486,657; Asia 158,327; Oceania 61,359; Africa 48,244; no region 239,813; more than one region 23,005

Result: women hold 26,986 P166 triples (need 4,000); regions meeting the rule: Americas, Europe

Decision: P166 stays primary (conditions 1-4 pass); region analysis on P166 limited to Americas and Europe, other regions via P106

Checks: parquet reads back (1,509,635 rows, 12 columns); Marie Curie (Q7186) verified

Caveats: counts are before step 04 filters; Asia borderline; North/South grouping open; 10,469 'missing' IDs and the QLever query still unresolved

Next: rebuild as people_v2 after the region review, then step 03 covariates