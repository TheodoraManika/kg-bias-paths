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