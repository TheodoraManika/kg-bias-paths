# Statistical Pitfalls in Fairness Experiments

Oct 21, 2026 · @Theodora Manika
Fairness numbers on small groups are noisy, and with many metrics, groups and methods, some differences will look important by chance. These rules keep our conclusions honest.
1. Where noise comes from
Every number we report would change somewhat if we repeated the experiment, and we need to know by how much before calling a difference real.
Source of variation
Example
How we measure it
Training randomness
Initial weights, neighbour sampling, negative sampling, GPU non-determinism
3–5 seeds per method
Which test examples we happen to have
A different test set would give a different MRR
Bootstrap over test queries
Data split and subsample
Another random split or subset
Repeat with 2–3 splits for the main result
Small groups
40 test people in a group: one extra hit moves Hits@10 by 2.5 points
Minimum group sizes; confidence intervals
Hyperparameter search
The best of 50 trials is optimistically biased on validation
Choose on validation, report on test once
2. Small groups
The uncertainty of a group's rate shrinks only with the square root of its size, so small groups give fairness numbers that are mostly noise.
For a rate p (e.g. Hits@10) measured on n people, the standard error is about:
SE = \sqrt{\frac{p(1-p)}{n}}
With p = 0.3, a 95% interval (about ±2 SE) is roughly ±13 points for n = 50, ±6.5 for n = 200 and ±3 for n = 1,000. A gap of 5 points between two groups of 50 tells us almost nothing.
Rules
	•	Set the minimum group size before results: e.g. 200 test queries per group for headline fairness metrics.
	•	Merge, do not drop: if a region is too small, merge it with a neighbouring region in the published scheme; report the merge.
	•	Report n next to every group metric, and a confidence interval.
	•	Bootstrap confidence intervals: resample test queries with replacement 1,000 times, recompute the metric each time, and take the 2.5th and 97.5th percentiles. For gaps, resample within each group and recompute the gap.
	•	Groups below the minimum are shown descriptively (counts and raw rates), clearly marked as unreliable.
3. Comparing methods
A method is better only if its advantage is larger than the noise from seeds and test sampling, and big enough to matter.
	•	Seeds: 3–5 per method, the same seed list for all methods. Report mean ± standard deviation over seeds.
	•	Paired comparison: because methods share seeds and test queries, compare them query by query (or seed by seed). Pairing removes the variation that is common to both, so real differences show up with fewer runs.
	•	Paired bootstrap test: resample test queries, recompute "metric of method A minus metric of method B" each time, and see whether the 95% interval of the difference excludes zero.
	•	Permutation test (alternative): randomly swap A's and B's scores per query many times; if the observed difference is rarely matched, it is unlikely to be chance.
	•	Effect size, not only significance: with many test queries, tiny differences can be "significant". Decide in advance what counts as meaningful (e.g. a parity gap reduced by at least 2 points, MRR loss below 0.01).
	•	Seed variance vs. method difference: if two methods' seed ranges overlap heavily, call the result "no clear difference".
	•	Equal tuning budget: comparing a tuned method to an untuned baseline is the most common way gains are exaggerated.
4. Many comparisons
With about 4 metrics × 2 attributes × several groups × 2 tasks × 6 methods we will make hundreds of comparisons, and at a 5% threshold roughly 1 in 20 true non-differences will look significant by chance.
Fix the primary comparison in advance
	•	Primary task: P166 (award received)
	•	Primary attribute: gender (largest groups)
	•	Primary metric: demographic parity gap @10 (does not depend on incomplete ground truth)
	•	Primary question: does our method reduce it more than the strongest baseline, at an MRR loss below the agreed threshold?
Everything else (P106, region, equal opportunity, per-group MRR) is secondary: reported fully, but described as supporting evidence, not as the main finding.
Corrections for secondary results
	•	Holm–Bonferroni: controls the chance of any false positive; good when there are a moderate number of tests.
	•	Benjamini–Hochberg: controls the expected share of false positives among the results you call significant; better when there are many tests.
	•	Both are available in statsmodels.stats.multitest.multipletests.
Never search through many metrics and groups and then report only those that look good.
5. Traps specific to bias analysis
These mistakes produce confident but wrong conclusions about where bias comes from.
Simpson's paradox. A gap in the total can shrink, vanish or reverse inside subgroups. Example: if women in Wikidata are concentrated in fields that give out fewer awards, the overall gender gap mixes field differences with gender. Always also look at the gap within field and birth decade.
Confounder, mediator, collider. Decide the role of each variable before controlling for it:
Variable
Role
What to do
Field / occupation group, birth decade
Confounder: related to gender and to awards, not caused by the model
Control for it (stratify or include in regression)
Degree (popularity)
Mediator: gender → degree → prediction
Use it to split direct vs. popularity effect; do not just "control it away" in the total
Having an award edge in training
Collider / outcome-related
Never control for it: it distorts the comparison
Unstable ratios. The amplification ratio (model gap ÷ data gap) explodes when the data gap is near zero and flips sign when it is negative. Report the two gaps and their difference as well, and give a bootstrap interval for the ratio.
Oaxaca–Blinder choices.
	•	The split depends on which group's coefficients are the reference (men's, women's or pooled). Report the pooled version and check that the others tell the same story.
	•	Hits@10 for one person is 0 or 1, so a linear model is only an approximation; use a non-linear variant (e.g. Fairlie decomposition) as a robustness check.
	•	Results depend on which covariates are included. Fix the list in the pre-registration.
Selection into Wikidata. Everyone in our data was notable enough to be in Wikidata, and the bar for inclusion may differ by gender and region. Conclusions apply to people in Wikidata, not to the population.
Fairness through unawareness. Removing the gender edge does not remove bias, because proxies remain. Always check with the probe.
6. Pre-registration light
Write this one-page plan before training the first baseline, commit it to the repository with a date, and do not edit it afterwards; later changes go in a separate "deviations" list with reasons.
# Analysis plan (frozen on YYYY-MM-DD, commit <hash>)  ## Hypotheses H1: Message passing amplifies the gender disparity in P166 predictions relative to the training data. H2: At least [X]% of the gender disparity is associated with degree (popularity path). H3: Our method reduces the targeted component more than the strongest baseline, with MRR loss below [T].  ## Primary outcome Demographic parity gap @10 on P166, gender, test set.  ## Secondary outcomes Equal-opportunity gap, per-group MRR, region, P106.  ## Groups Gender values with >= [N] test queries; others reported descriptively. Regions: [scheme]; merges: [list].  ## Decomposition Covariates: degree (mediator); field group, birth decade (confounders). Method: degree stratification + pooled Oaxaca-Blinder; Fairlie as robustness check.  ## Comparisons Methods: [list]. Seeds: [0-4]. Tuning budget: [K] trials each. Test: paired bootstrap, 1,000 resamples; Holm correction for secondary outcomes. Meaningful difference: parity gap change >= [D] points.  ## Deviations (added later, with date and reason)
7. How to report results
A reader should be able to judge every number's reliability without asking us.
	•	Every metric with uncertainty: mean ± standard deviation over seeds, plus a bootstrap 95% interval for fairness gaps.
	•	Every group metric with its n.
	•	Main table = primary outcome only; secondary outcomes in a second table or appendix, clearly labelled.
	•	Show the trade-off curve (accuracy vs. fairness) rather than one hand-picked setting.
	•	Report null and negative results in the same detail as positive ones: "no clear difference" is a finding.
	•	List deviations from the analysis plan, with reasons.
	•	Say what was not tested: other attributes, other graphs, causal claims beyond the decomposition.
Example row: Parity gap @10 (gender, P166): baseline 0.142 ± 0.006 [0.130, 0.155], ours 0.081 ± 0.009 [0.065, 0.098]; paired difference −0.061 [−0.078, −0.044]; n = 4,812 / 1,377. (Illustrative numbers only.)
Checklist and sources
	•	☐ Analysis plan written, dated and committed before the first baseline
	•	☐ Minimum group size and merge rules fixed
	•	☐ Primary task, attribute, metric and meaningful-difference threshold fixed
	•	☐ Confounders and mediators listed for the decomposition
	•	☐ Same seeds and tuning budget for all methods
	•	☐ Bootstrap intervals and paired tests implemented in the shared evaluation module
	•	☐ Multiple-comparison correction applied to secondary outcomes
	•	☐ Ratios reported together with their two parts
	•	☐ Within-field results checked for Simpson's paradox
	•	☐ Deviations from the plan listed in the report
Sources (from memory, not re-checked while writing; confirm before citing)
	•	Efron & Tibshirani, An Introduction to the Bootstrap (1993) — bootstrap intervals
	•	Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in NLP", ACL 2018 — paired tests for model comparison
	•	Benjamini & Hochberg (1995) — false discovery rate
	•	Holm (1979) — step-down multiple-testing correction
	•	Jann (2008), "The Blinder–Oaxaca decomposition for linear regression models", Stata Journal
	•	Fairlie (2005), non-linear decomposition for binary outcomes
	•	Pearl, Glymour & Jewell, Causal Inference in Statistics: A Primer (2016) — confounders, mediators, colliders, Simpson's paradox
	•	statsmodels multipletests documentation
