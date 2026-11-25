"""Compute full-graph degree and award covariates.

Per human (rows of people_v2.parquet):
  in_degree, out_degree, degree      edges of the full graph, P21 and P27 excluded
  degree_noaward                     same, and P166/P1411 excluded as well (robustness:
                                     an award edge is part of the outcome, so degree that
                                     counts it is partly outcome-related)

  n_facts                            statements about the person (person as head), same
                                     exclusions as degree; equals out_degree by definition

  n_awards                           distinct P166 awards in the graph (from the people table)

  birth_decade                       copied from the people table

  degree_decile, degree_noaward_decile   1..10, ties share a decile (stratification)

Leakage note: degrees include edges that the benchmark treats as valid/test. That is fine
for stratifying and describing results. If degree ever becomes a model INPUT, recompute it
on training edges only.

Run from the repository root: uv run python scripts/03_compute_covariates.py
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SENSITIVE = {'P21', 'P27'}
AWARD = {'P166', 'P1411'}

# Sanity numbers from the shell checks (relation counts over train+valid+test).
EXPECTED_TRIPLES = 20_624_575
EXPECTED_SENSITIVE_DROPPED = 1_151_085   # P27 1,146,848 + P21 4,237


def read_counts(w5m_dir, variants, chunksize=2_000_000):
    """Stream the benchmark files once and count in/out degree for each variant.

    variants: {name: set of relations to drop}. Returns ({name: (out, in)}, total, dropped)
    where dropped is the number of triples removed by the first variant.
    """
    files = sorted(Path(w5m_dir).glob('wikidata5m_transductive_*.txt'))
    if not files:
        sys.exit(f'No wikidata5m_transductive_*.txt files in {w5m_dir}')
    counts = {name: [pd.Series(dtype='int64'), pd.Series(dtype='int64')] for name in variants}
    total = dropped = 0
    first = next(iter(variants))
    for f in files:
        reader = pd.read_csv(f, sep='\t', header=None, usecols=[0, 1, 2],
                             names=['h', 'r', 't'], dtype=str, quoting=3,
                             chunksize=chunksize)
        for chunk in reader:
            total += len(chunk)
            for name, drop in variants.items():
                kept = chunk[~chunk['r'].isin(drop)]
                if name == first:
                    dropped += len(chunk) - len(kept)
                counts[name][0] = counts[name][0].add(kept['h'].value_counts(), fill_value=0)
                counts[name][1] = counts[name][1].add(kept['t'].value_counts(), fill_value=0)
    return {k: (v[0].astype('int64'), v[1].astype('int64')) for k, v in counts.items()}, total, dropped


def decile(s):
    """1..10 from the average rank, so people with the same value share a decile."""
    pct = s.rank(method='average', pct=True)
    return np.ceil(pct * 10).clip(1, 10).astype('int8')


def build_covariates(people, counts):
    ids = pd.Index(people['id'], name='id')
    cov = pd.DataFrame(index=ids)
    for name, (out, inn) in counts.items():
        o = out.reindex(ids).fillna(0).astype('int64')
        i = inn.reindex(ids).fillna(0).astype('int64')
        if name == '':
            cov['in_degree'], cov['out_degree'], cov['degree'] = i, o, i + o
        else:
            cov[f'degree_{name}'] = i + o
    cov['n_facts'] = cov['out_degree']
    cov['n_awards'] = pd.Series(people['awards'].str.len().to_numpy(), index=ids).astype('int32')
    cov['birth_decade'] = pd.Series(people['birth_decade'].to_numpy(), index=ids)
    cov['degree_decile'] = decile(cov['degree'])
    cov['degree_noaward_decile'] = decile(cov['degree_noaward'])
    return cov.reset_index()


def decile_edges(cov, col='degree', dec='degree_decile'):
    return {int(k): int(v) for k, v in cov.groupby(dec)[col].max().items()}


def audit_tables(cov, people):
    t = cov.merge(people[['id', 'gender', 'attrs_missing']], on='id')
    t = t[t['gender'].notna() & ~t['attrs_missing']]
    by_gender = t.groupby('gender').agg(
        humans=('id', 'size'),
        mean_degree=('degree', 'mean'), median_degree=('degree', 'median'),
        p90_degree=('degree', lambda s: s.quantile(0.9)),
        mean_degree_noaward=('degree_noaward', 'mean'),
        median_degree_noaward=('degree_noaward', 'median'),
        with_award=('n_awards', lambda s: int((s > 0).sum())),
        mean_n_awards=('n_awards', 'mean')).round(2).reset_index()
    dec = (t.groupby(['degree_decile', 'gender']).size().unstack(fill_value=0))
    dec = dec.div(dec.sum(axis=0), axis=1).round(4).reset_index()   # share of each gender per decile
    return by_gender, dec


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w5m', default='data/raw/wikidata5m')
    ap.add_argument('--people', default='data/processed/people_v2.parquet')
    ap.add_argument('--out', default='data/processed/covariates_v1.parquet')
    ap.add_argument('--audit-dir', default='docs/data_audit')
    ap.add_argument('--skip-expected-check', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    out = Path(args.out)
    if out.exists():
        sys.exit(f'{out} exists. Never overwrite a versioned file: bump the name, e.g. covariates_v2.parquet')
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = Path(args.audit_dir)
    audit.mkdir(parents=True, exist_ok=True)

    people = pd.read_parquet(args.people, columns=['id', 'gender', 'awards', 'birth_decade', 'attrs_missing'])
    print(f'people: {len(people):,}', flush=True)

    print('counting degrees on the full graph ...', flush=True)
    counts, total, dropped = read_counts(args.w5m, {'': SENSITIVE, 'noaward': SENSITIVE | AWARD})
    print(f'triples read: {total:,} (expected {EXPECTED_TRIPLES:,}); '
          f'P21/P27 triples excluded: {dropped:,} (expected {EXPECTED_SENSITIVE_DROPPED:,})')
    if not args.skip_expected_check and (total != EXPECTED_TRIPLES or dropped != EXPECTED_SENSITIVE_DROPPED):
        sys.exit('Counts differ from the shell checks: stop and investigate (or pass --skip-expected-check).')

    cov = build_covariates(people, counts)
    by_gender, dec = audit_tables(cov, people)
    by_gender.to_csv(audit / 'covariates_by_gender.csv', index=False)
    dec.to_csv(audit / 'covariates_degree_decile_by_gender.csv', index=False)
    (audit / 'covariates_decile_edges.json').write_text(json.dumps(
        {'degree': decile_edges(cov), 'degree_noaward': decile_edges(cov, 'degree_noaward', 'degree_noaward_decile')},
        indent=2))

    tmp = out.with_name(out.name + '.tmp')
    cov.to_parquet(tmp, index=False, compression='zstd')
    tmp.replace(out)
    (audit / 'covariates_run_log.json').write_text(json.dumps(
        {'git_commit': git_commit(), 'args': vars(args), 'triples': total, 'sensitive_excluded': dropped,
         'humans': len(cov), 'seconds': round(time.time() - t0, 1)}, indent=2))

    print(by_gender.to_string(index=False))
    print(dec.to_string(index=False))
    print(f'wrote {out} and audit tables in {audit} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
