"""Build a people attribute table from processed Wikidata5M and Wikidata attributes.

Humans      = entities with (P31, Q5) in the Wikidata5M graph (train+valid+test).
Occupations = P106 from the graph.   Awards = P166 from the graph.
Gender, citizenship (P27), birth date, sitelinks = 2026 attributes file.
Regions     = UN M49 codes from configs/data/regions.yaml (key 'entities').

Run from the repository root:
    uv run python scripts/02_build_people_table.py
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

MALE, FEMALE = 'Q6581097', 'Q6581072'
GRAPH_RELS = {'P31', 'P27', 'P106', 'P166'}

# Sanity numbers from the shell checks (Nov 2026). The script stops if they differ.
EXPECTED_HUMANS = 1_509_635
EXPECTED_ATTRS_MISSING = 10_469

# Decision rule for P166 vs P106.
# BEFORE THE FIRST FULL RUN. Do not change them after seeing the counts.
TEST_FRACTION = 0.10      # share of each group's triples that will end up in test
MIN_TEST_QUERIES = 200    # minimum test queries per headline group
SAFETY_FACTOR = 2         # margin for later filtering (step 04)
MIN_REGIONS = 2           # regions that must meet the group-size rule
MIN_AWARDS_WITH_5 = 200   # award entities that need at least 5 holders
MAX_TOP1_SHARE = 0.30     # largest single award, share of all P166 triples
MAX_TOP10_SHARE = 0.70    # ten largest awards, share of all P166 triples


# reading
def read_graph(w5m_dir):
    """Stream the three benchmark files in chunks; keep only the four relations."""
    keep = []
    files = sorted(Path(w5m_dir).glob('wikidata5m_transductive_*.txt'))
    if not files:
        sys.exit(f'No wikidata5m_transductive_*.txt files in {w5m_dir}')
    for f in files:
        reader = pd.read_csv(f, sep='\t', header=None, usecols=[0, 1, 2],
                             names=['h', 'r', 't'], dtype=str, quoting=3,
                             chunksize=2_000_000)
        for chunk in reader:
            keep.append(chunk[chunk['r'].isin(GRAPH_RELS)])
    return pd.concat(keep, ignore_index=True).drop_duplicates()


def read_attributes(path):
    a = pd.read_csv(path, sep='\t', dtype=str, quoting=3, keep_default_na=False)
    a.columns = ['qid', 'property', 'value', 'precision']
    return a


def load_regions(path):
    """Country Q-number -> M49 code. Entries without a region (policy) are skipped."""
    with open(path) as f:
        cfg = yaml.safe_load(f)['entities']
    mapping = {}
    for q, v in cfg.items():
        code = v.get('region_code')
        if code is None:
            continue
        assert isinstance(code, str), f'region_code of {q} must be a quoted string'
        mapping[q] = code
    return mapping


# building
def load_occupation_classes(path):
    """Occupation Q-number -> broad class, plus the class order used to break ties."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    occ = {q: v['class'] for q, v in doc['occupations'].items() if v.get('class')}
    order = list(doc['class_order'])
    unknown = sorted({c for c in occ.values() if c not in order and c != 'other'})
    assert not unknown, f'classes missing from class_order: {unknown}'
    return occ, order


def occupation_class_column(a, occ, order):
    """One class per person from the attributes-file P106 values.

    Rule (from the YAML): the class with the most distinct occupations wins; ties go to
    the class listed first in class_order; 'other' only if none of the occupations is
    mapped; people with no P106 at all get 'no_occupation'.
    """
    p = a.loc[a['property'] == 'P106', ['qid', 'value']].drop_duplicates()
    p['cls'] = p['value'].map(occ)
    mapped = p.dropna(subset=['cls'])
    mapped = mapped[mapped['cls'] != 'other']
    cnt = mapped.groupby(['qid', 'cls']).size().rename('n').reset_index()
    cnt['rank'] = cnt['cls'].map({c: i for i, c in enumerate(order)})
    cnt = cnt.sort_values(['qid', 'n', 'rank'], ascending=[True, False, True])
    best = cnt.drop_duplicates('qid').set_index('qid')['cls']
    out = pd.Series('other', index=pd.Index(sorted(set(p['qid'])), name='qid'))
    out.loc[best.index] = best
    return out.rename('occupation_class')


def humans(g):
    return sorted(g.loc[(g['r'] == 'P31') & (g['t'] == 'Q5'), 'h'].unique())


def graph_lists(g, ids, rel):
    s = g.loc[(g['r'] == rel) & g['h'].isin(ids), ['h', 't']]
    s = s.drop_duplicates().sort_values(['h', 't'])
    return s.groupby('h')['t'].agg(list)


def _lists(a, prop):
    s = a.loc[a['property'] == prop, ['qid', 'value']].drop_duplicates()
    s = s.sort_values(['qid', 'value'])
    return s.groupby('qid')['value'].agg(list)


def gender_columns(a):
    p = a.loc[a['property'] == 'P21', ['qid', 'value']].drop_duplicates()
    grp = p.groupby('qid')['value'].agg(['size', 'first'])
    status = pd.Series('other_value', index=grp.index)
    status[grp['first'] == MALE] = 'male'
    status[grp['first'] == FEMALE] = 'female'
    status[grp['size'] > 1] = 'multi_valued'
    out = pd.DataFrame({'gender_status': status})
    out['gender_qid'] = grp['first'].where(grp['size'] == 1)
    out['gender'] = out['gender_qid'].map({MALE: 'male', FEMALE: 'female'})
    return out


def region_columns(a, country_to_region):
    c = _lists(a, 'P27')
    out = pd.DataFrame({'countries': c})
    out['regions'] = c.map(
        lambda xs: sorted({country_to_region[q] for q in xs if q in country_to_region}))
    out['n_countries_unmapped'] = c.map(
        lambda xs: sum(q not in country_to_region for q in xs))
    return out


def birth_year(v):
    """Year from an ISO date string; handles BCE ('-0500-...') and a leading '+'."""
    v = v.lstrip('+')
    neg = v.startswith('-')
    y = int((v[1:] if neg else v).split('-')[0])
    return -y if neg else y


def birth_columns(a):
    b = a.loc[a['property'] == 'P569', ['qid', 'value', 'precision']].copy()
    b['prec'] = pd.to_numeric(b['precision'], errors='coerce')
    b = b[b['prec'] >= 8]                              # decade precision or better
    b['decade'] = b['value'].map(birth_year) // 10 * 10
    g = b.groupby('qid')['decade'].agg(['min', 'max'])
    d = g['min'].where(g['min'] == g['max'])           # conflicting dates -> no decade
    return d.rename('birth_decade')


def sitelinks_column(a):
    s = a.loc[a['property'] == 'sitelinks'].drop_duplicates('qid')
    return pd.to_numeric(s.set_index('qid')['value']).rename('sitelinks')


def build_people(g, a, country_to_region, occ_classes=None):
    ids = humans(g)
    people = pd.DataFrame(index=pd.Index(ids, name='id'))
    people = (people.join(gender_columns(a))
                    .join(region_columns(a, country_to_region))
                    .join(birth_columns(a))
                    .join(sitelinks_column(a)))
    if occ_classes is not None:
        people = people.join(occupation_class_column(a, *occ_classes))
        people['occupation_class'] = people['occupation_class'].fillna('no_occupation')
    people['occupations'] = graph_lists(g, ids, 'P106').reindex(people.index)
    people['awards'] = graph_lists(g, ids, 'P166').reindex(people.index)
    for c in ['countries', 'regions', 'occupations', 'awards']:
        people[c] = people[c].map(lambda x: x if isinstance(x, list) else [])
    people['gender_status'] = people['gender_status'].fillna('missing')
    people['n_countries_unmapped'] = people['n_countries_unmapped'].fillna(0).astype('int8')
    missing = set(a.loc[a['property'] == 'missing', 'qid'])
    people['attrs_missing'] = people.index.isin(missing)
    return people


def save_people(people, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    lst = pa.list_(pa.string())
    t = pa.table({
        'id': pa.array(people.index.tolist(), pa.string()),
        'gender': pa.array(people['gender'], pa.string(), from_pandas=True),
        'gender_status': pa.array(people['gender_status'], pa.string(), from_pandas=True),
        'regions': pa.array(people['regions'].tolist(), lst),
        'countries': pa.array(people['countries'].tolist(), lst),
        'occupations': pa.array(people['occupations'].tolist(), lst),
        'awards': pa.array(people['awards'].tolist(), lst),
        'birth_decade': pa.array(people['birth_decade'].astype('Int16'), pa.int16()),
        'sitelinks': pa.array(people['sitelinks'].astype('Int32'), pa.int32()),
        'n_countries_unmapped': pa.array(people['n_countries_unmapped'], pa.int8()),
        'attrs_missing': pa.array(people['attrs_missing'], pa.bool_()),
    })
    if 'occupation_class' in people.columns:
        t = t.append_column('occupation_class', pa.array(people['occupation_class'], pa.string()))
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    pq.write_table(t, tmp, compression='zstd')
    tmp.replace(path)


# audit
def coverage_table(people):
    n = len(people)
    ok = ~people['attrs_missing']
    rows = [
        ('humans (P31=Q5 in graph)', n),
        ('attrs_missing (no 2026 attributes)', int(people['attrs_missing'].sum())),
        ('P21 gender (attributes)', int((people['gender_status'] != 'missing').sum())),
        ('P27 citizenship (attributes)', int((people['countries'].str.len() > 0).sum())),
        ('P106 occupation (graph)', int((people['occupations'].str.len() > 0).sum())),
        ('P166 award (graph)', int((people['awards'].str.len() > 0).sum())),
        ('binary-gender humans, attributes present', int((people['gender'].notna() & ok).sum())),
        ('has birth decade', int(people['birth_decade'].notna().sum())),
    ]
    if 'occupation_class' in people.columns:
        has = people['occupation_class'] != 'no_occupation'
        rows.append(('P106 occupation (attributes, 2026)', int(has.sum())))
        rows.append(('occupation_class other than other/no_occupation',
                     int((~people['occupation_class'].isin(['other', 'no_occupation'])).sum())))
    out = pd.DataFrame(rows, columns=['measure', 'humans'])
    out['share_of_humans'] = (out['humans'] / n).round(4)
    return out


def gender_table(people):
    t = people.assign(has_award=people['awards'].str.len() > 0,
                      has_occ=people['occupations'].str.len() > 0)
    out = t.groupby('gender_status').agg(humans=('has_award', 'size'),
                                         with_award=('has_award', 'sum'),
                                         with_occupation=('has_occ', 'sum'))
    out['share_of_humans'] = (out['humans'] / len(people)).round(4)
    return out.reset_index()


def occupation_class_table(people):
    t = people[people['gender'].notna() & ~people['attrs_missing']]
    t = t.assign(has_award=t['awards'].str.len() > 0)
    out = t.groupby(['occupation_class', 'gender']).agg(
        humans=('has_award', 'size'), award_holders=('has_award', 'sum')).reset_index()
    return out


def other_gender_values(people):
    o = people.loc[people['gender_status'] == 'other_value', 'gender_qid']
    return o.value_counts().rename_axis('gender_qid').reset_index(name='humans')


def _exploded_regions(people):
    t = people.loc[~people['attrs_missing'], ['gender', 'awards', 'occupations', 'regions']].copy()
    t['n_awards'] = t['awards'].str.len()
    t['n_occ'] = t['occupations'].str.len()
    t['regions'] = t['regions'].map(lambda r: r if r else ['no_region'])
    return t.explode('regions')


def region_table(people):
    ex = _exploded_regions(people)
    out = ex.groupby('regions').agg(humans=('n_awards', 'size'),
                                    with_award=('n_awards', lambda s: int((s > 0).sum())),
                                    award_triples=('n_awards', 'sum'))
    n_people = int((~people['attrs_missing']).sum())
    multi = int((people.loc[~people['attrs_missing'], 'regions'].str.len() > 1).sum())
    out.loc['(people with more than one region)'] = [multi, 0, 0]
    out['share_of_humans'] = (out['humans'] / n_people).round(4)
    return out.reset_index()


def gender_region_table(people):
    ex = _exploded_regions(people)
    ex = ex[ex['gender'].notna()]
    out = ex.groupby(['regions', 'gender']).agg(
        humans=('n_awards', 'size'),
        award_holders=('n_awards', lambda s: int((s > 0).sum())),
        award_triples=('n_awards', 'sum'),
        occupation_triples=('n_occ', 'sum')).reset_index()
    return out


def award_diagnostics(people):
    aw = people.loc[people['awards'].str.len() > 0, ['gender', 'awards']].explode('awards')
    per_award = aw.groupby('awards').size().sort_values(ascending=False)
    total = int(per_award.sum())
    stats = {
        'p166_triples_total': total,
        'p166_holders_total': int((people['awards'].str.len() > 0).sum()),
        'distinct_awards': len(per_award),
        'awards_with_5plus_holders': int((per_award >= 5).sum()),
        'awards_with_10plus_holders': int((per_award >= 10).sum()),
        'top1_share': round(float(per_award.iloc[0] / total), 4) if total else None,
        'top10_share': round(float(per_award.iloc[:10].sum() / total), 4) if total else None,
        'p166_triples_male': int((aw['gender'] == 'male').sum()),
        'p166_triples_female': int((aw['gender'] == 'female').sum()),
    }
    return stats, per_award.head(25).rename_axis('award').reset_index(name='holders')


def _group_ok(triples_by_group):
    need = MIN_TEST_QUERIES * SAFETY_FACTOR
    return all(triples_by_group.get(k, 0) * TEST_FRACTION >= need for k in ('male', 'female'))


def decide(people, stats, gr_table):
    need = MIN_TEST_QUERIES * SAFETY_FACTOR
    result = {}
    for name, col, total_col in [('P166', 'award_triples', 'p166'), ('P106', 'occupation_triples', 'p106')]:
        src = 'awards' if name == 'P166' else 'occupations'
        counts = people.assign(n=people[src].str.len()).groupby('gender')['n'].sum()
        by_gender = pd.Series({'male': counts.get('male', 0), 'female': counts.get('female', 0)})
        c1 = bool(by_gender.get('female', 0) * TEST_FRACTION >= need)
        piv = (gr_table[gr_table['regions'] != 'no_region']
               .pivot_table(index='regions', columns='gender', values=col, aggfunc='sum', fill_value=0))
        n_regions = 0
        if {'male', 'female'} <= set(piv.columns):
            n_regions = int(((piv[['male', 'female']].min(axis=1) * TEST_FRACTION) >= need).sum())
        result[name] = {'cond1_women_triples': int(by_gender.get('female', 0)),
                        'cond1_pass': c1, 'cond2_regions_ok': n_regions,
                        'cond2_pass': n_regions >= MIN_REGIONS}
    p166 = result['P166']
    p166['cond3_pass'] = bool(stats['awards_with_5plus_holders'] >= MIN_AWARDS_WITH_5)
    p166['cond4_pass'] = bool((stats['top1_share'] or 1) <= MAX_TOP1_SHARE and
                              (stats['top10_share'] or 1) <= MAX_TOP10_SHARE)
    p166['cond5_award_coverage'] = round(stats['p166_holders_total'] / len(people), 4)
    all_pass = all(p166[k] for k in ('cond1_pass', 'cond2_pass', 'cond3_pass', 'cond4_pass'))
    if all_pass:
        rec = 'KEEP P166 as primary task.'
    elif not (p166['cond1_pass'] and p166['cond2_pass']) and \
            result['P106']['cond1_pass'] and result['P106']['cond2_pass']:
        rec = 'SWITCH primary task to P106 (keep P166 as secondary).'
    else:
        rec = 'BORDERLINE: keep P166 but report small-group results descriptively; discuss with teammate.'
    result['recommendation'] = rec
    return result


# main
def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w5m', default='data/raw/wikidata5m')
    ap.add_argument('--attrs', default='data/raw/wikidata_attributes/attributes_2026-10-25.tsv')
    ap.add_argument('--regions', default='configs/data/regions.yaml')
    ap.add_argument('--occupation-classes', default='configs/data/occupation_classes_p279.yaml')
    ap.add_argument('--out', default='data/processed/people_v1.parquet')
    ap.add_argument('--audit-dir', default='docs/data_audit')
    ap.add_argument('--skip-expected-check', action='store_true',
                    help='do not stop when counts differ from the shell checks')
    args = ap.parse_args()
    t0 = time.time()

    out = Path(args.out)
    if out.exists():
        sys.exit(f'{out} exists. Never overwrite a versioned file: bump the name, e.g. people_v2.parquet')
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = Path(args.audit_dir)
    audit.mkdir(parents=True, exist_ok=True)

    print('reading graph ...', flush=True)
    g = read_graph(args.w5m)
    print('reading attributes ...', flush=True)
    a = read_attributes(args.attrs)
    mapping = load_regions(args.regions)
    occ_classes = load_occupation_classes(args.occupation_classes)

    print('building people table ...', flush=True)
    people = build_people(g, a, mapping, occ_classes)
    n_h, n_m = len(people), int(people['attrs_missing'].sum())
    print(f'humans: {n_h:,} (expected {EXPECTED_HUMANS:,}); attrs_missing: {n_m:,} (expected {EXPECTED_ATTRS_MISSING:,})')
    if not args.skip_expected_check and (n_h != EXPECTED_HUMANS or n_m != EXPECTED_ATTRS_MISSING):
        sys.exit('Counts differ from the shell checks: stop and investigate (or pass --skip-expected-check).')

    cov = coverage_table(people)
    gen = gender_table(people)
    oth = other_gender_values(people)
    reg = region_table(people)
    grt = gender_region_table(people)
    stats, top_awards = award_diagnostics(people)
    decision = decide(people, stats, grt)

    occ_tab = occupation_class_table(people)
    n_p106_attrs = int((people['occupation_class'] != 'no_occupation').sum())
    print(f'humans with P106 in attributes: {n_p106_attrs:,} (the occupation YAML reports 1,440,740)')
    for name, df in [('occupation_class_counts', occ_tab), ('coverage', cov), ('gender_status_counts', gen), ('other_gender_values', oth),
                     ('region_counts', reg), ('gender_by_region', grt), ('top_awards', top_awards)]:
        df.to_csv(audit / f'people_{name}.csv', index=False)
    (audit / 'people_p166_decision.json').write_text(json.dumps(
        {'thresholds': {'TEST_FRACTION': TEST_FRACTION, 'MIN_TEST_QUERIES': MIN_TEST_QUERIES,
                        'SAFETY_FACTOR': SAFETY_FACTOR, 'MIN_REGIONS': MIN_REGIONS,
                        'MIN_AWARDS_WITH_5': MIN_AWARDS_WITH_5, 'MAX_TOP1_SHARE': MAX_TOP1_SHARE,
                        'MAX_TOP10_SHARE': MAX_TOP10_SHARE},
         'award_stats': stats, 'decision': decision}, indent=2))

    save_people(people, out)
    (audit / 'people_run_log.json').write_text(json.dumps(
        {'git_commit': git_commit(), 'args': vars(args), 'humans': n_h, 'attrs_missing': n_m,
         'seconds': round(time.time() - t0, 1)}, indent=2))

    print(cov.to_string(index=False))
    print(gen.to_string(index=False))
    print(json.dumps(decision, indent=2))
    print(f'wrote {out} and audit tables in {audit} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()