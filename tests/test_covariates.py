"""Toy-data tests for scripts/03_compute_covariates.py."""
import importlib.util
import pathlib

import pandas as pd

_path = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / '03_compute_covariates.py'
_spec = importlib.util.spec_from_file_location('cov03', _path)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

TRIPLES = [
    ('Q1', 'P31', 'Q5'), ('Q2', 'P31', 'Q5'),
    ('Q1', 'P166', 'Q100'), ('Q1', 'P106', 'Q200'), ('Q2', 'P106', 'Q200'),
    ('Q1', 'P27', 'Q30'), ('Q2', 'P27', 'Q30'),        # sensitive: must not count
    ('Q1', 'P21', 'Q6581072'),                         # sensitive: must not count
    ('Q3', 'P40', 'Q1'),                               # Q1 receives an in-edge
]


def write_graph(tmp_path):
    for name, rows in [('train', TRIPLES[:6]), ('valid', TRIPLES[6:8]), ('test', TRIPLES[8:])]:
        (tmp_path / f'wikidata5m_transductive_{name}.txt').write_text(
            ''.join(f'{h}\t{r}\t{t}\n' for h, r, t in rows))


def toy_counts(tmp_path):
    write_graph(tmp_path)
    return m.read_counts(tmp_path, {'': m.SENSITIVE, 'noaward': m.SENSITIVE | m.AWARD})


def test_sensitive_edges_are_excluded(tmp_path):
    counts, total, dropped = toy_counts(tmp_path)
    assert total == 9 and dropped == 3                   # two P27 and one P21
    out, inn = counts['']
    assert out['Q1'] == 3 and out['Q2'] == 2             # P31, P166, P106 / P31, P106
    assert inn['Q1'] == 1                                # the P40 edge from Q3


def test_award_free_degree(tmp_path):
    counts, _, _ = toy_counts(tmp_path)
    out, _inn = counts['noaward']
    assert out['Q1'] == 2                                # P166 dropped as well


def test_build_covariates_and_deciles(tmp_path):
    counts, _, _ = toy_counts(tmp_path)
    people = pd.DataFrame({'id': ['Q1', 'Q2', 'Q9'], 'gender': ['female', 'male', 'male'],
                           'awards': [['Q100'], [], []], 'birth_decade': [1860.0, None, 1900.0],
                           'attrs_missing': [False, False, False]})
    cov = m.build_covariates(people, counts).set_index('id')
    assert cov.loc['Q1', 'degree'] == 4 and cov.loc['Q1', 'in_degree'] == 1
    assert cov.loc['Q9', 'degree'] == 0                  # not in the graph: zero, not missing
    assert cov.loc['Q1', 'n_awards'] == 1 and cov.loc['Q2', 'n_awards'] == 0
    assert cov['degree_decile'].between(1, 10).all()
    assert cov.loc['Q1', 'degree_decile'] >= cov.loc['Q2', 'degree_decile']


def test_ties_share_a_decile():
    s = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 5, 9])
    d = m.decile(s)
    assert d.iloc[:8].nunique() == 1