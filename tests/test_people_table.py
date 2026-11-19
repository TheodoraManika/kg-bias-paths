"""Toy-data tests for scripts/02_build_people_table.py (run: uv run pytest tests/test_people_table.py)."""
import importlib.util
import pathlib

import pandas as pd

_path = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / '02_build_people_table.py'
_spec = importlib.util.spec_from_file_location('people02', _path)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

GRAPH = [
    ('Q1', 'P31', 'Q5'), ('Q2', 'P31', 'Q5'), ('Q3', 'P31', 'Q5'),
    ('Q4', 'P31', 'Q5'), ('Q5', 'P31', 'Q5'), ('Q6', 'P31', 'Q5'),
    ('Q9', 'P31', 'Q7'), ('Q9', 'P166', 'Q100'),          # non-human with an award
    ('Q1', 'P106', 'Q200'), ('Q1', 'P166', 'Q100'), ('Q1', 'P166', 'Q101'),
    ('Q1', 'P166', 'Q100'),                                # duplicate triple
    ('Q2', 'P166', 'Q100'), ('Q2', 'P106', 'Q200'),
    ('Q3', 'P106', 'Q201'), ('Q4', 'P166', 'Q102'),
]
ATTRS = [
    ('Q1', 'P21', 'Q6581072', ''), ('Q1', 'P27', 'Q30', ''),
    ('Q1', 'P569', '1860-07-20T00:00:00Z', '11'), ('Q1', 'sitelinks', '31', ''),
    ('Q2', 'P21', 'Q6581097', ''), ('Q2', 'P27', 'Q145', ''), ('Q2', 'P27', 'Q174193', ''),
    ('Q2', 'P569', '-0500-01-01T00:00:00Z', '9'),
    ('Q3', 'P21', 'Q6581097', ''), ('Q3', 'P21', 'Q6581072', ''),
    ('Q3', 'P569', '1800-01-01T00:00:00Z', '7'),
    ('Q4', 'P21', 'Q48270', ''), ('Q4', 'P27', 'Q999999', ''),
    ('Q5', 'missing', '', ''), ('Q6', 'sitelinks', '5', ''),
]


def toy():
    g = pd.DataFrame(GRAPH, columns=['h', 'r', 't']).drop_duplicates()
    a = pd.DataFrame(ATTRS, columns=['qid', 'property', 'value', 'precision'])
    return m.build_people(g, a, {'Q30': '019', 'Q145': '150', 'Q174193': '150'})


def test_humans_only():
    assert list(toy().index) == ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']


def test_lists_are_sorted_and_deduplicated():
    p = toy()
    assert p.loc['Q1', 'awards'] == ['Q100', 'Q101']
    assert p.loc['Q2', 'regions'] == ['150']


def test_gender_statuses():
    p = toy()
    assert p.loc['Q1', 'gender'] == 'female' and p.loc['Q2', 'gender'] == 'male'
    assert p.loc['Q3', 'gender_status'] == 'multi_valued' and pd.isna(p.loc['Q3', 'gender'])
    assert p.loc['Q4', 'gender_status'] == 'other_value' and pd.isna(p.loc['Q4', 'gender'])
    assert p.loc['Q6', 'gender_status'] == 'missing'


def test_missing_and_unmapped():
    p = toy()
    assert p.loc['Q5', 'attrs_missing'] and not p.loc['Q6', 'attrs_missing']
    assert p.loc['Q4', 'n_countries_unmapped'] == 1


def test_birth_decade():
    p = toy()
    assert p.loc['Q1', 'birth_decade'] == 1860      # day precision
    assert p.loc['Q2', 'birth_decade'] == -500      # BCE does not crash
    assert pd.isna(p.loc['Q3', 'birth_decade'])     # century precision -> no decade


def toy_with_classes():
    g = pd.DataFrame(GRAPH, columns=['h', 'r', 't']).drop_duplicates()
    extra = [('Q1', 'P106', 'Q200', ''), ('Q1', 'P106', 'Q201', ''),      # tie -> class_order
             ('Q2', 'P106', 'Q200', ''), ('Q2', 'P106', 'Q202', ''), ('Q2', 'P106', 'Q203', ''),
             ('Q3', 'P106', 'Q999', ''),                                    # unmapped -> other
             ]
    a = pd.DataFrame(ATTRS + extra, columns=['qid', 'property', 'value', 'precision'])
    occ = {'Q200': 'natural_sciences', 'Q201': 'medicine', 'Q202': 'medicine', 'Q203': 'medicine'}
    order = ['natural_sciences', 'medicine']
    return m.build_people(g, a, {'Q30': '019'}, (occ, order))


def test_occupation_class_rules():
    p = toy_with_classes()
    assert p.loc['Q1', 'occupation_class'] == 'natural_sciences'   # 1 vs 1, earlier class wins
    assert p.loc['Q2', 'occupation_class'] == 'medicine'           # 2 vs 1, majority wins
    assert p.loc['Q3', 'occupation_class'] == 'other'              # only unmapped occupations
    assert p.loc['Q4', 'occupation_class'] == 'no_occupation'      # no P106 at all