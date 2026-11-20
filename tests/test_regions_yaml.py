import collections
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def doc():
    with (ROOT / 'configs' / 'data' / 'regions.yaml').open() as f:
        return yaml.safe_load(f)


def test_codes_are_quoted_known_strings(doc):
    allowed = set(doc['regions'])
    for q, v in doc['entities'].items():
        code = v.get('region_code')
        assert code is None or (isinstance(code, str) and code in allowed), q


def test_status_agrees_with_region(doc):
    for q, v in doc['entities'].items():
        if v['status'] == 'approved_working_decision':
            assert v.get('region_code'), q
        else:
            assert not v.get('region_code'), q


def test_header_counts_match_entries(doc):
    ent = doc['entities'].values()
    assert dict(collections.Counter(v['status'] for v in ent)) == doc['decision_counts']
    assert dict(collections.Counter(v['region_code'] for v in ent if v.get('region_code'))) == doc['region_counts']