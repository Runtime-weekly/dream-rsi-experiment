import pytest
from dreamrsi.model import extract_source
from dreamrsi.upstream import classify


def test_extracts_actual_code_and_rejects_reasoning_only_or_reference_import():
    code, _ = extract_source('Here it is\n```python\ndef choose(obs):\n return []\n```', 'choose')
    assert 'def choose' in code
    with pytest.raises(ValueError):
        extract_source('I should return a function', 'choose')
    with pytest.raises(ValueError):
        extract_source('from sklearn import linear_model\ndef solve_path(*a): pass', 'solve_path')


def test_docs_only_release_is_not_mistaken_for_implementation():
    assert classify(['README.md', 'papers/Dream-RSI.pdf', 'assets/logo.png', '.github/build.yml']) == []
    assert classify(['src/policy.py', 'pyproject.toml']) == ['pyproject.toml', 'src/policy.py']
