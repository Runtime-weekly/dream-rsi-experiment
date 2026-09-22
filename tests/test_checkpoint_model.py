import json
import pytest
from dreamrsi.model import Ollama
from dreamrsi.cli import validate_config
from dreamrsi.core import save_json
from dreamrsi.engine import ROOT
import dreamrsi.model as model_module


def test_model_response_is_reused_without_another_request(tmp_path, monkeypatch):
    calls = []
    model = object.__new__(Ollama)
    model.url, model.name, model.tokens, model.timeout = 'http://127.0.0.1:11434', 'fixture', 100, 2
    model.identity = {'name':'fixture'}
    monkeypatch.setattr(model, 'inspect', lambda: model.identity)
    def respond(url, payload=None, timeout=10):
        calls.append(payload)
        return {'message': {'content': 'def choose(obs):\n    return []'}, 'done_reason':'stop'}
    monkeypatch.setattr(model_module, 'request_json', respond)
    first = model.generate('prompt', tmp_path, 'choose', 1)
    second = model.generate('prompt', tmp_path, 'choose', 1)
    assert first == second
    assert len(calls) == 1
    assert calls[0]['think'] is False  # native API, not the old nested-options mistake
    with pytest.raises(RuntimeError, match='differs'):
        model.generate('different', tmp_path, 'choose', 1)


def test_unconfirmed_request_cannot_be_repeated(tmp_path, monkeypatch):
    model = object.__new__(Ollama)
    save_json(tmp_path / 'request.json', {'prompt_sha256':'irrelevant'})
    with pytest.raises(RuntimeError, match='Unconfirmed'):
        model.generate('prompt', tmp_path, 'choose', 1)


@pytest.mark.parametrize('key,value', [('rounds',0), ('workers',-1), ('cost_weight',float('nan'))])
def test_invalid_experiment_budget_or_objective_rejected(key,value):
    config = json.loads((ROOT / 'configs/smoke.json').read_text())
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
