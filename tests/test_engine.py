import json
import pytest
from pathlib import Path
from dreamrsi.core import save_json
from dreamrsi.core import digest
from dreamrsi.engine import Experiment, ROOT
import dreamrsi.engine as engine


def test_full_loop_promotes_policy_then_resumes_without_new_calls(tmp_path, monkeypatch):
    # Explicit test fixture, never used or reported as a real-model experiment.
    monkeypatch.setattr(engine, "evaluate", lambda *a, **kw: {"valid": True, "score": 1.0})
    class Model:
        identity = {"name": "test-fixture"}
        calls = 0
        def generate(self, prompt, directory, function, seed):
            self.calls += 1
            directory.mkdir(parents=True, exist_ok=True)
            save_json(directory / "request.json", {"fixture": True})
            save_json(directory / "response.json", {"answer": {"eval_count": 1}, "seconds": 0})
            source = (ROOT / "tasks/lasso/baseline.py").read_text() if function == "solve_path" else "def choose(obs):\n    return []\n"
            return source, "fixture", {"model_calls": 1}
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    model = Model()
    experiment = Experiment(tmp_path, config, model)
    result = experiment.run()
    selection = json.loads((tmp_path / "adaptive/round-1/dream/selection.json").read_text())
    assert selection["selected"]["mean_value"] >= selection["evaluations"][0]["mean_value"]
    assert "revision-1" in selection["selected"]["path"]
    # A policy selected by replay controls the NEXT online rollout.
    assert len(json.loads((tmp_path / "adaptive/round-2/tree.json").read_text())) == 1
    assert result["arms"]["fixed"]["discovery_calls"] == 4
    assert result["arms"]["adaptive"]["discovery_calls"] == 2
    assert result["arms"]["adaptive"]["policy_development_calls"] == 2
    calls = model.calls
    experiment.run()
    assert model.calls == calls


def test_pre_request_failure_is_not_counted_as_a_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "evaluate", lambda *a, **kw: {"valid": True, "score": 1.0})
    class Model:
        identity = {"name": "test-fixture"}
        def generate(self, *a, **kw):
            raise ValueError("Prompt exceeds configured limit before any request")
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    result = Experiment(tmp_path, config, Model()).run()
    for arm in result["arms"].values():
        assert arm["candidate_attempts"] == 4
        assert arm["all_model_calls"] == 0
        assert arm["discovery_calls"] == 0
        assert arm["policy_development_calls"] == 0


def test_modified_selected_policy_cannot_be_resumed(tmp_path):
    policy = tmp_path / "policy.py"
    original = "def choose(obs): return []\n"
    policy.write_text(original)
    save_json(tmp_path / "selection.json", {"selected": {"path": str(policy), "sha256": digest(original)}})
    policy.write_text("def choose(obs): return ['root']\n")
    with pytest.raises(ValueError, match="selected policy changed"):
        Experiment(tmp_path, {}, None).dream(tmp_path, policy, [], 0)
