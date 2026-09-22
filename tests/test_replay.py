import pytest
from dreamrsi.core import Node, Tree, replay, select_policy, validate_batch
from dreamrsi.core import digest


def test_replay_hides_future_and_preserves_recorded_order():
    tree = Tree([Node("root", None, 0, True), Node("a", "root", 1, True, depth=1),
                 Node("b", "root", 2, True, depth=1), Node("c", "a", 5, True, depth=2)])
    class Policy:
        seen = []
        def choose(self, obs):
            self.seen.append([n["id"] for n in obs["nodes"]])
            assert all("artifact" not in n for n in obs["nodes"])
            return ["root"] if obs["decision"] == 0 else ["root", "a"]
    policy = Policy()
    result = replay(tree, policy, 2, 8, .5, .1)
    assert policy.seen == [["root"], ["root", "a"]]
    assert result["trajectory"] == [{"actions": ["root"], "revealed": ["a"]},
                                     {"actions": ["root", "a"], "revealed": ["b", "c"]}]
    assert result["represented_calls"] == 3
    assert result["value"] == pytest.approx(3.65)


def test_unrecorded_continuation_cannot_create_outcomes():
    tree = Tree([Node("root", None, 0, True), Node("a", "root", 1, True, depth=1),
                 Node("b", "root", 50, True, depth=1)])
    class Policy:
        def choose(self, obs):
            return ["root"] if obs["decision"] == 0 else ["a"]
    result = replay(tree, Policy(), 1, 3, 0, 0)
    assert result["best_score"] == 1
    assert result["represented_calls"] == 1
    assert result["rounds"] == 3


@pytest.mark.parametrize("actions", [["future"], ["root", "root"], ["root", "a"]])
def test_illegal_batches_rejected(actions):
    with pytest.raises(ValueError):
        validate_batch(actions, Tree(), 2)


def test_incumbent_survives_worse_tied_and_invalid_revisions():
    incumbent = {"valid": True, "mean_value": 2, "path": "original"}
    assert select_policy([incumbent, {"valid": True, "mean_value": 2},
                          {"valid": True, "mean_value": 1}, {"valid": False}]) is incumbent
    with pytest.raises(RuntimeError):
        select_policy([{"valid": False}, incumbent])


def test_nonroot_cannot_branch_or_invalid_score_become_reward():
    tree = Tree([Node("root", None, 0, True), Node("a", "root", 1, True, depth=1),
                 Node("b", "a", 2, True, depth=2)])
    with pytest.raises(ValueError):
        tree.add(Node("c", "a", 3, True, depth=2))
    with pytest.raises(ValueError):
        tree.add(Node("d", "root", 3, False, depth=1))


def test_empty_world_stops_without_asking_policy_for_impossible_work():
    class NeverCalled:
        def choose(self, obs):
            raise AssertionError("No recorded attempts to reveal")
    result = replay(Tree(), NeverCalled(), 2, 10, 1, 1)
    assert result["rounds"] == 0
    assert result["stop"] == "history_exhausted"


def test_changed_saved_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "candidate.py"
    artifact.write_text("original")
    tree = Tree([Node("root", None, 0, True, str(artifact), artifact_sha256=digest("original"))])
    assert Tree.load(tree.dump()).get("root").artifact_sha256 == digest("original")
    artifact.write_text("changed after scoring")
    with pytest.raises(ValueError, match="artifact changed"):
        Tree.load(tree.dump())
