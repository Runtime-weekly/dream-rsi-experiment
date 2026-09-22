import numpy as np
from dreamrsi.task import check_solution, evaluate, instance
from dreamrsi.engine import ROOT


def test_zero_solution_does_not_pass_path_correctness():
    data = instance(99, 64, 32)
    ok, _ = check_solution(data, np.zeros((8, 32)))
    assert not ok


def test_nan_and_wrong_shape_rejected():
    data = instance(2, 32, 16)
    assert not check_solution(data, np.full((8, 16), np.nan))[0]
    assert not check_solution(data, np.zeros((16, 8)))[0]


def test_baseline_passes_fresh_and_heldout_instances():
    source = ROOT / "tasks/lasso/baseline.py"
    assert evaluate(source, 934)["valid"]
    assert evaluate(source, 938, heldout=True)["valid"]

