"""Appendix A Problem 1, using a documented small local instance suite."""
import math
from pathlib import Path
import threading
import time
import numpy as np
from sklearn.linear_model import lasso_path
from .process import Worker

EVALUATION_LOCK = threading.Lock()
SPEC = """Optimize the Lasso regularization path.
Implement solve_path(X, y, lambdas) returning a finite NumPy array of shape (len(lambdas), X.shape[1]).
Minimize F(w) = ||y - Xw||^2 / (2*n) + lambda * ||w||_1, without an intercept.
The lambdas are positive and decreasing. Inputs are floating point NumPy arrays.
Every correctness case must achieve objective <= sklearn's tightly converged reference + 1e-6.
Correctness cases are fresh and distinct from timing cases. Invalid candidates score zero.
Valid candidates score the reciprocal geometric mean of complete-path wall times in milliseconds.
Use numpy, scipy or Python's standard library. Do not import sklearn or change the evaluator.
The sandbox has no network, writable files, subprocesses, or access to other attempts.
Return ONLY complete Python source. Put the mechanism and reasoning in a module docstring.
"""


def instance(seed, n, p, correlated=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if correlated:
        X = X * 0.65 + rng.normal(size=(n, 1)) * 0.35
    truth = rng.normal(size=p) * (rng.random(p) < 0.15)
    y = X @ truth + 0.05 * rng.normal(size=n)
    high = max(float(np.max(np.abs(X.T @ y))) / n, 1e-3)
    return X, y, np.geomspace(high, high * 0.03, 8)


def objective(X, y, lambdas, weights):
    return np.mean((y[:, None] - X @ weights.T) ** 2, axis=0) / 2 + lambdas * np.abs(weights).sum(axis=1)


def check_solution(data, result):
    X, y, lambdas = data
    W = np.asarray(result, dtype=float)
    if W.shape != (len(lambdas), X.shape[1]) or not np.isfinite(W).all():
        return False, "Wrong output shape or nonfinite coefficients"
    _, reference, _ = lasso_path(X, y, alphas=lambdas, tol=1e-12, max_iter=100000)
    differences = objective(X, y, lambdas, W) - objective(X, y, lambdas, reference.T)
    gap = float(np.max(differences))
    return bool(gap <= 1e-6), gap


def evaluate(source, seed, heldout=False):
    started = time.monotonic()
    # Fixed protocol; the candidate never receives its correctness references.
    shapes = [(72, 24), (100, 40), (140, 64)] if not heldout else [(180, 72), (220, 96), (160, 112)]
    result = {"valid": False, "score": 0.0, "correctness_seed": seed, "heldout": heldout}
    try:
        with EVALUATION_LOCK, Worker(source, "solver", timeout=30) as worker:
            gaps = []
            for i, (n, p) in enumerate(shapes):
                data = instance(seed + i, n, p, i % 2 == 1)
                worker.ask({"load": [a.tolist() for a in data]})
                ok, gap = check_solution(data, worker.ask({"run": True}))
                gaps.append(gap)
                if not ok:
                    raise ValueError("Correctness check failed: " + str(gap))
            times = []
            for i, (n, p) in enumerate(shapes):
                data = instance((88000 if heldout else 44000) + i, n, p, i % 2 == 1)
                worker.ask({"load": [a.tolist() for a in data]})
                worker.ask({"run": True})  # warm-up outside measurement
                repeats = []
                for _ in range(3):
                    begin = time.perf_counter()
                    W = worker.ask({"run": True})
                    repeats.append((time.perf_counter() - begin) * 1000)
                    ok, gap = check_solution(data, W)
                    if not ok:
                        raise ValueError("Timing-case correctness check failed: " + str(gap))
                times.append(float(np.median(repeats)))
            result.update(valid=True, score=math.exp(-sum(math.log(t) for t in times) / len(times)),
                          path_times_ms=times, correctness_max_gaps=gaps)
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["evaluation_seconds"] = time.monotonic() - started
    return result
