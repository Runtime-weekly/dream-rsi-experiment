"""Simple converged Gram-coordinate descent with warm starts along the path."""
import numpy as np


def solve_path(X, y, lambdas):
    n, p = X.shape
    gram = X.T @ X / n
    cross = X.T @ y / n
    w = np.zeros(p)
    result = []
    for lam in lambdas:
        for _ in range(20000):
            change = 0.0
            for j in range(p):
                z = cross[j] - gram[j] @ w + gram[j, j] * w[j]
                new = np.sign(z) * max(abs(z) - lam, 0.0) / gram[j, j]
                change = max(change, abs(new - w[j]))
                w[j] = new
            if change < 1e-10:
                break
        result.append(w.copy())
    return np.array(result)

