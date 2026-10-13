from __future__ import annotations

import numpy as np
import pandas as pd

from .types import normalize_probabilities


def weighted_quantile(values, probabilities, q: float) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    p = normalize_probabilities(probabilities, len(x))
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    order = np.argsort(x)
    xs, ps = x[order], p[order]
    cdf = np.cumsum(ps)
    idx = int(np.searchsorted(cdf, q, side="left"))
    return float(xs[min(idx, len(xs) - 1)])


def weighted_tail_masses(returns, probabilities, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    r = np.asarray(returns, dtype=float).reshape(-1)
    p = normalize_probabilities(probabilities, len(r))
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    order = np.argsort(r)
    take_sorted = np.zeros_like(p)
    remaining = float(alpha)
    for i in order:
        if remaining <= 1e-15:
            break
        mass = min(float(p[i]), remaining)
        take_sorted[i] = mass
        remaining -= mass
    if remaining > 1e-8:
        raise RuntimeError("unable to allocate requested tail probability")
    var_return = weighted_quantile(r, p, alpha)
    return take_sorted, var_return


def weighted_metrics(returns, probabilities, alpha: float = 0.05, target: float | None = None) -> pd.Series:
    r = np.asarray(returns, dtype=float).reshape(-1)
    p = normalize_probabilities(probabilities, len(r))
    mean = float(p @ r)
    centered = r - mean
    vol = float(np.sqrt(max(float(p @ (centered * centered)), 0.0)))
    tail_mass, var_return = weighted_tail_masses(r, p, alpha)
    es_return = float(tail_mass @ r / alpha)
    out = {
        "expected_return": mean,
        "volatility": vol,
        "median_return": weighted_quantile(r, p, 0.50),
        f"q_{int(alpha*100):02d}_return": var_return,
        f"VaR_{int((1-alpha)*100)}_loss": -var_return,
        f"ES_{int((1-alpha)*100)}_loss": -es_return,
    }
    if target is not None:
        out["probability_target_met"] = float(p[r >= target].sum())
        out["expected_shortfall_to_target"] = float(p @ np.maximum(target - r, 0.0))
    return pd.Series(out, dtype=float)


def weighted_mean_cov(returns: pd.DataFrame, probabilities) -> tuple[pd.Series, pd.DataFrame]:
    R = returns.to_numpy(dtype=float)
    p = normalize_probabilities(probabilities, len(R))
    mu = p @ R
    X = R - mu
    cov = (X.T * p) @ X
    return pd.Series(mu, index=returns.columns, name="expected_return"), pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
