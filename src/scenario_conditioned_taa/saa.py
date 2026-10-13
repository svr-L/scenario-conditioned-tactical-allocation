from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class SAAConfig:
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_aversion: float = 4.0
    target_return: float | None = None
    michaud_resamples: int = 250
    seed: int = 7


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    x = returns.astype(float).dropna(how="any")
    if len(x) < 3:
        raise ValueError("at least three complete observations are required")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("returns contain non-finite values")
    return x


def _bounds(n: int, cfg: SAAConfig):
    if cfg.min_weight * n > 1 + 1e-12 or cfg.max_weight * n < 1 - 1e-12:
        raise ValueError("weight bounds are infeasible")
    return [(cfg.min_weight, cfg.max_weight)] * n


def _solve_slsqp(fun, n: int, cfg: SAAConfig, *, jac=None, constraints_extra=()):
    x0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    constraints.extend(constraints_extra)
    res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=_bounds(n, cfg), constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 2000, "disp": False})
    if not res.success:
        raise RuntimeError(f"SAA optimization failed: {res.message}")
    w = np.asarray(res.x, dtype=float)
    w[np.abs(w) < 1e-12] = 0.0
    return w / w.sum()


def equal_weight(returns: pd.DataFrame, config: SAAConfig | None = None) -> pd.Series:
    x = _validate_returns(returns)
    cfg = config or SAAConfig()
    n = x.shape[1]
    if not (cfg.min_weight <= 1 / n <= cfg.max_weight):
        raise ValueError("equal weight violates configured bounds")
    return pd.Series(np.full(n, 1.0 / n), index=x.columns, name="equal_weight")


def global_minimum_variance(returns: pd.DataFrame, config: SAAConfig | None = None,
                            covariance: np.ndarray | None = None, name: str = "gmv") -> pd.Series:
    x = _validate_returns(returns)
    cfg = config or SAAConfig()
    cov = np.asarray(covariance if covariance is not None else x.cov().to_numpy(), dtype=float)
    n = x.shape[1]
    fun = lambda w: float(w @ cov @ w)
    jac = lambda w: 2.0 * cov @ w
    w = _solve_slsqp(fun, n, cfg, jac=jac)
    return pd.Series(w, index=x.columns, name=name)


def ledoit_wolf_gmv(returns: pd.DataFrame, config: SAAConfig | None = None) -> pd.Series:
    x = _validate_returns(returns)
    lw = LedoitWolf().fit(x.to_numpy(dtype=float))
    return global_minimum_variance(x, config, covariance=lw.covariance_, name="ledoit_wolf_gmv")


def mean_variance(returns: pd.DataFrame, config: SAAConfig | None = None,
                  mean: np.ndarray | None = None, covariance: np.ndarray | None = None,
                  name: str = "markowitz") -> pd.Series:
    x = _validate_returns(returns)
    cfg = config or SAAConfig()
    mu = np.asarray(mean if mean is not None else x.mean().to_numpy(), dtype=float)
    cov = np.asarray(covariance if covariance is not None else x.cov().to_numpy(), dtype=float)
    n = x.shape[1]
    lam = float(cfg.risk_aversion)
    fun = lambda w: float(0.5 * lam * (w @ cov @ w) - mu @ w)
    jac = lambda w: lam * (cov @ w) - mu
    extras = []
    if cfg.target_return is not None:
        extras.append({"type": "ineq", "fun": lambda w, m=mu, t=cfg.target_return: float(m @ w - t)})
    w = _solve_slsqp(fun, n, cfg, jac=jac, constraints_extra=extras)
    return pd.Series(w, index=x.columns, name=name)


def equal_risk_contribution(returns: pd.DataFrame, config: SAAConfig | None = None) -> pd.Series:
    x = _validate_returns(returns)
    cfg = config or SAAConfig()
    cov = x.cov().to_numpy(dtype=float)
    n = x.shape[1]

    def objective(w):
        v = float(w @ cov @ w)
        if v <= 0:
            return 1e6
        mrc_var = w * (cov @ w)
        target = v / n
        return float(np.sum((mrc_var - target) ** 2) / (v * v))

    w = _solve_slsqp(objective, n, cfg)
    return pd.Series(w, index=x.columns, name="equal_risk_contribution")


def michaud_resampled_efficiency(returns: pd.DataFrame, config: SAAConfig | None = None) -> pd.Series:
    """Michaud-style resampled mean-variance allocation.

    Estimate mean/covariance from history, draw parameter-consistent synthetic
    samples, solve the same constrained Markowitz problem in each resample, and
    average the resulting efficient allocations.  This is deliberately a compact,
    reproducible implementation rather than a claim to reproduce proprietary
    Michaud software choices.
    """
    x = _validate_returns(returns)
    cfg = config or SAAConfig()
    rng = np.random.default_rng(cfg.seed)
    mu = x.mean().to_numpy(dtype=float)
    cov = x.cov().to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    weights = []
    for _ in range(cfg.michaud_resamples):
        sim = rng.multivariate_normal(mu, cov, size=n_obs)
        sim_df = pd.DataFrame(sim, columns=x.columns)
        try:
            w = mean_variance(sim_df, cfg).to_numpy()
            weights.append(w)
        except RuntimeError:
            continue
    if not weights:
        raise RuntimeError("all Michaud resample optimizations failed")
    avg = np.mean(weights, axis=0)
    avg = avg / avg.sum()
    if np.any(avg < cfg.min_weight - 1e-8) or np.any(avg > cfg.max_weight + 1e-8):
        raise RuntimeError("averaged Michaud weights violate configured bounds")
    return pd.Series(avg, index=x.columns, name="michaud_resampled")


def build_saa(method: str, returns: pd.DataFrame, config: SAAConfig | None = None) -> pd.Series:
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    registry = {
        "ew": equal_weight,
        "equal_weight": equal_weight,
        "gmv": global_minimum_variance,
        "minimum_variance": global_minimum_variance,
        "ledoit_wolf": ledoit_wolf_gmv,
        "lw_gmv": ledoit_wolf_gmv,
        "markowitz": mean_variance,
        "mean_variance": mean_variance,
        "erc": equal_risk_contribution,
        "risk_parity": equal_risk_contribution,
        "michaud": michaud_resampled_efficiency,
    }
    if key not in registry:
        raise ValueError(f"unknown SAA method: {method}. Available: {sorted(registry)}")
    return registry[key](returns, config)
