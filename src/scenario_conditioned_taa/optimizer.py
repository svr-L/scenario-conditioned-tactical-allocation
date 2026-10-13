from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .risk import weighted_metrics
from .types import GroupConstraint, ScenarioSet, TAAConfig, TAAResult


class TargetInfeasibleError(RuntimeError):
    def __init__(self, target: float, maximum: float):
        self.target = float(target)
        self.maximum = float(maximum)
        super().__init__(f"target return {target:.8g} is infeasible; maximum feasible expected return is {maximum:.8g}")


def _asset_param(value, assets: pd.Index, name: str) -> pd.Series:
    if isinstance(value, Mapping):
        unknown = set(value) - set(assets)
        if unknown:
            raise ValueError(f"{name} contains unknown assets: {sorted(unknown)}")
        out = pd.Series({a: value.get(a, np.nan) for a in assets}, dtype=float)
        if out.isna().any():
            missing = out.index[out.isna()].tolist()
            raise ValueError(f"{name} missing assets: {missing}")
        return out
    return pd.Series(float(value), index=assets, dtype=float)


def tactical_bounds(saa_weights: pd.Series, config: TAAConfig) -> tuple[pd.Series, pd.Series]:
    assets = saa_weights.index
    band = _asset_param(config.band, assets, "band")
    lo_global = _asset_param(config.min_weight, assets, "min_weight")
    hi_global = _asset_param(config.max_weight, assets, "max_weight")
    if (band < 0).any():
        raise ValueError("bands must be non-negative")
    lo = pd.concat([saa_weights - band, lo_global], axis=1).max(axis=1)
    hi = pd.concat([saa_weights + band, hi_global], axis=1).min(axis=1)
    if (lo > hi + 1e-12).any():
        raise ValueError("inconsistent tactical/global bounds")
    if lo.sum() > 1 + 1e-12 or hi.sum() < 1 - 1e-12:
        raise ValueError("tactical bands make the fully-invested portfolio infeasible")
    return lo, hi


def _group_rows(assets: pd.Index, groups: Mapping[str, GroupConstraint], total_vars: int):
    rows, rhs = [], []
    pos = {a: i for i, a in enumerate(assets)}
    for name, spec in groups.items():
        unknown = set(spec.members) - set(assets)
        if unknown:
            raise ValueError(f"group {name!r} contains unknown assets: {sorted(unknown)}")
        base = np.zeros(total_vars)
        for a in spec.members:
            base[pos[a]] = 1.0
        if spec.upper is not None:
            rows.append(base.copy()); rhs.append(float(spec.upper))
        if spec.lower is not None:
            rows.append(-base.copy()); rhs.append(-float(spec.lower))
    return rows, rhs


def maximum_feasible_expected_return(scenarios: ScenarioSet, saa_weights, config: TAAConfig,
                                     previous_weights=None) -> tuple[float, pd.Series]:
    """Maximum expected return attainable under the same hard TAA constraints.

    When a turnover cap is active, auxiliary absolute-deviation variables ensure
    the feasibility frontier respects it as well. Penalty rates do not affect the
    feasible set and are therefore omitted.
    """
    assets = scenarios.assets
    saa = scenarios.aligned_weights(saa_weights)
    prev = scenarios.aligned_weights(previous_weights if previous_weights is not None else saa)
    lo, hi = tactical_bounds(saa, config)
    mu = scenarios.probabilities @ scenarios.returns.to_numpy(dtype=float)
    n = len(assets)

    use_turn = config.max_turnover is not None
    m = n + (n if use_turn else 0)
    c = np.zeros(m); c[:n] = -mu
    rows, rhs = [], []

    group_rows, group_rhs = _group_rows(assets, config.group_constraints, m)
    rows.extend(group_rows); rhs.extend(group_rhs)

    if use_turn:
        t0 = n
        for j in range(n):
            row = np.zeros(m); row[j] = 1.0; row[t0 + j] = -1.0
            rows.append(row); rhs.append(float(prev.iloc[j]))
            row = np.zeros(m); row[j] = -1.0; row[t0 + j] = -1.0
            rows.append(row); rhs.append(float(-prev.iloc[j]))
        row = np.zeros(m); row[t0:] = 1.0
        rows.append(row); rhs.append(float(config.max_turnover))

    A_eq = np.zeros((1, m)); A_eq[0, :n] = 1.0
    bounds = list(zip(lo, hi)) + ([(0.0, None)] * n if use_turn else [])
    res = linprog(c, A_ub=np.asarray(rows) if rows else None,
                  b_ub=np.asarray(rhs) if rhs else None,
                  A_eq=A_eq, b_eq=np.array([1.0]), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"feasibility LP failed: {res.message}")
    w = pd.Series(res.x[:n], index=assets, name="max_expected_return")
    return float(mu @ res.x[:n]), w


def optimize_taa(scenarios: ScenarioSet, saa_weights, config: TAAConfig | None = None,
                 previous_weights=None) -> TAAResult:
    """Minimum weighted-CVaR tactical allocation around a strategic anchor.

    The LP solves the global optimum of the declared convex problem:

        min CVaR_alpha(loss) + active_penalty * ||w-w_SAA||_1
                             + transaction_cost_rate * ||w-w_prev||_1

    subject to full investment, asset tactical bands, optional group bounds,
    optional turnover cap, and an optional probability-weighted expected-return
    floor.  Median/quantile/target-hit metrics are diagnostics, not chance
    constraints in v0.1.
    """
    cfg = config or TAAConfig()
    R = scenarios.returns.to_numpy(dtype=float)
    p = scenarios.probabilities
    assets = scenarios.assets
    S, n = R.shape
    saa = scenarios.aligned_weights(saa_weights)
    if not np.isclose(saa.sum(), 1.0, atol=1e-8):
        raise ValueError("SAA weights must sum to one")
    prev = scenarios.aligned_weights(previous_weights if previous_weights is not None else saa)
    if not np.isclose(prev.sum(), 1.0, atol=1e-8):
        raise ValueError("previous weights must sum to one")
    lo, hi = tactical_bounds(saa, cfg)

    max_ret, _ = maximum_feasible_expected_return(scenarios, saa, cfg, previous_weights=prev)
    if cfg.target_return is not None and cfg.target_return > max_ret + cfg.target_tolerance:
        raise TargetInfeasibleError(cfg.target_return, max_ret)

    # Variables x = [w(n), zeta(1), u(S), a_active(n), t_turnover(n)]
    i_w = slice(0, n)
    i_z = n
    i_u = slice(n + 1, n + 1 + S)
    i_a = slice(n + 1 + S, n + 1 + S + n)
    i_t = slice(n + 1 + S + n, n + 1 + S + 2 * n)
    m = n + 1 + S + 2 * n

    c = np.zeros(m)
    c[i_z] = 1.0
    c[i_u] = p / cfg.alpha
    c[i_a] = cfg.active_penalty
    c[i_t] = cfg.transaction_cost_rate

    rows, rhs = [], []

    # Scenario losses: -(R w) - zeta - u <= 0
    A_loss = sparse.hstack([
        sparse.csr_matrix(-R),
        sparse.csr_matrix(-np.ones((S, 1))),
        -sparse.eye(S, format="csr"),
        sparse.csr_matrix((S, 2 * n)),
    ], format="csr")
    rows_sparse = [A_loss]
    rhs_np = [np.zeros(S)]

    # Expected return floor.
    mu = p @ R
    if cfg.target_return is not None:
        row = np.zeros(m); row[i_w] = -mu
        rows.append(row); rhs.append(-float(cfg.target_return))

    # |w - saa| <= a
    for j in range(n):
        row = np.zeros(m); row[j] = 1.0; row[i_a.start + j] = -1.0
        rows.append(row); rhs.append(float(saa.iloc[j]))
        row = np.zeros(m); row[j] = -1.0; row[i_a.start + j] = -1.0
        rows.append(row); rhs.append(float(-saa.iloc[j]))

    # |w - previous| <= turnover auxiliary
    for j in range(n):
        row = np.zeros(m); row[j] = 1.0; row[i_t.start + j] = -1.0
        rows.append(row); rhs.append(float(prev.iloc[j]))
        row = np.zeros(m); row[j] = -1.0; row[i_t.start + j] = -1.0
        rows.append(row); rhs.append(float(-prev.iloc[j]))

    if cfg.max_turnover is not None:
        row = np.zeros(m); row[i_t] = 1.0
        rows.append(row); rhs.append(float(cfg.max_turnover))

    group_rows, group_rhs = _group_rows(assets, cfg.group_constraints, m)
    rows.extend(group_rows); rhs.extend(group_rhs)

    if rows:
        rows_sparse.append(sparse.csr_matrix(np.vstack(rows)))
        rhs_np.append(np.asarray(rhs, dtype=float))
    A_ub = sparse.vstack(rows_sparse, format="csr")
    b_ub = np.concatenate(rhs_np)

    A_eq = np.zeros((1, m)); A_eq[0, i_w] = 1.0
    bounds = list(zip(lo.to_numpy(), hi.to_numpy())) + [(None, None)] + [(0.0, None)] * S + [(0.0, None)] * (2 * n)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=sparse.csr_matrix(A_eq), b_eq=np.array([1.0]),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"TAA LP failed: {res.message}")

    w = pd.Series(res.x[i_w], index=assets, name="taa_weight")
    active = (w - saa).rename("active_weight")
    port = scenarios.portfolio_returns(w)
    metrics = weighted_metrics(port, p, alpha=cfg.alpha, target=cfg.target_return)
    diagnostics = pd.Series({
        "lp_objective": float(res.fun),
        "cvar_zeta": float(res.x[i_z]),
        "sum_weights": float(w.sum()),
        "gross_active_weight": float(np.abs(active).sum()),
        "turnover": float(np.abs(w - prev).sum()),
        "estimated_transaction_cost": float(cfg.transaction_cost_rate * np.abs(w - prev).sum()),
        "target_return": np.nan if cfg.target_return is None else float(cfg.target_return),
        "max_feasible_expected_return": max_ret,
    }, dtype=float)
    return TAAResult(w, active, port, metrics, diagnostics, max_ret, res.message)
