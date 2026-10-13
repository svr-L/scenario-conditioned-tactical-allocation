from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import weighted_mean_cov, weighted_metrics, weighted_tail_masses
from .types import ScenarioSet


def _risk_components(scenarios: ScenarioSet, weights: pd.Series, alpha: float):
    R = scenarios.returns
    p = scenarios.probabilities
    w = scenarios.aligned_weights(weights)
    mu, cov = weighted_mean_cov(R, p)
    port = scenarios.portfolio_returns(w).to_numpy()
    port_mean = float(p @ port)
    port_vol = float(np.sqrt(max(float(p @ ((port - port_mean) ** 2)), 0.0)))
    covw = cov.to_numpy() @ w.to_numpy()
    vol_component = np.zeros(len(w)) if port_vol <= 1e-15 else w.to_numpy() * covw / port_vol
    tail_mass, _ = weighted_tail_masses(port, p, alpha)
    tail_asset_mean = (tail_mass[:, None] * R.to_numpy()).sum(axis=0) / alpha
    es_component = -w.to_numpy() * tail_asset_mean
    return mu, cov, pd.Series(vol_component, index=w.index), pd.Series(es_component, index=w.index)


def decision_attribution(scenarios: ScenarioSet, saa_weights, taa_weights,
                         alpha: float = 0.05, target: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explain a tactical decision relative to its strategic anchor.

    The asset table separates expected-return contribution, Euler volatility
    contribution, exact scenario ES contribution, and active tracking-error
    contribution.  The summary compares the full scenario distributions.
    """
    saa = scenarios.aligned_weights(saa_weights)
    taa = scenarios.aligned_weights(taa_weights)
    if not np.isclose(saa.sum(), 1.0) or not np.isclose(taa.sum(), 1.0):
        raise ValueError("SAA and TAA weights must sum to one")
    p = scenarios.probabilities
    mu, cov, saa_vol_c, saa_es_c = _risk_components(scenarios, saa, alpha)
    _, _, taa_vol_c, taa_es_c = _risk_components(scenarios, taa, alpha)
    active = taa - saa
    active_ret = scenarios.returns.to_numpy() @ active.to_numpy()
    a_mean = float(p @ active_ret)
    te = float(np.sqrt(max(float(p @ ((active_ret - a_mean) ** 2)), 0.0)))
    cov_active = cov.to_numpy() @ active.to_numpy()
    te_comp = np.zeros(len(active)) if te <= 1e-15 else active.to_numpy() * cov_active / te

    table = pd.DataFrame(index=scenarios.assets)
    table["saa_weight"] = saa
    table["taa_weight"] = taa
    table["active_weight"] = active
    table["asset_expected_return"] = mu
    table["delta_expected_return_contribution"] = active * mu
    table["saa_component_volatility"] = saa_vol_c
    table["taa_component_volatility"] = taa_vol_c
    table["delta_component_volatility"] = taa_vol_c - saa_vol_c
    table["saa_component_ES_loss"] = saa_es_c
    table["taa_component_ES_loss"] = taa_es_c
    table["delta_component_ES_loss"] = taa_es_c - saa_es_c
    table["active_component_tracking_error"] = te_comp

    saa_metrics = weighted_metrics(scenarios.portfolio_returns(saa), p, alpha=alpha, target=target)
    taa_metrics = weighted_metrics(scenarios.portfolio_returns(taa), p, alpha=alpha, target=target)
    summary = pd.concat([saa_metrics.rename("SAA"), taa_metrics.rename("TAA")], axis=1)
    summary["TAA_minus_SAA"] = summary["TAA"] - summary["SAA"]
    summary.loc["tracking_error_TAA_vs_SAA"] = [0.0, te, te]
    summary.loc["gross_active_weight"] = [0.0, float(np.abs(active).sum()), float(np.abs(active).sum())]
    return table, summary
