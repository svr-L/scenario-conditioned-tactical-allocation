from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .optimizer import optimize_taa
from .saa import SAAConfig, build_saa
from .types import MultiAnchorResult, ScenarioSet, TAAConfig, WalkForwardResult

ScenarioProvider = Callable[[pd.Timestamp, pd.DataFrame], ScenarioSet]


def _get_support(provider, date, hist):
    if callable(provider):
        return provider(date, hist)
    if date not in provider:
        return None
    return provider[date]


def _align_support(support: ScenarioSet, columns: pd.Index) -> ScenarioSet:
    missing = set(columns) - set(support.assets)
    extra = set(support.assets) - set(columns)
    if missing or extra:
        raise ValueError(f"scenario/realized universe mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return ScenarioSet(support.returns.reindex(columns=columns), support.probabilities,
                       support.features, support.metadata)


def run_walk_forward(realized_returns: pd.DataFrame,
                     scenario_provider: ScenarioProvider | Mapping[pd.Timestamp, ScenarioSet],
                     *, saa_method: str = "erc", saa_config: SAAConfig | None = None,
                     taa_config: TAAConfig | None = None, min_history: int = 36,
                     saa_rebalance_every: int = 12) -> WalkForwardResult:
    """Strict point-in-time one-step walk-forward harness.

    At row t, SAA estimation uses rows strictly before t. The support for t is
    assumed to contain only information available at t. TAA weights are applied to
    realized row t. `taa_returns` are net of the configured proportional
    transaction-cost rate; gross returns and costs are retained separately.
    """
    rr = pd.DataFrame(realized_returns).astype(float).sort_index()
    if rr.index.has_duplicates:
        raise ValueError("realized_returns index must be unique")
    if min_history < 3:
        raise ValueError("min_history must be at least 3")
    cfg_saa = saa_config or SAAConfig()
    cfg_taa = taa_config or TAAConfig()

    net_ret, gross_ret, costs, saa_ret, taa_w, saa_w, diag = [], [], [], [], [], [], []
    current_saa = None
    previous_taa = None
    rebalance_count = 0

    for pos in range(min_history, len(rr)):
        date = rr.index[pos]
        hist = rr.iloc[:pos]
        if current_saa is None or rebalance_count % saa_rebalance_every == 0:
            current_saa = build_saa(saa_method, hist, cfg_saa)
        rebalance_count += 1

        support = _get_support(scenario_provider, date, hist)
        if support is None:
            continue
        support = _align_support(support, rr.columns)
        current_saa = current_saa.reindex(rr.columns)
        result = optimize_taa(support, current_saa, cfg_taa, previous_weights=previous_taa)
        realized = rr.iloc[pos]
        gross = float(result.weights @ realized)
        cost = float(result.diagnostics["estimated_transaction_cost"])
        net_ret.append((date, gross - cost))
        gross_ret.append((date, gross))
        costs.append((date, cost))
        saa_ret.append((date, float(current_saa @ realized)))
        taa_w.append(result.weights.rename(date))
        saa_w.append(current_saa.rename(date))
        d = result.diagnostics.copy(); d.name = date; diag.append(d)
        previous_taa = result.weights

    if not net_ret:
        raise ValueError("walk-forward produced no evaluation observations")
    return WalkForwardResult(
        taa_returns=pd.Series(dict(net_ret), name="TAA_net"),
        taa_gross_returns=pd.Series(dict(gross_ret), name="TAA_gross"),
        transaction_costs=pd.Series(dict(costs), name="transaction_cost"),
        saa_returns=pd.Series(dict(saa_ret), name="SAA"),
        taa_weights=pd.DataFrame(taa_w),
        saa_weights=pd.DataFrame(saa_w),
        diagnostics=pd.DataFrame(diag),
    )


def run_multi_anchor_walk_forward(realized_returns: pd.DataFrame,
                                  scenario_provider: ScenarioProvider | Mapping[pd.Timestamp, ScenarioSet],
                                  *, saa_methods: Sequence[str] = ("equal_weight", "gmv", "ledoit_wolf", "markowitz", "erc", "michaud"),
                                  saa_config: SAAConfig | None = None,
                                  taa_config: TAAConfig | None = None,
                                  min_history: int = 36,
                                  saa_rebalance_every: int = 12) -> MultiAnchorResult:
    """Run the identical TAA rule across multiple SAA anchors on the same supports."""
    rr = pd.DataFrame(realized_returns).astype(float).sort_index()
    cfg_saa = saa_config or SAAConfig()
    cfg_taa = taa_config or TAAConfig()
    methods = list(dict.fromkeys(saa_methods))
    current = {m: None for m in methods}
    previous = {m: None for m in methods}
    net = {m: {} for m in methods}; gross = {m: {} for m in methods}; costs = {m: {} for m in methods}; base = {m: {} for m in methods}
    tw = {m: [] for m in methods}; sw = {m: [] for m in methods}; dg = {m: [] for m in methods}
    rebalance_count = 0

    for pos in range(min_history, len(rr)):
        date = rr.index[pos]
        hist = rr.iloc[:pos]
        if rebalance_count % saa_rebalance_every == 0:
            for m in methods:
                current[m] = build_saa(m, hist, cfg_saa).reindex(rr.columns)
        rebalance_count += 1
        support = _get_support(scenario_provider, date, hist)
        if support is None:
            continue
        support = _align_support(support, rr.columns)
        realized = rr.iloc[pos]
        for m in methods:
            result = optimize_taa(support, current[m], cfg_taa, previous_weights=previous[m])
            g = float(result.weights @ realized)
            c = float(result.diagnostics["estimated_transaction_cost"])
            net[m][date] = g - c; gross[m][date] = g; costs[m][date] = c
            base[m][date] = float(current[m] @ realized)
            tw[m].append(result.weights.rename(date)); sw[m].append(current[m].rename(date))
            d = result.diagnostics.copy(); d.name = date; dg[m].append(d)
            previous[m] = result.weights

    if not any(net[m] for m in methods):
        raise ValueError("multi-anchor walk-forward produced no evaluation observations")
    return MultiAnchorResult(
        taa_returns=pd.DataFrame({m: pd.Series(net[m]) for m in methods}),
        taa_gross_returns=pd.DataFrame({m: pd.Series(gross[m]) for m in methods}),
        transaction_costs=pd.DataFrame({m: pd.Series(costs[m]) for m in methods}),
        saa_returns=pd.DataFrame({m: pd.Series(base[m]) for m in methods}),
        taa_weights={m: pd.DataFrame(tw[m]) for m in methods},
        saa_weights={m: pd.DataFrame(sw[m]) for m in methods},
        diagnostics={m: pd.DataFrame(dg[m]) for m in methods},
    )
