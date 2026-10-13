import numpy as np
import pandas as pd
import pytest

from scenario_conditioned_taa import (
    ScenarioSet, TAAConfig, GroupConstraint, optimize_taa,
    maximum_feasible_expected_return, TargetInfeasibleError, decision_attribution,
)


def support(seed=4, n=2500):
    rng = np.random.default_rng(seed)
    # Equity highest mean/tail; Rates defensive; Credit intermediate.
    z = rng.standard_normal((n, 3))
    r = np.column_stack([
        0.010 + 0.050 * z[:, 0],
        0.003 + 0.015 * (0.2*z[:, 0] + np.sqrt(0.96)*z[:, 1]),
        0.006 + 0.030 * (0.4*z[:, 0] + np.sqrt(0.84)*z[:, 2]),
    ])
    return ScenarioSet(pd.DataFrame(r, columns=["Equity", "Rates", "Credit"]))


def test_taa_respects_bands_target_and_full_investment():
    s = support()
    saa = pd.Series([0.40, 0.35, 0.25], index=s.assets)
    cfg = TAAConfig(alpha=0.05, target_return=0.0060, band=0.10, active_penalty=0.001)
    out = optimize_taa(s, saa, cfg)
    assert np.isclose(out.weights.sum(), 1.0)
    assert (np.abs(out.active_weights) <= 0.100001).all()
    assert out.metrics["expected_return"] >= 0.0060 - 1e-8


def test_infeasible_target_reports_maximum():
    s = support()
    saa = pd.Series([0.40, 0.35, 0.25], index=s.assets)
    cfg = TAAConfig(target_return=0.50, band=0.05)
    maximum, _ = maximum_feasible_expected_return(s, saa, cfg)
    with pytest.raises(TargetInfeasibleError) as err:
        optimize_taa(s, saa, cfg)
    assert np.isclose(err.value.maximum, maximum)


def test_group_and_turnover_constraints():
    s = support()
    saa = pd.Series([0.40, 0.35, 0.25], index=s.assets)
    groups = {"Risky": GroupConstraint(("Equity", "Credit"), upper=0.67)}
    cfg = TAAConfig(target_return=0.0045, band=0.15, max_turnover=0.10, group_constraints=groups)
    out = optimize_taa(s, saa, cfg, previous_weights=saa)
    assert out.weights[["Equity", "Credit"]].sum() <= 0.670001
    assert out.diagnostics["turnover"] <= 0.100001


def test_decision_attribution_reconciles_es_and_expected_return():
    s = support(n=1200)
    saa = pd.Series([0.40, 0.35, 0.25], index=s.assets)
    out = optimize_taa(s, saa, TAAConfig(target_return=0.0055, band=0.10))
    table, summary = decision_attribution(s, saa, out.weights, alpha=0.05, target=0.0055)
    assert np.isclose(table["delta_expected_return_contribution"].sum(),
                      summary.loc["expected_return", "TAA_minus_SAA"], atol=1e-10)
    assert np.isclose(table["taa_component_ES_loss"].sum(),
                      summary.loc["ES_95_loss", "TAA"], atol=1e-10)


def test_feasibility_frontier_respects_turnover_cap():
    s = support()
    saa = pd.Series([0.40, 0.35, 0.25], index=s.assets)
    loose = TAAConfig(band=0.20)
    tight = TAAConfig(band=0.20, max_turnover=0.02)
    max_loose, _ = maximum_feasible_expected_return(s, saa, loose)
    max_tight, _ = maximum_feasible_expected_return(s, saa, tight, previous_weights=saa)
    assert max_tight <= max_loose + 1e-12
