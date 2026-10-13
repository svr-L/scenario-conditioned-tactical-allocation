from __future__ import annotations

import numpy as np
import pandas as pd

from .types import ScenarioSet


def from_scenario_engine_outputs(instrument_returns: pd.DataFrame, *, prior=None,
                                 scenario_features: pd.DataFrame | None = None,
                                 metadata: dict | None = None) -> ScenarioSet:
    """Thin adapter for `multi-asset-scenario-engine` repriced instrument returns."""
    return ScenarioSet(instrument_returns, prior, scenario_features, metadata or {"source": "multi_asset_scenario_engine"})


def apply_posterior(scenarios: ScenarioSet, posterior_probabilities, *, label: str = "entropy_pooling") -> ScenarioSet:
    """Replace the measure while preserving the exact scenario support/repricing."""
    return scenarios.with_probabilities(posterior_probabilities, conditioning=label)


def to_entropy_pooling_support(scenarios: ScenarioSet):
    """Create the companion EP `ScenarioSupport` if that package is installed."""
    try:
        from state_conditioned_entropy_pooling import ScenarioSupport
    except ImportError as exc:
        raise ImportError("Install the companion state-conditioned-entropy-pooling package to use this adapter") from exc
    features = scenarios.features
    if features is None:
        features = pd.DataFrame(index=scenarios.returns.index)
    return ScenarioSupport(
        instrument_returns=scenarios.returns,
        scenario_features=features,
        prior=np.asarray(scenarios.probabilities),
    )


def run_external_attribution(realized_returns: pd.DataFrame, taa_weights: pd.Series,
                             saa_weights: pd.Series, classifications: pd.DataFrame,
                             factor_returns: pd.DataFrame | None = None):
    """Bridge to the companion performance/risk attribution package when installed."""
    try:
        from portfolio_attribution_engine import PortfolioAttributionEngine
    except ImportError as exc:
        raise ImportError("Install/fix the companion portfolio_attribution_engine package to use this adapter") from exc
    return PortfolioAttributionEngine(
        returns=realized_returns,
        portfolio_weights=taa_weights,
        benchmark_weights=saa_weights,
        classifications=classifications,
        factor_returns=factor_returns,
    )


def aggregate_to_sleeves(scenarios: ScenarioSet, sleeve_weights: pd.DataFrame) -> ScenarioSet:
    """Aggregate instrument-level scenario returns into investable sleeves.

    `sleeve_weights` has instrument names on rows and sleeve names on columns. Each
    sleeve column must sum to one (within tolerance); zeros are allowed. This is
    useful when the full repricer emits security-level returns while the strategic
    allocation is defined on Equity/Rates/Credit/etc. sleeves.
    """
    W = pd.DataFrame(sleeve_weights).astype(float)
    unknown = set(W.index) - set(scenarios.assets)
    if unknown:
        raise ValueError(f"sleeve map contains unknown instruments: {sorted(unknown)}")
    if W.columns.has_duplicates or W.empty:
        raise ValueError("sleeve_weights must contain unique non-empty sleeve columns")
    if not np.isfinite(W.to_numpy()).all():
        raise ValueError("sleeve weights contain non-finite values")
    if (W < -1e-12).any().any():
        raise ValueError("sleeve aggregation weights must be non-negative")
    sums = W.sum(axis=0)
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-8):
        raise ValueError(f"each sleeve must sum to one; got {sums.to_dict()}")
    R = scenarios.returns.reindex(columns=W.index).to_numpy() @ W.to_numpy()
    sleeve_returns = pd.DataFrame(R, index=scenarios.returns.index, columns=W.columns)
    md = dict(scenarios.metadata); md["aggregation"] = "instrument_to_sleeve"
    return ScenarioSet(sleeve_returns, scenarios.probabilities, scenarios.features, md)
