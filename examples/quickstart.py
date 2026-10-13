import numpy as np
import pandas as pd

from scenario_conditioned_taa import (
    ScenarioSet, SAAConfig, TAAConfig, build_saa,
    optimize_taa, decision_attribution,
)

rng = np.random.default_rng(42)
assets = ["Equity", "Rates", "IG_Credit", "HY_Credit", "Cash"]

# Historical monthly returns used only to form the slow SAA anchor.
hist = pd.DataFrame(
    rng.normal([0.007, 0.0025, 0.0035, 0.0045, 0.0015],
               [0.040, 0.015, 0.018, 0.028, 0.002], size=(180, 5)),
    columns=assets,
)
saa = build_saa("erc", hist, SAAConfig(max_weight=0.45))

# Replace this synthetic block with repriced instrument_returns from the
# companion Multi-Asset Scenario Engine.
scenario_returns = pd.DataFrame(
    rng.normal([0.008, 0.002, 0.0035, 0.0040, 0.0015],
               [0.045, 0.015, 0.019, 0.030, 0.002], size=(4000, 5)),
    columns=assets,
)
support = ScenarioSet(scenario_returns)

cfg = TAAConfig(
    alpha=0.05,
    target_return=0.0030,
    band=0.07,
    max_weight=0.50,
    active_penalty=0.0005,
    transaction_cost_rate=0.0002,
)
result = optimize_taa(support, saa, cfg)
asset_attr, summary = decision_attribution(support, saa, result.weights, alpha=cfg.alpha,
                                           target=cfg.target_return)

print("\nSAA")
print(saa.round(4))
print("\nTAA")
print(result.weights.round(4))
print("\nDistribution metrics")
print(result.metrics.round(6))
print("\nDecision attribution")
print(asset_attr.round(6))
print("\nSAA -> TAA distribution change")
print(summary.round(6))
