import numpy as np
import pandas as pd

from scenario_conditioned_taa import ScenarioSet, TAAConfig, SAAConfig, run_walk_forward


def test_walkforward_uses_only_pre_t_history_and_runs():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    realized = pd.DataFrame(rng.normal([0.006, 0.003, 0.004], [0.035, 0.012, 0.020], size=(72, 3)),
                            index=idx, columns=["Equity", "Rates", "Credit"])

    def provider(date, history):
        assert history.index.max() < date
        mu = history.tail(24).mean().to_numpy()
        cov = history.tail(24).cov().to_numpy()
        scen = rng.multivariate_normal(mu, cov, size=500)
        return ScenarioSet(pd.DataFrame(scen, columns=realized.columns))

    out = run_walk_forward(realized, provider, saa_method="erc", min_history=36,
                           saa_rebalance_every=12,
                           saa_config=SAAConfig(),
                           taa_config=TAAConfig(target_return=None, band=0.05))
    assert len(out.taa_returns) == 36
    assert out.taa_weights.shape == (36, 3)
    assert np.allclose(out.taa_weights.sum(axis=1), 1.0)


def test_walkforward_returns_are_net_of_costs():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2018-01-31", periods=48, freq="ME")
    realized = pd.DataFrame(rng.normal([0.006, 0.003, 0.004], [0.03, 0.012, 0.018], size=(48,3)), index=idx, columns=["Equity","Rates","Credit"])
    def provider(date, history):
        scen = rng.multivariate_normal(history.tail(24).mean(), history.tail(24).cov(), size=300)
        return ScenarioSet(pd.DataFrame(scen, columns=realized.columns))
    out = run_walk_forward(realized, provider, min_history=24, saa_method="erc",
                           taa_config=TAAConfig(band=0.05, transaction_cost_rate=0.001))
    assert np.allclose((out.taa_gross_returns - out.transaction_costs).values, out.taa_returns.values)
