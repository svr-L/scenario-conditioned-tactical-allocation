import numpy as np
import pandas as pd

from scenario_conditioned_taa import SAAConfig, build_saa


def sample_returns(seed=1, n=180):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0025, 0.0003, 0.0001], [0.0003, 0.0012, 0.00015], [0.0001, 0.00015, 0.0007]])
    mu = np.array([0.007, 0.004, 0.003])
    return pd.DataFrame(rng.multivariate_normal(mu, cov, size=n), columns=["Equity", "Rates", "Credit"])


def test_all_saa_methods_are_fully_invested_and_bounded():
    r = sample_returns()
    cfg = SAAConfig(max_weight=0.80, michaud_resamples=20, seed=9)
    for method in ["equal_weight", "gmv", "ledoit_wolf", "markowitz", "erc", "michaud"]:
        w = build_saa(method, r, cfg)
        assert np.isclose(w.sum(), 1.0, atol=1e-7)
        assert (w >= -1e-9).all()
        assert (w <= 0.800001).all()


def test_erc_risk_contributions_are_close():
    r = sample_returns(n=500)
    w = build_saa("erc", r, SAAConfig())
    cov = r.cov().to_numpy()
    rc_var = w.to_numpy() * (cov @ w.to_numpy())
    shares = rc_var / rc_var.sum()
    assert np.max(np.abs(shares - 1 / len(w))) < 0.02
