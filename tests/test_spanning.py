import numpy as np
import pandas as pd

from scenario_conditioned_taa import spanning_regression


def test_spanning_restrictions_not_rejected_for_convex_combo():
    rng = np.random.default_rng(123)
    n = 240
    b1 = rng.normal(0.006, 0.03, n)
    b2 = rng.normal(0.003, 0.015, n)
    # Population-spanned return: convex combination plus tiny independent noise.
    # The noise avoids a singular HAC covariance from a literally perfect fit.
    y = 0.6*b1 + 0.4*b2 + rng.normal(0.0, 1e-5, n)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    res = spanning_regression(pd.Series(y, index=idx), pd.DataFrame({"B1": b1, "B2": b2}, index=idx))
    assert res["spanning_wald_pvalue"] > 0.05
    assert abs(res["alpha_per_period"]) < 5e-6
