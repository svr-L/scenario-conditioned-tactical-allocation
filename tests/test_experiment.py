import numpy as np
import pandas as pd
from scenario_conditioned_taa import performance_summary, tactical_value_table


def test_experiment_tables_run():
    rng = np.random.default_rng(22)
    idx = pd.date_range("2000-01-31", periods=180, freq="ME")
    s1 = rng.normal(0.004, 0.02, len(idx)); s2 = rng.normal(0.003, 0.015, len(idx))
    saa = pd.DataFrame({"erc": s1, "gmv": s2}, index=idx)
    taa = pd.DataFrame({"erc": 0.9*s1 + 0.1*s2 + rng.normal(0.0005,0.003,len(idx)),
                        "gmv": 0.2*s1 + 0.8*s2 + rng.normal(0.0003,0.002,len(idx))}, index=idx)
    perf = performance_summary(taa)
    val = tactical_value_table(taa, saa)
    assert set(perf.index) == {"erc","gmv"}
    assert set(val.index) == {"erc","gmv"}
    assert "multi_SAA_spanning_p" in val.columns
