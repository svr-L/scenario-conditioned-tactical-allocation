import numpy as np
import pandas as pd
from scenario_conditioned_taa import ScenarioSet, aggregate_to_sleeves


def test_instrument_to_sleeve_aggregation():
    R = pd.DataFrame({"EQ1":[0.1,-0.1], "EQ2":[0.0,0.2], "UST":[0.02,0.01], "HY":[-0.03,0.04]})
    s = ScenarioSet(R)
    W = pd.DataFrame({"Equity":[0.5,0.5,0,0], "Rates":[0,0,1,0], "Credit":[0,0,0,1]}, index=R.columns)
    out = aggregate_to_sleeves(s, W)
    assert list(out.returns.columns) == ["Equity","Rates","Credit"]
    assert np.isclose(out.returns.loc[0,"Equity"], 0.05)
    assert np.allclose(out.probabilities, s.probabilities)
