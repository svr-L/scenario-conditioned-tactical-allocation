# Integration with the companion engines

## 1. Multi-Asset Scenario Engine -> TAA

The scenario engine already constructs `instrument_returns` after full horizon
repricing. Pass that DataFrame directly:

```python
from scenario_conditioned_taa import from_scenario_engine_outputs

support = from_scenario_engine_outputs(
    instrument_returns,
    scenario_features=scenario_features,  # optional
    prior=prior_probabilities,             # optional; uniform by default
)
```

Rows must be the same repriced scenarios across all columns.

### Instrument-level outputs versus SAA sleeves

If strategic allocation is defined on broader sleeves, aggregate explicitly:

```python
from scenario_conditioned_taa import aggregate_to_sleeves

# rows = scenario-engine instruments, columns = strategic sleeves
W = pd.DataFrame({
    "Equity":    {...},
    "Rates":     {...},
    "IG_Credit": {...},
    "HY_Credit": {...},
}).fillna(0.0)

support_sleeves = aggregate_to_sleeves(support, W)
```

Each sleeve column must sum to one. The operation changes the investable return
columns but preserves scenario probabilities and state features.

## 2. Entropy Pooling -> TAA

The support is not regenerated. Convert the exact support to the companion EP
interface, solve views, and replace only the probability vector:

```python
from scenario_conditioned_taa import to_entropy_pooling_support, apply_posterior
from state_conditioned_entropy_pooling import EventView, solve_event_views

ep_support = to_entropy_pooling_support(support)
view = EventView("credit_widen", event_indicator, 0.60)
posterior = solve_event_views(ep_support.prior, [view]).probabilities
conditional_support = apply_posterior(support, posterior)
```

This preserves the key ownership rule:

```text
scenario engine: states + repricing
EP engine:       probabilities
TAA engine:      portfolio decision
```

## 3. TAA -> realized Performance/Risk Attribution

After a holding period, feed realized asset returns, TAA weights and the SAA anchor
as benchmark weights into the companion attribution engine:

```python
from scenario_conditioned_taa import run_external_attribution

engine = run_external_attribution(
    realized_returns,
    taa_weights,
    saa_weights,
    classifications,
    factor_returns,
)
```

The TAA package itself performs *ex-ante decision attribution* on the scenario
support (mean / volatility / ES / active TE). The companion attribution engine
owns *realized* performance, factor, risk, drawdown and stress attribution.

The supplied v1.1 development notebook extends the realized factor layer so the
same native-frequency factor decomposition is carried through expected return,
risk/tracking error and Expected Shortfall with an explicit residual. That logic
should remain in the attribution repository rather than be duplicated here.

## 4. Recommended first integrated research run

Use a common sleeve universe and a 1M horizon:

```text
Equity / Rates / IG Credit / HY Credit / Cash
```

1. estimate six slow SAA anchors from trailing history;
2. generate/reprice one-month physical scenarios;
3. optionally reweight the same support with EP;
4. solve identical TAA constraints around each anchor;
5. hold for one month and deduct transaction costs;
6. attribute the realized outcome;
7. repeat strictly point in time;
8. run own-anchor and multi-SAA spanning tests.
