# Scenario-Conditioned Tactical Allocation

A research-grade **Tactical Asset Allocation (TAA) decision layer** that sits
between strategic allocation, a repriced physical cross-asset scenario
distribution, and realized performance/risk attribution.

The central object is not an unconstrained "optimal portfolio". It is a tactical
deviation around a defensible Strategic Asset Allocation anchor:

```text
w_TAA,t = w_SAA,t + delta_t
```

with user-declared bands, target-return feasibility, tail-risk optimization,
turnover/cost controls, decision attribution, strict walk-forward validation, and
formal spanning tests against multiple SAA benchmarks.

## Why this repository exists

It is the orchestration layer above three separate research engines:

1. **multi-asset-scenario-engine**: physical joint scenarios and full horizon
   instrument repricing;
2. **state-conditioned-entropy-pooling**: optional state/view conditioning of the
   same support via probability reweighting;
3. **portfolio-performance-risk-attribution**: realized return, factor, volatility,
   tracking-error, Expected-Shortfall, drawdown and stress attribution.

This repository deliberately does not duplicate those engines. It owns the
portfolio decision problem connecting them.

## Core optimization

Given repriced scenario returns `R[s,i]`, scenario probabilities `p[s]`, a strategic
anchor `w_SAA`, and tactical bands, v0.1 solves the global optimum of

```text
min_w  weighted-CVaR_alpha(w)
       + lambda_active * ||w - w_SAA||_1
       + transaction_cost_rate * ||w - w_prev||_1
```

subject to

```text
1' w = 1
w_SAA,i - band_i <= w_i <= w_SAA,i + band_i
portfolio/sleeve bounds
optional turnover cap
E_p[R_p] >= target_return
```

The Rockafellar-Uryasev representation makes this a linear program, solved with
HiGHS through SciPy. Before solving, a separate feasibility LP computes the
maximum expected return reachable inside the tactical constraint set. An
unreachable target therefore fails explicitly with the attainable maximum.

**Important:** the return floor is a model-implied scenario constraint, not a
promise of realized future return.

## Strategic anchors

Built in:

- Equal Weight;
- Global Minimum Variance;
- Ledoit-Wolf GMV;
- Markowitz mean-variance;
- Equal Risk Contribution / Risk Parity;
- Michaud-style resampled efficiency.

The research design runs the *same* TAA layer across independently constructed
anchors. This tests whether tactical value is robust to SAA architecture rather
than being an artifact of one benchmark.

## Distribution-aware decision attribution

For SAA and TAA on the same scenario support, the package reports:

- expected-return contribution;
- Euler volatility contribution;
- exact weighted Expected-Shortfall contribution using partial tail mass at VaR;
- active tracking-error contribution;
- median and lower-quantile changes;
- probability of meeting the requested target;
- gross active weight and turnover.

This separates "what portfolio was chosen" from "why the scenario distribution
made the tilt attractive".

## Entropy Pooling integration

The scenario support is invariant. Conditioning replaces only the measure:

```text
physical support R
      + prior q
          |
          v
state / views / Entropy Pooling
          |
          v
posterior p
          |
          v
same TAA optimizer
```

Use `to_entropy_pooling_support()` and `apply_posterior()` to connect the companion
package without regenerating or repricing scenarios.

## Walk-forward design

`run_walk_forward()` enforces a strict one-step protocol:

```text
history strictly before t -> SAA
information available at t -> scenario support / posterior
                           -> TAA decision
realized row t             -> realized TAA and SAA returns
```

SAA can be refreshed slowly while TAA is rebalanced at every evaluation date.

## Spanning, not just Sharpe

`spanning_regression()` tests whether TAA performance is replicable by the SAA
benchmark set. It reports Newey-West/HAC alpha and a joint Wald test of

```text
alpha = 0
sum(beta_SAA) = 1
```

for one tested TAA return stream against multiple strategic benchmark returns.
Rejecting the joint restrictions is evidence against mean-variance spanning by the
benchmark set.

## Quick start

```bash
pip install -e '.[test]'
pytest
python examples/quickstart.py
```

Minimal use:

```python
from scenario_conditioned_taa import ScenarioSet, TAAConfig, optimize_taa

support = ScenarioSet(instrument_returns, probabilities=posterior_or_prior)
result = optimize_taa(
    support,
    saa_weights,
    TAAConfig(target_return=0.01, band=0.05, alpha=0.05),
)
print(result.weights)
print(result.metrics)
```

`instrument_returns` is intended to be the full-repricing output of the companion
Multi-Asset Scenario Engine.

## Research roadmap

### v0.1 — implemented here
- six SAA anchors;
- exact weighted-CVaR tactical LP;
- tactical bands, sleeve constraints, turnover and costs;
- expected-return feasibility frontier;
- distribution/decision attribution;
- strict single-anchor and multi-anchor walk-forward harnesses;
- net-of-cost realized TAA returns with gross/cost decomposition;
- HAC alpha and multi-SAA spanning test;
- instrument-to-sleeve aggregation bridge;
- thin Scenario Engine / Entropy Pooling / Attribution adapters.

### v0.2
- canonical 1M/3M support exported directly from the scenario engine;
- production ETF/sleeve universe alignment;
- transaction-cost calibration;
- richer multi-anchor experiment tables / plots;
- factor-adjusted spanning using the v1.1 attribution factor layer.

### v0.3
- state-conditioned prior / channel-view EP factorial inside the TAA walk-forward;
- distribution calibration diagnostics attached to portfolio decisions;
- frozen OOS protocol / lockbox.

### v0.4
- Bayesian feedback from ex-ante forecasts to ex-post calibration and view
  reliability, kept separate from portfolio-decision quality.

## Methodological non-goals

v0.1 intentionally does **not** pretend that a median or target-hit-probability
constraint is a linear program. Those are reported as diagnostics. Exact scenario
chance constraints would require mixed-integer optimization (or a clearly stated
approximation) and are deferred rather than hidden behind a heuristic solver.
