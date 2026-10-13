# Changelog

## 0.1.0

- Added six strategic anchors: EW, GMV, Ledoit-Wolf GMV, Markowitz, ERC/Risk
  Parity, and Michaud-style resampled efficiency.
- Added exact weighted Rockafellar-Uryasev CVaR tactical LP around SAA anchors.
- Added user tactical bands, global bounds, sleeve/group constraints, turnover cap,
  active regularization, transaction costs and expected-return floor.
- Added target-feasibility LP respecting hard turnover and sleeve constraints.
- Added distribution metrics: mean, volatility, median, lower quantile, VaR, ES,
  target-hit probability and expected target shortfall.
- Added decision attribution for expected return, Euler volatility, exact ES and
  active tracking error.
- Added instrument-to-sleeve aggregation.
- Added strict single-anchor and multi-anchor walk-forward harnesses with net and
  gross returns plus transaction-cost decomposition.
- Added performance tables, HAC alpha and mean-variance spanning Wald tests.
- Added thin adapters for the physical scenario engine, state-conditioned Entropy
  Pooling and realized attribution engine.
