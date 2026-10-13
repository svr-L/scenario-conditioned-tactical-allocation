# Architecture

```text
Historical returns -----------------------> SAA Engine
                                             |
                                             v
                                       strategic anchor w_SAA
                                             |
Physical Cross-Asset Scenario Engine          |
  joint scenarios -> full horizon repricing   |
                   |                         |
                   v                         |
             ScenarioSet R[S,N] <------------+
                   |
                   +--> optional Flexible Probabilities / Entropy Pooling
                   |        same support, q -> p
                   v
             Tactical CVaR Optimizer
             w_TAA = w_SAA + delta_w
                   |
                   +--> Decision Attribution (mean / vol / ES / TE)
                   |
                   +--> realized holding-period return
                              |
                              v
             Performance & Risk Attribution Engine
                              |
                              v
                 Walk-forward / Spanning tests
```

## Ownership boundaries

- **multi-asset-scenario-engine** owns physical support, market-state reconstruction,
  full horizon repricing, and baseline risk validation.
- **state-conditioned-entropy-pooling** owns probability conditioning on a fixed
  support, diagnostics, and conditional-tail validation.
- **portfolio-performance-risk-attribution** owns realized performance/risk/factor
  attribution and reconciliation.
- **scenario-conditioned-tactical-allocation** owns strategic anchors, tactical
  constraints, portfolio decision optimization, decision attribution, and
  cross-anchor OOS/spanning validation.

No upstream engine is copied into this repository.

## Universe bridge: instruments -> strategic sleeves

The current scenario repricer can emit instrument-level returns while SAA is often
specified on broader sleeves. `aggregate_to_sleeves()` applies an explicit,
user-auditable within-sleeve weight matrix to every repriced scenario before TAA.
This avoids silently mixing unrelated proxy universes and keeps the same scenario
probabilities/features.
