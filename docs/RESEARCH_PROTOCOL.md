# Research protocol

## Core hypothesis

A scenario-conditioned tactical layer can add economically meaningful and
non-spanned value around independently constructed SAA anchors without relying on
unconstrained re-optimization of the whole portfolio.

## Strategic anchors

Primary: Equal Weight, GMV, Ledoit-Wolf GMV, Markowitz, ERC/Risk Parity, and
Michaud-style resampled efficiency.

The SAA is estimated from slow-moving historical information. Tactical decisions
are deviations from the strategic anchor, not replacements for it.

## Tactical decision rule

For each date t and SAA j:

    w_TAA[j,t] = w_SAA[j,t] + delta[j,t]

subject to user-declared bands, full investment, optional sleeve bounds, turnover
limits and a model-implied expected-return floor. The default objective is exact
probability-weighted CVaR plus L1 active and transaction-cost penalties.

A requested return is guaranteed only inside the model-implied scenario problem;
it is never described as a guarantee of realized future return.

## Validation ladder

1. distribution improvement vs own SAA;
2. realized net value-added vs own SAA;
3. HAC alpha vs own SAA;
4. spanning against the complete SAA benchmark set;
5. factor-adjusted alpha using the companion attribution engine;
6. stability across SAA anchors, tactical bands, horizons and stress regimes.

All hyperparameters used for the locked OOS evaluation should be frozen before
opening the holdout. The Entropy-Pooling project lockbox can be reused for this.

## V0.1 limitations by design

- Expected-return floors are linear constraints.
- Median, lower quantiles and target-hit probabilities are reported but are not
  exact chance constraints. Exact sample chance constraints require a MILP or a
  different approximation and belong in a later extension.
- Tracking error is attributed and reported; an exact quadratic TE cap would move
  the core problem from LP to QP/SOCP and is deferred deliberately.
