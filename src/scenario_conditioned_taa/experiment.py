from __future__ import annotations

import numpy as np
import pandas as pd

from .spanning import spanning_regression


def performance_summary(returns: pd.DataFrame, periods_per_year: int = 12, risk_free_per_period: float = 0.0) -> pd.DataFrame:
    """Compact realized performance table for OOS strategy comparisons."""
    x = pd.DataFrame(returns).astype(float)
    rows = {}
    for col in x:
        r = x[col].dropna()
        if r.empty:
            continue
        wealth = (1.0 + r).cumprod()
        ann_return = float(wealth.iloc[-1] ** (periods_per_year / len(r)) - 1.0) if (wealth > 0).all() else np.nan
        ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
        excess = r - risk_free_per_period
        sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)) if excess.std(ddof=1) > 0 else np.nan
        dd = wealth / wealth.cummax() - 1.0
        rows[col] = {
            "CAGR": ann_return,
            "annualized_volatility": ann_vol,
            "Sharpe": sharpe,
            "max_drawdown": float(dd.min()),
            "mean_return_annualized": float(r.mean() * periods_per_year),
            "n_obs": len(r),
        }
    return pd.DataFrame(rows).T


def tactical_value_table(taa_returns: pd.DataFrame, saa_returns: pd.DataFrame,
                         *, periods_per_year: int = 12, hac_lags: int | None = None) -> pd.DataFrame:
    """Own-anchor value add plus spanning against the complete SAA benchmark set."""
    taa = pd.DataFrame(taa_returns)
    saa = pd.DataFrame(saa_returns)
    common_methods = [c for c in taa.columns if c in saa.columns]
    rows = []
    for m in common_methods:
        joined = pd.concat([taa[m].rename("taa"), saa[m].rename("saa")], axis=1).dropna()
        own = spanning_regression(joined["taa"], joined[["saa"]], hac_lags=hac_lags, periods_per_year=periods_per_year)
        multi = spanning_regression(taa[m], saa, hac_lags=hac_lags, periods_per_year=periods_per_year)
        diff = joined["taa"] - joined["saa"]
        rows.append({
            "anchor": m,
            "net_value_add_ann_mean": float(diff.mean() * periods_per_year),
            "own_anchor_alpha_ann": own["alpha_annualized"],
            "own_anchor_alpha_p_HAC": own["alpha_pvalue_HAC"],
            "own_anchor_spanning_p": own["spanning_wald_pvalue"],
            "multi_SAA_alpha_ann": multi["alpha_annualized"],
            "multi_SAA_alpha_p_HAC": multi["alpha_pvalue_HAC"],
            "multi_SAA_spanning_p": multi["spanning_wald_pvalue"],
            "multi_SAA_spanning_rejected_5pct": multi["spanning_rejected_5pct"],
        })
    return pd.DataFrame(rows).set_index("anchor") if rows else pd.DataFrame()
