from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2



def spanning_regression(test_returns: pd.Series, benchmark_returns: pd.DataFrame,
                        *, risk_free: pd.Series | float | None = None,
                        hac_lags: int | None = None, periods_per_year: int = 12) -> dict:
    """HAC spanning regression of one TAA return stream on multiple SAA benchmarks.

    Reports alpha and a Huberman-Kandel-style joint Wald test of
        alpha = 0 and sum(beta) = 1.
    Rejection means the tested return stream is not mean-variance spanned by the
    benchmark set under these regression restrictions.
    """
    y = pd.Series(test_returns, name="test").astype(float)
    X = pd.DataFrame(benchmark_returns).astype(float)
    data = pd.concat([y, X], axis=1).dropna()
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    if len(data) <= X.shape[1] + 3:
        raise ValueError("insufficient observations for spanning regression")
    if risk_free is not None:
        if np.isscalar(risk_free):
            rf = pd.Series(float(risk_free), index=data.index)
        else:
            rf = pd.Series(risk_free).reindex(data.index).astype(float)
        y = y - rf
        X = X.sub(rf, axis=0)
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc).fit()
    if hac_lags is None:
        # Small, transparent Newey-West default for monthly data; caller can override.
        hac_lags = max(1, int(np.floor(4 * (len(data) / 100.0) ** (2 / 9))))
    robust = model.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
    params = pd.Series(robust.params, index=Xc.columns)
    bse = pd.Series(robust.bse, index=Xc.columns)
    pvals = pd.Series(robust.pvalues, index=Xc.columns)

    # Without an explicit risk-free asset, Huberman-Kandel spanning requires
    # alpha=0 and sum(beta)=1. With returns expressed in excess of an available
    # risk-free asset, alpha=0 is the relevant restriction.
    k = len(params)
    if risk_free is None:
        R = np.zeros((2, k))
        R[0, 0] = 1.0
        R[1, 1:] = 1.0
        q = np.array([0.0, 1.0])
    else:
        R = np.zeros((1, k)); R[0, 0] = 1.0
        q = np.array([0.0])
    diff = R @ params.to_numpy() - q
    cov = np.asarray(robust.cov_params())
    V = R @ cov @ R.T
    stat = float(diff.T @ np.linalg.pinv(V) @ diff)
    df_wald = int(R.shape[0])
    p_joint = float(chi2.sf(stat, df=df_wald))

    resid = np.asarray(robust.resid)
    resid_vol = float(np.std(resid, ddof=k)) if len(resid) > k else np.nan
    alpha = float(params["const"])
    ann_alpha = alpha * periods_per_year
    appraisal = np.nan if not np.isfinite(resid_vol) or resid_vol <= 0 else alpha / resid_vol * np.sqrt(periods_per_year)
    return {
        "params": params,
        "standard_errors_HAC": bse,
        "pvalues_HAC": pvals,
        "alpha_per_period": alpha,
        "alpha_annualized": ann_alpha,
        "alpha_pvalue_HAC": float(pvals["const"]),
        "spanning_wald_chi2": stat,
        "spanning_wald_pvalue": p_joint,
        "spanning_wald_df": df_wald,
        "spanning_rejected_5pct": bool(p_joint < 0.05),
        "appraisal_ratio_annualized": float(appraisal) if np.isfinite(appraisal) else np.nan,
        "n_obs": int(len(data)),
        "hac_lags": int(hac_lags),
    }
