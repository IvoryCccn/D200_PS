"""
src/var_models.py
=================
VAR fitting, rolling forecast (multi-step), and evaluation metrics.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Constants ─────────────────────────────────────────────────────────
EQUITY_COLS = [
    "SP500_vol", "STOXX600_vol", "FTSE100_vol", "DAX_vol",
    "Nikkei_vol", "HangSeng_vol", "MSCI_EM_vol", "CSI300_vol",
]
DEFAULT_MAXLAGS = 10
HORIZONS = [1, 5, 10, 22]


# ═══════════════════════════════════════════════════════════════════════
# VAR — Fit
# ═══════════════════════════════════════════════════════════════════════

def fit_var(endog, maxlags=DEFAULT_MAXLAGS, verbose=True):
    """
    Fit a VAR model with AIC lag selection.

    Parameters
    ----------
    endog   : pd.DataFrame — endogenous vol series (train period only)
    maxlags : int
    verbose : bool

    Returns
    -------
    result       : VARResultsWrapper
    selected_lag : int
    """
    model = VAR(endog)
    lag_order = model.select_order(maxlags=maxlags)
    selected_lag = max(1, lag_order.aic)
    result = model.fit(selected_lag)

    if verbose:
        print(f"  AIC selected lag : {selected_lag}")
        print(f"  Log-likelihood   : {result.llf:.2f}")
        print(f"  AIC              : {result.aic:.2f}")
        print(f"  N obs used       : {result.nobs}")
    return result, selected_lag


# ═══════════════════════════════════════════════════════════════════════
# VAR — Multi-step rolling forecast  h ∈ {1, 5, 10, 22}
# ═══════════════════════════════════════════════════════════════════════

def rolling_forecast_var_multistep(vol_full, forecast_start, forecast_end,
                                    horizons=None, maxlags=DEFAULT_MAXLAGS,
                                    equity_cols=None, verbose_step=50):
    """
    Multi-step expanding-window VAR forecast.

    At each origin t, forecasts max(horizons) steps in one VAR call, then
    retains step h for each h in horizons (direct-horizon evaluation).
    Actual value used is vol at t+h (not rolling mean).

    Parameters
    ----------
    vol_full       : pd.DataFrame
    forecast_start : str
    forecast_end   : str
    horizons       : list[int]  default [1,5,10,22]
    maxlags        : int
    equity_cols    : list[str]
    verbose_step   : int

    Returns
    -------
    dict {h: (df_actual_h, df_forecast_h)}. Both DataFrames are indexed by origin date t.
    """
    if horizons is None:
        horizons = HORIZONS
    if equity_cols is None:
        equity_cols = EQUITY_COLS

    max_h   = max(horizons)
    origins = vol_full.loc[forecast_start:forecast_end].index
    n       = len(origins)

    store = {h: (np.full((n, len(equity_cols)), np.nan),
                 np.full((n, len(equity_cols)), np.nan))
             for h in horizons}

    print(f"VAR multi-step: {forecast_start} ~ {forecast_end} | "
          f"horizons={horizons} | origins={n}")

    for i, t in enumerate(origins):
        before = vol_full.index[vol_full.index < t]
        if len(before) < DEFAULT_MAXLAGS + max_h:
            continue
        endog_w = vol_full.loc[:before[-1], equity_cols]
        try:
            m      = VAR(endog_w)
            lag    = max(1, m.select_order(maxlags=maxlags).aic)
            res    = m.fit(lag)
            fc_all = res.forecast(endog_w.values[-lag:], steps=max_h)
        except Exception:
            fc_all = np.tile(endog_w.values[-1], (max_h, 1))

        for h in horizons:
            future = vol_full.index[vol_full.index > t]
            if len(future) < h:
                continue
            t_h = future[h - 1]
            store[h][0][i] = vol_full.loc[t_h, equity_cols].values
            store[h][1][i] = fc_all[h - 1]

        if (i + 1) % verbose_step == 0 or i == n - 1:
            print(f"  {i+1}/{n}  ({t.date()})")

    out = {}
    for h in horizons:
        acts, fcs = store[h]
        valid = ~np.isnan(acts[:, 0])
        out[h] = (
            pd.DataFrame(acts[valid], index=origins[valid], columns=equity_cols),
            pd.DataFrame(fcs[valid],  index=origins[valid], columns=equity_cols),
        )
    return out


# ═══════════════════════════════════════════════════════════════════════
# Metrics — single horizon
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(actual, forecast, label="", equity_cols=None):
    """
    MSE / MAE / RMSE / MAPE / DirAcc per market + AVERAGE row.

    Parameters
    ----------
    actual, forecast : pd.DataFrame — same shape
    label            : str
    equity_cols      : list[str]

    Returns
    -------
    pd.DataFrame [Market, MSE, MAE, RMSE, MAPE, Dir_Acc, Period]
    """
    if equity_cols is None:
        equity_cols = EQUITY_COLS

    rows = []
    for col in equity_cols:
        yt = actual[col].values
        yp = forecast[col].values
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt, yp = yt[mask], yp[mask]
        if len(yt) == 0:
            continue

        mse  = mean_squared_error(yt, yp)
        mae  = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mse)
        nz   = yt > 1e-8
        mape = (np.abs((yt[nz] - yp[nz]) / yt[nz]).mean() * 100
                if nz.sum() > 0 else np.nan)
        dir_acc = (np.sign(np.diff(yt)) == np.sign(yp[1:] - yt[:-1])).mean()

        rows.append({"Market": col.replace("_vol", ""),
                     "MSE": mse, "MAE": mae, "RMSE": rmse,
                     "MAPE": mape, "Dir_Acc": dir_acc, "Period": label})

    df  = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df[["MSE", "MAE", "RMSE", "MAPE", "Dir_Acc"]].mean()
    return pd.concat([df, pd.DataFrame([{
        "Market": "AVERAGE", "Period": label,
        **{k: agg[k] for k in ["MSE","MAE","RMSE","MAPE","Dir_Acc"]}
    }])], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Metrics — multi-horizon
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics_multistep(results_dict, label="", equity_cols=None):
    """
    Compute metrics for every horizon in a multi-step results dict.

    Parameters
    ----------
    results_dict : dict {h: (df_actual, df_forecast)}
    label        : str
    equity_cols  : list[str]

    Returns
    -------
    pd.DataFrame with columns [Market, MSE, MAE, RMSE, MAPE, Dir_Acc, Period, Horizon, Label]
    """
    parts = []
    for h, (act, fc) in sorted(results_dict.items()):
        df = compute_metrics(act, fc, label=f"h={h}", equity_cols=equity_cols)
        df["Horizon"] = h
        df["Label"]   = label
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

