"""
src/spillover.py
==========
Spillover Index construction functions for the WarShock-Spillover project.
Called by notebooks/03_spillover_index.ipynb.

Key functions:
    fit_var                 — fit VAR(p) and select lag order via AIC
    fevd_matrix             — compute H-step FEVD spillover matrix
    spillover_summary       — total / directional / net spillover from one window
    rolling_spillover       — rolling-window spillover index time series
    plot_spillover_ts       — Fig 6: total spillover index + war events
    plot_spillover_heatmaps — Fig 7: calm vs war spillover matrices
    plot_net_spillover      — Fig 8: net spillover bar chart (who transmits/receives)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS, add_constant

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR   = os.path.join(ROOT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT_DIR, "outputs", "tables")
PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"       : 150,
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
})

EQUITY_COLS = ["SP500", "STOXX600", "FTSE100", "DAX",
               "Nikkei", "HangSeng", "CSI300", "MSCI_EM"]

HIGH_INTENSITY = [
    "Israel-Lebanon 2006",
    "Gaza Cast Lead 2008",
    "Israel-Hamas 2023",
    "Israel-Iran 2024",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Core DY Spillover Functions
# ══════════════════════════════════════════════════════════════════════════════

def fit_var(vol_window: pd.DataFrame,
            maxlags: int = 10,
            exog: pd.DataFrame = None) -> tuple:
    """
    Fit a VAR(p) or VARX(p) model on a volatility window.

    Parameters
    ----------
    vol_window : pd.DataFrame         -- shape (T, N), endogenous vars, no NaN
    maxlags    : int                  -- maximum lag order for AIC selection
    exog       : pd.DataFrame or None -- exogenous regressors (e.g. crisis dummies).

    Returns
    -------
    result    : VARResults
    lag_order : int
    """
    if exog is not None:
        # ── Partial out exog via OLS on each endogenous column ────────────────
        exog_aligned = exog.reindex(vol_window.index).fillna(0)
        X = add_constant(exog_aligned)
        residuals = pd.DataFrame(index=vol_window.index)
        for col in vol_window.columns:
            ols_res = OLS(vol_window[col], X).fit()
            residuals[col] = ols_res.resid
        data_for_var = residuals
    else:
        data_for_var = vol_window

    model     = VAR(data_for_var)
    lag_order = model.select_order(maxlags=maxlags).aic
    lag_order = max(1, lag_order)
    result    = model.fit(lag_order)
    return result, lag_order


def fevd_matrix(var_result, H: int = 10) -> np.ndarray:
    """
    Compute the generalised forecast error variance decomposition (GFEVD)
    matrix following Diebold-Yilmaz (2012).

    Parameters
    ----------
    var_result : VARResults
    H          : int  — forecast horizon (default 10 days)

    Returns
    -------
    fevd : np.ndarray, shape (N, N) e.g. fevd[i, j] = % of market i's forecast error variance explained by shocks from market j
    """
    n      = var_result.neqs
    coefs  = var_result.coefs          # shape (p, N, N)
    sigma  = var_result.sigma_u        # residual covariance (N, N)

    # Build MA coefficient matrices Psi_h via recursion
    p    = len(coefs)
    psi  = [np.eye(n)]                 # Psi_0 = I
    for h in range(1, H + 1):
        ph = np.zeros((n, n))
        for j in range(min(h, p)):
            ph += psi[h - j - 1] @ coefs[j]
        psi.append(ph)

    # Generalised FEVD (Pesaran & Shin, 1998) — order-invariant
    sigma_diag = np.diag(sigma)
    fevd       = np.zeros((n, n))

    for i in range(n):
        denom = sum(psi[h][i, :] @ sigma @ psi[h][i, :] for h in range(H + 1))
        for j in range(n):
            ej      = np.zeros(n); ej[j] = 1.0
            numer   = sum((psi[h][i, :] @ sigma @ ej) ** 2
                          for h in range(H + 1)) / sigma_diag[j]
            fevd[i, j] = numer / denom

    # Row-normalise so each row sums to 1
    row_sums = fevd.sum(axis=1, keepdims=True)
    fevd     = fevd / row_sums
    return fevd


def spillover_summary(fevd: np.ndarray,
                      labels: list[str]) -> dict:
    """
    Compute total, directional, and net spillover indices from a FEVD matrix.

    Parameters
    ----------
    fevd   : np.ndarray (N, N)  — row-normalised FEVD matrix
    labels : list[str]          — market names

    Returns
    -------
    dict with keys:
        total       : float — Total Spillover Index (%)
        to          : pd.Series — spillover transmitted TO others
        from_       : pd.Series — spillover received FROM others
        net         : pd.Series — net = to - from_ (positive = net transmitter)
        matrix_df   : pd.DataFrame — full FEVD matrix with row/col labels
    """
    n      = len(labels)
    off_d  = fevd.copy()
    np.fill_diagonal(off_d, 0)

    total  = off_d.sum() / n * 100   # Total Spillover Index

    to_    = pd.Series(off_d.sum(axis=0) / n * 100, index=labels)  # col sums
    from_  = pd.Series(off_d.sum(axis=1) / n * 100, index=labels)  # row sums
    net    = to_ - from_

    df = pd.DataFrame(fevd * 100, index=labels, columns=labels)
    df["TO others"] = to_.values
    from_row = from_.values.tolist() + [total]
    df.loc["FROM others"] = from_row

    return {
        "total"    : total,
        "to"       : to_,
        "from_"    : from_,
        "net"      : net,
        "matrix_df": df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Rolling Spillover Index
# ══════════════════════════════════════════════════════════════════════════════

def rolling_spillover(vol_df: pd.DataFrame,
                      window: int  = 200,
                      H: int       = 10,
                      step: int    = 1,
                      maxlags: int = 5) -> pd.DataFrame:
    """
    Compute rolling Diebold-Yilmaz Total Spillover Index.

    Parameters
    ----------
    vol_df  : pd.DataFrame  — daily annualised volatility (T × N)
    window  : int           — rolling window size in trading days (default 200)
    H       : int           — FEVD forecast horizon (default 10)
    step    : int           — step between windows (default 1 = daily)
    maxlags : int           — max VAR lag order for AIC selection

    Returns
    -------
    pd.DataFrame with columns:
        total_spillover : rolling total spillover index (%)
        lag_order       : VAR lag selected each window
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    labels   = [c.replace("_vol", "") for c in vol_cols]
    data     = vol_df[vol_cols].dropna()

    records = []
    dates   = data.index

    print(f"  Rolling spillover: window={window}, H={H}, "
          f"n_windows≈{len(dates)//step}")

    for i in range(window, len(dates), step):
        win = data.iloc[i - window: i]

        # Skip windows with any NaN column
        if win.isnull().any().any():
            continue

        try:
            result, lag = fit_var(win, maxlags=maxlags)
            fevd        = fevd_matrix(result, H=H)
            summary     = spillover_summary(fevd, labels)
            records.append({
                "Date"           : dates[i],
                "total_spillover": summary["total"],
                "lag_order"      : lag,
            })
        except Exception as e:
            # Silently skip failed windows
            continue

        if len(records) % 200 == 0:
            print(f"    ... {len(records)} windows done "
                  f"(latest: {dates[i].date()})")

    result_df = pd.DataFrame(records).set_index("Date")
    print(f"  Done. {len(result_df)} spillover observations.")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Static Spillover (Full-Sample & Sub-Sample)
# ══════════════════════════════════════════════════════════════════════════════

def static_spillover(vol_df: pd.DataFrame,
                     label: str = "Full sample",
                     H: int = 10,
                     maxlags: int = 5,
                     exog_df: pd.DataFrame = None) -> dict:
    """
    Compute a single spillover summary over the entire vol_df period.
    Used for calm-period vs war-period comparison.

    Parameters
    ----------
    vol_df   : pd.DataFrame            -- daily annualised volatility (T x N)
    label    : str                     -- label for print output
    H        : int                     -- FEVD forecast horizon
    maxlags  : int                     -- max VAR lag order
    exog_df  : pd.DataFrame or None    -- exogenous regressors aligned to vol_df index.
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    labels   = [c.replace("_vol", "") for c in vol_cols]
    data     = vol_df[vol_cols].dropna()

    exog_aligned = exog_df.reindex(data.index).fillna(0) if exog_df is not None else None
    result, lag  = fit_var(data, maxlags=maxlags, exog=exog_aligned)
    method_tag   = "VARX" if exog_df is not None else "VAR"

    fevd     = fevd_matrix(result, H=H)
    summary  = spillover_summary(fevd, labels)
    summary["label"]     = label
    summary["lag_order"] = lag
    print(f"  [{label}] Total Spillover = {summary['total']:.2f}%  "
          f"({method_tag}, lag={lag})")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 4. Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _shade_wars(ax, events: pd.DataFrame, end_date: str) -> list:
    patches = []
    for _, row in events.iterrows():
        end   = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp(end_date)
        color = "#d62728" if row["event_name"] in HIGH_INTENSITY else "#ff7f0e"
        ax.axvspan(row["start_date"], end, color=color, alpha=0.13, zorder=0)
    patches = [
        mpatches.Patch(color="#d62728", alpha=0.4, label="High-intensity"),
        mpatches.Patch(color="#ff7f0e", alpha=0.4, label="Other / Russia-Ukraine"),
    ]
    return patches


def plot_spillover_ts(spillover_df: pd.DataFrame, events: pd.DataFrame, end_date: str = "2025-12-31", save: bool = True) -> None:
    """
    Fig 6: Rolling Total Spillover Index time series with war event shading.
    Core result figure for the paper.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(spillover_df.index, spillover_df["total_spillover"],
            color="#1f77b4", linewidth=1.0, alpha=0.9,
            label="Total Spillover Index (%)")

    # Smooth trend line (21-day MA)
    smooth = spillover_df["total_spillover"].rolling(21).mean()
    ax.plot(smooth.index, smooth,
            color="#d62728", linewidth=1.8, linestyle="--",
            label="21-day MA")

    war_patches = _shade_wars(ax, events, end_date)

    # Annotate war event starts
    for _, row in events.iterrows():
        if row["start_date"] >= spillover_df.index[0]:
            short = (row["event_name"]
                     .replace("Israel-", "IL-")
                     .replace(" War", "")
                     .replace(" Direct 2024", "-Iran 2024"))
            ax.axvline(row["start_date"], color="grey",
                       linewidth=0.8, linestyle=":", alpha=0.7)
            ax.text(row["start_date"],
                    spillover_df["total_spillover"].quantile(0.97),
                    short, fontsize=5.5, rotation=75,
                    ha="left", va="top", color="darkred", alpha=0.85)

    ax.set_title("Diebold-Yilmaz Total Spillover Index (200-Day Rolling Window)\n"
                 "with Middle East War Events",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Spillover Index (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles = [plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5,
                           label="Total Spillover Index"),
               plt.Line2D([0], [0], color="#d62728", linewidth=1.8,
                           linestyle="--", label="21-day MA")]
    ax.legend(handles=handles + war_patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig6_spillover_timeseries.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


def plot_spillover_heatmaps(summary_calm: dict, summary_war: dict, save: bool = True) -> None:
    """
    Fig 7: Side-by-side spillover matrices for calm vs war periods.
    Shows directional spillover between all market pairs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, summary, title in zip(
        axes,
        [summary_calm, summary_war],
        [f"Calm Periods  (Total={summary_calm['total']:.1f}%)",
         f"War Periods   (Total={summary_war['total']:.1f}%)"]
    ):
        n      = len(summary["to"])
        labels = list(summary["to"].index)
        # Use core N×N matrix only (exclude summary row/col)
        mat    = summary["matrix_df"].iloc[:n, :n].values.astype(float)

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=mat.max(), aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Shock FROM", fontsize=9)
        ax.set_ylabel("Impact ON", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")

        for r in range(n):
            for c in range(n):
                ax.text(c, r, f"{mat[r,c]:.1f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if mat[r, c] > mat.max() * 0.6 else "black")

        plt.colorbar(im, ax=ax, shrink=0.85, label="% variance explained")

    fig.suptitle("Directional Volatility Spillover Matrices: Calm vs War\n"
                 "(Row i, Col j = % of market i's forecast error from market j)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig7_spillover_heatmaps.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


def plot_net_spillover(summary_calm: dict, summary_war: dict, save: bool = True) -> None:
    """
    Fig 8: Net spillover bar chart (positive = net transmitter of risk).
    Compares calm vs war period net spillover per market.
    """
    labels     = list(summary_calm["net"].index)
    net_calm   = summary_calm["net"].values
    net_war    = summary_war["net"].values

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, net_calm, width,
                   color="#1f77b4", alpha=0.75, label="Calm periods")
    bars2 = ax.bar(x + width/2, net_war,  width,
                   color="#d62728", alpha=0.75, label="War periods")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Net Spillover (% — positive = net transmitter)")
    ax.set_title("Net Volatility Spillover by Market: Calm vs War Periods\n"
                 "(Positive = risk transmitter,  Negative = risk receiver)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig8_net_spillover.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_spillover(df: pd.DataFrame, filename: str = "spillover_index.xlsx") -> None:
    path = os.path.join(PROC_DIR, filename)
    df.to_excel(path)
    print(f"  Saved → {path}  {df.shape}")
