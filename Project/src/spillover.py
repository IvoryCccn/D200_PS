"""
src/spillover.py
================
Spillover Index construction and visualisation for the WarShock-Spillover project.
Called by notebooks/03_spillover_index.ipynb.

Key functions:
    fit_var               — fit VAR(p), lag selected via AIC
    fevd_matrix           — H-step generalised FEVD (Pesaran & Shin 1998)
    spillover_summary     — total / directional / net spillover from one FEVD
    static_spillover      — single-window spillover summary (training sub-samples)
    rolling_spillover     — rolling-window total spillover time series
    plot_spillover_heatmaps   — Fig 7: calm vs war FEVD matrices
    plot_net_spillover        — Fig 8: net spillover bar chart
    plot_spillover_ts         — Fig 9: rolling index with war-event shading
    plot_war_structural_break — Fig 10: regime means + Welch t-test
    save_spillover            — persist rolling series to Excel
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from statsmodels.tsa.api import VAR

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

# ── Constants (updated for new 8-index universe) ───────────────────────────────
EQUITY_COLS = ["SP500", "DAX", "CAC40", "FTSE100",
               "Nikkei", "KOSPI", "HangSeng", "SSE"]

# High-intensity events matching war_events.xlsx event_name exactly
HIGH_INTENSITY = [
    "Gaza War 2008",
    "Israel-Hamas War",
]

MARKET_COLORS = {
    "SP500"   : "#1f77b4",
    "DAX"     : "#ff7f0e",
    "CAC40"   : "#2ca02c",
    "FTSE100" : "#d62728",
    "Nikkei"  : "#9467bd",
    "KOSPI"   : "#8c564b",
    "HangSeng": "#e377c2",
    "SSE"     : "#7f7f7f",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Core DY Functions
# ══════════════════════════════════════════════════════════════════════════════

def fit_var(vol_window: pd.DataFrame, maxlags: int = 10) -> tuple:
    """
    Fit VAR(p) on vol_window, lag order selected by AIC.

    Returns
    -------
    result    : VARResults
    lag_order : int
    """
    model     = VAR(vol_window)
    lag_order = max(1, model.select_order(maxlags=maxlags).aic)
    return model.fit(lag_order), lag_order


def fevd_matrix(var_result, H: int = 10) -> np.ndarray:
    """
    Generalised FEVD (Pesaran & Shin 1998) — order-invariant.

    Returns
    -------
    fevd : np.ndarray (N, N)  row-normalised; fevd[i,j] = % of market i's
           H-step forecast error variance explained by shocks from market j.
    """
    n, coefs, sigma = var_result.neqs, var_result.coefs, var_result.sigma_u
    p = len(coefs)

    psi = [np.eye(n)]
    for h in range(1, H + 1):
        ph = sum(psi[h - j - 1] @ coefs[j] for j in range(min(h, p)))
        psi.append(ph)

    sigma_diag = np.diag(sigma)
    fevd = np.zeros((n, n))
    for i in range(n):
        denom = sum(psi[h][i, :] @ sigma @ psi[h][i, :] for h in range(H + 1))
        for j in range(n):
            ej    = np.zeros(n); ej[j] = 1.0
            numer = sum((psi[h][i, :] @ sigma @ ej) ** 2
                        for h in range(H + 1)) / sigma_diag[j]
            fevd[i, j] = numer / denom

    fevd /= fevd.sum(axis=1, keepdims=True)   # row-normalise
    return fevd


def spillover_summary(fevd: np.ndarray, labels: list) -> dict:
    """
    Compute total, directional, and net spillover from a FEVD matrix.

    Returns dict with keys: total, to, from_, net, matrix_df
    """
    n     = len(labels)
    off_d = fevd.copy(); np.fill_diagonal(off_d, 0)

    total = off_d.sum() / n * 100
    to_   = pd.Series(off_d.sum(axis=0) / n * 100, index=labels)
    from_ = pd.Series(off_d.sum(axis=1) / n * 100, index=labels)
    net   = to_ - from_

    df = pd.DataFrame(fevd * 100, index=labels, columns=labels)
    df["TO others"] = to_.values
    df.loc["FROM others"] = from_.values.tolist() + [total]

    return {"total": total, "to": to_, "from_": from_, "net": net, "matrix_df": df}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Static & Rolling Spillover
# ══════════════════════════════════════════════════════════════════════════════

def static_spillover(vol_df: pd.DataFrame,
                     label: str = "Full sample",
                     H: int = 10,
                     maxlags: int = 5) -> dict:
    """Single-window DY spillover summary (used for sub-sample comparisons)."""
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    labels   = [c.replace("_vol", "") for c in vol_cols]
    data     = vol_df[vol_cols].dropna()

    result, lag = fit_var(data, maxlags=maxlags)
    fevd        = fevd_matrix(result, H=H)
    summary     = spillover_summary(fevd, labels)
    summary["label"]     = label
    summary["lag_order"] = lag
    print(f"  [{label}] Total Spillover = {summary['total']:.2f}%  (VAR, lag={lag})")
    return summary


def rolling_spillover(vol_df: pd.DataFrame,
                      window: int  = 200,
                      H: int       = 10,
                      step: int    = 1,
                      maxlags: int = 5) -> pd.DataFrame:
    """
    Rolling Diebold-Yilmaz Total Spillover Index.
 
    Returns
    -------
    pd.DataFrame  columns: [total_spillover, lag_order]
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    labels   = [c.replace("_vol", "") for c in vol_cols]
    data     = vol_df[vol_cols].dropna()
    dates    = data.index
 
    print(f"  Rolling spillover: window={window}, H={H}, "
          f"n_windows≈{len(dates) // step}")
 
    records = []
    for i in range(window, len(dates), step):
        win = data.iloc[i - window: i]
        if win.isnull().any().any():
            continue
        try:
            result, lag = fit_var(win, maxlags=maxlags)
            fevd        = fevd_matrix(result, H=H)
            summary     = spillover_summary(fevd, labels)
            rec = {
                "Date"           : dates[i],
                "total_spillover": summary["total"],
                "lag_order"      : lag,
            }
            # Net spillover per market (positive = net transmitter)
            for mkt, val in summary["net"].items():
                rec[f"net_{mkt}"] = val
            records.append(rec)
        except Exception:
            continue
 
        if len(records) % 200 == 0:
            print(f"    ... {len(records)} windows done "
                  f"(latest: {dates[i].date()})")
 
    result_df = pd.DataFrame(records).set_index("Date")
    print(f"  Done. {len(result_df)} spillover observations.")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _shade_wars(ax, events: pd.DataFrame, end_date: str) -> list:
    """Helper: shade war event windows onto an axes."""
    for _, row in events.iterrows():
        end   = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp(end_date)
        color = "#d62728" if row["event_name"] in HIGH_INTENSITY else "#ff7f0e"
        ax.axvspan(row["start_date"], end, color=color, alpha=0.13, zorder=0)
    return [
        mpatches.Patch(color="#d62728", alpha=0.4, label="High-intensity"),
        mpatches.Patch(color="#ff7f0e", alpha=0.4, label="Other Middle East conflict"),
    ]


def plot_spillover_ts(spillover_df: pd.DataFrame,
                      events: pd.DataFrame,
                      end_date: str = "2025-12-31",
                      save: bool = True) -> None:
    """Fig 9: Rolling Total Spillover Index with war-event shading."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(spillover_df.index, spillover_df["total_spillover"],
            color="#1f77b4", linewidth=1.0, alpha=0.9,
            label="Total Spillover Index (%)")

    smooth = spillover_df["total_spillover"].rolling(21).mean()
    ax.plot(smooth.index, smooth,
            color="#d62728", linewidth=1.8, linestyle="--", label="21-day MA")

    war_patches = _shade_wars(ax, events, end_date)

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
        path = os.path.join(FIG_DIR, "fig9_spillover_timeseries.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


def plot_spillover_heatmaps(summary_calm: dict,
                            summary_war: dict,
                            save: bool = True) -> None:
    """Fig 7: Side-by-side FEVD matrices — Pure Calm vs War Only."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, summary, title in zip(
        axes,
        [summary_calm, summary_war],
        [f"Pure Calm  (Total={summary_calm['total']:.1f}%)",
         f"War Only   (Total={summary_war['total']:.1f}%)"]
    ):
        n      = len(summary["to"])
        labels = list(summary["to"].index)
        mat    = summary["matrix_df"].iloc[:n, :n].values.astype(float)

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=mat.max(), aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Shock FROM", fontsize=9)
        ax.set_ylabel("Impact ON",  fontsize=9)
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
        plt.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    plt.show()


def plot_net_spillover(summary_calm: dict,
                       summary_war: dict,
                       save: bool = True) -> None:
    """Fig 8: Net spillover bar chart — Calm vs War (positive = net transmitter)."""
    labels   = list(summary_calm["net"].index)
    x, width = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, summary_calm["net"].values, width,
           color="#1f77b4", alpha=0.75, label="Calm periods")
    ax.bar(x + width/2, summary_war["net"].values,  width,
           color="#d62728", alpha=0.75, label="War periods")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Net Spillover (%) — positive = net transmitter")
    ax.set_title("Net Volatility Spillover by Market: Calm vs War\n"
                 "(Positive = risk transmitter,  Negative = risk receiver)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig8_net_spillover.png")
        plt.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    plt.show()


def plot_war_structural_break(spill: pd.DataFrame,
                              save: bool = True) -> dict:
    """
    Fig 10: War-shock structural break analysis.
 
    Plots the rolling spillover index with:
      - Regime colour bands (Calm / War)
      - Regime-mean horizontal lines
      - Key war-event vertical markers
      - Welch t-test result annotated on chart
 
    Parameters
    ----------
    spill : rolling spillover DataFrame (output of rolling_spillover),
            must contain columns: total_spillover, mideast_war
            optionally: any_crisis (used to define Pure Calm if present)
 
    Returns
    -------
    dict  with keys: mean_calm, mean_war, t_stat, p_val, n_calm, n_war
    """
    # ── Regime masks ──────────────────────────────────────────────────────────
    # War-dominant: rolling window with >30% war days (war_dominant=1)
    # Pure Calm:    war_dominant=0 AND any_crisis=0
    # This alignment ensures spillover value and regime label cover the same window.
    war_col = "war_dominant" if "war_dominant" in spill.columns else "mideast_war"
    if "any_crisis" in spill.columns:
        is_calm = (spill[war_col] == 0) & (spill["any_crisis"] == 0)
    else:
        is_calm = (spill[war_col] == 0)
    is_war = (spill[war_col] == 1)
 
    calm_vals = spill.loc[is_calm, "total_spillover"].dropna()
    war_vals  = spill.loc[is_war,  "total_spillover"].dropna()
 
    mean_calm = calm_vals.mean()
    mean_war  = war_vals.mean()
    t_stat, p_val = scipy_stats.ttest_ind(calm_vals, war_vals, equal_var=False)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01
          else ("*" if p_val < 0.05 else "n.s."))
 
    print(f"War Structural Break — Welch t-test")
    print(f"  Calm : {mean_calm:.2f}%  (n={len(calm_vals)})")
    print(f"  War  : {mean_war:.2f}%  (n={len(war_vals)})")
    print(f"  Δ = {mean_war - mean_calm:+.2f} pp   "
          f"t = {t_stat:.3f}   p = {p_val:.4f}  {sig}")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 6))
 
    # Raw series + 63-day smooth
    ax.plot(spill.index, spill["total_spillover"],
            color="#aec7e8", linewidth=0.6, alpha=0.7, zorder=1)
    smooth = spill["total_spillover"].rolling(63, center=True).mean()
    ax.plot(smooth.index, smooth,
            color="#1f77b4", linewidth=1.8, label="63-day MA", zorder=2)
 
    # Regime shading
    y_max = spill["total_spillover"].max() * 1.05
    ax.fill_between(spill.index, 0, y_max,
                    where=is_calm, color="#2ca02c", alpha=0.08,
                    label="Calm regime", zorder=0)
    ax.fill_between(spill.index, 0, y_max,
                    where=is_war,  color="#ff7f0e", alpha=0.10,
                    label="War regime", zorder=0)
 
    # Regime mean lines — span the full calm/war index range
    calm_idx = spill.index[is_calm]
    war_idx  = spill.index[is_war]
    if len(calm_idx):
        ax.hlines(mean_calm, calm_idx[0], calm_idx[-1],
                  colors="#2ca02c", linewidths=1.6, linestyles="--",
                  label=f"Calm mean: {mean_calm:.1f}%")
    if len(war_idx):
        ax.hlines(mean_war, war_idx[0], war_idx[-1],
                  colors="#d62728", linewidths=1.6, linestyles="--",
                  label=f"War mean: {mean_war:.1f}%")
 
    # Key event vertical markers (actual events in war_events.xlsx)
    key_events = [
        ("2003-03-20", "Iraq War"),
        ("2008-12-27", "Gaza 2008"),
        ("2023-10-07", "Israel-Hamas 2023"),
    ]
    y_top = spill["total_spillover"].quantile(0.97)
    for date_str, label_text in key_events:
        dt = pd.Timestamp(date_str)
        if dt < spill.index[0] or dt > spill.index[-1]:
            continue
        ax.axvline(dt, color="darkred", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.text(dt, y_top, label_text, fontsize=7, rotation=0,
                ha="left", va="top", color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
 
    # Welch t-test annotation box
    ax.text(0.01, 0.97,
            f"Welch t-test: Calm vs War\n"
            f"Calm {mean_calm:.1f}%  →  War {mean_war:.1f}%\n"
            f"Δ = {mean_war - mean_calm:+.1f} pp   p {sig}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="grey", alpha=0.9))
 
    ax.set_title(
        "Fig 10 · DY(2012) Rolling Spillover Index — War-Shock Structural Analysis\n"
        "(200-day window, H=10 | Regime means + Welch t-test: Calm vs War)",
        fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Spillover Index (%)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
 
    if save:
        path = os.path.join(FIG_DIR, "fig10_war_structural_break.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()
 
    return dict(mean_calm=mean_calm, mean_war=mean_war,
                t_stat=t_stat, p_val=p_val,
                n_calm=len(calm_vals), n_war=len(war_vals))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_spillover(df: pd.DataFrame,
                   filename: str = "spillover_index.xlsx") -> None:
    path = os.path.join(PROC_DIR, filename)
    df.to_excel(path)
    print(f"  Saved → {path}  {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VIX-conditional war effect analysis
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_vix_conditional_spillover(
        spill_df: pd.DataFrame,
        all_vars_df: pd.DataFrame,
        war_col: str    = "mideast_war",
        vix_col: str    = "VIX",
        n_quantiles: int = 3,
        fig_filename: str = "fig10_vix_conditional_spillover.png",
        save: bool = True) -> pd.DataFrame:
    """
    VIX-conditional war-effect analysis on rolling total spillover.
 
    Splits the sample into VIX quantile buckets (low / mid / high by default),
    then within each bucket compares war vs non-war spillover means via
    Welch t-test. If the war premium persists across all VIX regimes, it is
    not driven by VIX-war collinearity.
 
    Parameters
    ----------
    spill_df    : rolling spillover DataFrame with war_col attached
    all_vars_df : all_variables_aligned.xlsx
    war_col     : column in spill_df flagging war days  (default 'mideast_war')
    vix_col     : VIX column in all_vars_df             (default 'VIX')
    n_quantiles : number of VIX quantile buckets        (default 3: low/mid/high)
    fig_filename: output figure name
    save        : save figure to FIG_DIR
 
    Returns
    -------
    results_df  : DataFrame with columns
                  VIX_quantile | VIX_mean | war_mean | nonwar_mean |
                  war_premium | war_premium_pct | t_stat | p_val |
                  n_war | n_nonwar | sig
    """
    # ── 1. Align VIX to spillover index ───────────────────────────────
    vix = all_vars_df[[vix_col]].rename(columns={vix_col: "VIX"})
    df  = spill_df[["total_spillover", war_col]].copy()
    df  = df.join(vix, how="left")
    df["VIX"] = df["VIX"].ffill(limit=5)
    df = df.dropna(subset=["total_spillover", "VIX"])
 
    # ── 2. Assign VIX quantile labels ─────────────────────────────────
    preset = {3: ["Low VIX", "Mid VIX", "High VIX"],
              4: ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]}
    qlabels = preset.get(n_quantiles,
                         [f"Q{i+1}" for i in range(n_quantiles)])
    df["vix_q"] = pd.qcut(df["VIX"], q=n_quantiles, labels=qlabels)
 
    # ── 3. Compute war premium per VIX bucket ─────────────────────────
    rows = []
    for ql in qlabels:
        sub     = df[df["vix_q"] == ql]
        war_v   = sub.loc[sub[war_col] == 1, "total_spillover"].dropna()
        nowar_v = sub.loc[sub[war_col] == 0, "total_spillover"].dropna()
        if len(war_v) < 5 or len(nowar_v) < 5:
            continue
        t, p = scipy_stats.ttest_ind(war_v, nowar_v, equal_var=False)
        wm, nm = war_v.mean(), nowar_v.mean()
        rows.append({
            "VIX_quantile"    : ql,
            "VIX_mean"        : round(sub["VIX"].mean(), 1),
            "war_mean"        : round(wm, 2),
            "nonwar_mean"     : round(nm, 2),
            "war_premium"     : round(wm - nm, 2),
            "war_premium_pct" : round((wm - nm) / nm * 100, 1),
            "t_stat"          : round(t, 3),
            "p_val"           : round(p, 4),
            "n_war"           : len(war_v),
            "n_nonwar"        : len(nowar_v),
            "sig"             : ("***" if p < 0.01 else
                                 "**"  if p < 0.05 else
                                 "*"   if p < 0.10 else "ns"),
        })
    results_df = pd.DataFrame(rows)
 
    # ── 4. Print table ─────────────────────────────────────────────────
    print("=" * 72)
    print("VIX-CONDITIONAL WAR EFFECT ON TOTAL SPILLOVER")
    print("=" * 72)
    print(f"{'VIX Regime':<12} {'VIX avg':>8} {'War':>7} {'Non-War':>9} "
          f"{'Premium':>9} {'Prem%':>7} {'t':>7} {'p':>7} {'sig':>4} "
          f"{'n_war':>6} {'n_nw':>6}")
    print("-" * 72)
    for _, r in results_df.iterrows():
        print(f"{r['VIX_quantile']:<12} {r['VIX_mean']:>8.1f} "
              f"{r['war_mean']:>7.2f} {r['nonwar_mean']:>9.2f} "
              f"{r['war_premium']:>9.2f} {r['war_premium_pct']:>6.1f}% "
              f"{r['t_stat']:>7.3f} {r['p_val']:>7.4f} {r['sig']:>4} "
              f"{r['n_war']:>6} {r['n_nonwar']:>6}")
 
    # ── 5. Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Fig 10 · VIX-Conditional War Effect on Total Spillover\n"
        "(Controls for VIX-war collinearity — war premium within each VIX regime)",
        fontsize=12, fontweight="bold")
 
    qlabels_avail = results_df["VIX_quantile"].tolist()
    x = np.arange(len(qlabels_avail))
    w = 0.35
 
    # Left: war vs non-war means by VIX quantile
    ax = axes[0]
    b1 = ax.bar(x - w/2, results_df["war_mean"],    w,
                label="War",     color="#d7191c", alpha=0.85)
    b2 = ax.bar(x + w/2, results_df["nonwar_mean"], w,
                label="Non-War", color="#4575b4", alpha=0.85)
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    for i, (_, r) in enumerate(results_df.iterrows()):
        if r["sig"] != "ns":
            ymax = max(r["war_mean"], r["nonwar_mean"])
            ax.text(i, ymax + 0.8, r["sig"],
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(qlabels_avail)
    ax.set_xlabel("VIX Regime")
    ax.set_ylabel("Total Spillover (%)")
    ax.set_title("Mean Spillover: War vs Non-War\nby VIX Quantile",
                 fontweight="bold")
    ax.legend(fontsize=9)
 
    # Right: war premium (pp) by VIX quantile
    ax = axes[1]
    bar_colors = ["#d7191c" if v > 0 else "#4575b4"
                  for v in results_df["war_premium"]]
    bars = ax.bar(x, results_df["war_premium"],
                  color=bar_colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%.2f pp", fontsize=8, padding=3)
    ax.axhline(0, color="gray", lw=1.0, linestyle="--")
    for i, (_, r) in enumerate(results_df.iterrows()):
        if r["sig"] != "ns":
            ypos = r["war_premium"] + (0.3 if r["war_premium"] >= 0 else -0.8)
            ax.text(i, ypos, r["sig"], ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(qlabels_avail)
    ax.set_xlabel("VIX Regime")
    ax.set_ylabel("War Premium (pp)")
    ax.set_title("War Spillover Premium (War − Non-War)\nby VIX Quantile",
                 fontweight="bold")
 
    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()
 
    return results_df