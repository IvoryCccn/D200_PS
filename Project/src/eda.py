"""
src/eda.py
==========
EDA functions for the WarShock-Spillover project.
Called by notebooks/02_eda.ipynb.

Functions:
    plot_volatility_timeseries   — Fig 1: rolling vol + war shading
    plot_return_distributions    — Fig 2: return histograms vs normal
    plot_correlation_heatmaps    — Fig 3: calm vs war correlation matrices
    plot_event_windows           — Fig 4: cumulative returns around events
    plot_macro_war_linkage       — Fig 5: Oil / Gold / VIX vs war events
    run_adf_tests                — Table: stationarity of vol series
    war_vs_calm_stats            — Table: descriptive stats by regime
    plot_crisis_decomposition    - Fig 5a: 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# ── Output path ────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR     = os.path.join(ROOT_DIR, "outputs", "figures")
TABLE_DIR   = os.path.join(ROOT_DIR, "outputs", "tables")
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.size"       : 10,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
})

EQUITY_COLS = ["SP500", "STOXX600", "FTSE100", "DAX",
               "Nikkei", "HangSeng", "CSI300", "MSCI_EM"]

HIGH_INTENSITY = [
    "Israel-Lebanon 2006",
    "Gaza Cast Lead 2008",
    "Israel-Hamas 2023",
    "Israel-Iran 2024",
]

MARKET_COLORS = {
    "SP500"   : "#1f77b4",
    "STOXX600": "#ff7f0e",
    "FTSE100" : "#2ca02c",
    "DAX"     : "#d62728",
    "Nikkei"  : "#9467bd",
    "HangSeng": "#8c564b",
    "CSI300"  : "#e377c2",
    "MSCI_EM" : "#7f7f7f",
}


# ── Helper: shade war periods on an axis ──────────────────────────────────────
def _shade_wars(ax, events: pd.DataFrame, end_date: str) -> list:
    """Add war period shading to an existing axis. Returns legend patches."""
    patches = []
    for _, row in events.iterrows():
        end   = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp(end_date)
        color = "#d62728" if row["event_name"] in HIGH_INTENSITY else "#ff7f0e"
        alpha = 0.13
        ax.axvspan(row["start_date"], end, color=color, alpha=alpha, zorder=0)

    patches = [
        mpatches.Patch(color="#d62728", alpha=0.35, label="High-intensity war"),
        mpatches.Patch(color="#ff7f0e", alpha=0.35, label="Other conflict / Russia-Ukraine"),
    ]
    return patches


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 · Rolling Volatility Time-Series
# ══════════════════════════════════════════════════════════════════════════════
def plot_volatility_timeseries(vol_df: pd.DataFrame,
                               events: pd.DataFrame,
                               end_date: str = "2025-12-31",
                               save: bool = True) -> None:
    """
    Line chart of annualised 21-day rolling volatility for all equity indices,
    with war-period shading.
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in vol_cols:
        mkt = col.replace("_vol", "")
        ax.plot(vol_df.index, vol_df[col],
                linewidth=0.9, alpha=0.85,
                color=MARKET_COLORS.get(mkt, "grey"),
                label=mkt)

    war_patches = _shade_wars(ax, events, end_date)

    ax.set_title("Annualised 21-Day Rolling Volatility — Global Equity Indices\n"
                 "with Middle East War Events", fontsize=12, fontweight="bold")
    ax.set_ylabel("Volatility (annualised)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    market_handles = [plt.Line2D([0], [0], color=MARKET_COLORS.get(c.replace("_vol",""), "grey"),
                                  linewidth=1.5, label=c.replace("_vol",""))
                      for c in vol_cols]
    ax.legend(handles=market_handles + war_patches,
              fontsize=7, ncol=5, loc="upper right")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig1_volatility_timeseries.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 · Return Distributions
# ══════════════════════════════════════════════════════════════════════════════
def plot_return_distributions(returns_df: pd.DataFrame,
                               save: bool = True) -> None:
    """
    Histogram of daily log returns for each equity index,
    overlaid with a fitted normal distribution curve.
    Demonstrates fat tails → motivates deep learning over linear VAR.
    """
    ret_cols = [c + "_ret" for c in EQUITY_COLS if c + "_ret" in returns_df.columns]
    n        = len(ret_cols)
    ncols    = 4
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(ret_cols):
        ax  = axes[i]
        mkt = col.replace("_ret", "")
        data = returns_df[col].dropna()

        ax.hist(data, bins=80, density=True, color=MARKET_COLORS.get(mkt, "steelblue"),
                alpha=0.6, edgecolor="none")

        # Fitted normal
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma),
                color="black", linewidth=1.5, linestyle="--", label="Normal fit")

        # Annotations
        kurt = data.kurt()
        ax.set_title(f"{mkt}\nKurt={kurt:.2f}", fontsize=9)
        ax.set_xlabel("Log Return", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Daily Log Return Distributions — Fat Tails vs Normal\n"
                 "(Excess kurtosis motivates non-linear deep learning models)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig2_return_distributions.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 · Correlation Heatmaps: Calm vs War
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmaps(returns_df: pd.DataFrame,
                               war_dummy: pd.DataFrame,
                               save: bool = True) -> None:
    """
    Side-by-side correlation heatmaps of equity returns:
    left = calm periods, right = mideast war periods.
    """
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ret_cols = [c + "_ret" for c in EQUITY_COLS if c + "_ret" in returns_df.columns]
    labels   = [c.replace("_ret", "") for c in ret_cols]

    calm_mask = war_dummy["mideast_war"] == 0
    war_mask  = war_dummy["mideast_war"] == 1

    corr_calm = returns_df.loc[calm_mask, ret_cols].corr()
    corr_war  = returns_df.loc[war_mask,  ret_cols].corr()

    cmap = plt.cm.RdYlGn
    vmin, vmax = -1, 1

    # Reserve space for colorbar on the right via gridspec
    fig = plt.figure(figsize=(15, 5.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax_calm = fig.add_subplot(gs[0])
    ax_war  = fig.add_subplot(gs[1])
    ax_cb   = fig.add_subplot(gs[2])   # dedicated colorbar axes

    for ax, corr, title in zip(
        [ax_calm, ax_war],
        [corr_calm, corr_war],
        ["Calm Periods", "Middle East War Periods"]
    ):
        im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for r in range(len(labels)):
            for c in range(len(labels)):
                val = corr.values[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(val) > 0.6 else "black")

    # Colorbar in its own dedicated axes — no overlap
    fig.colorbar(im, cax=ax_cb, label="Pearson Correlation")

    fig.suptitle("Equity Return Correlations: Calm vs Middle East War Periods\n"
                 "(Higher correlations during war → stronger risk contagion)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig3_correlation_heatmaps.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 · Event Study Windows
# ══════════════════════════════════════════════════════════════════════════════
def plot_event_windows(returns_df: pd.DataFrame,
                       events: pd.DataFrame,
                       pre: int = 20,
                       post: int = 60,
                       save: bool = True) -> None:
    """
    Cumulative returns in [-pre, +post] trading days around each war event start.
    Compares Middle East wars vs Russia-Ukraine as a reference.
    """
    ret_cols  = [c + "_ret" for c in EQUITY_COLS if c + "_ret" in returns_df.columns]
    avg_ret   = returns_df[ret_cols].mean(axis=1)   # equal-weight global average

    mideast_events   = events[events["region"] == "Middle East"]
    russia_events    = events[events["event_name"].str.contains("Russia", na=False)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    def _plot_group(ax, group, title, color):
        for _, row in group.iterrows():
            start_dt = row["start_date"]
            if start_dt not in avg_ret.index:
                start_dt = avg_ret.index[avg_ret.index.searchsorted(start_dt)]
            idx = avg_ret.index.get_loc(start_dt)
            window_idx = avg_ret.index[max(0, idx - pre): idx + post + 1]
            window_ret = avg_ret.loc[window_idx]
            cum_ret    = (1 + window_ret).cumprod() - 1
            x = range(-min(pre, idx), len(cum_ret) - min(pre, idx))
            ax.plot(list(x), cum_ret.values * 100,
                    linewidth=1.2, alpha=0.75, label=row["event_name"])

        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Event start")
        ax.axhline(0, color="grey",  linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trading days relative to event start")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(fontsize=6.5, loc="lower left")

    _plot_group(axes[0], mideast_events,
                "Middle East War Events\n(Equal-weight global index)", "#d62728")
    _plot_group(axes[1], russia_events,
                "Russia-Ukraine War (Reference)\n(Equal-weight global index)", "#1f77b4")

    fig.suptitle(f"Event Study: Cumulative Returns [{-pre}, +{post}] Trading Days",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig4_event_windows.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 · Oil / Gold / VIX Linkage During War Periods
# ══════════════════════════════════════════════════════════════════════════════
def plot_macro_war_linkage(combined: pd.DataFrame,
                            events: pd.DataFrame,
                            end_date: str = "2025-12-31",
                            save: bool = True) -> None:
    """
    Three-panel time series: Brent oil, Gold, VIX — with war shading.
    Highlights the anomalous gold decline during Israel-Iran 2024.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    series_cfg = [
        ("Brent", "Brent Crude Oil ($/barrel)", "#8B4513"),
        ("Gold",  "Gold Price ($/oz)",           "#DAA520"),
        ("VIX",   "VIX — Fear Index",            "#4B0082"),
    ]

    for ax, (col, ylabel, color) in zip(axes, series_cfg):
        if col not in combined.columns:
            ax.set_visible(False)
            continue
        ax.plot(combined.index, combined[col],
                color=color, linewidth=0.9, alpha=0.9)
        war_patches = _shade_wars(ax, events, end_date)
        ax.set_ylabel(ylabel, fontsize=9)

        # Annotate key events
        for _, row in events.iterrows():
            ax.axvline(row["start_date"], color="grey",
                       linewidth=0.6, linestyle=":", alpha=0.7)

    # Label war events on top panel
    for _, row in events.iterrows():
        short = row["event_name"].replace("Israel-", "IL-").replace("Gaza ", "")
        axes[0].text(row["start_date"], axes[0].get_ylim()[1] * 0.92,
                     short, fontsize=5.5, rotation=75,
                     ha="left", va="top", color="darkred", alpha=0.8)

    axes[0].legend(handles=war_patches, fontsize=7, loc="upper left")
    axes[0].set_title("Oil, Gold & VIX During Middle East War Periods\n"
                       "(Note: anomalous Gold decline during Israel-Iran 2024)",
                       fontsize=11, fontweight="bold")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig5_macro_war_linkage.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 · ADF Stationarity Tests
# ══════════════════════════════════════════════════════════════════════════════
def run_adf_tests(vol_df: pd.DataFrame,
                  save: bool = True) -> pd.DataFrame:
    """
    ADF unit root test on each equity volatility series.
    Required before VAR modelling.
    Returns a summary DataFrame.
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    results  = []

    for col in vol_cols:
        series = vol_df[col].dropna()
        adf_stat, p_val, _, _, crit, _ = adfuller(series, autolag="AIC")
        results.append({
            "Series"       : col.replace("_vol", ""),
            "ADF Statistic": round(adf_stat, 4),
            "p-value"      : round(p_val,    4),
            "1% Critical"  : round(crit["1%"],  4),
            "5% Critical"  : round(crit["5%"],  4),
            "Stationary?"  : "✅ Yes" if p_val < 0.05 else "❌ No",
        })

    df = pd.DataFrame(results).set_index("Series")
    print("\n[ADF Test Results — Volatility Series]")
    print(df.to_string())

    if save:
        path = os.path.join(TABLE_DIR, "table_adf_tests.xlsx")
        df.to_excel(path)
        print(f"\n  Saved → {path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Table 2 · War vs Calm Descriptive Statistics (crisis-controlled)
# ══════════════════════════════════════════════════════════════════════════════
def war_vs_calm_stats(vol_df: pd.DataFrame,
                      war_dummy: pd.DataFrame,
                      save: bool = True) -> pd.DataFrame:
    """
    Compare mean volatility across four regimes:

        Regime A — Pure calm   : mideast_war=0 AND any_crisis=0
        Regime B — War only    : mideast_war=1 AND any_crisis=0
        Regime C — Crisis only : mideast_war=0 AND any_crisis=1
        Regime D — War+Crisis  : mideast_war=1 AND any_crisis=1

    The key comparison for the paper is B vs A (war effect net of crises).
    Crises (GFC + COVID) are controlled for so their outsized volatility
    does not suppress the apparent war effect.

    Requires war_dummy to contain columns: mideast_war, any_crisis.
    (Produced by data_loader.build_war_dummies() v2+)
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]

    # ── Check crisis columns exist (backward-compat) ────────────────────────
    has_crisis = "any_crisis" in war_dummy.columns
    if not has_crisis:
        print("  [WARN] war_dummy has no 'any_crisis' column")

    aligned = vol_df[vol_cols].join(war_dummy, how="inner")

    rows = []
    for col in vol_cols:
        mkt = col.replace("_vol", "")

        if has_crisis:
            # Four-regime decomposition
            pure_calm   = aligned.loc[(aligned["mideast_war"] == 0) &
                                      (aligned["any_crisis"]  == 0), col].dropna()
            war_only    = aligned.loc[(aligned["miany_crisis"]  == 0), col].dropna()
            crisis_only = aligned.loc[(aligned["mideast_war"] == 0) &
                                      (aligned["any_crisis"]  == 1), col].dropna()
            war_crisis  = aligned.loc[(aligned["mideast_war"] == 1) &
                                      (aligned["any_crisis"]  == 1), col].dropna()
        else:
            # Fallback: original two-regime split
            pure_calm   = aligned.loc[aligned["mideast_war"] == 0, col].dropna()
            war_only    = aligned.loc[aligned["mideast_war"] == 1, col].dropna()
            crisis_only = pd.Series(dtype=float)
            war_crisis  = pd.Series(dtype=float)

        hi = aligned.loc[aligned.get("high_intensity", pd.Series(0,
             index=aligned.index)) == 1, col].dropna()

        # t-test: war-only vs pure-calm (the clean comparison)
        t_stat, p_val = (stats.ttest_ind(war_only, pure_calm, equal_var=False)
                         if len(war_only) > 1 and len(pure_calm) > 1
                         else (np.nan, np.nan))

        rows.append({
            "Market"               : mkt,
            "A: Pure Calm Mean"    : round(pure_calm.mean(),   4),
            "B: War-Only Mean"     : round(war_only.mean(),    4),
            "C: Crisis-Only Mean"  : round(crisis_only.mean(), 4) if len(crisis_only) > 0 else np.nan,
            "D: War+Crisis Mean"   : round(war_crisis.mean(),  4) if len(war_crisis)  > 0 else np.nan,
            "High-Int Mean"        : round(hi.mean(),          4) if len(hi) > 0 else np.nan,
            "B/A Ratio"            : round(war_only.mean() / pure_calm.mean(), 3)
                                     if pure_calm.mean() != 0 else np.nan,
            "C/A Ratio"            : round(crisis_only.mean() / pure_calm.mean(), 3)
                                     if len(crisis_only) > 0 and pure_calm.mean() != 0 else np.nan,
            "t-stat (B vs A)"      : round(t_stat, 3) if not np.isnan(t_stat) else np.nan,
            "p-value"              : round(p_val,  4) if not np.isnan(p_val)  else np.nan,
            "Significant?"         : ("✅" if (not np.isnan(p_val) and p_val < 0.05) else "❌"),
        })

    df = pd.DataFrame(rows).set_index("Market")

    print("\n[War vs Calm Volatility — Crisis-Controlled Summary]")
    print("  Regime A = Pure calm")
    print("  Regime B = War only")
    print("  Regime C = Crisis only")
    print("  Regime D = War + Crisis")
    print()
    print(df.to_string())

    if save:
        path = os.path.join(TABLE_DIR, "table_war_vs_calm.xlsx")
        df.to_excel(path)
        print(f"\n  Saved → {path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5b · Crisis Decomposition Plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_crisis_decomposition(vol_df: pd.DataFrame,
                              war_dummy: pd.DataFrame,
                              save: bool = True) -> None:
    """
    Fig 5b: Bar chart comparing mean volatility across four regimes
    (Pure Calm / War Only / Crisis Only / War+Crisis) for each market.

    Visually shows that the naive 'war period has lower vol' finding in Table 2
    (original) was driven by GFC/COVID dominating the 'calm' baseline.
    After crisis control, war-only periods show elevated vol relative to
    pure-calm — consistent with the war-shock narrative.
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    labels   = [c.replace("_vol", "") for c in vol_cols]

    if "any_crisis" not in war_dummy.columns:
        print("  [SKIP] plot_crisis_decomposition requires 'any_crisis' column")
        return

    aligned = vol_df[vol_cols].join(war_dummy, how="inner")

    regimes = {
        "A: Pure Calm"   : (aligned["mideast_war"] == 0) & (aligned["any_crisis"] == 0),
        "B: War Only"    : (aligned["mideast_war"] == 1) & (aligned["any_crisis"] == 0),
        "C: Crisis Only" : (aligned["mideast_war"] == 0) & (aligned["any_crisis"] == 1),
        "D: War+Crisis"  : (aligned["mideast_war"] == 1) & (aligned["any_crisis"] == 1),
    }
    colors = ["#4878cf", "#d65f5f", "#e8a838", "#6acc65"]

    x     = np.arange(len(labels))
    width = 0.20
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (regime_name, mask) in enumerate(regimes.items()):
        means = [aligned.loc[mask, c].mean() for c in vol_cols]
        ax.bar(x + (i - 1.5) * width, means, width,
               label=f"{regime_name}  (n={mask.sum()})",
               color=colors[i], alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Annualised Volatility")
    ax.set_title(
        "Volatility Regime Decomposition: War vs Crisis Effects\n"
        "(Regime A = baseline; B = war net of crises; C = GFC/COVID; D = overlap)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper right")

    # Annotate C/A ratio to show crisis is the dominant driver
    for i, col in enumerate(vol_cols):
        calm_mean   = aligned.loc[regimes["A: Pure Calm"],   col].mean()
        crisis_mean = aligned.loc[regimes["C: Crisis Only"], col].mean()
        war_mean    = aligned.loc[regimes["B: War Only"],    col].mean()
        if calm_mean > 0:
            ax.text(x[i], crisis_mean + 0.005,
                    f"C/A={crisis_mean/calm_mean:.1f}x",
                    ha="center", va="bottom", fontsize=6.5, color="#e8a838")
            ax.text(x[i] - 0.20, war_mean + 0.005,
                    f"B/A={war_mean/calm_mean:.2f}x",
                    ha="center", va="bottom", fontsize=6.5, color="#d65f5f")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig5a_crisis_decomposition.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()
