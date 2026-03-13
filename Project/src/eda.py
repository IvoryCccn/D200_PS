"""
src/eda.py
==========
EDA functions for the WarShock-Spillover project.
Called by notebooks/02_eda.ipynb.

Functions:
    plot_volatility_timeseries   — Fig 1: rolling vol + war shading
    plot_return_distributions    — Fig 2: return histograms vs normal (fat tails)
    plot_correlation_heatmaps    — Fig 3: calm vs war correlation matrices
    plot_event_windows           — Fig 4: cumulative returns around Middle East events
    plot_macro_war_linkage       — Fig 5: Brent / Gold / VIX vs war events
    plot_crisis_decomposition    — Fig 6: bar chart of regime-mean volatilities
    run_adf_tests                — Table 1: stationarity tests on vol series
    war_vs_calm_stats            — Table 2: four-regime volatility decomposition
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# ── Output paths ───────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR   = os.path.join(ROOT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT_DIR, "outputs", "tables")
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

EQUITY_COLS = ["SP500", "DAX", "CAC40", "FTSE100",
               "Nikkei", "KOSPI", "HangSeng", "SSE"]

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


# ── Helper ─────────────────────────────────────────────────────────────────────
def _shade_wars(ax, events: pd.DataFrame, end_date: str) -> list:
    """Shade war event windows on an axis. Returns legend patches."""
    for _, row in events.iterrows():
        end   = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp(end_date)
        color = "#d62728" if row["event_name"] in HIGH_INTENSITY else "#ff7f0e"
        ax.axvspan(row["start_date"], end, color=color, alpha=0.13, zorder=0)
    return [
        mpatches.Patch(color="#d62728", alpha=0.35, label="High-intensity war"),
        mpatches.Patch(color="#ff7f0e", alpha=0.35, label="Other Middle East conflict"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 · Rolling Volatility Time-Series
# ══════════════════════════════════════════════════════════════════════════════
def plot_volatility_timeseries(vol_df: pd.DataFrame,
                               events: pd.DataFrame,
                               end_date: str = "2025-12-31",
                               save: bool = True) -> None:
    """
    Fig 1: Annualised 21-day rolling volatility for all equity indices,
    with Middle East war-period shading.

    Research link: establishes that war events coincide with vol spikes,
    motivating the war-shock regime analysis in nb03/nb08.
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

    ax.set_title("Fig 1 · Annualised 21-Day Rolling Volatility — Global Equity Indices\n"
                 "with Middle East War Events (2000–2025)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Volatility (annualised %)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    market_handles = [
        plt.Line2D([0], [0], color=MARKET_COLORS.get(c.replace("_vol", ""), "grey"),
                   linewidth=1.5, label=c.replace("_vol", ""))
        for c in vol_cols
    ]
    ax.legend(handles=market_handles + war_patches,
              fontsize=7, ncol=5, loc="upper right")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig1_volatility_timeseries.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 · Return Distributions (Fat Tails)
# ══════════════════════════════════════════════════════════════════════════════
def plot_return_distributions(returns_df: pd.DataFrame,
                              save: bool = True) -> None:
    """
    Fig 2: Daily log return histograms overlaid with normal fit.

    Research link: excess kurtosis in all markets invalidates the Gaussian
    assumption of linear VAR — directly motivates LSTM and GRU.
    """
    ret_cols = [c + "_ret" for c in EQUITY_COLS if c + "_ret" in returns_df.columns]
    ncols    = 4
    nrows    = (len(ret_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(ret_cols):
        ax   = axes[i]
        mkt  = col.replace("_ret", "")
        data = returns_df[col].dropna()

        ax.hist(data, bins=80, density=True,
                color=MARKET_COLORS.get(mkt, "steelblue"),
                alpha=0.6, edgecolor="none")

        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma),
                color="black", linewidth=1.5, linestyle="--")

        kurt = data.kurt()
        skew = data.skew()
        ax.set_title(f"{mkt}\nKurt={kurt:.2f}  Skew={skew:.2f}", fontsize=9)
        ax.set_xlabel("Log Return", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Fig 2 · Daily Log Return Distributions — Fat Tails vs Normal\n"
                 "(Excess kurtosis > 0 in all markets → linear VAR mis-specified → LSTM/GRU justified)",
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
    Fig 3: Side-by-side Pearson correlation matrices — calm vs war periods.

    Research link: higher correlations during war periods suggest increased
    risk contagion / co-movement, motivating cross-market spillover analysis.
    """
    ret_cols = [c + "_ret" for c in EQUITY_COLS if c + "_ret" in returns_df.columns]
    labels   = [c.replace("_ret", "") for c in ret_cols]

    aligned   = returns_df[ret_cols].join(war_dummy[["mideast_war"]], how="inner")
    calm_mask = aligned["mideast_war"] == 0
    war_mask  = aligned["mideast_war"] == 1

    corr_calm = aligned.loc[calm_mask, ret_cols].corr()
    corr_war  = aligned.loc[war_mask,  ret_cols].corr()

    fig = plt.figure(figsize=(15, 5.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax_calm = fig.add_subplot(gs[0])
    ax_war  = fig.add_subplot(gs[1])
    ax_cb   = fig.add_subplot(gs[2])

    for ax, corr, title in zip(
        [ax_calm, ax_war],
        [corr_calm, corr_war],
        [f"Calm Periods (n={calm_mask.sum()})",
         f"Middle East War Periods (n={war_mask.sum()})"]
    ):
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")

        for r in range(len(labels)):
            for c in range(len(labels)):
                val = corr.values[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(val) > 0.6 else "black")

    fig.colorbar(im, cax=ax_cb, label="Pearson Correlation")
    fig.suptitle("Fig 3 · Equity Return Correlations: Calm vs Middle East War Periods\n"
                 "(Higher off-diagonal values during war → stronger risk contagion)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig3_correlation_heatmaps.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 · Event Study: Cumulative Returns Around War Starts
# ══════════════════════════════════════════════════════════════════════════════
def plot_event_windows(returns_df: pd.DataFrame,
                       events: pd.DataFrame,
                       pre: int = 20,
                       post: int = 60,
                       save: bool = True) -> None:
    """
    Fig 4: Cumulative equal-weight global returns in [-pre, +post] trading days
    around each Middle East war event start.

    Research link: short-term drawdowns at event onset demonstrate that war
    shocks transmit quickly across markets — the speed and depth vary by event,
    evidence of non-linear, regime-dependent dynamics.
    """
    ret_cols = [c + "_ret" for c in EQUITY_COLS if c + "_ret" in returns_df.columns]
    avg_ret  = returns_df[ret_cols].mean(axis=1)

    # Only Middle East events — Russia-Ukraine excluded (different research scope)
    mideast_events = events[events.get("region", pd.Series("Middle East",
                     index=events.index)) == "Middle East"] \
                     if "region" in events.columns else events

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(mideast_events)))
    for (_, row), color in zip(mideast_events.iterrows(), colors):
        start_dt = row["start_date"]
        if start_dt not in avg_ret.index:
            loc = avg_ret.index.searchsorted(start_dt)
            if loc >= len(avg_ret.index):
                continue
            start_dt = avg_ret.index[loc]

        idx        = avg_ret.index.get_loc(start_dt)
        window_idx = avg_ret.index[max(0, idx - pre): idx + post + 1]
        window_ret = avg_ret.loc[window_idx]
        cum_ret    = (1 + window_ret).cumprod() - 1
        x          = range(-min(pre, idx), len(cum_ret) - min(pre, idx))

        ax.plot(list(x), cum_ret.values * 100,
                linewidth=1.3, alpha=0.8, color=color,
                label=row["event_name"])

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Event start")
    ax.axhline(0, color="grey",  linewidth=0.8, linestyle=":")
    ax.set_xlabel(f"Trading days relative to event start")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        f"Fig 4 · Event Study: Cumulative Returns [{-pre}, +{post}] Days\n"
        "Middle East War Events — Equal-Weight Global Index",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig4_event_windows.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 · Brent / Gold / VIX Linkage During War Periods
# ══════════════════════════════════════════════════════════════════════════════
def plot_macro_war_linkage(combined: pd.DataFrame,
                           events: pd.DataFrame,
                           end_date: str = "2025-12-31",
                           save: bool = True) -> None:
    """
    Fig 5: Three-panel time series — Brent crude, Gold, VIX — with war shading.

    Research link: this is the core transmission channel of the original
    research motivation (Middle East war → energy disruption → financial markets).
    Brent spike + VIX spike confirms the oil-fear pathway.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    series_cfg = [
        ("Brent", "Brent Crude ($/barrel)",  "#8B4513"),
        ("Gold",  "Gold ($/oz)",              "#DAA520"),
        ("VIX",   "VIX — Fear Index",         "#4B0082"),
    ]

    for ax, (col, ylabel, color) in zip(axes, series_cfg):
        if col not in combined.columns:
            ax.text(0.5, 0.5, f"{col} not found", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            continue
        ax.plot(combined.index, combined[col],
                color=color, linewidth=0.9, alpha=0.9)
        _shade_wars(ax, events, end_date)
        ax.set_ylabel(ylabel, fontsize=9)
        for _, row in events.iterrows():
            ax.axvline(row["start_date"], color="grey",
                       linewidth=0.6, linestyle=":", alpha=0.7)

    # Event labels on top panel
    for _, row in events.iterrows():
        short = (row["event_name"]
                 .replace("Israel-", "IL-")
                 .replace("Gaza ", "")
                 .replace(" War", ""))
        axes[0].text(row["start_date"],
                     axes[0].get_ylim()[1] * 0.92,
                     short, fontsize=5.5, rotation=75,
                     ha="left", va="top", color="darkred", alpha=0.8)

    war_patches = [
        mpatches.Patch(color="#d62728", alpha=0.35, label="High-intensity war"),
        mpatches.Patch(color="#ff7f0e", alpha=0.35, label="Other conflict"),
    ]
    axes[0].legend(handles=war_patches, fontsize=7, loc="upper left")
    axes[0].set_title(
        "Fig 5 · Brent Oil, Gold & VIX During Middle East War Periods\n"
        "(Core transmission channel: war → energy → financial markets)",
        fontsize=11, fontweight="bold"
    )
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
def run_adf_tests(vol_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Table 1: Augmented Dickey-Fuller test on each equity volatility series.

    Research link: VAR requires stationarity — if any series has a unit root,
    it must be first-differenced before modelling.
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
            "1% Critical"  : round(crit["1%"], 4),
            "5% Critical"  : round(crit["5%"], 4),
            "Stationary?"  : "✅ Yes" if p_val < 0.05 else "❌ No",
        })

    df = pd.DataFrame(results).set_index("Series")
    print("\n[Table 1 — ADF Stationarity Tests on Volatility Series]")
    print(df.to_string())

    if save:
        path = os.path.join(TABLE_DIR, "table_adf_tests.xlsx")
        df.to_excel(path)
        print(f"\n  Saved → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Table 2 · Four-Regime Volatility Decomposition
# ══════════════════════════════════════════════════════════════════════════════
def war_vs_calm_stats(vol_df: pd.DataFrame,
                      war_dummy: pd.DataFrame,
                      save: bool = True) -> pd.DataFrame:
    """
    Table 2: Mean volatility across four regimes per market.

        A — Pure Calm  : mideast_war=0, any_crisis=0   (true baseline)
        B — War Only   : mideast_war=1, any_crisis=0   (clean war effect)
        C — Crisis Only: mideast_war=0, any_crisis=1   (GFC / COVID)
        D — War+Crisis : mideast_war=1, any_crisis=1   (overlap)

    Key ratios:
        B/A — war premium net of crises
        C/A — crisis premium (expected >> B/A)
    Welch t-test (B vs A) gives statistical significance of the war effect.

    Research link: B/A ratio feeds directly into the regime-performance
    comparison in nb08 Table 3.
    """
    vol_cols   = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    has_crisis = "any_crisis" in war_dummy.columns

    if not has_crisis:
        print("  [WARN] war_dummy has no 'any_crisis' column — falling back to two-regime")

    aligned = vol_df[vol_cols].join(war_dummy, how="inner")
    rows    = []

    for col in vol_cols:
        mkt = col.replace("_vol", "")

        if has_crisis:
            pure_calm   = aligned.loc[(aligned["mideast_war"] == 0) &
                                      (aligned["any_crisis"]  == 0), col].dropna()
            war_only    = aligned.loc[(aligned["mideast_war"] == 1) &
                                      (aligned["any_crisis"]  == 0), col].dropna()
            crisis_only = aligned.loc[(aligned["mideast_war"] == 0) &
                                      (aligned["any_crisis"]  == 1), col].dropna()
            war_crisis  = aligned.loc[(aligned["mideast_war"] == 1) &
                                      (aligned["any_crisis"]  == 1), col].dropna()
        else:
            pure_calm   = aligned.loc[aligned["mideast_war"] == 0, col].dropna()
            war_only    = aligned.loc[aligned["mideast_war"] == 1, col].dropna()
            crisis_only = pd.Series(dtype=float)
            war_crisis  = pd.Series(dtype=float)

        t_stat, p_val = (stats.ttest_ind(war_only, pure_calm, equal_var=False)
                         if len(war_only) > 1 and len(pure_calm) > 1
                         else (np.nan, np.nan))

        calm_mean   = pure_calm.mean()
        crisis_mean = crisis_only.mean() if len(crisis_only) > 0 else np.nan
        war_mean    = war_only.mean()

        rows.append({
            "Market"             : mkt,
            "A: Pure Calm"       : round(calm_mean, 4),
            "B: War Only"        : round(war_mean,  4),
            "C: Crisis Only"     : round(crisis_mean, 4) if not np.isnan(crisis_mean) else np.nan,
            "D: War+Crisis"      : round(war_crisis.mean(), 4) if len(war_crisis) > 0 else np.nan,
            "B/A Ratio"          : round(war_mean / calm_mean, 3) if calm_mean > 0 else np.nan,
            "C/A Ratio"          : round(crisis_mean / calm_mean, 3)
                                   if (not np.isnan(crisis_mean) and calm_mean > 0) else np.nan,
            "t-stat (B vs A)"    : round(t_stat, 3) if not np.isnan(t_stat) else np.nan,
            "p-value"            : round(p_val,  4) if not np.isnan(p_val)  else np.nan,
            "Significant?"       : "✅" if (not np.isnan(p_val) and p_val < 0.05) else "❌",
        })

    df = pd.DataFrame(rows).set_index("Market")

    print("\n[Table 2 — Four-Regime Volatility Decomposition]")
    print("  A=Pure Calm  B=War Only  C=Crisis Only  D=War+Crisis")
    print()
    print(df.to_string())

    if save:
        path = os.path.join(TABLE_DIR, "table_war_vs_calm.xlsx")
        df.to_excel(path)
        print(f"\n  Saved → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5b · Crisis Decomposition Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_crisis_decomposition(vol_df: pd.DataFrame,
                              war_dummy: pd.DataFrame,
                              save: bool = True) -> None:
    """
    Fig 5b: Bar chart of mean volatility across four regimes per market.

    Research link: visually confirms that C/A >> B/A (crises dominate),
    while B/A > 1 shows a real but modest war premium — this heterogeneity
    motivates the regime-performance analysis in nb08.
    """
    vol_cols = [c + "_vol" for c in EQUITY_COLS if c + "_vol" in vol_df.columns]
    labels   = [c.replace("_vol", "") for c in vol_cols]

    if "any_crisis" not in war_dummy.columns:
        print("  [SKIP] requires 'any_crisis' column in war_dummy")
        return

    aligned = vol_df[vol_cols].join(war_dummy, how="inner")

    regimes = {
        "A: Pure Calm"   : (aligned["mideast_war"] == 0) & (aligned["any_crisis"] == 0),
        "B: War Only"    : (aligned["mideast_war"] == 1) & (aligned["any_crisis"] == 0),
        "C: Crisis Only" : (aligned["mideast_war"] == 0) & (aligned["any_crisis"] == 1),
        "D: War+Crisis"  : (aligned["mideast_war"] == 1) & (aligned["any_crisis"] == 1),
    }
    colors = ["#4878cf", "#d65f5f", "#e8a838", "#6acc65"]

    x, width = np.arange(len(labels)), 0.20
    fig, ax  = plt.subplots(figsize=(13, 5))

    for i, (regime_name, mask) in enumerate(regimes.items()):
        means = [aligned.loc[mask, c].mean() for c in vol_cols]
        ax.bar(x + (i - 1.5) * width, means, width,
               label=f"{regime_name}  (n={mask.sum()})",
               color=colors[i], alpha=0.82)

    # Annotate B/A and C/A ratios
    for i, col in enumerate(vol_cols):
        calm_mean   = aligned.loc[regimes["A: Pure Calm"],   col].mean()
        war_mean    = aligned.loc[regimes["B: War Only"],    col].mean()
        crisis_mean = aligned.loc[regimes["C: Crisis Only"], col].mean()
        if calm_mean > 0:
            ax.text(x[i] - 0.20, war_mean + 0.003,
                    f"B/A={war_mean/calm_mean:.2f}x",
                    ha="center", va="bottom", fontsize=6, color="#d65f5f")
            ax.text(x[i] + 0.20, crisis_mean + 0.003,
                    f"C/A={crisis_mean/calm_mean:.1f}x",
                    ha="center", va="bottom", fontsize=6, color="#e8a838")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Annualised Volatility")
    ax.set_title(
        "Fig 5b · Volatility Regime Decomposition: War vs Crisis Effects\n"
        "(B/A = war premium net of crises;  C/A = GFC/COVID premium)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "fig6_crisis_decomposition.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()
