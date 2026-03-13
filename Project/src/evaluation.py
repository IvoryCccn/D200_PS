"""
src/evaluation.py
=================
Plotting and table-building utilities for nb08 · Three-Model Comparison.

Public API
----------
Tables:
    build_master_table(all_metrics, period_label, ...)
    build_average_summary(all_metrics)
    build_regime_table(regime_dict)
    build_ratio_table(regime_dict)

Plots:
    plot_mse_bars(all_metrics, ...)
    plot_mape_lines(all_metrics, ...)
    plot_diracc_lines(all_metrics, ...)
    plot_db_ratio(regime_dict, ...)
    plot_mse_heatmap(all_metrics, ...)

Regime:
    compute_regime_metrics(act_df, fc_df, war_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants shared with notebooks ───────────────────────────────────
MODELS       = ["VAR", "LSTM", "GRU"]
MODEL_COLORS = {"VAR": "#4575b4", "LSTM": "#d7191c", "GRU": "#1a9641"}
MARKERS      = {"VAR": "s", "LSTM": "o", "GRU": "^"}

MARKET_LABELS = {
    "SP500_vol":    "S&P 500",
    "DAX_vol":      "DAX",
    "CAC40_vol":    "CAC 40",
    "FTSE100_vol":  "FTSE 100",
    "Nikkei_vol":   "Nikkei",
    "KOSPI_vol":    "KOSPI",
    "HangSeng_vol": "Hang Seng",
    "SSE_vol":      "SSE",
}

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_ROOT, "outputs", "figures")


# ══════════════════════════════════════════════════════════════════════
# 1. Regime metrics
# ══════════════════════════════════════════════════════════════════════

def compute_regime_metrics(act_df: pd.DataFrame,
                            fc_df:  pd.DataFrame,
                            war_df: pd.DataFrame) -> dict:
    """
    Compute mean-across-markets MSE / RMSE / MAPE for each regime
    using the TEST period actuals/forecasts.

    Regimes
    -------
    B: War-Only    mideast_war=1, any_crisis=0
    C: Crisis-Only mideast_war=0, any_crisis=1
    D: War+Crisis  mideast_war=1, any_crisis=1
    Full Test      all observations

    Returns
    -------
    dict  {regime_name: {"n": int, "MSE": float, "RMSE": float, "MAPE": float}}
    """
    war    = war_df["mideast_war"].reindex(act_df.index).fillna(0).astype(int)
    crisis = war_df["any_crisis"].reindex(act_df.index).fillna(0).astype(int)

    masks = {
        "B: War-Only"   : (war == 1) & (crisis == 0),
        "C: Crisis-Only": (war == 0) & (crisis == 1),
        "D: War+Crisis" : (war == 1) & (crisis == 1),
        "Full Test"     : pd.Series(True, index=act_df.index),
    }

    results = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n < 5:
            results[name] = {"n": n, "MSE": np.nan,
                             "RMSE": np.nan, "MAPE": np.nan}
            continue
        err  = act_df[mask].values - fc_df[mask].values
        mse  = float(np.mean(err ** 2))
        rmse = float(np.sqrt(np.mean(err ** 2, axis=0)).mean())
        # MAPE: avoid division by zero
        act_v = act_df[mask].values
        mape  = float(np.mean(
            np.abs(err) / np.where(act_v == 0, np.nan, np.abs(act_v))
        ) * 100)
        results[name] = {"n": n,
                         "MSE" : round(mse,  6),
                         "RMSE": round(rmse, 6),
                         "MAPE": round(mape, 2)}
    return results


# ══════════════════════════════════════════════════════════════════════
# 2. Table builders
# ══════════════════════════════════════════════════════════════════════

def build_average_summary(all_metrics: pd.DataFrame,
                           horizons: list = None) -> pd.DataFrame:
    """
    Clean summary: AVERAGE market, Val + Test, all horizons.

    Columns: Period | Horizon | VAR_MSE | VAR_MAPE | VAR_DirAcc |
             LSTM_MSE | LSTM_MAPE | LSTM_DirAcc | GRU_MSE | GRU_MAPE | GRU_DirAcc
    """
    if horizons is None:
        horizons = sorted(all_metrics["Horizon"].unique())

    rows = []
    for period in ["Val", "Test"]:
        for h in horizons:
            row = {"Period": period, "Horizon": f"h={h}"}
            for model in MODELS:
                cell = all_metrics[
                    (all_metrics["Model"]   == model) &
                    (all_metrics["Horizon"] == h) &
                    (all_metrics["Market"]  == "AVERAGE") &
                    (all_metrics["Label"].str.contains(period, na=False))
                ]
                if len(cell) > 0:
                    r = cell.iloc[0]
                    row[f"{model}_MSE"]    = round(r["MSE"],    6)
                    row[f"{model}_MAPE"]   = round(r["MAPE"],   2)
                    row[f"{model}_DirAcc"] = round(r["Dir_Acc"],3)
                else:
                    row[f"{model}_MSE"]    = np.nan
                    row[f"{model}_MAPE"]   = np.nan
                    row[f"{model}_DirAcc"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_master_table(all_metrics: pd.DataFrame,
                        period_label: str,
                        equity_cols: list,
                        metric_primary: str   = "MSE",
                        metric_secondary: str = "MAPE",
                        horizons: list = None) -> pd.DataFrame:
    """
    Wide-format table: rows = markets, cols = Model × Horizon.
    Cell = primary (secondary%).
    """
    if horizons is None:
        horizons = sorted(all_metrics["Horizon"].unique())

    sub = all_metrics[
        all_metrics["Label"].str.contains(period_label, na=False)
    ].copy()

    market_keys = [c.replace("_vol", "") for c in equity_cols] + ["AVERAGE"]
    rows = []
    for mkt_key in market_keys:
        vol_col = mkt_key + "_vol" if mkt_key != "AVERAGE" else "AVERAGE"
        display = MARKET_LABELS.get(vol_col, mkt_key)
        row = {"Market": display}
        for model in MODELS:
            for h in horizons:
                cell = sub[
                    (sub["Model"]   == model) &
                    (sub["Horizon"] == h) &
                    (sub["Market"]  == mkt_key)
                ]
                if len(cell) > 0:
                    r   = cell.iloc[0]
                    p   = r[metric_primary]
                    s   = r[metric_secondary]
                    row[f"{model}_h{h}"] = f"{p:.6f} ({s:.1f}%)"
                else:
                    row[f"{model}_h{h}"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def build_regime_table(regime_dict: dict) -> pd.DataFrame:
    """
    Regime-conditional metrics table.

    Parameters
    ----------
    regime_dict : {model_name: {regime_name: {"n", "MSE", "RMSE", "MAPE"}}}

    Returns
    -------
    DataFrame with columns: Model | Regime | n | MSE | RMSE | MAPE
    """
    rows = []
    for model, res in regime_dict.items():
        for regime, vals in res.items():
            rows.append({"Model": model, "Regime": regime, **vals})
    return pd.DataFrame(rows)


def build_ratio_table(regime_dict: dict) -> pd.DataFrame:
    """
    B/Full ratio table: RMSE(B: War-Only) / RMSE(Full Test).

    Interpretation: ratio < 1 means war-period errors are LOWER than the
    test-period average (market has already priced in war risk);
    ratio > 1 means war-period errors are amplified.

    Note: D: War+Crisis has 0 obs in the 2022-2025 test period (no
    systemic financial crisis coincides with the Israel-Hamas war),
    so B vs Full Test is used as the war-shock comparison.

    Returns
    -------
    DataFrame with columns:
        Model | B_n | B_RMSE | Full_n | Full_RMSE | B_Full_ratio
    """
    rows = []
    for model, res in regime_dict.items():
        b    = res.get("B: War-Only", {})
        full = res.get("Full Test",   {})
        b_rmse    = b.get("RMSE", np.nan)
        full_rmse = full.get("RMSE", np.nan)
        ratio = (round(b_rmse / full_rmse, 3)
                 if (full_rmse and not np.isnan(full_rmse) and full_rmse > 0)
                 else np.nan)
        rows.append({
            "Model"        : model,
            "B_n"          : b.get("n", np.nan),
            "B_RMSE"       : b_rmse,
            "Full_n"       : full.get("n", np.nan),
            "Full_RMSE"    : full_rmse,
            "B_Full_ratio" : ratio,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 3. Plots
# ══════════════════════════════════════════════════════════════════════

def _get_avg_metric(all_metrics, model, h, period, col):
    cell = all_metrics[
        (all_metrics["Model"]   == model) &
        (all_metrics["Horizon"] == h) &
        (all_metrics["Market"]  == "AVERAGE") &
        (all_metrics["Label"].str.contains(period, na=False))
    ]
    return cell.iloc[0][col] if len(cell) > 0 else np.nan


def plot_mse_bars(all_metrics: pd.DataFrame,
                  horizons: list,
                  fig_filename: str = "fig14_mse_comparison.png",
                  save: bool = True) -> None:
    """
    Grouped bar chart: MSE by horizon for Val and Test.
    """
    x     = np.arange(len(horizons))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 14 · MSE by Forecast Horizon — VAR vs LSTM vs GRU",
                 fontsize=13, fontweight="bold")

    for ax, period in zip(axes, ["Val", "Test"]):
        for i, model in enumerate(MODELS):
            vals = [_get_avg_metric(all_metrics, model, h, period, "MSE")
                    for h in horizons]
            bars = ax.bar(x + i * width, vals, width,
                          label=model, color=MODEL_COLORS[model], alpha=0.85)
            ax.bar_label(bars, fmt="%.5f", fontsize=6, padding=2)

        ax.set_title(f"{period} Period", fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_ylabel("MSE (avg across 8 markets)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_mape_lines(all_metrics: pd.DataFrame,
                    horizons: list,
                    fig_filename: str = "fig15_mape_degradation.png",
                    save: bool = True) -> None:
    """
    Line chart: MAPE degradation with forecast horizon for Val and Test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 15 · MAPE Degradation with Forecast Horizon — VAR vs LSTM vs GRU",
                 fontsize=13, fontweight="bold")

    for ax, period in zip(axes, ["Val", "Test"]):
        for model in MODELS:
            vals = [_get_avg_metric(all_metrics, model, h, period, "MAPE")
                    for h in horizons]
            ax.plot(horizons, vals,
                    color=MODEL_COLORS[model], marker=MARKERS[model],
                    lw=2.0, ms=7, label=model)

        ax.set_title(f"{period} Period", fontweight="bold")
        ax.set_xlabel("Forecast Horizon (h)")
        ax.set_ylabel("MAPE % (avg across 8 markets)")
        ax.set_xticks(horizons)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_diracc_lines(all_metrics: pd.DataFrame,
                      horizons: list,
                      fig_filename: str = "fig16_diracc.png",
                      save: bool = True) -> None:
    """
    Direction accuracy vs random walk (0.50) benchmark.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 16 · Direction Accuracy vs Random Walk Benchmark (0.50)",
                 fontsize=13, fontweight="bold")

    for ax, period in zip(axes, ["Val", "Test"]):
        for model in MODELS:
            vals = [_get_avg_metric(all_metrics, model, h, period, "Dir_Acc")
                    for h in horizons]
            ax.plot(horizons, vals,
                    color=MODEL_COLORS[model], marker=MARKERS[model],
                    lw=2.0, ms=7, label=model)

        ax.axhline(0.50, color="gray", lw=1.2, linestyle="--",
                   label="Random (0.50)")
        ax.set_title(f"{period} Period", fontweight="bold")
        ax.set_xlabel("Forecast Horizon (h)")
        ax.set_ylabel("Direction Accuracy")
        ax.set_xticks(horizons)
        ax.set_ylim(0.40, 0.65)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_db_ratio(regime_dict: dict,
                  fig_filename: str = "fig17_war_rmse.png",
                  save: bool = True) -> None:
    """
    Left : B/Full RMSE ratio per model (< 1 = war period better than avg).
    Right: absolute RMSE for B: War-Only vs Full Test per model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Fig 17 · War-Period Forecast Error — "
        "B: War-Only vs Full Test RMSE (Test, h=1)",
        fontsize=13, fontweight="bold")

    models_avail = [m for m in MODELS if m in regime_dict]
    x      = np.arange(len(models_avail))
    colors = [MODEL_COLORS[m] for m in models_avail]

    # ── Left: B / Full ratio ──────────────────────────────────────────
    ax = axes[0]
    ratios = []
    for model in models_avail:
        b    = regime_dict[model].get("B: War-Only", {}).get("RMSE", np.nan)
        full = regime_dict[model].get("Full Test",   {}).get("RMSE", np.nan)
        ratios.append(
            round(b / full, 3)
            if (full and not np.isnan(full) and full > 0) else np.nan
        )

    bars = ax.bar(x, ratios, color=colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.axhline(1.0, color="gray", lw=1.2, linestyle="--",
               label="Avg level (1.0)")
    ax.set_title("B / Full RMSE Ratio\n(< 1: war-period error below avg)",
                 fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models_avail)
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # ── Right: absolute RMSE B vs Full ───────────────────────────────
    ax   = axes[1]
    w    = 0.3
    x2   = np.arange(len(models_avail))
    for i, (regime_key, label, ls) in enumerate([
            ("B: War-Only", "War-Only (B)", 0),
            ("Full Test",   "Full Test",    1),
    ]):
        vals = [regime_dict[m].get(regime_key, {}).get("RMSE", np.nan)
                for m in models_avail]
        offset = (i - 0.5) * w
        bars2  = ax.bar(x2 + offset, vals, w,
                        label=label,
                        color=[MODEL_COLORS[m] for m in models_avail],
                        alpha=0.85 - 0.3 * i,
                        hatch="" if i == 0 else "//")
        ax.bar_label(bars2, fmt="%.4f", fontsize=7, padding=2)

    ax.set_title("Absolute RMSE: War-Only vs Full Test",
                 fontweight="bold")
    ax.set_xticks(x2); ax.set_xticklabels(models_avail)
    ax.set_ylabel("RMSE (avg across 8 markets)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_mse_heatmap(all_metrics: pd.DataFrame,
                     equity_cols: list,
                     horizons: list = None,
                     fig_filename: str = "fig18_mse_heatmap.png",
                     save: bool = True) -> None:
    """
    Per-market MSE heatmap for h = 1 / 5 / 22 (Test period).
    Rows = models, Cols = markets.
    """
    if horizons is None:
        horizons = [1, 5, 22]

    mkt_display = [MARKET_LABELS.get(c, c) for c in equity_cols]

    fig, axes = plt.subplots(1, len(horizons),
                              figsize=(6 * len(horizons), 4), sharey=True)
    fig.suptitle("Fig 18 · Per-Market MSE Heatmap (Test Period)",
                 fontsize=13, fontweight="bold")

    for ax, h in zip(axes, horizons):
        matrix = []
        for model in MODELS:
            row_vals = []
            for mkt in equity_cols:
                mkt_key = mkt.replace("_vol", "")
                cell = all_metrics[
                    (all_metrics["Model"]   == model) &
                    (all_metrics["Horizon"] == h) &
                    (all_metrics["Market"]  == mkt_key) &
                    (all_metrics["Label"].str.contains("Test", na=False))
                ]
                row_vals.append(cell.iloc[0]["MSE"] if len(cell) > 0 else np.nan)
            matrix.append(row_vals)

        mat = np.array(matrix, dtype=float)
        im  = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(equity_cols)))
        ax.set_xticklabels(mkt_display, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS)
        ax.set_title(f"h = {h}", fontweight="bold")

        for r in range(len(MODELS)):
            for c in range(len(equity_cols)):
                v = mat[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.4f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v > np.nanmax(mat) * 0.6 else "black")

    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
