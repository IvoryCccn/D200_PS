"""
src/data_loader.py
==================
All data loading & processing functions for the WarShock-Spillover project.
Called by notebooks/01_data_collection.ipynb.

Data flow:
    Yahoo Finance (equity + safe-haven)  ─┐
    Local Excel   (FRED_data.xlsx)        ├─▶ merged & aligned ─▶ returns ─▶ volatility
    Local Excel   (war_events.xlsx)      ─┘                                ─▶ war dummies

Equity universe (8 indices, all via Yahoo Finance):
    SP500     ^GSPC      US
    DAX       ^GDAXI     Germany
    CAC40     ^FCHI      France
    FTSE100   ^FTSE      UK
    Nikkei    ^N225      Japan
    KOSPI     ^KS11      South Korea
    HangSeng  ^HSI       Hong Kong
    SSE       000001.SS  China (Shanghai)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ── Project-level paths ────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR  = os.path.join(ROOT_DIR, "data")
DATA_PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(DATA_PROC_DIR, exist_ok=True)

# ── Index / series definitions ─────────────────────────────────────────────────
EQUITY_TICKERS = {
    "SP500"   : "^GSPC",      # US S&P 500              (USD)
    "DAX"     : "^GDAXI",     # German DAX              (EUR)
    "CAC40"   : "^FCHI",      # French CAC 40           (EUR)
    "FTSE100" : "^FTSE",      # UK FTSE 100             (GBP)
    "Nikkei"  : "^N225",      # Japan Nikkei 225        (JPY)
    "KOSPI"   : "^KS11",      # South Korea KOSPI       (KRW)
    "HangSeng": "^HSI",       # Hong Kong Hang Seng     (HKD)
    "SSE"     : "000001.SS",  # China SSE Composite     (CNY)
}

CONTROL_TICKERS = {
    "Gold"  : "GC=F",      # Gold futures
    "DXY"   : "DX-Y.NYB",  # US Dollar Index
    "Silver": "SI=F",      # Silver futures
}

FRED_COL_MAP = {
    "VIXCLS"       : "VIX",    # CBOE Volatility Index
    "DCOILBRENTEU" : "Brent",  # Brent crude oil
    "DCOILWTICO"   : "WTI",    # WTI crude oil
    "DGS10"        : "US10Y",  # US 10-Year Treasury yield
    "BAMLH0A0HYM2" : "HY_OAS", # ICE BofA HY OAS
}

# Middle East conflicts classified as high-intensity
HIGH_INTENSITY_EVENTS = [
    "Lebanon War 2006",
    "Gaza War 2008",
    "Israel-Hamas War 2023",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_yahoo_equity(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download equity index closing prices from Yahoo Finance."""
    frames = {}
    for name, ticker in EQUITY_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False,
                             multi_level_index=False)
            if df.empty:
                print(f"  [WARNING] {name} ({ticker}): no data returned")
                continue
            close = df[["Close"]].rename(columns={"Close": name})
            pct   = close[name].isna().mean() * 100
            frames[name] = close
            print(f"  [OK] {name:10s} ({ticker:12s}): "
                  f"{len(close)} rows | "
                  f"{close.index[0].date()} ~ {close.index[-1].date()} | "
                  f"missing={pct:.1f}%")
        except Exception as e:
            print(f"  [ERROR] {name} ({ticker}): {e}")
    return frames


def load_yahoo_controls(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download safe-haven / control variable prices from Yahoo Finance."""
    frames = {}
    for name, ticker in CONTROL_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False,
                             multi_level_index=False)
            if df.empty:
                print(f"  [WARNING] {name}: no data returned")
                continue
            close = df[["Close"]].rename(columns={"Close": name})
            frames[name] = close
            print(f"  [OK] {name:8s} ({ticker:12s}): {len(close)} rows")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    return frames


def load_fred(start: str, end: str) -> pd.DataFrame:
    """Load FRED macro variables from local Excel file."""
    path = os.path.join(DATA_RAW_DIR, "FRED_data.xlsx")
    try:
        raw = pd.read_excel(path, parse_dates=["Date"], index_col="Date")

        present = [c for c in FRED_COL_MAP if c in raw.columns]
        missing = [c for c in FRED_COL_MAP if c not in raw.columns]
        if missing:
            print(f"  [WARNING] columns not found in FRED Excel: {missing}")

        raw = raw[present].rename(columns=FRED_COL_MAP)

        # Replace 0 with NaN (holiday placeholders in oil / yield series)
        for col in raw.columns:
            n = (raw[col] == 0).sum()
            if n:
                raw[col] = raw[col].replace(0, np.nan)
                print(f"  [FIX] {col}: replaced {n} zero(s) → NaN")

        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index().loc[start:end]

        print(f"  [OK] FRED: {len(raw)} rows | "
              f"{raw.index[0].date()} ~ {raw.index[-1].date()}")
        for col in raw.columns:
            pct  = raw[col].isna().mean() * 100
            flag = "⚠️ " if pct > 5 else "  "
            print(f"  {flag}{col:10s}: {pct:.2f}% missing")
        return raw

    except FileNotFoundError:
        print(f"  [ERROR] {path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"  [ERROR] FRED load failed: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Merging & Alignment
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_align(equity_frames: dict,
                    control_frames: dict,
                    fred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all data sources and align to S&P 500 trading days.
    Forward-fill up to 3 days to handle cross-market holiday gaps.
    """
    all_frames = list(equity_frames.values()) + list(control_frames.values())
    yahoo_df   = pd.concat(all_frames, axis=1) if all_frames else pd.DataFrame()

    combined = (pd.concat([yahoo_df, fred_df], axis=1)
                if not fred_df.empty else yahoo_df.copy())
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    # Align to S&P 500 trading days
    if "SP500" in combined.columns:
        trading_days = combined["SP500"].dropna().index
        combined     = combined.reindex(trading_days)
        print(f"  Trading days (S&P 500 basis): {len(trading_days)}")

    combined = combined.ffill(limit=3)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def compute_returns(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for price series and first differences for rates."""
    price_cols = (list(EQUITY_TICKERS.keys()) +
                  [c for c in CONTROL_TICKERS if c in combined.columns] +
                  ["Brent", "WTI"])
    price_cols = [c for c in price_cols if c in combined.columns]

    ret = np.log(combined[price_cols]).diff()
    ret.columns = [c + "_ret" for c in price_cols]

    for col in ["VIX", "US10Y", "HY_OAS"]:
        if col in combined.columns:
            ret[col + "_chg"] = combined[col].diff()

    return ret


def compute_volatility(returns_df: pd.DataFrame,
                       window: int = 21) -> pd.DataFrame:
    """
    Annualised rolling volatility for equity indices only.
    window=21 trading days ≈ 1 calendar month.
    """
    equity_ret_cols = [c + "_ret" for c in EQUITY_TICKERS
                       if c + "_ret" in returns_df.columns]
    vol = returns_df[equity_ret_cols].rolling(window).std() * np.sqrt(252)
    vol.columns = [c.replace("_ret", "_vol") for c in equity_ret_cols]
    return vol


def build_war_dummies(index: pd.DatetimeIndex, end_date: str) -> pd.DataFrame:
    """
    Build dummy columns from war_events.xlsx plus hard-coded crisis periods.

    War events covered (Middle East only, from war_events.xlsx)
    -----------------------------------------------------------
    Iraq War                  2003-03-20 ~ 2003-05-01
    Lebanon War 2006          2006-07-12 ~ 2006-08-14
    Gaza War 2008             2008-12-27 ~ 2009-01-18
    Gaza War 2014             2014-07-08 ~ 2014-08-26
    ISIS Iraq escalation      2014-06-10 ~ 2014-12-09
    US-Iran crisis            2020-01-03 ~ 2020-01-08
    Israel-Hamas War 2023     2023-10-07 ~ ongoing

    Dummy columns
    -------------
    mideast_war    : 1 if any Middle East conflict active
    high_intensity : 1 for high-intensity events only
                     (Lebanon War 2006 / Gaza War 2008 / Israel-Hamas War 2023)
    gfc_crisis     : 1 for Global Financial Crisis  (2008-09-01 – 2009-06-30)
    covid_crisis   : 1 for COVID market shock       (2020-02-01 – 2020-09-30)
    any_crisis     : 1 if gfc OR covid (convenience union flag)

    Design note
    -----------
    GFC window : Lehman collapse (Sep 2008) through trough recovery (Jun 2009).
    COVID window: First global sell-off (Feb 2020) through initial stabilisation (Sep 2020).
    """
    CRISIS_PERIODS = {
        "gfc_crisis"  : ("2008-09-01", "2009-06-30"),
        "covid_crisis": ("2020-02-01", "2020-09-30"),
    }

    path  = os.path.join(DATA_RAW_DIR, "war_events.xlsx")
    dummy = pd.DataFrame(
        {"mideast_war": 0, "high_intensity": 0,
         "gfc_crisis":  0, "covid_crisis":   0, "any_crisis": 0},
        index=index,
    )

    # ── War dummies from Excel ──────────────────────────────────────────────────
    try:
        events = pd.read_excel(path, parse_dates=["start_date", "end_date"])
        for _, row in events.iterrows():
            end  = (row["end_date"] if pd.notna(row["end_date"])
                    else pd.Timestamp(end_date))
            mask = (index >= row["start_date"]) & (index <= end)
            dummy.loc[mask, "mideast_war"] = 1
            if row["event_name"] in HIGH_INTENSITY_EVENTS:
                dummy.loc[mask, "high_intensity"] = 1
    except FileNotFoundError:
        print(f"  [ERROR] {path} not found — war dummies set to 0")
    except Exception as e:
        print(f"  [ERROR] war dummies: {e}")

    # ── Crisis dummies (hard-coded) ─────────────────────────────────────────────
    for col, (start, end) in CRISIS_PERIODS.items():
        mask = (index >= start) & (index <= end)
        dummy.loc[mask, col] = 1

    dummy["any_crisis"] = ((dummy["gfc_crisis"] == 1) |
                           (dummy["covid_crisis"] == 1)).astype(int)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"  mideast_war    days : {dummy['mideast_war'].sum()}")
    print(f"  high_intensity days : {dummy['high_intensity'].sum()}")
    print(f"  gfc_crisis     days : {dummy['gfc_crisis'].sum()}")
    print(f"  covid_crisis   days : {dummy['covid_crisis'].sum()}")
    print(f"  any_crisis     days : {dummy['any_crisis'].sum()}")

    return dummy


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to data/processed/ as Excel."""
    path = os.path.join(DATA_PROC_DIR, filename)
    df.to_excel(path)
    print(f"  Saved → {path}  {df.shape}")
