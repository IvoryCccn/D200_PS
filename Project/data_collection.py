"""
Global Financial Risk Network - Data Collection Script v2
Period: 2006-01-01 to 2025-12-31
Focus: Middle East War Shocks & Financial Risk Spillover
Indices: 8 global equity indices (all available from 2006)
"""

# %%
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os

# ============================================================
# Setting
# ============================================================
START_DATE = "2006-01-01"
END_DATE   = "2025-12-31"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# ============================================================
# Part 1: Yahoo Finance — Stock Index
# ============================================================
EQUITY_INDICES = {
    "SP500"    : "^GSPC",      # US S&P 500
    "STOXX600" : "^STOXX",     # Europe STOXX 600
    "FTSE100"  : "^FTSE",      # UK FTSE 100
    "DAX"      : "^GDAXI",     # German DAX
    "Nikkei"   : "^N225",      # Japan Nikkei 225
    "HangSeng" : "^HSI",       # HongKong HengSeng
    # "CSI300" : "000300.SS",  # China HS 300 (loaded from local document)
    "MSCI_EM"  : "EEM",        # Emerging Market MSCI EM ETF
}

print("=" * 60)
print("Step 1: Downloading Equity Indices (Yahoo Finance)...")
print(f"Period: {START_DATE} → {END_DATE}")
print("=" * 60)

equity_frames = {}
for name, ticker in EQUITY_INDICES.items():
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARNING] {name} ({ticker}): No data returned")
            continue
        close = df[["Close"]].copy()
        close.columns = [name]
        equity_frames[name] = close
        pct_missing = close.isna().mean().values[0] * 100
        print(f"  [OK] {name:10s} ({ticker:12s}): "
              f"{len(close)} rows | "
              f"{close.index[0].date()} ~ {close.index[-1].date()} | "
              f"missing={pct_missing:.1f}%")
    except Exception as e:
        print(f"  [ERROR] {name} ({ticker}): {e}")

# load CSI300 from local document
CSI300_FILE = "data/CSI300_data.xlsx"
try:
    csi = pd.read_excel(
        CSI300_FILE,
        usecols=["日期", "收盘"],
        dtype={"日期": str},
    )
    csi["Date"]   = pd.to_datetime(csi["日期"], format="%Y%m%d")
    csi           = csi.set_index("Date")[["收盘"]].rename(columns={"收盘": "CSI300"})
    csi["CSI300"] = pd.to_numeric(csi["CSI300"], errors="coerce")
    csi           = csi.sort_index().loc[START_DATE:END_DATE]

    pct_missing = csi["CSI300"].isna().mean() * 100
    equity_frames["CSI300"] = csi
    print(f"  [OK] CSI300     (local CSV   ): "
          f"{len(csi)} rows | "
          f"{csi.index[0].date()} ~ {csi.index[-1].date()} | "
          f"missing={pct_missing:.1f}%")
except FileNotFoundError:
    print(f"  [ERROR] CSI300: {CSI300_FILE} not found — place file in data/ folder")
except Exception as e:
    print(f"  [ERROR] CSI300: {e}")

# %%
# ============================================================
# Part 2: Yahoo Finance — Control Variables & Safe-haven Assets
# ============================================================
CONTROL_YAHOO = {
    "Gold"  : "GC=F",      # Gold Future
    "DXY"   : "DX-Y.NYB",  # Dollar Index
    "Silver": "SI=F",      # Sliver
}

print("\n" + "=" * 60)
print("Step 2: Downloading Control Variables (Yahoo Finance)...")
print("=" * 60)

control_frames = {}
for name, ticker in CONTROL_YAHOO.items():
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARNING] {name}: No data returned")
            continue
        close = df[["Close"]].copy()
        close.columns = [name]
        control_frames[name] = close
        print(f"  [OK] {name:8s} ({ticker:12s}): {len(close)} rows")
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

# %%
# ============================================================
# Part 3: FRED — Load from Local Excel File
# ============================================================
# Expected file : data/FRED_data.xlsx
# Expected cols : Date, VIXCLS, DCOILBRENTEU, DCOILWTICO, DGS10, BAMLH0A0HYM2

FRED_COL_MAP = {
    "VIXCLS"       : "VIX",
    "DCOILBRENTEU" : "Brent",
    "DCOILWTICO"   : "WTI",
    "DGS10"        : "US10Y",
    "BAMLH0A0HYM2" : "HY_OAS",
}

FRED_FILE = "data/FRED_data.xlsx"

print("\n" + "=" * 60)
print("Step 3: Loading Macro Variables from Local Excel (FRED)...")
print("=" * 60)

try:
    fred_raw = pd.read_excel(FRED_FILE, parse_dates=["Date"], index_col="Date")

    cols_present = [c for c in FRED_COL_MAP.keys() if c in fred_raw.columns]
    missing_cols = [c for c in FRED_COL_MAP.keys() if c not in fred_raw.columns]
    if missing_cols:
        print(f"  [WARNING] Columns not found in Excel: {missing_cols}")

    fred_raw = fred_raw[cols_present].copy()
    fred_raw.rename(columns=FRED_COL_MAP, inplace=True)

    # replace 0 with NaN
    for col in fred_raw.columns:
        zero_count = (fred_raw[col] == 0).sum()
        if zero_count > 0:
            fred_raw[col] = fred_raw[col].replace(0, np.nan)
            print(f"  [FIX] {col:8s}: replaced {zero_count} zero(s) → NaN")

    # sort
    fred_raw.index = pd.to_datetime(fred_raw.index)
    fred_raw = fred_raw.sort_index().loc[START_DATE:END_DATE]

    print(f"  [OK] {len(fred_raw)} rows loaded | "
          f"{fred_raw.index[0].date()} ~ {fred_raw.index[-1].date()}")
    print(f"  Columns : {list(fred_raw.columns)}")
    for col in fred_raw.columns:
        pct  = fred_raw[col].isna().mean() * 100
        print(f"  {col:10s}: {pct:.2f}% missing")

except FileNotFoundError:
    print(f"  [ERROR] File not found → {FRED_FILE}")
    fred_raw = pd.DataFrame()
except Exception as e:
    print(f"  [ERROR] {e}")
    fred_raw = pd.DataFrame()

# %%
# ============================================================
# Part 4: Merging & Aligning to S&P 500 Trading Days
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Merging & Aligning to S&P 500 Trading Days...")
print("=" * 60)

# Merge all Yahoo data (equity indices + control variables)
all_yahoo = list(equity_frames.values()) + list(control_frames.values())
yahoo_df  = pd.concat(all_yahoo, axis=1) if all_yahoo else pd.DataFrame()

# Merge all into one DataFrame
if not fred_raw.empty:
    combined = pd.concat([yahoo_df, fred_raw], axis=1)
else:
    print("  [WARNING] FRED data is empty — proceeding with Yahoo data only")
    combined = yahoo_df.copy()

combined.index = pd.to_datetime(combined.index)
combined = combined.sort_index()

# Align to S&P 500 trading days (drops weekends & non-US holidays)
if "SP500" in combined.columns:
    trading_days = combined["SP500"].dropna().index
    combined     = combined.reindex(trading_days)
    print(f"  Trading days (S&P500 basis): {len(trading_days)}")

# Forward-fill up to 3 days (handles cross-market holiday gaps)
combined = combined.ffill(limit=3)

combined.to_excel(f"{OUTPUT_DIR}/all_variables_aligned.xlsx")
print(f"  Aligned shape : {combined.shape}")
print(f"  Date range    : {combined.index[0].date()} ~ {combined.index[-1].date()}")
print(f"  Columns       : {list(combined.columns)}")

# %%
# ============================================================
# Part 5: Computing Log Returns & Rolling Volatility
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Computing Log Returns & Rolling Volatility...")
print("=" * 60)

# 5-1 Log Returns
price_cols = (list(EQUITY_INDICES.keys()) +
              list(CONTROL_YAHOO.keys()) +
              ["Brent", "WTI"])
price_cols = [c for c in price_cols if c in combined.columns]

returns_df = np.log(combined[price_cols]).diff()
returns_df.columns = [c + "_ret" for c in price_cols]

# 5-2 Different
for col in ["VIX", "US10Y", "HY_OAS"]:
    if col in combined.columns:
        returns_df[col + "_chg"] = combined[col].diff()

# 5-3 21 Days Rolling Volatility
equity_ret_cols = [c + "_ret" for c in EQUITY_INDICES.keys()
                   if c + "_ret" in returns_df.columns]
vol_df = returns_df[equity_ret_cols].rolling(21).std() * np.sqrt(252)  # 年化
vol_df.columns = [c.replace("_ret", "_vol") for c in equity_ret_cols]

# Save
returns_df.to_excel(f"{OUTPUT_DIR}/log_returns.xlsx")
vol_df.to_excel(f"{OUTPUT_DIR}/rolling_volatility.xlsx")

print(f"  Returns shape    : {returns_df.shape}")
print(f"  Volatility shape : {vol_df.shape}")

# %%
# ============================================================
# Part 6: Loading War Event Dummy Variables
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Loading War Event Dummy Variables...")
print("=" * 60)

war_excel_path = "data/war_events.xlsx"
if os.path.exists(war_excel_path):
    war_events = pd.read_excel(war_excel_path, parse_dates=["start_date", "end_date"])

    # create dummy variable for each trading day
    war_dummy = pd.DataFrame(index=combined.index)
    war_dummy["mideast_war"] = 0
    war_dummy["high_intensity"] = 0

    HIGH_INTENSITY = ["Israel-Lebanon 2006", "Gaza Cast Lead 2008",
                      "Israel-Hamas 2023",   "Israel-Iran 2024"]

    for _, row in war_events.iterrows():
        end = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp(END_DATE)
        mask = (war_dummy.index >= row["start_date"]) & (war_dummy.index <= end)
        war_dummy.loc[mask, "mideast_war"] = 1
        if row["event_name"] in HIGH_INTENSITY:
            war_dummy.loc[mask, "high_intensity"] = 1

    war_dummy.to_excel(f"{OUTPUT_DIR}/war_dummies.xlsx")
    n_war_days = war_dummy["mideast_war"].sum()
    print(f"  War dummy created: {n_war_days} trading days flagged as war periods")
    print(f"  High-intensity days: {war_dummy['high_intensity'].sum()}")
else:
    print(f"  [SKIP] {war_csv_path} not found — run after creating war_events.csv")

# %%
# ============================================================
# Part 7: Data Quality Report
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Data Quality Report")
print("=" * 60)

print("\n[Equity Indices — Missing Rate after ffill]")
for col in EQUITY_INDICES.keys():
    if col in combined.columns:
        pct = combined[col].isna().mean() * 100
        print(f"  {col:12s}: {pct:.2f}% missing")

print("\n[Returns — Descriptive Stats (Equity only)]")
desc = returns_df[equity_ret_cols].describe().round(6)
print(desc.to_string())

# %%
