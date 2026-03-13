# WarShock-Spillover

> **Can LSTM/GRU better capture nonlinear volatility spillovers under war-shock regimes than a linear VAR baseline?**

---

## Overview

This project investigates whether war shocks alter cross-market volatility transmission dynamics, and whether deep learning models (LSTM/GRU) better capture nonlinear spillovers under conflict regimes compared to a linear VAR baseline.

**Reference framework:**
- Diebold & Yilmaz (2009, 2012) — Generalised FEVD spillover index methodology
- Kumar et al. (2024, arXiv:2410.16858) — Temporal GAT benchmark

---

## Data

| Item | Detail |
|------|--------|
| Sample period | 2006-03-08 to 2025-12-30 (4,016 trading days) |
| Train / Val / Test | 2006–2018 / 2019–2021 / 2022–2025 |
| Endogenous variables | Realised volatility of 8 equity indices: S&P 500, STOXX 600, FTSE 100, DAX, Nikkei, Hang Seng, MSCI EM, CSI 300 |
| Control variables (lag-1) | VIX level, Brent return, Gold return |
| War dummy | `mideast_war` = 1 from 2011-03-15 onward |
| Crisis window | Soleimani/COVID: 2020-01-03 to 2020-03-31 |

---

## Models

| Model | Description |
|-------|-------------|
| **VAR** | AIC-selected lag=10, expanding-window rolling forecast |
| **LSTM** | 1-layer, 128 hidden units, Optuna 30-trial tuning |
| **GRU** | 1-layer, 128 hidden units, same tuning procedure |

All three models share identical 11-feature inputs. War dummies are **excluded** from model inputs to ensure fair comparison; they are used only for regime-conditional evaluation.

**Forecast horizons:** h = 1 / 5 / 10 / 22 trading days (direct multi-output strategy)  
**Metrics:** MSE, MAE, RMSE, MAPE, DirAcc (random benchmark = 0.50)

---

## Project Structure

```
WarShock-Spillover/
├── notebooks/
│   ├── 01_data_collection.ipynb    # Data download and preprocessing
│   ├── 02_eda.ipynb                # Exploratory data analysis
│   ├── 03_spillover_index.ipynb    # DY(2012) spillover index
│   ├── 04_var_baseline.ipynb       # VAR rolling forecast
│   ├── 05_tuning.ipynb             # Optuna hyperparameter search
│   ├── 06_lstm_model.ipynb         # LSTM training and evaluation
│   ├── 07_gru_model.ipynb          # GRU training and evaluation
│   └── 08_results_comparison.ipynb # Three-model comparison and war shock analysis
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── spillover.py
│   ├── var_models.py               # fit_var, rolling_forecast, compute_metrics
│   ├── lstm_models.py              # VolatilityLSTM, train_lstm, tune_lstm_optuna
│   └── gru_models.py               # VolatilityGRU, train_gru, tune_gru_optuna
├── data/
├── outputs/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Run notebooks in order (01 → 08). Each notebook saves outputs to `outputs/tables/` and `outputs/figures/`.
