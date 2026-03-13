"""
src/deep_models.py
==================
Unified LSTM + GRU utilities for WarShock-Spillover project.

Replaces lstm_models.py and gru_models.py.

Public API
----------
Models:
    VolatilityRNN(rnn_type="LSTM"|"GRU", ...)

Data:
    make_sequences(data, lookback, n_targets)
    build_multistep_sequences(data, lookback, horizons, n_targets)
    build_loaders(X_train, y_train, X_val, y_val, batch_size)
    inverse_transform_vol(scaled_preds, scaler, n_vol, n_features)

Training / Inference:
    train_model(model, train_loader, val_loader, ...)
    predict(model, X, device)

Tuning:
    tune_optuna(rnn_type, feat_train_scaled, feat_val_scaled, ...)

Plotting:
    plot_forecast(results_val_ms, results_test_ms, test_start, ...)
    plot_training_history(history, model_name, horizon, ...)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ── Defaults (overridden by Optuna best params at runtime) ─────────────
LOOKBACK    = 60
N_FEATURES  = 19   # 8 vol + VIX_level + Brent_ret + Gold_ret + 8 net spillover
N_MARKETS   = 8
HORIZONS    = [1, 5, 10, 22]
HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10

# Plot defaults
PLOT_MARKETS  = ["SP500_vol", "DAX_vol", "Nikkei_vol", "KOSPI_vol"]
MARKET_LABELS = ["S&P 500", "DAX", "Nikkei", "KOSPI"]

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(_ROOT, "outputs", "figures")


# ══════════════════════════════════════════════════════════════════════
# 1. Model — single class handles both LSTM and GRU
# ══════════════════════════════════════════════════════════════════════

class VolatilityRNN(nn.Module):
    """
    Unified LSTM / GRU for multivariate volatility forecasting.

    Parameters
    ----------
    rnn_type   : "LSTM" or "GRU"
    n_features : int   input dimension
    hidden_size: int
    n_layers   : int
    dropout    : float
    n_outputs  : int   output dimension (= N_MARKETS for h=1)

    Forward
    -------
    Input  : (batch, seq_len, n_features)
    Output : (batch, n_outputs)
    """
    def __init__(self, rnn_type: str = "LSTM",
                 n_features: int  = N_FEATURES,
                 hidden_size: int = HIDDEN_SIZE,
                 n_layers: int    = N_LAYERS,
                 dropout: float   = DROPOUT,
                 n_outputs: int   = N_MARKETS):
        super().__init__()
        rnn_type = rnn_type.upper()
        assert rnn_type in ("LSTM", "GRU"), "rnn_type must be 'LSTM' or 'GRU'"
        self.rnn_type = rnn_type

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ══════════════════════════════════════════════════════════════════════
# 2. Data preparation
# ══════════════════════════════════════════════════════════════════════

def make_sequences(data: np.ndarray,
                   lookback: int = LOOKBACK,
                   n_targets: int = N_MARKETS):
    """
    Single-step sequences: X = [t-lookback, t), y = vol at t.

    Returns
    -------
    X : (T-lookback, lookback, n_features)
    y : (T-lookback, n_targets)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(data[i, :n_targets])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_multistep_sequences(data: np.ndarray,
                               lookback: int  = LOOKBACK,
                               horizons: list = None,
                               n_targets: int = N_MARKETS):
    """
    Multi-step sequences: for each origin t, Y contains vol at t+h for
    every h in horizons. Used for direct multi-output training.

    Returns
    -------
    X          : (N, lookback, n_features)
    Y_dict     : {h: np.ndarray (N, n_targets)}
    dates_mask : np.ndarray (N,) — valid origin indices
    """
    if horizons is None:
        horizons = HORIZONS
    max_h = max(horizons)
    X, Y_lists, valid = [], {h: [] for h in horizons}, []

    for i in range(lookback, len(data) - max_h):
        X.append(data[i - lookback:i, :])
        for h in horizons:
            Y_lists[h].append(data[i + h - 1, :n_targets])
        valid.append(i)

    X      = np.array(X, dtype=np.float32)
    Y_dict = {h: np.array(v, dtype=np.float32) for h, v in Y_lists.items()}
    return X, Y_dict, np.array(valid)


def build_loaders(X_train, y_train, X_val, y_val,
                  batch_size: int = BATCH_SIZE):
    """Wrap arrays into DataLoaders."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))


def inverse_transform_vol(scaled_preds: np.ndarray, scaler,
                           n_vol: int = N_MARKETS,
                           n_features: int = N_FEATURES) -> np.ndarray:
    """
    Inverse-scale vol predictions (first n_vol columns of scaler).
    Pads zeros for remaining feature columns before inverse_transform.
    """
    n = scaled_preds.shape[0]
    padded = np.zeros((n, n_features), dtype=np.float32)
    padded[:, :n_vol] = scaled_preds
    return scaler.inverse_transform(padded)[:, :n_vol]


# ══════════════════════════════════════════════════════════════════════
# 3. Training loop
# ══════════════════════════════════════════════════════════════════════

def train_model(model: VolatilityRNN,
                train_loader, val_loader,
                epochs: int   = EPOCHS,
                lr: float     = LR,
                patience: int = PATIENCE,
                device        = None,
                verbose: bool = True):
    """
    Adam + ReduceLROnPlateau + gradient clipping (norm=1) + early stopping.
    Works for both LSTM and GRU (VolatilityRNN).

    Returns
    -------
    model      : restored to best val weights
    history    : {'train_loss': [...], 'val_loss': [...]}
    best_epoch : int
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=False)

    best_val, best_state, best_ep = float("inf"), None, 0
    no_improve = 0
    t_losses, v_losses = [], []

    if verbose:
        print(f"[{model.rnn_type}] Device={device} | epochs={epochs} "
              f"| lr={lr} | patience={patience}")
        print("=" * 60)

    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item() * xb.size(0)
        vl /= len(val_loader.dataset)

        t_losses.append(tl); v_losses.append(vl)
        sched.step(vl)

        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  Epoch {ep:3d}/{epochs}  train={tl:.6f}  val={vl:.6f}")

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep    = ep
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping ep={ep} "
                          f"(best={best_ep}, val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    if verbose:
        print(f"\nBest val_loss: {best_val:.6f}  (epoch {best_ep})")
    return model, {"train_loss": t_losses, "val_loss": v_losses}, best_ep


# ══════════════════════════════════════════════════════════════════════
# 4. Inference
# ══════════════════════════════════════════════════════════════════════

def predict(model: VolatilityRNN,
            X: np.ndarray,
            device=None) -> np.ndarray:
    """Forward pass → scaled numpy array."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 5. Optuna tuning
# ══════════════════════════════════════════════════════════════════════

def tune_optuna(rnn_type: str,
                feat_train_scaled: np.ndarray,
                feat_val_scaled: np.ndarray,
                n_trials: int   = 30,
                n_epochs: int   = 30,
                patience: int   = 7,
                n_features: int = N_FEATURES,
                n_markets: int  = N_MARKETS,
                batch_size: int = BATCH_SIZE,
                device          = None,
                seed: int       = 42):
    """
    Bayesian hyperparameter optimisation via Optuna TPE sampler.
    Works for both LSTM and GRU via rnn_type parameter.

    Search space
    ------------
    hidden_size : {32, 64, 128}
    n_layers    : {1, 2}
    dropout     : {0.1, 0.2, 0.3}
    lr          : {1e-4, 1e-3, 1e-2}
    lookback    : {22, 44, 60}

    Objective: minimise validation MSE at h=1.

    Returns
    -------
    best_params : dict
    study       : optuna.Study
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    rnn_type = rnn_type.upper()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        hidden = trial.suggest_categorical("hidden_size", [32, 64, 128])
        layers = trial.suggest_int("n_layers", 1, 2)
        drop   = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
        lr     = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2])
        look   = trial.suggest_categorical("lookback", [22, 44, 60])

        X_tr, y_tr = make_sequences(feat_train_scaled, lookback=look,
                                    n_targets=n_markets)
        X_vl, y_vl = make_sequences(feat_val_scaled,   lookback=look,
                                    n_targets=n_markets)
        if len(X_tr) == 0 or len(X_vl) == 0:
            return float("inf")

        tr_ldr, vl_ldr = build_loaders(X_tr, y_tr, X_vl, y_vl,
                                        batch_size=batch_size)
        model = VolatilityRNN(rnn_type=rnn_type,
                               n_features=n_features, hidden_size=hidden,
                               n_layers=layers, dropout=drop,
                               n_outputs=n_markets)
        _, history, _ = train_model(model, tr_ldr, vl_ldr,
                                     epochs=n_epochs, lr=lr,
                                     patience=patience,
                                     device=device, verbose=False)
        return min(history["val_loss"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n{'='*55}")
    print(f"{rnn_type} Optuna best params (val MSE={study.best_value:.6f})")
    print(f"{'='*55}")
    for k, v in best.items():
        print(f"  {k:15s}: {v}")

    return best, study


# ══════════════════════════════════════════════════════════════════════
# 6. Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_forecast(results_val_ms: dict,
                  results_test_ms: dict,
                  test_start: str,
                  model_name: str       = "LSTM",
                  fig_filename: str     = None,
                  plot_markets: list    = None,
                  market_labels: list   = None,
                  forecast_color: str   = "#d7191c",
                  save: bool            = True) -> None:
    """
    Plot h=1 forecast vs actual for val + test periods.

    Parameters
    ----------
    results_val_ms  : {h: (df_actual, df_forecast)}
    results_test_ms : {h: (df_actual, df_forecast)}
    test_start      : str  vertical divider date
    model_name      : "LSTM" or "GRU"
    fig_filename    : if None, auto-named fig10_lstm / fig11_gru
    plot_markets    : vol column names  (default PLOT_MARKETS)
    market_labels   : display labels    (default MARKET_LABELS)
    forecast_color  : hex colour for forecast line
    save            : bool
    """
    if plot_markets is None:
        plot_markets = PLOT_MARKETS
    if market_labels is None:
        market_labels = MARKET_LABELS
    if fig_filename is None:
        idx = "10" if model_name.upper() == "LSTM" else "11"
        fig_filename = f"fig{idx}_{model_name.lower()}_forecast.png"

    act_v, fc_v = results_val_ms[1]
    act_t, fc_t = results_test_ms[1]

    fig, axes = plt.subplots(len(plot_markets), 1,
                              figsize=(14, 3 * len(plot_markets)),
                              sharex=False)
    if len(plot_markets) == 1:
        axes = [axes]

    fig.suptitle(f"{model_name} One-Step-Ahead Forecast vs Actual\n"
                 "(Val: 2019–2021 | Test: 2022–2025)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, col, lbl in zip(axes, plot_markets, market_labels):
        ax.plot(act_v.index, act_v[col],
                color="#2c7bb6", lw=1.0, alpha=0.85, label="Actual")
        ax.plot(fc_v.index, fc_v[col],
                color=forecast_color, lw=0.9, alpha=0.75,
                linestyle="--", label=f"{model_name} Forecast")
        ax.plot(act_t.index, act_t[col],
                color="#2c7bb6", lw=1.0, alpha=0.85)
        ax.plot(fc_t.index, fc_t[col],
                color=forecast_color, lw=0.9, alpha=0.75, linestyle="--")
        ax.axvline(pd.Timestamp(test_start),
                   color="gray", lw=0.8, linestyle=":")
        ax.set_ylabel("Ann. Vol", fontsize=9)
        ax.set_title(lbl, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_training_history(history: dict,
                           model_name: str = "LSTM",
                           horizon: int    = 1,
                           fig_filename: str = None,
                           save: bool      = True) -> None:
    """
    Plot train vs val loss curve for a single horizon.

    Parameters
    ----------
    history     : {'train_loss': [...], 'val_loss': [...]}
    model_name  : str
    horizon     : int
    fig_filename: if None, auto-named
    save        : bool
    """
    if fig_filename is None:
        fig_filename = (f"fig_training_history_"
                        f"{model_name.lower()}_h{horizon}.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"],
            color="#1f77b4", lw=1.5, label="Train loss")
    ax.plot(epochs, history["val_loss"],
            color="#d62728", lw=1.5, linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.set_title(f"{model_name} Training History  (h={horizon})",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, fig_filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
