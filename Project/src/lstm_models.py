"""
src/lstm_models.py
==================
LSTM utilities for WarShock-Spillover project.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Defaults (overridden by Optuna best params at runtime) ────────────
LOOKBACK    = 60
N_FEATURES  = 11  # 8 vol + VIX_level + Brent_ret + Gold_ret
N_MARKETS   = 8
HORIZONS    = [1, 5, 10, 22]
HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class VolatilityLSTM(nn.Module):
    """
    2-layer LSTM for multivariate volatility forecasting.

    Supports both single-step (n_outputs = n_markets) and multi-step
    direct output (n_outputs = n_markets * H).

    Forward
    -------
    Input  : (batch, seq_len, n_features)
    Output : (batch, n_outputs)
    """
    def __init__(self, n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 n_layers=N_LAYERS, dropout=DROPOUT, n_outputs=N_MARKETS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def make_sequences(data, lookback=LOOKBACK, n_targets=N_MARKETS):
    """
    Single-step sequences: X = window of lookback days, y = next day vol.

    Parameters
    ----------
    data      : np.ndarray (T, n_features) — scaled
    lookback  : int
    n_targets : int — first n_targets columns are vol (prediction targets)

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


def build_multistep_sequences(data, lookback=LOOKBACK,
                               horizons=None, n_targets=N_MARKETS):
    """
    Multi-step sequences: for each origin t, Y contains vol at t+h for
    every h in horizons. Used for direct multi-output training.

    Parameters
    ----------
    data     : np.ndarray (T, n_features) — scaled
    lookback : int
    horizons : list[int]  default [1,5,10,22]
    n_targets: int

    Returns
    -------
    X      : (N, lookback, n_features)
    Y_dict : {h: np.ndarray (N, n_targets)}  — target at each horizon
    dates_mask : np.ndarray (N,) — valid origin indices into original array
    """
    if horizons is None:
        horizons = HORIZONS
    max_h = max(horizons)
    X, Y_dict_lists, valid = [], {h: [] for h in horizons}, []

    for i in range(lookback, len(data) - max_h):
        X.append(data[i - lookback:i, :])
        for h in horizons:
            Y_dict_lists[h].append(data[i + h - 1, :n_targets])
        valid.append(i)

    X = np.array(X, dtype=np.float32)
    Y_dict = {h: np.array(v, dtype=np.float32) for h, v in Y_dict_lists.items()}
    return X, Y_dict, np.array(valid)


def build_loaders(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """Wrap arrays into DataLoaders (single-step or one horizon at a time)."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))


# ═══════════════════════════════════════════════════════════════════════
# Training loop (shared by single-step and per-horizon training)
# ═══════════════════════════════════════════════════════════════════════

def train_lstm(model, train_loader, val_loader,
               epochs=EPOCHS, lr=LR, patience=PATIENCE,
               device=None, verbose=True):
    """
    Train with Adam + ReduceLROnPlateau + gradient clipping + early stopping.

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
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=False)

    best_val, best_state, best_ep = float("inf"), None, 0
    no_improve = 0
    t_losses, v_losses = [], []

    if verbose:
        print(f"Device={device} | epochs={epochs} | lr={lr} | patience={patience}")
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
                    print(f"  Early stopping ep={ep} (best={best_ep}, val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    if verbose:
        print(f"\nBest val_loss: {best_val:.6f}  (epoch {best_ep})")
    return model, {"train_loss": t_losses, "val_loss": v_losses}, best_ep


# ═══════════════════════════════════════════════════════════════════════
# Inference & inverse transform
# ═══════════════════════════════════════════════════════════════════════

def predict_lstm(model, X, device=None):
    """Forward pass to scaled numpy."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


def inverse_transform_vol(scaled_preds, scaler,
                           n_vol=N_MARKETS, n_features=N_FEATURES):
    """
    Inverse-scale vol predictions (first n_vol columns of scaler).
    Pads zeros for remaining feature columns before calling inverse_transform.
    """
    n = scaled_preds.shape[0]
    padded = np.zeros((n, n_features), dtype=np.float32)
    padded[:, :n_vol] = scaled_preds
    return scaler.inverse_transform(padded)[:, :n_vol]


# ═══════════════════════════════════════════════════════════════════════
# Optuna hyperparameter tuning
# ═══════════════════════════════════════════════════════════════════════

def tune_lstm_optuna(feat_train_scaled, feat_val_scaled,
                     n_trials=30, n_epochs=30, patience=7,
                     n_features=N_FEATURES, n_markets=N_MARKETS,
                     batch_size=BATCH_SIZE, device=None, seed=42):
    """
    Bayesian hyperparameter optimisation via Optuna TPE sampler.

    Search space (mirrors Kumar et al. 2024 grid + standard extensions):
        hidden_size : categorical {32, 64, 128}
        n_layers    : int         {1, 2}
        dropout     : categorical {0.1, 0.2, 0.3}
        lr          : categorical {1e-4, 1e-3, 1e-2}
        lookback    : categorical {22, 44, 60}

    Objective: minimize validation MSE on h=1 (single-step) task.

    Parameters
    ----------
    feat_train_scaled : np.ndarray (T_train, n_features)
    feat_val_scaled   : np.ndarray (T_val,   n_features)
    n_trials          : int   default 30
    n_epochs          : int   default 30 (reduced for speed during search)
    patience          : int   default 7
    n_features        : int
    n_markets         : int
    batch_size        : int
    device            : torch.device
    seed              : int

    Returns
    -------
    best_params : dict  — ready to pass into VolatilityLSTM / train_lstm
    study       : optuna.Study
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

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
        model = VolatilityLSTM(n_features=n_features, hidden_size=hidden,
                                n_layers=layers, dropout=drop,
                                n_outputs=n_markets)
        _, history, _ = train_lstm(model, tr_ldr, vl_ldr,
                                    epochs=n_epochs, lr=lr,
                                    patience=patience,
                                    device=device, verbose=False)
        return min(history["val_loss"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n{'='*55}")
    print(f"LSTM Optuna best params (val MSE={study.best_value:.6f})")
    print(f"{'='*55}")
    for k, v in best.items():
        print(f"  {k:15s}: {v}")

    return best, study
