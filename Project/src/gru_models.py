"""
src/gru_models.py
=================
GRU utilities for WarShock-Spillover project.
"""

import torch
import torch.nn as nn

# Re-use constants from lstm_models
from src.lstm_models import (
    LOOKBACK, N_FEATURES, N_MARKETS, HORIZONS,
    BATCH_SIZE, EPOCHS, PATIENCE,
    make_sequences, build_multistep_sequences,
    build_loaders, inverse_transform_vol,
)

HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.2
LR          = 1e-3


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class VolatilityGRU(nn.Module):
    """
    2-layer GRU for multivariate volatility forecasting.

    GRU uses reset + update gates (vs LSTM's input/forget/output).

    Forward
    -------
    Input  : (batch, seq_len, n_features)
    Output : (batch, n_outputs)
    """
    def __init__(self, n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 n_layers=N_LAYERS, dropout=DROPOUT, n_outputs=N_MARKETS):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ═══════════════════════════════════════════════════════════════════════
# Training loop (identical logic to train_lstm)
# ═══════════════════════════════════════════════════════════════════════

def train_gru(model, train_loader, val_loader,
              epochs=EPOCHS, lr=LR, patience=PATIENCE,
              device=None, verbose=True):
    """
    Adam + ReduceLROnPlateau + gradient clipping + early stopping.

    Returns
    -------
    model, history, best_epoch
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
# Inference
# ═══════════════════════════════════════════════════════════════════════

def predict_gru(model, X, device=None):
    """Forward pass to scaled numpy."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# Optuna tuning (identical search space to LSTM)
# ═══════════════════════════════════════════════════════════════════════

def tune_gru_optuna(feat_train_scaled, feat_val_scaled,
                    n_trials=30, n_epochs=30, patience=7,
                    n_features=N_FEATURES, n_markets=N_MARKETS,
                    batch_size=BATCH_SIZE, device=None, seed=42):
    """
    Bayesian hyperparameter optimisation for GRU via Optuna TPE.

    Search space:
        hidden_size : {32, 64, 128}
        n_layers    : {1, 2}
        dropout     : {0.1, 0.2, 0.3}
        lr          : {1e-4, 1e-3, 1e-2}
        lookback    : {22, 44, 60}

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
        model = VolatilityGRU(n_features=n_features, hidden_size=hidden,
                               n_layers=layers, dropout=drop,
                               n_outputs=n_markets)
        _, history, _ = train_gru(model, tr_ldr, vl_ldr,
                                   epochs=n_epochs, lr=lr,
                                   patience=patience,
                                   device=device, verbose=False)
        return min(history["val_loss"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n{'='*55}")
    print(f"GRU Optuna best params (val MSE={study.best_value:.6f})")
    print(f"{'='*55}")
    for k, v in best.items():
        print(f"  {k:15s}: {v}")

    return best, study
