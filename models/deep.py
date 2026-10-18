"""Phase 5: PyTorch sequence models (LSTM and 1D-CNN), deterministic training, and MC-dropout prediction.

Everything runs on ONE CPU thread with PyTorch's deterministic algorithms and seeded shuffling, because
Phase 4 showed that "same seed" only means "same result" when the thread count and code path are fixed too.
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features.build import RUL_CAP

THREADS = 1
BATCH = 256


def set_determinism(seed: int) -> None:
    """Fix every source of randomness and the thread count."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


class LSTMNet(nn.Module):
    """Reads the window cycle by cycle; the memory after the last cycle is turned into an RUL estimate."""

    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):                      # x: (batch, window, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNNNet(nn.Module):
    """Slides 5-cycle filters along the window to detect local shapes, then maps them to an RUL estimate."""

    def __init__(self, n_features: int, hidden: int, dropout: float, window: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, hidden, kernel_size=5, padding=2), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden * window, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):                      # Conv1d wants (batch, channels, time)
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


def build(arch: str, n_features: int, hidden: int, dropout: float, window: int) -> nn.Module:
    if arch == "LSTM":
        return LSTMNet(n_features, hidden, dropout)
    if arch == "CNN":
        return CNNNet(n_features, hidden, dropout, window)
    raise ValueError(arch)


def _loader(X, y, seed):
    g = torch.Generator().manual_seed(seed)  # seeded shuffling: same batches in the same order every run
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy((y / RUL_CAP).astype(np.float32)))
    return DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)


def _epoch(model, loader, opt):
    model.train()
    loss_fn = nn.MSELoss()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()


@torch.no_grad()
def predict(model, X) -> np.ndarray:
    """Point prediction with dropout OFF, in cycles, clipped to 0-125."""
    model.eval()
    p = model(torch.from_numpy(X)).numpy().astype(np.float64) * RUL_CAP
    return np.clip(p, 0, RUL_CAP)


def train_early_stopping(cfg: dict, seed: int, Xtr, ytr, Xva, yva, max_epochs: int = 60, patience: int = 8):
    """Train on the 80 training engines; keep the epoch with the lowest validation RMSE (all validation rows)."""
    set_determinism(seed)
    model = build(cfg["arch"], Xtr.shape[2], cfg["hidden"], cfg["dropout"], Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = _loader(Xtr, ytr, seed)
    best, best_rmse, best_epoch, curve, waited = None, float("inf"), 0, [], 0
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        _epoch(model, loader, opt)
        r = float(np.sqrt(np.mean((predict(model, Xva) - yva) ** 2)))
        curve.append(r)
        if r < best_rmse - 1e-9:
            best, best_rmse, best_epoch, waited = copy.deepcopy(model.state_dict()), r, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best)
    return model, {"best_epoch": best_epoch, "val_rmse": best_rmse, "curve": curve,
                   "epochs_run": len(curve), "seconds": time.perf_counter() - t0}


def train_fixed(cfg: dict, seed: int, X, y, epochs: int):
    """Refit on all 100 training engines for a fixed number of epochs (chosen on validation)."""
    set_determinism(seed)
    model = build(cfg["arch"], X.shape[2], cfg["hidden"], cfg["dropout"], X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = _loader(X, y, seed)
    t0 = time.perf_counter()
    for _ in range(epochs):
        _epoch(model, loader, opt)
    return model, time.perf_counter() - t0


@torch.no_grad()
def mc_dropout(model, X, passes: int, seed: int) -> np.ndarray:
    """Monte Carlo dropout: `passes` predictions with dropout ON. Returns an array (passes, n) in cycles.

    The networks have no batch-norm, so train mode changes nothing except switching dropout on.
    """
    torch.manual_seed(seed)
    model.train()
    xt = torch.from_numpy(X)
    draws = np.stack([model(xt).numpy().astype(np.float64) * RUL_CAP for _ in range(passes)])
    model.eval()
    return np.clip(draws, 0, RUL_CAP)
