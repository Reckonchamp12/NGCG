"""
ngcg.utils
==========
Shared neural network components and training utilities.
"""

from __future__ import annotations
import copy
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT   = torch.float32
USE_AMP = DEVICE == "cuda"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark  = True
    torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth, width, and activation.

    Parameters
    ----------
    in_d    : input dimension
    out_d   : output dimension
    hidden  : tuple of hidden layer widths
    act     : activation class (default: Tanh)
    """
    def __init__(
        self,
        in_d: int,
        out_d: int,
        hidden: tuple[int, ...] = (256, 256),
        act: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        dims = [in_d] + list(hidden) + [out_d]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers.pop()  # remove final activation
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_pairs(traj: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert (N, T, D) trajectory array to (x_t, x_{t+1}) GPU tensors.
    Shape of each output: (N*(T-1), D).
    """
    x = torch.tensor(traj[:, :-1].reshape(-1, traj.shape[-1]), dtype=FLOAT, device=DEVICE)
    y = torch.tensor(traj[:,  1:].reshape(-1, traj.shape[-1]), dtype=FLOAT, device=DEVICE)
    return x, y


def train_mlp(
    model: nn.Module,
    tr_traj: np.ndarray,
    va_traj: np.ndarray,
    epochs: int,
    patience: int,
    lr: float,
    batch: int = 2048,
    verbose: bool = True,
) -> tuple[float, int]:
    """
    Train a model with MSE loss, OneCycleLR, optional AMP.
    All trajectory pairs are kept GPU-resident for speed.

    Returns
    -------
    (best_val_mse, epochs_run)
    """
    model = model.to(DEVICE)
    xtr, ytr = make_pairs(tr_traj)
    xva, yva = make_pairs(va_traj)
    N = len(xtr)
    if N == 0:
        return float("inf"), 0

    bs      = min(batch, N)
    n_batch = max(1, N // bs)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=n_batch, pct_start=0.1
    )
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    best   = float("inf")
    pat    = 0
    bw     = copy.deepcopy(model.state_dict())
    perm   = torch.randperm(N, device=DEVICE)

    for ep in range(epochs):
        model.train()
        perm = perm[torch.randperm(N, device=DEVICE)]
        for i in range(n_batch):
            sl = perm[i * bs: (i + 1) * bs]
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = F.mse_loss(model(xtr[sl]), ytr[sl])
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                F.mse_loss(model(xtr[sl]), ytr[sl]).backward(); opt.step()
            sched.step()

        model.eval()
        with torch.no_grad(), (torch.cuda.amp.autocast() if USE_AMP
                               else contextlib.nullcontext()):
            vl = F.mse_loss(model(xva), yva).item()

        if vl < best - 1e-7:
            best = vl; pat = 0; bw = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= patience:
                break
        if verbose and (ep + 1) % 20 == 0:
            print(f"    ep {ep+1:3d}/{epochs}  val_mse={vl:.5f}  best={best:.5f}")

    model.load_state_dict(bw)
    return best, ep + 1


def rollout(
    model: nn.Module,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    Autoregressive model rollout.

    Parameters
    ----------
    model : trained MLP, eval mode
    x0    : (N, D) initial conditions
    steps : number of rollout steps

    Returns
    -------
    (N, steps, D) numpy array
    """
    model.eval()
    x   = torch.tensor(x0, dtype=FLOAT, device=DEVICE)
    out = []
    ctx = torch.cuda.amp.autocast() if USE_AMP else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        for _ in range(steps):
            x = model(x); out.append(x)
    return torch.stack(out, dim=1).cpu().numpy()
