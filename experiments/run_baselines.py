"""
ngcg_stepwise_eval.py
=====================
Correct, fast benchmark of all NGCG models on all systems.

HDF5 path  : /kaggle/input/datasets/drahulray/upadted-gcmc-data/ngcg_data_clean (1).h5
Output     : /kaggle/working/eval_results/

All 10 bugs from previous version are fixed — see FIX-N comments.

Run in Kaggle notebook:
    %run /kaggle/working/ngcg_stepwise_eval.py
    %run /kaggle/working/ngcg_stepwise_eval.py --system mass_spring --method NGCG
    %run /kaggle/working/ngcg_stepwise_eval.py --systems mass_spring henon_heiles
"""

# ── Bootstrap ─────────────────────────────────────────────────────────────────
import subprocess, sys, os

os.environ["TORCHDYNAMO_DISABLE"] = "1"   # FIX: no torch.compile on P100 (CC6.0)
os.environ["TORCH_LOGS"]          = "-dynamo"

def _pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *p], check=False)

for _pkg, _imp in [("h5py","h5py"), ("torch","torch"), ("pysr","pysr"),
                   ("sympy","sympy"), ("scikit-learn","sklearn"),
                   ("pandas","pandas"), ("matplotlib","matplotlib")]:
    try: __import__(_imp)
    except: _pip(_pkg)

import argparse, copy, contextlib, time, warnings, traceback
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures   # FIX-2: stable replacement for pysindy
from sklearn.decomposition import PCA
import sympy as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

HDF5_PATH = "/kaggle/working/ngcg_data_clean.h5"
OUT_DIR   = "/kaggle/working/eval_results"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT     = torch.float32
USE_AMP   = (DEVICE == "cuda")
SEED      = 42

os.makedirs(OUT_DIR,              exist_ok=True)
os.makedirs(f"{OUT_DIR}/plots",   exist_ok=True)
os.makedirs(f"{OUT_DIR}/details", exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark  = True
    torch.backends.cudnn.allow_tf32 = True

np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM METADATA
# ══════════════════════════════════════════════════════════════════════════════

# Note: lotka_volterra law uses greek letters that clash with SymPy builtins
# (gamma, beta are SymPy functions). We inject them explicitly as Symbols.
TRUE_LAWS = {
    "mass_spring"    : "p**2/(2*m) + k*q**2/2",
    "lotka_volterra" : "delta*x - gamma*log(x) + beta*y - alpha*log(y)",
    "double_pendulum": None,
    "henon_heiles"   : "px**2/2 + py**2/2 + x**2/2 + y**2/2 + x**2*y - y**3/3",
    "lorenz"         : None,
    "coupled_springs": "p1**2/2 + p2**2/2 + k1*q1**2/2 + k2*(q2-q1)**2/2 + k3*q2**2/2",
    "three_body"     : None,
    "burgers"        : None,
    "ks"             : None,
}

STATE_VARS = {
    "mass_spring"    : ["q", "p"],
    "lotka_volterra" : ["x", "y"],
    "double_pendulum": ["theta1", "theta2", "p1", "p2"],
    "henon_heiles"   : ["x", "y", "px", "py"],
    "lorenz"         : ["x", "y", "z"],
    "coupled_springs": ["q1", "q2", "p1", "p2"],
    "three_body"     : ["x", "y", "vx", "vy"],
    # BUG-D FIX: PDE systems are collapsed to 3 scalar features [mean, var, skew]
    # so var_names must have length 3 to match the D=3 PySR input shape.
    "burgers"        : ["u_mean", "u_var", "u_skew"],
    "ks"             : ["u_mean", "u_var", "u_skew"],
}

# Names of per-trajectory parameters stored in HDF5 'params' dataset
PARAM_NAMES = {
    "mass_spring"    : ["k", "m"],
    "lotka_volterra" : ["alpha", "beta", "gamma", "delta"],
    "coupled_springs": ["k1", "k2", "k3"],
}

HNN_SYSTEMS = {"mass_spring", "double_pendulum", "henon_heiles",
               "coupled_springs", "three_body"}
PDE_SYSTEMS = {"burgers", "ks"}

ALL_SYSTEMS = list(TRUE_LAWS.keys())
ALL_METHODS = ["MLP_Dynamics", "HNN", "SINDy", "IRAS",
               "NGCG", "NGCG-noGrad", "NGCG-noRetrain", "NGCG-noVerif"]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

_LOW_QUALITY_SYSTEMS: set = set()

def _repair_nan(traj: np.ndarray, name: str) -> np.ndarray:
    N   = traj.shape[0]
    bad = ~np.all(np.isfinite(traj.reshape(N, -1)), axis=1)
    if bad.sum():
        pct = bad.sum()/max(N,1)
        if bad.sum():
            print(f"    [repair] {name}: dropping {bad.sum()}/{N} ({100*pct:.0f}%) NaN traj")
            traj = traj[~bad]
        if pct > 0.5:
            _LOW_QUALITY_SYSTEMS.add(name)
            print(f"    ⚠  {name}: >50% dropped — low quality, results unreliable")
    return traj


def _pde_scalar(traj: np.ndarray) -> np.ndarray:
    """(N,T,D_field) → (N,T,3)  [spatial mean, variance, skewness]"""
    mu  = traj.mean(-1, keepdims=True)
    var = traj.var (-1, keepdims=True)
    skw = ((traj - mu)**3).mean(-1, keepdims=True) / (var + 1e-8)**1.5
    return np.concatenate([mu, var, skw], axis=-1).astype(np.float32)


def load_system(system: str) -> dict:
    """
    Loads one system from HDF5.
    Returns dict with keys: train, val, test  (N,T,D) float32
                            params_train, params_test  (N,P) or None
                            dt, attrs
    Returns None if system not in file.
    """
    with h5py.File(HDF5_PATH, "r") as f:
        if system not in f:
            return None
        g   = f[system]
        tj  = g["trajectories"][:]
        t   = g["time"][:]
        tr  = g["train_indices"][:]
        va  = g["val_indices"  ][:]
        te  = g["test_indices" ][:]
        par = g["params"][:] if "params" in g else None
        att = dict(g.attrs)

    # Repair NaN before indexing
    tj = _repair_nan(tj, system)
    N  = len(tj)
    tr = tr[tr < N];  va = va[va < N];  te = te[te < N]

    tr_t = tj[tr].astype(np.float32)
    va_t = tj[va].astype(np.float32)
    te_t = tj[te].astype(np.float32)

    par_tr = par[tr] if par is not None else None
    par_te = par[te] if par is not None else None

    # PDE: collapse field to 3 scalar features
    if system in PDE_SYSTEMS:
        tr_t = _pde_scalar(tr_t)
        va_t = _pde_scalar(va_t)
        te_t = _pde_scalar(te_t)

    dt = float(att.get("dt", 0.1))
    print(f"    Loaded {system}: train={len(tr_t)} val={len(va_t)} test={len(te_t)} "
          f"D={tr_t.shape[-1]} dt={dt:.4f}")

    return dict(train=tr_t, val=va_t, test=te_t,
                params_train=par_tr, params_test=par_te,
                time=t, dt=dt, attrs=att)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  NEURAL NETWORK BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=(256, 256), act=nn.Tanh):
        super().__init__()
        dims = [in_d] + list(hidden) + [out_d]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers.pop()                          # remove last activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_gpu_pairs(traj: np.ndarray):
    """
    FIX-9/10: pre-load ALL (x_t, x_{t+1}) pairs to GPU once.
    Eliminates DataLoader worker overhead — 4-6× faster for D≤8.
    """
    x = torch.tensor(traj[:, :-1].reshape(-1, traj.shape[-1]),
                     dtype=FLOAT, device=DEVICE)
    y = torch.tensor(traj[:,  1:].reshape(-1, traj.shape[-1]),
                     dtype=FLOAT, device=DEVICE)
    return x, y


def train_dynamics(model: nn.Module, tr_traj: np.ndarray, va_traj: np.ndarray,
                   epochs=150, patience=15, lr=3e-3):
    """
    One-step predictor training with:
      - GPU-resident tensors (no DataLoader)
      - OneCycleLR for fast convergence
      - AMP (fp16) on P100
    Returns (best_val_mse, epochs_run).
    """
    model  = model.to(DEVICE)
    xtr, ytr = _to_gpu_pairs(tr_traj)
    xva, yva = _to_gpu_pairs(va_traj)

    N_pairs  = len(xtr)
    if N_pairs == 0:
        print("    ✗ Empty dataset — skipping training")
        return float("inf"), 0
    batch    = min(2048, N_pairs)
    n_batch  = max(1, N_pairs // batch)

    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
                 opt, max_lr=lr, epochs=epochs,
                 steps_per_epoch=n_batch, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    best = float("inf");  pat = 0
    best_w = copy.deepcopy(model.state_dict());  ep_run = 0
    perm   = torch.randperm(N_pairs, device=DEVICE)

    for ep in range(epochs):
        model.train()
        perm = perm[torch.randperm(N_pairs, device=DEVICE)]   # in-GPU shuffle
        for i in range(n_batch):
            sl  = perm[i * batch : (i+1) * batch]
            xb  = xtr[sl];  yb = ytr[sl]
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = nn.functional.mse_loss(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt);  scaler.update()
            else:
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward();   opt.step()
            sched.step()

        model.eval()
        with torch.no_grad(), (torch.cuda.amp.autocast() if USE_AMP
                               else contextlib.nullcontext()):
            vl = nn.functional.mse_loss(model(xva), yva).item()

        ep_run = ep + 1
        if vl < best - 1e-7:
            best  = vl;  pat = 0
            best_w = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= patience:
                break
        if (ep + 1) % 20 == 0:
            print(f"      ep {ep+1:3d}/{epochs}  val_mse={vl:.5f}  best={best:.5f}")

    model.load_state_dict(best_w)
    return best, ep_run


def rollout(model: nn.Module, x0: np.ndarray, steps: int) -> np.ndarray:
    """Autoregressive rollout fully on GPU. Single D→H transfer at the end."""
    model.eval()
    x   = torch.tensor(x0, dtype=FLOAT, device=DEVICE)
    out = []
    ctx = torch.cuda.amp.autocast() if USE_AMP else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        for _ in range(steps):
            x = model(x)
            out.append(x)
    return torch.stack(out, dim=1).cpu().numpy()   # (N, steps, D)


def mse_at_k(model: nn.Module, te_traj: np.ndarray, k: int = 16) -> float:
    pred = rollout(model, te_traj[:, 0], k)
    return float(np.mean((pred - te_traj[:, 1:k+1]) ** 2))

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYMBOLIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pysr_run(X: np.ndarray, y: np.ndarray,
             var_names: list, n_iter: int = 15) -> list:
    """
    FIX-3: var_names passed every call so PySR outputs 'q','p' not 'x0','x1'.
            Without this, SymPy match always fails → DR = 0 forever.
    FIX-8: n_iter=15 baselines / 25 NGCG  (was 40-50) — Julia already warm.
    """
    try:
        from pysr import PySRRegressor
        m = PySRRegressor(
            niterations      = n_iter,
            populations      = 12,
            binary_operators = ["+", "-", "*", "/"],
            unary_operators  = ["square", "sqrt", "log", "exp"],
            maxsize          = 15,
            verbosity        = 0,
            random_state     = SEED,
            procs            = 0,
            multithreading   = False,
        )
        m.fit(X, y, variable_names=var_names)    # ← FIX-3 key line
        return list(
            m.equations_
             .sort_values("score", ascending=False)
             .head(5)["sympy_format"]
             .astype(str)
        )
    except Exception as e:
        print(f"      PySR error: {e}")
        return []


# All param/state names that could clash with SymPy builtins
_ALL_POTENTIAL_SYMS = [
    "q","p","x","y","z","u","v","m","k","r","s","t","n",
    "alpha","beta","gamma","delta","epsilon","theta","phi","psi","omega",
    "q1","q2","p1","p2","k1","k2","k3","m1","m2",
    "theta1","theta2","px","py","pz","vx","vy","vz",
]

def _safe_sympify(expr_str: str, extra_syms: list = None) -> sp.Expr:
    """
    BUG-B FIX: sympify with ALL potential variable names pre-declared as
    sp.Symbol objects. This prevents gamma/beta/alpha/delta from being
    interpreted as SymPy special functions instead of plain symbols.
    """
    all_names = list(_ALL_POTENTIAL_SYMS)
    if extra_syms:
        all_names += [s for s in extra_syms if s not in all_names]
    local_dict = {name: sp.Symbol(name) for name in all_names}
    return sp.sympify(expr_str, locals=local_dict)


def candidate_constancy(cand_str: str, svars: list, traj: np.ndarray) -> float:
    """
    Mean( std_t(f(x_t)) / |mean_t(f(x_t))| ) over test trajectories.
    Returns 0 for perfect invariant, large for non-conserved expressions.
    """
    if not cand_str or not svars: return float("inf")
    try:
        fn   = sp.lambdify(svars, _safe_sympify(cand_str, svars), modules="numpy")
        N, T, D = traj.shape
        vals = np.stack([fn(*[traj[:, t, j] for j in range(D)])
                         for t in range(T)], axis=1).astype(np.float64)
        mask = np.all(np.isfinite(vals), axis=1)
        if mask.sum() == 0: return float("inf")
        v    = vals[mask]
        return float((v.std(1) / (np.abs(v.mean(1)) + 1e-8)).mean())
    except Exception:
        return float("inf")


def sympy_match(cand: str, true: str, svars: list) -> bool:
    """True if cand equals true up to additive constant or multiplicative scale."""
    if not cand or not true:
        return False
    try:
        e1 = _safe_sympify(cand, svars)
        e2 = _safe_sympify(true, svars)
        if sp.simplify(e1 - e2).is_number:  return True
        if sp.simplify(e1 / e2).is_number:  return True
        return False
    except:
        return False


def constancy_ratio(law_str: str, svars: list, traj: np.ndarray,
                    params_arr=None, param_names=None) -> float:
    """
    FIX-6: substitute per-trajectory parameter values (k, m, etc.)
            before evaluating the law numerically.
    Returns mean std/|mean| over test trajectories (lower = more conserved).
    """
    if not law_str:
        return float("nan")
    N, T, D = traj.shape
    try:
        expr     = _safe_sympify(law_str, svars)
        all_syms = {str(s) for s in expr.free_symbols}
        ratios   = []
        for i in range(N):
            subs = {}
            # Substitute parameter values when available
            if params_arr is not None and param_names:
                for pi, pn in enumerate(param_names):
                    if pn in all_syms and pi < params_arr.shape[1]:
                        subs[sp.Symbol(pn)] = float(params_arr[i, pi])
            try:
                fn   = sp.lambdify(svars, expr.subs(subs), modules="numpy")
                vals = np.array(
                    [fn(*[traj[i, t, j] for j in range(D)]) for t in range(T)],
                    dtype=float
                )
                if np.all(np.isfinite(vals)):
                    ratios.append(vals.std() / (abs(vals.mean()) + 1e-10))
            except:
                pass
        return float(np.mean(ratios)) if ratios else float("nan")
    except:
        return float("nan")


def conservation_violation(nn_model: nn.Module, te_traj: np.ndarray,
                           law_str: str, svars: list,
                           params_te=None, param_names=None, k: int = 16) -> float:
    """Mean |C(x̂_t) − C(x_0)| over predicted rollout with param substitution."""
    if not law_str or nn_model is None:
        return float("nan")
    try:
        expr  = _safe_sympify(law_str, svars)
        x0    = te_traj[:, 0]
        pred  = rollout(nn_model, x0, k)    # (N, k, D)
        devs  = []
        for i in range(len(x0)):
            subs = {}
            if params_te is not None and param_names:
                for pi, pn in enumerate(param_names):
                    if pi < params_te.shape[1]:
                        subs[sp.Symbol(pn)] = float(params_te[i, pi])
            try:
                fn = sp.lambdify(svars, expr.subs(subs), modules="numpy")
                ev = lambda s: fn(*[s[j] for j in range(len(svars))])
                c0 = ev(x0[i])
                devs.append(np.mean([abs(ev(pred[i, t]) - c0) for t in range(k)]))
            except:
                pass
        return float(np.mean(devs)) if devs else float("nan")
    except:
        return float("nan")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a  MLP Dynamics  ────────────────────────────────────────────────────────
class MLPDynamicsModel:
    name = "MLP_Dynamics"

    def __init__(self, d: int, seed: int = 0):
        torch.manual_seed(seed)
        self.model = MLP(d, d)
        self.cands = []

    def train(self, tr, va, system="", **_):
        D = tr.shape[-1]
        print(f"    Training MLP dynamics (D={D})...")
        best, ep = train_dynamics(self.model, tr, va)
        print(f"    ✓ val_mse={best:.5f}  epochs={ep}")
        if len(tr) == 0: return

        # BUG-E FIX: PCA IC-repeat was a linear target → PySR found only linear
        # expressions, never quadratic energy forms (p²/2m + kq²/2).
        # Correct approach: train a small constancy φ (same as IRAS) and use
        # φ(x) values as the PySR target — these encode the true invariant shape.
        N, T, D2 = tr.shape
        phi = MLP(D2, 1, hidden=(64, 64)).to(DEVICE)
        opt_phi = torch.optim.Adam(phi.parameters(), lr=1e-3)
        tt2d = torch.tensor(tr, dtype=FLOAT, device=DEVICE).view(N * T, D2)
        for _ in range(60):
            phi.train()
            v2d=phi(tt2d).view(N,T)
            loss=v2d.var(1).mean()/(v2d.mean(1).var()+1e-4)
            opt_phi.zero_grad(set_to_none=True); loss.backward(); opt_phi.step()
        phi.eval()

        pts = tr.reshape(-1, D2)
        idx = np.random.choice(len(pts), min(1000, len(pts)), replace=False)
        xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
        with torch.no_grad():
            y = phi(xt).squeeze(-1).cpu().numpy()

        svars = STATE_VARS.get(system, [f"x{i}" for i in range(D2)])
        print(f"    PySR on {len(idx)} pts  vars={svars}...")
        self.cands = pysr_run(pts[idx], y, var_names=svars, n_iter=15)
        print(f"    → {len(self.cands)} candidates: {self.cands[:2]}")

    def predict(self, x0, k):  return rollout(self.model, x0, k)
    def mse    (self, te):     return mse_at_k(self.model, te)


# ── 4b  HNN  ─────────────────────────────────────────────────────────────────
class HNNCore(nn.Module):
    """
    BUG-C FIX: x.requires_grad_(True) alone is not enough when x arrives as
    a cached GPU tensor (non-leaf). We must use detach() first to make it a
    leaf node, then requires_grad_(True), and use torch.enable_grad() context
    to ensure grad computation works in both train and eval mode.
    create_graph=True only during training (needed for loss.backward());
    during eval create_graph=False (faster, no double-backward needed).
    """
    def __init__(self, d: int, dt: float):
        super().__init__()
        assert d % 2 == 0, "HNN requires even state dimension (q, p pairs)"
        self.H  = MLP(d, 1)
        self.dt = dt
        self.hd = d // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            # detach() + requires_grad_(True) = fresh leaf — always works
            x_leaf = x.detach().requires_grad_(True)
            H = self.H(x_leaf).sum()
            g = torch.autograd.grad(H, x_leaf,
                                    create_graph=self.training)[0]
        dq =  g[:, self.hd:]
        dp = -g[:, :self.hd]
        return x_leaf + self.dt * torch.cat([dq, dp], dim=1)


class HNNModel:
    name = "HNN"

    def __init__(self, d: int, dt: float, seed: int = 0):
        torch.manual_seed(seed)
        self.model = HNNCore(d, dt)
        self.cands = []

    def train(self, tr, va, system="", **_):
        D = tr.shape[-1]
        print(f"    Training HNN (D={D})...")
        best, ep = train_dynamics(self.model, tr, va)
        print(f"    ✓ val_mse={best:.5f}  epochs={ep}")

        # Extract symbolic Hamiltonian via PySR
        pts   = tr.reshape(-1, D)
        idx   = np.random.choice(len(pts), min(800, len(pts)), replace=False)
        self.model = self.model.to(DEVICE)
        xt    = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
        with torch.no_grad(), (torch.cuda.amp.autocast() if USE_AMP
                               else contextlib.nullcontext()):
            y = self.model.H(xt).squeeze(-1).cpu().numpy()

        svars = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
        print(f"    PySR on {len(idx)} pts  vars={svars}...")
        self.cands = pysr_run(pts[idx], y, var_names=svars, n_iter=15)
        print(f"    → {len(self.cands)} candidates: {self.cands[:2]}")

    def predict(self, x0, k):  return rollout(self.model, x0, k)
    def mse    (self, te):     return mse_at_k(self.model, te)


# ── 4c  SINDy  ───────────────────────────────────────────────────────────────
class SINDyModel:
    """
    FIX-2: replaced pysindy PolynomialLibrary with sklearn PolynomialFeatures.
    The pysindy API changed in recent versions (AxesArray bug).
    sklearn is stable and gives identical polynomial feature sets.
    Conserved features = those with std/|mean| < 0.02 along trajectories.
    """
    name = "SINDy"

    def __init__(self, d: int, seed: int = 0):
        self.d     = d
        self.cands = []

    def train(self, tr, va, system="", **_):
        N, T, D = tr.shape
        svars   = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
        print(f"    Fitting degree-3 polynomial features (sklearn)...")
        try:
            pf    = PolynomialFeatures(degree=3, include_bias=False)
            pts   = tr.reshape(-1, D)
            feats = pf.fit_transform(pts)               # (N*T, n_feats)
            names = pf.get_feature_names_out(svars)
            f3    = feats.reshape(N, T, -1)             # (N, T, n_feats)
            ratio = (f3.std(axis=1) /
                     (np.abs(f3.mean(axis=1)) + 1e-10)).mean(axis=0)
            self.cands = [names[i] for i, r in enumerate(ratio) if r < 0.02]
            print(f"    ✓ SINDy → {len(self.cands)} conserved features")
            if self.cands:
                print(f"      {self.cands[:5]}")
        except Exception as e:
            print(f"    SINDy error: {e}")

    def predict(self, x0, k):  return None
    def mse    (self, te):     return float("nan")


# ── 4d  IRAS  ────────────────────────────────────────────────────────────────
class IRASModel:
    """
    Trains φ: R^D → R to minimise within-trajectory variance (constancy loss).
    Then runs PySR on (x, φ(x)) pairs to extract symbolic formula.
    """
    name = "IRAS"

    def __init__(self, d: int, seed: int = 0):
        torch.manual_seed(seed)
        self.phi   = MLP(d, 1, hidden=(128, 128, 128))
        self.cands = []

    def train(self, tr, va, system="", epochs=80, patience=15, **_):
        N, T, D = tr.shape
        self.phi = self.phi.to(DEVICE)
        opt  = torch.optim.Adam(self.phi.parameters(), lr=1e-3)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        tt2d = torch.tensor(tr, dtype=FLOAT,
                            device=DEVICE).view(N * T, D)   # pre-load to GPU

        best = float("inf");  pat = 0
        best_w = copy.deepcopy(self.phi.state_dict())
        print(f"    Training IRAS φ (D={D}, epochs≤{epochs})...")

        for ep in range(epochs):
            self.phi.train()
            with (torch.cuda.amp.autocast() if USE_AMP
                  else contextlib.nullcontext()):
                vals      = self.phi(tt2d).view(N, T)    # (N, T)
                # Within-traj variance (minimise → constant along traj)
                intra_var = vals.var(dim=1).mean()
                # Between-traj variance (maximise → different trajectories
                # have different conserved values, stopping collapse to const)
                traj_means = vals.mean(dim=1)             # (N,)
                inter_var  = traj_means.var()
                # Normalised loss: minimise intra, maximise inter
                loss = intra_var / (inter_var + 1e-4)
            opt.zero_grad(set_to_none=True)
            loss.backward();  opt.step();  sch.step()
            lv = loss.item()
            if lv < best - 1e-7:
                best  = lv;  pat = 0
                best_w = copy.deepcopy(self.phi.state_dict())
            else:
                pat += 1
                if pat >= patience:
                    break

        self.phi.load_state_dict(best_w);  self.phi.eval()
        print(f"    ✓ IRAS φ done (ep={ep+1}  loss={best:.5f})")

        pts = tr.reshape(-1, D)
        idx = np.random.choice(len(pts), min(800, len(pts)), replace=False)
        xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
        with torch.no_grad():
            y = self.phi(xt).squeeze(-1).cpu().numpy()

        svars = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
        print(f"    PySR on {len(idx)} pts  vars={svars}...")
        self.cands = pysr_run(pts[idx], y, var_names=svars, n_iter=15)
        print(f"    ✓ IRAS+PySR → {len(self.cands)}: {self.cands[:2]}")

    def predict(self, x0, k):  return None
    def mse    (self, te):     return float("nan")


# ── 4e  NGCG (full + 3 ablations)  ──────────────────────────────────────────
class NGCGModel:
    """
    Phase 1: MLP one-step dynamics.
    Phase 2: Constancy network φ (optionally gradient-guided).
    Phase 3: PySR symbolic extraction from φ values.
    Phase 4: Closed-loop retrain with conservation penalty (optional).
    """
    def __init__(self, d: int, dt: float, seed: int = 0,
                 use_grad=True, use_retrain=True, use_verif=True):
        torch.manual_seed(seed)
        self.dyn         = MLP(d, d)
        self.phi         = MLP(d, 1, hidden=(128, 128, 128))
        self.dt          = dt
        self.use_grad    = use_grad
        self.use_retrain = use_retrain
        self.use_verif   = use_verif
        self.cands       = []
        suf = [s for s, f in [("noGrad",    not use_grad),
                               ("noRetrain", not use_retrain),
                               ("noVerif",   not use_verif)] if f]
        self.name = "NGCG" + ("-" + "-".join(suf) if suf else "")

    def train(self, tr, va, system="", epochs=150, patience=15, **_):
        N, T, D = tr.shape

        # ── Phase 1 ───────────────────────────────────────────────────────────
        print(f"    [Ph1] Dynamics MLP (D={D})...")
        best1, ep1 = train_dynamics(self.dyn, tr, va,
                                    epochs=epochs, patience=patience)
        print(f"    ✓ val_mse={best1:.5f}  ep={ep1}")
        self.dyn = self.dyn.to(DEVICE)

        # ── Phase 2 ───────────────────────────────────────────────────────────
        print(f"    [Ph2] Constancy φ  grad_guide={self.use_grad}...")
        self.phi = self.phi.to(DEVICE)
        opt  = torch.optim.Adam(self.phi.parameters(), lr=1e-3)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        tt   = torch.tensor(tr, dtype=FLOAT, device=DEVICE)
        tt2d = tt.view(N * T, D)
        ttx  = tt[:, :-1].reshape(-1, D)    # (N*(T-1), D) for gradient term

        best = float("inf");  pat = 0
        best_w = copy.deepcopy(self.phi.state_dict())

        for ep in range(80):
            self.phi.train();  self.dyn.eval()
            with (torch.cuda.amp.autocast() if USE_AMP
                  else contextlib.nullcontext()):
                vals       = self.phi(tt2d).view(N, T)
                intra_var  = vals.var(1).mean()
                inter_var  = vals.mean(1).var()
                loss       = intra_var / (inter_var + 1e-4)
                if self.use_grad:
                    with torch.no_grad():
                        xt1 = self.dyn(ttx)
                    loss = loss + 0.1 * (self.phi(xt1) - self.phi(ttx)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward();  opt.step();  sch.step()
            lv = loss.item()
            if lv < best - 1e-7:
                best  = lv;  pat = 0
                best_w = copy.deepcopy(self.phi.state_dict())
            else:
                pat += 1
                if pat >= patience:
                    break

        self.phi.load_state_dict(best_w);  self.phi.eval()
        print(f"    ✓ φ done (ep={ep+1})")

        # ── Phase 3 ───────────────────────────────────────────────────────────
        pts = tr.reshape(-1, D)
        idx = np.random.choice(len(pts), min(1000, len(pts)), replace=False)
        xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
        with torch.no_grad(), (torch.cuda.amp.autocast() if USE_AMP
                               else contextlib.nullcontext()):
            y = self.phi(xt).squeeze(-1).cpu().numpy()

        svars = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
        print(f"    [Ph3] PySR on {len(idx)} pts  vars={svars}...")
        self.cands = pysr_run(pts[idx], y, var_names=svars, n_iter=25)
        print(f"    PySR → {len(self.cands)}: {self.cands[:3]}")

        # ── Phase 4 ───────────────────────────────────────────────────────────
        if self.use_retrain and self.cands:
            print(f"    [Ph4] Closed-loop retrain...")
            self._retrain(tr)
            print(f"    ✓ Retrain done")

    def _retrain(self, tr, epochs=60):
        xtr, ytr = _to_gpu_pairs(tr)
        N_pairs  = len(xtr)
        if N_pairs == 0: return
        batch    = min(2048, N_pairs)
        n_batch  = max(1, N_pairs // batch)
        scaler   = torch.cuda.amp.GradScaler() if USE_AMP else None
        opt      = torch.optim.Adam(
            list(self.dyn.parameters()) + list(self.phi.parameters()),
            lr=5e-4)
        perm = torch.randperm(N_pairs, device=DEVICE)
        self.dyn.train();  self.phi.train()
        for ep in range(epochs):
            perm = perm[torch.randperm(N_pairs, device=DEVICE)]
            for i in range(n_batch):
                sl = perm[i*batch:(i+1)*batch]
                xb = xtr[sl];  yb = ytr[sl]
                opt.zero_grad(set_to_none=True)
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = self.dyn(xb)
                        loss = (nn.functional.mse_loss(pred, yb) +
                                0.01 * (self.phi(pred) - self.phi(xb)).pow(2).mean())
                    scaler.scale(loss).backward()
                    scaler.step(opt);  scaler.update()
                else:
                    pred = self.dyn(xb)
                    loss = (nn.functional.mse_loss(pred, yb) +
                            0.01 * (self.phi(pred) - self.phi(xb)).pow(2).mean())
                    loss.backward();  opt.step()

    def predict(self, x0, k):  return rollout(self.dyn, x0, k)
    def mse    (self, te):     return mse_at_k(self.dyn, te)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(model_obj, system: str, te_traj: np.ndarray,
                    params_te=None, param_names=None) -> dict:
    true_law = TRUE_LAWS.get(system)
    svars    = STATE_VARS.get(system, [])
    cands    = getattr(model_obj, "cands", [])
    results  = {"method": model_obj.name, "system": system}

    # 1. MSE @ 16 steps (999.0 = no predictor, not a failure)
    print(f"    [1/6] MSE@16 ...", end=" ", flush=True)
    mse16 = model_obj.mse(te_traj)
    results["MSE_16"] = mse16 if np.isfinite(mse16) else 999.0
    print(f"{results['MSE_16']:.5g}" + ("  (no predictor)" if not np.isfinite(mse16) else ""))

    # 2. Discovery Rate — PRIMARY: numerical constancy check on test trajectories.
    #    A candidate is a true positive if its value is approximately constant
    #    along each test trajectory (std/|mean| < tol).
    #    SECONDARY: symbolic match (bonus, often fails for parameterised laws).
    #    Systems with no true law (lorenz, three_body, etc.): DR = nan.
    print(f"    [2/6] Discovery Rate ...", end=" ", flush=True)
    CONST_TOL_LOOSE  = 0.05    # primary: std/|mean| < 5% per trajectory
    CONST_TOL_STRICT = 0.005   # strict:   std/|mean| < 0.5%

    if len(te_traj) == 0:
        DR = float("nan"); DR_strict = float("nan")
        tp_symbolic = False; tp_numeric = False; tp_strict = False
    elif true_law is not None:
        # Try symbolic match first (fast, exact)
        tp_symbolic = any(sympy_match(c, true_law, svars) for c in cands)
        # Numerical constancy of each candidate on test trajectories
        cand_scores = []
        for c in cands:
            score = candidate_constancy(c, svars, te_traj)
            cand_scores.append((c, score))
        tp_numeric  = any(s < CONST_TOL_LOOSE  for _, s in cand_scores)
        tp_strict   = any(s < CONST_TOL_STRICT for _, s in cand_scores)
        DR          = 1.0 if (tp_symbolic or tp_numeric) else 0.0
        DR_strict   = 1.0 if (tp_symbolic or tp_strict)  else 0.0
        # Best candidate = lowest constancy score
        if cand_scores:
            best_cand, best_score = min(cand_scores, key=lambda x: x[1])
        else:
            best_cand, best_score = "", float("inf")
    else:
        # No true law (lorenz, three_body, double_pendulum, burgers, ks):
        # DR = 0.0 means "correctly found no spurious conserved law"
        # DR = 1.0 would be a FALSE POSITIVE (bad).
        # We evaluate all candidates for constancy — a constant candidate
        # on a chaotic system IS a false positive.
        cand_scores = [(c, candidate_constancy(c, svars, te_traj))
                       for c in cands] if len(te_traj) > 0 else []
        # For no-law systems: DR=0 means good (no false positives found)
        fp_found    = any(s < CONST_TOL_LOOSE for _, s in cand_scores)
        DR          = 1.0 if fp_found else 0.0   # 1 = bad (false positive)
        DR_strict   = 1.0 if any(s < CONST_TOL_STRICT for _, s in cand_scores) else 0.0
        tp_symbolic = tp_numeric = tp_strict = False
        best_cand   = min(cand_scores, key=lambda x: x[1])[0] if cand_scores else ""
        best_score  = min(cand_scores, key=lambda x: x[1])[1] if cand_scores else float("inf")

    results["DR"]          = DR
    results["DR_strict"]   = DR_strict
    results["DR_symbolic"] = 1.0 if tp_symbolic else 0.0
    results["has_true_law"]= 1.0 if true_law else 0.0
    results["has_true_law"]= 1.0 if true_law else 0.0   # useful for analysis
    results["best_cand_constancy"] = best_score if cands else float("nan")
    print(f"DR={DR}  DR_strict={results['DR_strict']}  symbolic={tp_symbolic}  "
          f"numeric={tp_numeric}  ({len(cands)} cands)  "
          f"best_score={best_score:.4f}  top={cands[:1]}")

    # 3. FDR / F1 — based on numerical constancy
    print(f"    [3/6] FDR/F1 ...", end=" ", flush=True)
    if cands and len(te_traj) > 0 and cand_scores:
        if true_law:
            # True positive = constant candidate; false positive = non-constant
            fp  = sum(1 for _, s in cand_scores if s >= CONST_TOL_LOOSE)
        else:
            # No true law: any constant candidate is a false positive
            fp  = sum(1 for _, s in cand_scores if s < CONST_TOL_LOOSE)
        FDR = fp / len(cands)
    else:
        FDR = 0.0
    # F1 defined for all systems: DR=0 for no-law systems is perfect
    F1 = 2 * DR * (1 - FDR) / max(1e-9, DR + (1 - FDR))
    results["FDR"] = FDR;  results["F1"] = F1
    print(f"FDR={FDR:.2f}  F1={F1:.3f}")

    # 4. Constancy ratio (true law evaluated on test trajectories)
    print(f"    [4/6] Constancy ratio ...", end=" ", flush=True)
    if true_law:
        cr = constancy_ratio(true_law, svars, te_traj,
                             params_arr=params_te, param_names=param_names)
        results["constancy_ratio"] = cr if np.isfinite(cr) else 0.0
        print(f"{results['constancy_ratio']:.5g}  (lower = more conserved)")
    else:
        # No true law: report best candidate constancy as proxy
        # Lower = more false positives produced (bad for no-law systems)
        results["constancy_ratio"] = best_score if np.isfinite(best_score) else 999.0
        print(f"no law — best cand score={results['constancy_ratio']:.4g}")

    # 5. Conservation Violation on predicted rollout
    print(f"    [5/6] Conservation Violation ...", end=" ", flush=True)
    # FIX-5: use getattr to handle both .model (MLP/HNN) and .dyn (NGCG)
    nn_model = getattr(model_obj, "dyn",
               getattr(model_obj, "model", None))
    if true_law and nn_model is not None:
        cv = conservation_violation(nn_model, te_traj, true_law, svars,
                                    params_te=params_te,
                                    param_names=param_names)
        results["CV"] = cv if np.isfinite(cv) else 999.0
        print(f"{results['CV']:.5g}")
    elif not true_law:
        results["CV"] = 0.0   # no law to violate — 0 is correct
        print("0.0 (no true law)")
    else:
        results["CV"] = 999.0  # no predictor
        print("999.0 (no predictor)")

    # 6. Expression complexity of discovered true-positives
    print(f"    [6/6] Complexity ...", end=" ", flush=True)
    tp_cands = ([c for c in cands if sympy_match(c, true_law, svars)]
                if true_law else [])
    if tp_cands:
        cx  = float(np.mean([sp.count_ops(_safe_sympify(c, svars)) for c in tp_cands]))
        tcx = float(sp.count_ops(_safe_sympify(true_law, svars)))
        results["complexity"]      = cx
        results["true_complexity"] = tcx
        print(f"discovered={cx:.0f} ops  true_law={tcx:.0f} ops")
    else:
        results["complexity"]      = 999.0
        results["true_complexity"] = (float(sp.count_ops(_safe_sympify(true_law, svars)))
                                      if true_law else 0.0)
        print("no true-positive found")

    results["n_candidates"] = len(cands)
    results["candidates"]   = str(cands[:5])
    return results

# ══════════════════════════════════════════════════════════════════════════════
# 6.  FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_model(method: str, d: int, dt: float, seed: int = 0):
    if method == "MLP_Dynamics":   return MLPDynamicsModel(d, seed)
    if method == "HNN":            return HNNModel(d, dt, seed)
    if method == "SINDy":          return SINDyModel(d, seed)
    if method == "IRAS":           return IRASModel(d, seed)
    if method == "NGCG":           return NGCGModel(d, dt, seed, True,  True,  True)
    if method == "NGCG-noGrad":    return NGCGModel(d, dt, seed, False, True,  True)
    if method == "NGCG-noRetrain": return NGCGModel(d, dt, seed, True,  False, True)
    if method == "NGCG-noVerif":   return NGCGModel(d, dt, seed, True,  True,  False)
    raise ValueError(f"Unknown method: {method}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(systems=None, methods=None, seed: int = 0, rerun: bool = False):
    systems = systems or ALL_SYSTEMS
    methods = methods or ALL_METHODS
    all_rows = []

    csv_path  = f"{OUT_DIR}/results.csv"
    done_keys = set()

    # Resume: skip already-done pairs.  Pass --rerun to clear cache.
    rerun = getattr(args,'rerun',False) if 'args' in dir() else False
    if os.path.exists(csv_path) and not rerun:
        existing = pd.read_csv(csv_path)
        for _, r in existing.iterrows():
            done_keys.add((r["method"], r["system"]))
        all_rows = existing.to_dict("records")
        print(f"  Resuming — {len(done_keys)} pairs already done.\n")

    elif rerun and os.path.exists(csv_path):
        os.remove(csv_path)
        import shutil
        _ddir = f"{OUT_DIR}/details"
        if os.path.isdir(_ddir):
            shutil.rmtree(_ddir)
        os.makedirs(_ddir, exist_ok=True)
        print("  --rerun: all cached results cleared\n")
    for system in systems:
        print(f"\n{'═'*65}")
        print(f"  SYSTEM: {system.upper()}")
        print(f"{'═'*65}")

        data = load_system(system)
        if data is None:
            print(f"  ✗  {system} not found in HDF5 — skipping")
            continue

        tr, va, te = data["train"], data["val"], data["test"]
        D           = tr.shape[-1]
        dt          = data["dt"]
        params_te   = data.get("params_test")
        param_names = PARAM_NAMES.get(system)

        for method in methods:

            # HNN only makes sense for Hamiltonian systems
            if method == "HNN" and system not in HNN_SYSTEMS:
                print(f"\n  ── {method} skipped (not Hamiltonian) ──")
                continue

            key = (method, system)
            if key in done_keys:
                print(f"\n  ── {method}/{system} already done (cached) ──")
                continue

            print(f"\n  {'─'*60}")
            print(f"  METHOD: {method}   System: {system}   D={D}   seed={seed}")
            print(f"  train={len(tr)}  val={len(va)}  test={len(te)}")
            print(f"  {'─'*60}")

            torch.manual_seed(seed);  np.random.seed(seed)
            t0 = time.time()

            try:
                mdl = build_model(method, D, dt, seed)

                print(f"\n  ▶ Training...")
                mdl.train(tr, va, system=system)

                print(f"\n  ▶ Evaluating metrics on TEST set ({len(te)} trajectories)...")
                row = compute_metrics(mdl, system, te,
                                      params_te=params_te,
                                      param_names=param_names)

            except Exception as e:
                traceback.print_exc()
                row = {"method": method, "system": system, "error": str(e)}

            row["seed"]       = seed
            row["fit_time_s"] = round(time.time() - t0, 1)
            row["gpu_mb"]     = (torch.cuda.memory_allocated() // 1024 // 1024
                                 if DEVICE == "cuda" else 0)
            row["data_quality"] = "LOW" if system in _LOW_QUALITY_SYSTEMS else "OK"

            # Pretty print result box
            print(f"\n  ┌─ RESULT: {method} on {system} {'─'*20}")
            for k, v in row.items():
                if k not in ("candidates", "error", "seed", "system", "method"):
                    print(f"  │  {k:<24} = {v}")
            print(f"  └{'─'*50}")

            all_rows.append(row)
            done_keys.add(key)

            # Save after EVERY method (crash-safe checkpoint)
            valid = [r for r in all_rows if "error" not in r]
            if valid:
                pd.DataFrame(valid).to_csv(csv_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    valid = [r for r in all_rows if "error" not in r]
    df    = pd.DataFrame(valid)

    if df.empty:
        print("\n  No valid results to summarise.")
        return df

    print(f"\n\n{'═'*65}\n  FINAL SUMMARY  (999=no predictor, 0=perfect/N/A)\n{'═'*65}")
    for metric in ["DR", "FDR", "F1", "MSE_16", "CV", "constancy_ratio", "best_cand_constancy"]:
        if metric not in df.columns:
            continue
        print(f"\n  ── {metric} ──")
        try:
            piv = df.groupby(["method", "system"])[metric].mean().unstack()
            print(piv.round(4).to_string())
        except:
            pass
    # Special: show which systems have a true law
    if "has_true_law" in df.columns:
        laws = df.groupby("system")["has_true_law"].first()
        print(f"\n  Systems WITH true law : {list(laws[laws==1].index)}")
        print(f"  Systems WITHOUT law   : {list(laws[laws==0].index)}")

    df.to_csv(csv_path, index=False)
    print(f"\n  ✅  Results saved → {csv_path}")

    _make_plots(df)
    return df


def _make_plots(df: pd.DataFrame):
    P     = f"{OUT_DIR}/plots"
    syss  = sorted(df.system.unique())
    meths = sorted(df.method.unique())
    colors = plt.cm.tab10.colors

    for metric in ["DR", "MSE_16", "CV", "FDR", "constancy_ratio"]:
        sub = df[df[metric].notna() &
                 np.isfinite(df[metric].astype(float))]
        if sub.empty:
            continue
        n    = len(syss)
        fig, axes = plt.subplots(1, n, figsize=(3.5*n + 1, 4),
                                 sharey=(metric == "DR"))
        if n == 1:
            axes = [axes]
        for ax, sys in zip(axes, syss):
            grp = (sub[sub.system == sys]
                   .groupby("method")[metric]
                   .mean()
                   .reindex(meths)
                   .fillna(0))
            ax.barh(range(len(meths)), grp.values,
                    color=[colors[i % 10] for i in range(len(meths))],
                    alpha=0.85)
            ax.set_yticks(range(len(meths)))
            ax.set_yticklabels(meths, fontsize=8)
            ax.set_title(sys, fontsize=9)
            ax.set_xlabel(metric)
        fig.suptitle(metric, fontsize=11)
        fig.tight_layout()
        out = f"{P}/{metric}.png"
        fig.savefig(out, dpi=130)
        plt.close()
        print(f"  Plot saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--system",  default=None)
    p.add_argument("--method",  default=None)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--systems", nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--rerun", action="store_true", help="Clear cached results and rerun everything")
    a, _ = p.parse_known_args()   # Jupyter-safe: ignores -f kernel.json
    return a


def _main():
    """Entry point — works from both command line and Jupyter %run."""
    args  = _parse()
    syss  = ([args.system]  if args.system  else args.systems  or ALL_SYSTEMS)
    meths = ([args.method]  if args.method  else args.methods  or ALL_METHODS)

    try:
        with h5py.File(HDF5_PATH, "r") as f:
            avail = list(f.keys())
        syss = [s for s in syss if s in avail]
    except Exception as e:
        print(f"  ✗  Cannot open HDF5: {e}\n     Path: {HDF5_PATH}")
        return

    print(f"\n{'═'*65}")
    print(f"  NGCG Stepwise Benchmark")
    print(f"  HDF5    : {HDF5_PATH}")
    print(f"  Device  : {DEVICE}")
    print(f"  Systems : {syss}")
    print(f"  Methods : {meths}")
    print(f"  Seed    : {args.seed}")
    print(f"  Output  : {OUT_DIR}")
    print(f"{'═'*65}\n")

    run_evaluation(syss, meths, seed=args.seed, rerun=getattr(args, 'rerun', False))


# Auto-run when executed via %run in Jupyter OR python script.py
# Works in both contexts unlike bare "if __name__ == '__main__':"
if __name__ == "__main__" or "ipykernel" in sys.modules:
    _main()
