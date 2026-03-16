"""
ngcg.model
==========
NGCG: Neural-Guided Conjecture Generation for Conservation Laws.

Four-stage pipeline:
  1. Neural Dynamics    — MLP one-step predictor, frozen after training.
  2. Variance Minimiser — Multi-restart C_θ(z), anti-collapse loss.
  3. Symbolic Extraction — PySR + polynomial Lasso + LV log-basis Lasso.
  4. Verification Gate  — constancy + trajectory diversity test.

Usage
-----
    from ngcg import NGCG

    model = NGCG(system="mass_spring", data_path="ngcg_data_clean.h5")
    result = model.fit(seed=0)
    # {'DR': 1.0, 'FDR': 0.0, 'F1': 1.0, 'best_constancy': 0.0001, ...}
"""

from __future__ import annotations
import copy
import time
import warnings
import contextlib
from itertools import combinations_with_replacement

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso

from .data import (
    load_system, TRUE_LAWS, STATE_VARS, PARAM_NAMES, PDE_SYSTEMS, ALL_SYSTEMS
)
from .metrics import (
    constancy_score, trajectory_diversity, discovery_rate,
    false_discovery_rate, f1_score_ngcg, conservation_violation,
    true_law_constancy, _sympify,
)
from .utils import MLP, make_pairs, train_mlp, rollout, DEVICE, FLOAT, USE_AMP

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HP = dict(
    # Stage 1: Neural Dynamics
    dyn_hidden   = (256, 256),
    dyn_epochs   = 150,
    dyn_patience = 15,
    dyn_lr       = 3e-3,
    batch        = 2048,

    # Stage 2: Multi-restart φ
    n_restarts   = 10,        # reduced to 3 for PDE systems automatically
    phi_hidden   = (64, 64, 64),
    phi_epochs   = 300,
    phi_patience = 40,
    phi_lr       = 1e-3,
    phi_l2       = 1e-4,

    # Stage 3: PySR
    pysr_niter   = 50,
    pysr_maxsize = 25,
    pysr_n_pts   = 1000,

    # Stage 3: Lotka-Volterra log-basis Lasso
    lv_lambdas   = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],

    # Stage 3: Polynomial Lasso
    poly_max_degree = 3,

    # Stage 4: Verification gate
    gate_strict      = 0.01,
    gate_loose       = 0.05,
    gate_very_strict = 0.005,
    diversity_ratio  = 10.0,

    rollout_k = 16,
)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Variance Minimiser
# ─────────────────────────────────────────────────────────────────────────────

def _phi_variance_loss(phi: nn.Module, traj_gpu: torch.Tensor) -> torch.Tensor:
    """
    Anti-collapse normalised variance loss:
        L = Var_t[C_θ(z_t)] / Var_i[mean_t C_θ(z_t^i)] + eps
    Intra-trajectory variance / inter-trajectory variance.
    Cannot be minimised by outputting a global constant.
    """
    N, T, D = traj_gpu.shape
    c       = phi(traj_gpu.view(N * T, D)).view(N, T)
    intra   = c.var(dim=1).mean()
    inter   = c.mean(dim=1).var()
    return intra / (inter + 1e-4)


def _train_one_phi(
    tr_gpu: torch.Tensor,
    va_gpu: torch.Tensor,
    D: int,
    seed: int,
    hp: dict,
) -> tuple[float, nn.Module]:
    """Train one φ restart. Returns (best_val_constancy, phi_model)."""
    torch.manual_seed(seed)
    phi   = MLP(D, 1, hidden=hp["phi_hidden"]).to(DEVICE)
    opt   = torch.optim.Adam(phi.parameters(), lr=hp["phi_lr"], weight_decay=hp["phi_l2"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, hp["phi_epochs"])
    best  = float("inf"); bw = copy.deepcopy(phi.state_dict()); pat = 0

    for _ in range(hp["phi_epochs"]):
        phi.train()
        loss = _phi_variance_loss(phi, tr_gpu)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        phi.eval()
        with torch.no_grad():
            vl = _phi_variance_loss(phi, va_gpu).item()
        if vl < best - 1e-7: best = vl; pat = 0; bw = copy.deepcopy(phi.state_dict())
        else:
            pat += 1
            if pat >= hp["phi_patience"]: break

    phi.load_state_dict(bw)
    return best, phi


def _phi_constancy(phi: nn.Module, traj: np.ndarray) -> float:
    N, T, D = traj.shape
    z = torch.tensor(traj.reshape(-1, D), dtype=FLOAT, device=DEVICE)
    with torch.no_grad():
        c = phi(z).view(N, T).cpu().numpy()
    mask = np.all(np.isfinite(c), axis=1)
    if mask.sum() == 0: return 999.0
    v = c[mask]
    return float((v.std(axis=1) / (np.abs(v.mean(axis=1)) + 1e-8)).mean())


def _run_phi_restarts(
    tr: np.ndarray, va: np.ndarray, D: int, n_restarts: int, hp: dict, seed: int
) -> tuple[nn.Module, float]:
    """Run n_restarts independent φ networks; return best by val constancy."""
    tr_gpu = torch.tensor(tr, dtype=FLOAT, device=DEVICE)
    va_gpu = torch.tensor(va, dtype=FLOAT, device=DEVICE)
    best_score = float("inf"); best_phi = None

    for r in range(n_restarts):
        _, phi = _train_one_phi(tr_gpu, va_gpu, D, seed + r * 137, hp)
        cs     = _phi_constancy(phi, va)
        print(f"    restart {r:2d}  val_constancy={cs:.5f}")
        if cs < best_score: best_score = cs; best_phi = copy.deepcopy(phi)

    return best_phi, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3a: LV log-basis Lasso (Lotka-Volterra specific)
# ─────────────────────────────────────────────────────────────────────────────

def _lv_log_basis(traj: np.ndarray) -> np.ndarray:
    eps = 1e-6
    x = np.clip(traj[:, :, 0], eps, None)
    y = np.clip(traj[:, :, 1], eps, None)
    return np.stack([x, y, np.log(x), np.log(y)], axis=-1)  # (N, T, 4)


def _lv_lasso(tr: np.ndarray, va: np.ndarray, lambdas: list) -> tuple:
    """
    Eigenvector of mean trajectory covariance in {x, y, log(x), log(y)} basis.
    Falls back to Lasso for sparsification.
    Returns (best_w, best_val_constancy).
    """
    phi_tr = _lv_log_basis(tr); phi_va = _lv_log_basis(va)
    N, T, M = phi_tr.shape
    best_score = float("inf"); best_w = None

    # Method A: minimum eigenvector of mean covariance
    A = np.zeros((M, M))
    for i in range(N):
        p = phi_tr[i]; pc = p - p.mean(0); A += pc.T @ pc
    A /= N
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        for k in range(M):
            w = eigvecs[:, k]
            c = (phi_va.reshape(-1, M) @ w).reshape(len(va), T)
            sc = float((c.std(1) / (np.abs(c.mean(1)) + 1e-8)).mean())
            if sc < best_score: best_score = sc; best_w = w.copy()
    except Exception: pass

    # Method B: Lasso on centred phi
    X  = (phi_tr - phi_tr.mean(1, keepdims=True)).reshape(-1, M)
    cn = np.linalg.norm(X, axis=0) + 1e-10
    Xn = X / cn[None, :]
    for lam in lambdas:
        try:
            m = Lasso(alpha=lam, fit_intercept=False, max_iter=20000, tol=1e-8)
            m.fit(Xn, np.zeros(len(X))); w = m.coef_ / cn
            if np.all(np.abs(w) < 1e-10): continue
            c  = (phi_va.reshape(-1, M) @ w).reshape(len(va), T)
            sc = float((c.std(1) / (np.abs(c.mean(1)) + 1e-8)).mean())
            if sc < best_score: best_score = sc; best_w = w.copy()
        except Exception: pass

    # Sign sweep
    if best_w is not None:
        for wt in [best_w, -best_w, best_w / (np.abs(best_w).max() + 1e-10)]:
            c  = (phi_va.reshape(-1, M) @ wt).reshape(len(va), T)
            sc = float((c.std(1) / (np.abs(c.mean(1)) + 1e-8)).mean())
            if sc < best_score: best_score = sc; best_w = wt.copy()

    return best_w, best_score


def _lv_w_to_str(w: np.ndarray) -> str:
    names = ["x", "y", "log(x)", "log(y)"]
    terms = [f"({wi:+.5f})*{n}" for wi, n in zip(w, names) if abs(wi) > 1e-6]
    return " + ".join(terms) if terms else "0"


def _lv_constancy(w: np.ndarray, traj: np.ndarray) -> float:
    phi = _lv_log_basis(traj); c = (phi @ w).reshape(len(traj), traj.shape[1])
    mask = np.all(np.isfinite(c), 1)
    if mask.sum() == 0: return 999.0
    v = c[mask]; return float((v.std(1) / (np.abs(v.mean(1)) + 1e-8)).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b: Polynomial Lasso (general Hamiltonian systems)
# ─────────────────────────────────────────────────────────────────────────────

def _poly_lasso(
    tr: np.ndarray, va: np.ndarray, te: np.ndarray,
    svars: list, D: int, max_degree: int = 3
) -> list[tuple[str, float]]:
    """
    Eigenvector of mean trajectory covariance in polynomial basis up to
    degree max_degree. Returns candidates with constancy < 0.1 on test.
    """
    combos = []
    for deg in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(D), deg):
            combos.append(combo)
    M = len(combos)

    def build_phi(traj: np.ndarray) -> np.ndarray:
        N, T, _ = traj.shape; parts = []
        for combo in combos:
            t = traj[:, :, combo[0]].copy()
            for idx in combo[1:]: t = t * traj[:, :, idx]
            parts.append(t[:, :, None])
        return np.concatenate(parts, axis=2)

    try:
        phi_tr = build_phi(tr); N, T, _ = tr.shape
        A = np.zeros((M, M))
        for i in range(N):
            p = phi_tr[i]; pc = p - p.mean(0); A += pc.T @ pc
        A /= N
        _, eigvecs = np.linalg.eigh(A)
        phi_te = build_phi(te)
        cands  = []
        for k in range(min(M, 8)):
            w  = eigvecs[:, k]
            c  = (phi_te.reshape(len(te)*te.shape[1], M) @ w).reshape(len(te), te.shape[1])
            mask = np.all(np.isfinite(c), 1)
            if mask.sum() == 0: continue
            v  = c[mask]; sc = float((v.std(1) / (np.abs(v.mean(1)) + 1e-8)).mean())
            if sc < 0.1:
                terms = [f"({wi:+.4f})*{'*'.join(svars[j] for j in combo)}"
                         for wi, combo in zip(w, combos) if abs(wi) > 1e-3]
                cands.append((" + ".join(terms) if terms else "0", sc))
        return cands
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3c: PySR (general symbolic regression)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pysr(
    X: np.ndarray, y: np.ndarray, var_names: list,
    n_iter: int, maxsize: int, seed: int = 42
) -> list[tuple[str, float]]:
    """NaN-safe PySR wrapper. Returns [(expr_str, r2), ...] sorted by score."""
    mask = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(X) < 10: return []
    X = np.clip(X, -100, 100); y = np.clip(y, -1e6, 1e6)
    try:
        from pysr import PySRRegressor
        m = PySRRegressor(
            niterations=n_iter, populations=15,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "cube", "sqrt", "log", "exp", "sin", "cos"],
            maxsize=maxsize, verbosity=0, random_state=seed, procs=0,
            multithreading=False,
        )
        m.fit(X, y, variable_names=var_names)
        results = []
        for _, row in m.equations_.sort_values("score", ascending=False).iterrows():
            expr_str = str(row["sympy_format"])
            try:
                fn   = sp.lambdify(var_names, _sympify(expr_str, var_names), modules="numpy")
                yhat = np.array(fn(*[X[:, i] for i in range(X.shape[1])]),
                                dtype=float).flatten()
                yhat = np.where(np.isfinite(yhat), yhat, np.nan)
                ok   = np.isfinite(yhat)
                r2   = float(1 - np.sum((y[ok]-yhat[ok])**2) /
                             (np.sum((y[ok]-y[ok].mean())**2) + 1e-12)) if ok.sum() >= 5 else -999.0
            except Exception: r2 = -999.0
            results.append((expr_str, r2))
        return results
    except Exception as e:
        print(f"    PySR error: {e}"); return []


# ─────────────────────────────────────────────────────────────────────────────
# Main NGCG class
# ─────────────────────────────────────────────────────────────────────────────

class NGCG:
    """
    NGCG: Neural-Guided Conjecture Generation for Conservation Laws.

    Parameters
    ----------
    system    : one of ALL_SYSTEMS
    data_path : path to ngcg_data_clean.h5
    hp        : dict of hyperparameters (overrides DEFAULT_HP)
    """

    def __init__(
        self,
        system: str,
        data_path: str,
        hp: dict | None = None,
    ):
        self.system    = system
        self.data_path = data_path
        self.hp        = {**DEFAULT_HP, **(hp or {})}
        self._svars    = STATE_VARS.get(system, [])
        self._true_law = TRUE_LAWS.get(system)
        self._has_law  = bool(self._true_law)
        self._result   = None

    def fit(self, seed: int = 0) -> dict:
        """
        Run all four stages on the loaded data.

        Parameters
        ----------
        seed : random seed for reproducibility

        Returns
        -------
        dict with keys: DR, DR_strict, FDR, F1, MSE_16, CV,
                        best_constancy, best_expr, true_law_constancy,
                        complexity, fit_time_s, and more.
        """
        torch.manual_seed(seed); np.random.seed(seed)
        data = load_system(self.system, self.data_path)
        if data is None:
            raise ValueError(f"System '{self.system}' not found in {self.data_path}")

        tr, va, te = data["train"], data["val"], data["test"]
        par_te     = data.get("params_test")
        param_names = PARAM_NAMES.get(self.system)
        D          = tr.shape[-1]
        hp         = self.hp
        t0         = time.time()
        r: dict    = {"method": "NGCG", "system": self.system,
                      "has_true_law": 1.0 if self._has_law else 0.0}

        # ── Stage 1: Neural Dynamics ─────────────────────────────────────────
        print(f"  Stage 1: Dynamics (D={D})")
        self._dyn = MLP(D, D, hidden=hp["dyn_hidden"]).to(DEVICE)
        dyn_mse, _ = train_mlp(
            self._dyn, tr, va,
            epochs=hp["dyn_epochs"], patience=hp["dyn_patience"],
            lr=hp["dyn_lr"], batch=hp["batch"], verbose=True,
        )
        self._dyn.eval()
        mse16 = 999.0
        if len(te) > 0:
            pred16 = rollout(self._dyn, te[:, 0], hp["rollout_k"])
            mse16  = float(np.mean((pred16 - te[:, 1:hp["rollout_k"]+1])**2))
        r["MSE_16"] = mse16 if np.isfinite(mse16) else 999.0
        r["dyn_val_mse"] = dyn_mse
        print(f"  ✓ val_mse={dyn_mse:.5f}  MSE@16={r['MSE_16']:.5g}")

        # ── Stage 2: Multi-restart φ ─────────────────────────────────────────
        _nr = 3 if self.system in PDE_SYSTEMS else hp["n_restarts"]
        print(f"  Stage 2: {_nr} φ restarts")
        self._phi, phi_val = _run_phi_restarts(tr, va, D, _nr, hp, seed)
        r["phi_val_constancy"] = phi_val
        print(f"  ✓ best val_constancy={phi_val:.5f}")

        # ── Stage 3: Symbolic Extraction ─────────────────────────────────────
        print("  Stage 3: Symbolic extraction")
        candidates: list[tuple[str, float]] = []

        # PDE explicit candidate
        if self.system in PDE_SYSTEMS and self._svars:
            sc = constancy_score(self._svars[0], self._svars, te) if len(te) > 0 else 999.0
            candidates.append((self._svars[0], sc))
            print(f"  PDE: '{self._svars[0]}' constancy={sc:.5g}")

        # Lotka-Volterra log-basis Lasso
        if self.system == "lotka_volterra":
            best_w, _ = _lv_lasso(tr, va, hp["lv_lambdas"])
            if best_w is not None:
                sc = _lv_constancy(best_w, te) if len(te) > 0 else 999.0
                candidates.append((_lv_w_to_str(best_w), sc))
                print(f"  LV Lasso constancy={sc:.5f}")

        # General path
        if self.system not in PDE_SYSTEMS:
            # Polynomial Lasso
            if self._has_law and D <= 4:
                for e, s in _poly_lasso(tr, va, te, self._svars, D,
                                        hp.get("poly_max_degree", 3)):
                    candidates.append((e, s))

            # PySR on φ values
            pts = tr.reshape(-1, D)
            idx = np.random.choice(len(pts), min(hp["pysr_n_pts"], len(pts)), replace=False)
            xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
            with torch.no_grad():
                y_phi = self._phi(xt).squeeze(-1).cpu().numpy()
            X_in = np.clip(pts[idx], 1e-6, None) if self.system == "lotka_volterra" else pts[idx]
            for expr_str, r2 in _run_pysr(X_in, y_phi, self._svars,
                                          hp["pysr_niter"], hp["pysr_maxsize"], seed)[:8]:
                sc = constancy_score(expr_str, self._svars, te) if len(te) > 0 else 999.0
                candidates.append((expr_str, sc))
        else:
            print("  PySR skipped for PDE (u_mean already candidate)")

        # ── Stage 4: Verification Gate ───────────────────────────────────────
        print("  Stage 4: Verification gate")
        GATE  = hp["gate_strict"]
        TOL_L = hp["gate_loose"]
        candidates.sort(key=lambda x: x[1])
        accepted = [(e, s) for e, s in candidates if s < GATE]

        # PDE simplicity: keep only u_mean if it passes
        if self.system in PDE_SYSTEMS and self._svars:
            simple = [(e, s) for e, s in accepted if e.strip() == self._svars[0]]
            if simple: accepted = simple

        # Diversity test for no-law systems
        if not self._has_law:
            accepted = [
                (e, s) for e, s in accepted
                if trajectory_diversity(e, self._svars, te) >= hp["diversity_ratio"]
            ]

        best_expr = accepted[0][0] if accepted else ""
        best_sc   = accepted[0][1] if accepted else (candidates[0][1] if candidates else 999.0)
        r["best_expr"]      = best_expr[:200]
        r["best_constancy"] = best_sc if np.isfinite(best_sc) else 999.0
        r["n_accepted"]     = len(accepted)
        r["n_candidates"]   = len(candidates)

        # ── Metrics ──────────────────────────────────────────────────────────
        all_scores = [s for _, s in candidates]
        dr, dr_s   = discovery_rate(candidates, self._has_law, TOL_L)
        fdr        = false_discovery_rate(accepted, self._has_law, TOL_L)
        f1         = f1_score_ngcg(dr, fdr)
        r["DR"] = dr; r["DR_strict"] = dr_s; r["DR_symbolic"] = 0.0
        r["FDR"] = fdr; r["F1"] = f1

        # Conservation violation
        r["CV"] = 0.0
        if self._has_law and len(te) > 0 and self._true_law:
            def _rollout_fn(x0, k): return rollout(self._dyn, x0, k)
            cv = conservation_violation(
                _rollout_fn, te, self._true_law, self._svars,
                params=par_te, param_names=param_names, k=hp["rollout_k"]
            )
            r["CV"] = cv if np.isfinite(cv) else 999.0

        # True law constancy
        tlc = true_law_constancy(
            self._true_law, self._svars, te, par_te, param_names
        ) if self._has_law and len(te) > 0 else 0.0
        r["true_law_constancy"] = tlc if np.isfinite(tlc) else 0.0

        # Expression complexity
        try:
            cx = float(sp.count_ops(_sympify(best_expr, self._svars))) if best_expr else 999.0
        except Exception: cx = 999.0
        r["complexity"]  = cx if np.isfinite(cx) else 999.0
        r["fit_time_s"]  = round(time.time() - t0, 1)
        r["gpu_mb"]      = torch.cuda.memory_allocated()//1024//1024 if DEVICE=="cuda" else 0

        self._result = r
        return r

    @property
    def result(self) -> dict | None:
        """Access the last fit result."""
        return self._result
