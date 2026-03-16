"""
ngcg.metrics
============
All evaluation metrics for the NGCG benchmark.

Definitions
-----------
constancy(C, traj):
    mean_i [ std_t[C(z_t^i)] / |mean_t[C(z_t^i)]| ]
    Lower is better. A perfect invariant scores 0.

DR (Discovery Rate):
    1 if any accepted candidate has constancy < tol_loose on test trajectories.
    For no-law systems, DR should be 0.

FDR (False Discovery Rate):
    (false positives among accepted) / (total accepted)
    For has-law systems: false positive = accepted candidate with constancy >= tol_loose.
    For no-law systems: every accepted candidate is a false positive.

F1:
    2 * DR * (1 - FDR) / (DR + (1 - FDR))

CV (Conservation Violation):
    mean_i mean_t |C(x_hat_t^i) - C(x_0^i)|
    Measures how well the discovered law is conserved under the learned dynamics.

trajectory_diversity(C, traj):
    std_i[mean_t C(z_t^i)] / mean_i[std_t C(z_t^i)]
    Ratio of inter-trajectory to intra-trajectory variation.
    Genuine invariants have high ratio (varies between trajectories, constant within).
    Spurious near-constants have ratio ≈ 1.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Optional

# SymPy symbol registry to avoid gamma/beta name clashes
_SYM_NAMES = [
    "q","p","x","y","z","u","v","m","k","r","s",
    "alpha","beta","gamma","delta","epsilon","theta","phi",
    "q1","q2","p1","p2","k1","k2","k3","m1","m2",
    "theta1","theta2","px","py","vx","vy","u_mean","u_var","u_skew",
]


def _sympify(expr_str: str, extra: Optional[list] = None) -> sp.Expr:
    loc = {n: sp.Symbol(n) for n in _SYM_NAMES + (list(extra) if extra else [])}
    return sp.sympify(expr_str, locals=loc)


# ─────────────────────────────────────────────────────────────────────────────
# Core scalar metric
# ─────────────────────────────────────────────────────────────────────────────

def constancy_score(
    expr_str: str,
    state_vars: list[str],
    trajectories: np.ndarray,
) -> float:
    """
    Evaluate constancy of a symbolic expression along test trajectories.

    Parameters
    ----------
    expr_str     : SymPy-parseable string, e.g. "p**2/2 + q**2/2"
    state_vars   : list of variable names matching trajectory dimensions
    trajectories : (N, T, D) array of test trajectories

    Returns
    -------
    float : mean_i [ std_t[C(z_t^i)] / |mean_t[C(z_t^i)]| ]
            Lower = more conserved. Returns 999.0 on parse failure.
    """
    if not expr_str or expr_str.strip() in ("0", ""):
        return 999.0
    try:
        fn      = sp.lambdify(state_vars, _sympify(expr_str, state_vars), modules="numpy")
        N, T, D = trajectories.shape
        vals    = np.stack(
            [fn(*[trajectories[:, t, j] for j in range(D)]) for t in range(T)],
            axis=1
        ).astype(np.float64)
        mask = np.all(np.isfinite(vals), axis=1)
        if mask.sum() == 0:
            return 999.0
        v = vals[mask]
        return float((v.std(axis=1) / (np.abs(v.mean(axis=1)) + 1e-8)).mean())
    except Exception:
        return 999.0


def trajectory_diversity(
    expr_str: str,
    state_vars: list[str],
    trajectories: np.ndarray,
) -> float:
    """
    Compute inter-trajectory / intra-trajectory variation ratio.

    A genuine invariant varies between trajectories (different ICs → different
    conserved values) but is constant within each trajectory.
    Ratio >= 10 is required for acceptance (see Stage 4 gate).

    Returns
    -------
    float : inter_std / intra_std.  Higher = more genuine invariant.
    """
    if not expr_str or expr_str.strip() in ("0", ""):
        return 0.0
    try:
        fn      = sp.lambdify(state_vars, _sympify(expr_str, state_vars), modules="numpy")
        N, T, D = trajectories.shape
        vals    = np.stack(
            [fn(*[trajectories[:, t, j] for j in range(D)]) for t in range(T)],
            axis=1
        ).astype(np.float64)
        mask = np.all(np.isfinite(vals), axis=1)
        if mask.sum() < 5:
            return 0.0
        v         = vals[mask]
        intra_std = v.std(axis=1).mean()
        inter_std = v.mean(axis=1).std()
        return float(inter_std / (intra_std + 1e-10))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark metrics
# ─────────────────────────────────────────────────────────────────────────────

def discovery_rate(
    candidates: list[tuple[str, float]],
    has_true_law: bool,
    tol_loose: float = 0.05,
) -> tuple[float, float]:
    """
    Compute DR and DR_strict.

    Parameters
    ----------
    candidates   : list of (expr_str, test_constancy) tuples
    has_true_law : whether a true conservation law exists for this system
    tol_loose    : DR threshold (default 0.05)

    Returns
    -------
    (DR, DR_strict) : both in {0.0, 1.0}
    """
    if not candidates:
        return 0.0, 0.0
    all_scores = [s for _, s in candidates]
    if has_true_law:
        DR       = 1.0 if any(s < tol_loose    for s in all_scores) else 0.0
        DR_strict= 1.0 if any(s < tol_loose/10 for s in all_scores) else 0.0
    else:
        # For no-law systems DR should always be 0 (any 1 = false positive)
        DR        = 0.0
        DR_strict = 0.0
    return DR, DR_strict


def false_discovery_rate(
    accepted: list[tuple[str, float]],
    has_true_law: bool,
    tol_loose: float = 0.05,
) -> float:
    """
    Compute FDR over accepted candidates only.

    For has-law systems:
        FDR = (accepted with constancy >= tol_loose) / total accepted
    For no-law systems:
        FDR = 1.0 if any accepted, else 0.0
    """
    if not accepted:
        return 0.0
    if not has_true_law:
        return 1.0 if accepted else 0.0
    tp  = sum(1 for _, s in accepted if s < tol_loose)
    fp  = len(accepted) - tp
    return fp / max(1, len(accepted))


def f1_score_ngcg(dr: float, fdr: float) -> float:
    """F1 = 2 * DR * (1 - FDR) / (DR + (1 - FDR))"""
    return 2 * dr * (1 - fdr) / max(1e-9, dr + (1 - fdr))


def conservation_violation(
    rollout_fn,
    test_trajectories: np.ndarray,
    expr_str: str,
    state_vars: list[str],
    params: Optional[np.ndarray] = None,
    param_names: Optional[list[str]] = None,
    k: int = 16,
) -> float:
    """
    Compute mean |C(x_hat_t) - C(x_0)| over k-step predicted rollout.

    Parameters
    ----------
    rollout_fn        : callable(x0, steps) → (N, steps, D)
    test_trajectories : (N, T, D) test set
    expr_str          : discovered expression string
    state_vars        : variable names
    params            : (N, n_params) per-trajectory parameters, optional
    param_names       : parameter name strings, optional
    k                 : rollout horizon

    Returns
    -------
    float : mean conservation violation. 0.0 if no expression.
    """
    if not expr_str:
        return 0.0
    try:
        expr     = _sympify(expr_str, state_vars)
        x0       = test_trajectories[:, 0]
        pred     = rollout_fn(x0, k)          # (N, k, D)
        devs     = []
        for i in range(len(x0)):
            subs = {}
            if params is not None and param_names:
                for pi, pn in enumerate(param_names):
                    if pi < params.shape[1]:
                        subs[sp.Symbol(pn)] = float(params[i, pi])
            try:
                fn  = sp.lambdify(state_vars, expr.subs(subs), modules="numpy")
                ev  = lambda s: fn(*[s[j] for j in range(len(state_vars))])  # noqa: E731
                c0  = ev(x0[i])
                devs.append(np.mean([abs(ev(pred[i, t]) - c0) for t in range(k)]))
            except Exception:
                pass
        return float(np.mean(devs)) if devs else 999.0
    except Exception:
        return 999.0


def true_law_constancy(
    law_str: str,
    state_vars: list[str],
    test_trajectories: np.ndarray,
    params: Optional[np.ndarray] = None,
    param_names: Optional[list[str]] = None,
) -> float:
    """
    Evaluate constancy of the ground-truth conservation law on test data.
    Should be ≈ 0 for well-conserved systems (sanity check).
    Handles per-trajectory parameter substitution (k, m, alpha, etc.).
    """
    if not law_str:
        return 0.0
    try:
        expr     = _sympify(law_str, state_vars)
        all_syms = {str(s) for s in expr.free_symbols}
        N, T, D  = test_trajectories.shape
        ratios   = []
        for i in range(N):
            subs = {}
            if params is not None and param_names:
                for pi, pn in enumerate(param_names):
                    if pn in all_syms and pi < params.shape[1]:
                        subs[sp.Symbol(pn)] = float(params[i, pi])
            try:
                fn   = sp.lambdify(state_vars, expr.subs(subs), modules="numpy")
                vals = np.array(
                    [fn(*test_trajectories[i, t, :]) for t in range(T)], dtype=float
                )
                if np.all(np.isfinite(vals)):
                    ratios.append(vals.std() / (abs(vals.mean()) + 1e-8))
            except Exception:
                pass
        return float(np.mean(ratios)) if ratios else 999.0
    except Exception:
        return 999.0
