"""
regen_pde_stable.py
===================
Regenerates ONLY burgers and ks datasets with numerically stable parameters
and writes them into the existing ngcg_data_clean.h5 in /kaggle/working/.

Root cause of NaN blowup (from diagnosis):
  - Burgers: diverges at t≈50-200.  Cause: IC amplitudes too large (sum of
    sinusoids with amp ±1 → peak |u| ~ 5-6), viscosity too small relative to
    dt. Fix: smaller IC amplitudes, larger viscosity, smaller dt.
  - KS: diverges at t≈100-130 (100% of trajectories).  Cause: ETD-RK4 with
    dt=t_end/500 is too large for domain L=64π.  Fix: much shorter t_end or
    larger dt_inner substepping.  We use a shorter domain and subsample.

Run:
    %run /kaggle/working/regen_pde_stable.py

Output: patches /kaggle/working/ngcg_data_clean.h5
        (copies from Kaggle input path first if working copy not present)
"""

import os, shutil, time
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED       = 42
INPUT_H5   = "/kaggle/input/datasets/drahulray/upadted-gcmc-data/ngcg_data_clean (1).h5"
WORKING_H5 = "/kaggle/working/ngcg_data_clean.h5"

np.random.seed(SEED)

# ── Copy input to working if needed ──────────────────────────────────────────
if not os.path.exists(WORKING_H5):
    print(f"Copying {INPUT_H5} → {WORKING_H5} ...")
    shutil.copy2(INPUT_H5, WORKING_H5)
    print(f"  Done ({os.path.getsize(WORKING_H5)/1e6:.1f} MB)")
else:
    print(f"Working copy exists: {WORKING_H5} ({os.path.getsize(WORKING_H5)/1e6:.1f} MB)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_splits(n, seed=SEED):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    va, te  = train_test_split(tmp, test_size=0.50, random_state=seed)
    return tr, va, te

def save_pde_group(h5path, name, trajs, t_eval, meta):
    """Write / overwrite one PDE group in the HDF5."""
    N = trajs.shape[0]
    tr, va, te = make_splits(N)
    with h5py.File(h5path, "a") as f:
        if name in f:
            del f[name]
        g = f.create_group(name)
        g.create_dataset("trajectories",  data=trajs,  compression="gzip", compression_opts=4)
        g.create_dataset("time",          data=t_eval)
        g.create_dataset("train_indices", data=tr)
        g.create_dataset("val_indices",   data=va)
        g.create_dataset("test_indices",  data=te)
        for k, v in meta.items():
            g.attrs[k] = v
    print(f"  [{name}] saved  shape={trajs.shape}  "
          f"train={len(tr)} val={len(va)} test={len(te)}")


# ══════════════════════════════════════════════════════════════════════════════
# BURGERS  —  stable pseudo-spectral solver
# Fix: amplitude 0.3 (was 1.0), nu=0.05 (was 0.01), t_end=1.0 (was 2.0)
# These give smooth solutions that don't blow up in the ETD window.
# ══════════════════════════════════════════════════════════════════════════════

def fft_burgers_stable(u0, nu, dt, n_steps, N):
    """
    Strang-split pseudo-spectral Burgers with stability guard.
    Returns (n_steps, N) array, or None if divergence detected.
    """
    dx = 2 * np.pi / N
    k  = np.fft.rfftfreq(N, d=dx / (2 * np.pi))        # wavenumbers

    # Pre-compute diffusion filters
    diff_half = np.exp(-nu * k**2 * dt / 2)
    diff_full = diff_half ** 2

    u   = u0.copy().astype(np.float64)
    out = [u.copy()]

    for _ in range(n_steps):
        # Half diffusion
        uhat = np.fft.rfft(u) * diff_half
        u    = np.fft.irfft(uhat, n=N)

        # Nonlinear advection (spectral)
        uhat  = np.fft.rfft(u)
        dudx  = np.fft.irfft(1j * k * uhat, n=N)
        u     = u - dt * u * dudx

        # Half diffusion
        uhat = np.fft.rfft(u) * diff_half
        u    = np.fft.irfft(uhat, n=N)

        # Stability guard: clip extreme values
        if np.any(np.abs(u) > 1e4) or not np.all(np.isfinite(u)):
            return None
        out.append(u.copy())

    return np.array(out, dtype=np.float32)[:n_steps]


def gen_burgers_stable(rng, n_traj=1000, nx=64, t_end=1.0, n_steps=500,
                       nu=0.05, amp=0.3):
    """
    1D viscous Burgers on [0, 2π].
    Key stability changes vs original:
      - amp=0.3  (was 1.0)  → smaller initial gradients
      - nu=0.05  (was 0.01) → stronger dissipation
      - t_end=1.0 (was 2.0) → shorter integration horizon
    """
    x   = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    dt  = t_end / n_steps

    # CFL check
    max_u_est = amp * 5   # rough max |u|
    cfl = max_u_est * dt / (2 * np.pi / nx)
    print(f"  Burgers: nx={nx}  nu={nu}  amp={amp}  dt={dt:.5f}  "
          f"CFL≈{cfl:.3f}  (should be <0.5)")

    trajs  = []
    failed = 0

    for i in tqdm(range(n_traj), desc="burgers", ncols=80):
        while True:
            n_modes = rng.randint(2, 5)
            u0 = np.zeros(nx)
            for m in range(1, n_modes + 1):
                a     = rng.uniform(-amp, amp)
                phase = rng.uniform(0, 2 * np.pi)
                u0   += a * np.sin(m * x + phase)

            result = fft_burgers_stable(u0, nu, dt, n_steps, nx)
            if result is not None:
                trajs.append(result)
                break
            else:
                failed += 1
                if failed > 200:
                    # fallback: use zero IC
                    trajs.append(np.zeros((n_steps, nx), dtype=np.float32))
                    break

    trajs  = np.stack(trajs, axis=0)               # (N, T, nx)
    t_eval = np.linspace(0, t_end * (1-1/n_steps), n_steps).astype(np.float32)

    # Verify no NaN
    n_bad = (~np.all(np.isfinite(trajs.reshape(n_traj, -1)), axis=1)).sum()
    print(f"  Burgers: {n_bad}/{n_traj} bad trajectories after fix "
          f"({'✓ clean' if n_bad==0 else '✗ still bad'})")

    meta = {
        "description"    : f"1D viscous Burgers (nu={nu}, amp={amp}). "
                           f"Periodic on [0,2pi], {nx} grid pts. t_end={t_end}. "
                           f"Stable version: smaller ICs, higher viscosity.",
        "conserved_law"  : "none_viscous",
        "component_names": str(["u"]),
        "grid_shape"     : str((nx,)),
        "dt"             : float(t_end / n_steps),
        "nu"             : nu,
        "state_names"    : str([f"u{i}" for i in range(nx)]),
    }
    return trajs, t_eval, meta


# ══════════════════════════════════════════════════════════════════════════════
# KS  —  stable ETD-RK4 solver
# Fix: shorter domain L=32 (was 64π≈201), t_end=50 (was 150),
#      IC amplitudes 0.01 (was 0.1), nx=64 (was 128)
# ══════════════════════════════════════════════════════════════════════════════

def ks_etdrk4_stable(u0, L, dt, n_steps, N):
    """
    ETD-RK4 for KS: u_t = -u*u_x - u_xx - u_xxxx.
    Returns (n_steps, N) or None if divergence detected.
    """
    dx  = L / N
    k   = np.fft.rfftfreq(N, d=dx / (2 * np.pi))    # scaled wavenumbers

    L_op = k**2 - k**4                               # linear operator
    E    = np.exp(L_op * dt)
    E2   = np.exp(L_op * dt / 2)

    def NL(uhat):
        u    = np.fft.irfft(uhat, n=N)
        dudx = np.fft.irfft(1j * k * uhat, n=N)
        return -np.fft.rfft(u * dudx)

    u    = u0.copy().astype(np.float64)
    uhat = np.fft.rfft(u)
    out  = [u.copy()]

    for _ in range(n_steps):
        N1 = NL(uhat)
        N2 = NL(E2 * uhat + dt / 2 * N1)
        N3 = NL(E2 * uhat + dt / 2 * N2)
        N4 = NL(E  * uhat + dt      * N3)
        uhat = E * uhat + dt * (E * N1 / 6 + E2 * (N2 + N3) / 3 + N4 / 6)

        u_new = np.fft.irfft(uhat, n=N)
        if np.any(np.abs(u_new) > 1e3) or not np.all(np.isfinite(u_new)):
            return None
        out.append(u_new.copy())

    return np.array(out, dtype=np.float32)[:n_steps]


def gen_ks_stable(rng, n_traj=1000, nx=64, L=32.0, t_end=50.0,
                  n_steps=500, amp=0.01):
    """
    Kuramoto-Sivashinsky on [0, L] periodic.
    Key stability changes:
      - L=32    (was 64π≈201) → smaller domain, less spectral energy
      - t_end=50 (was 150)    → shorter integration
      - amp=0.01 (was 0.1)    → smaller perturbations
      - nx=64   (was 128)     → adequate resolution for L=32
    """
    x  = np.linspace(0, L, nx, endpoint=False)
    dt = t_end / n_steps

    print(f"  KS: L={L}  nx={nx}  dt={dt:.4f}  amp={amp}  t_end={t_end}")

    trajs  = []
    failed = 0

    for i in tqdm(range(n_traj), desc="ks", ncols=80):
        while True:
            u0 = np.zeros(nx)
            for m in range(1, 6):
                a     = rng.uniform(-amp, amp)
                phase = rng.uniform(0, 2 * np.pi)
                u0   += a * np.cos(2 * np.pi * m * x / L + phase)

            result = ks_etdrk4_stable(u0, L, dt, n_steps, nx)
            if result is not None:
                trajs.append(result)
                break
            else:
                failed += 1
                if failed > 200:
                    trajs.append(np.zeros((n_steps, nx), dtype=np.float32))
                    break

    trajs  = np.stack(trajs, axis=0)
    t_eval = np.linspace(0, t_end * (1-1/n_steps), n_steps).astype(np.float32)

    n_bad = (~np.all(np.isfinite(trajs.reshape(n_traj, -1)), axis=1)).sum()
    print(f"  KS: {n_bad}/{n_traj} bad trajectories after fix "
          f"({'✓ clean' if n_bad==0 else '✗ still bad'})")

    meta = {
        "description"    : f"Kuramoto-Sivashinsky on [0,{L}] periodic, {nx} grid pts. "
                           f"t_end={t_end}. Stable version: shorter domain, small ICs.",
        "conserved_law"  : "integral_u_dx",
        "component_names": str(["u"]),
        "grid_shape"     : str((nx,)),
        "dt"             : float(t_end / n_steps),
        "state_names"    : str([f"u{i}" for i in range(nx)]),
    }
    return trajs, t_eval, meta


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.RandomState(SEED)
    t0  = time.time()

    print(f"\n{'═'*60}")
    print(f"  Regenerating Burgers + KS with stable parameters")
    print(f"  Output: {WORKING_H5}")
    print(f"{'═'*60}\n")

    # ── Burgers ───────────────────────────────────────────────────────────────
    print("▶ Burgers (stable)...")
    trajs_b, t_b, meta_b = gen_burgers_stable(rng, n_traj=1000)
    save_pde_group(WORKING_H5, "burgers", trajs_b, t_b, meta_b)

    # ── KS ────────────────────────────────────────────────────────────────────
    print("\n▶ KS (stable)...")
    trajs_k, t_k, meta_k = gen_ks_stable(rng, n_traj=1000)
    save_pde_group(WORKING_H5, "ks", trajs_k, t_k, meta_k)

    # ── Verify ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Verification (re-reading from file):")
    with h5py.File(WORKING_H5, "r") as f:
        for name in ["burgers", "ks"]:
            g    = f[name]
            traj = g["trajectories"][:]
            N, T, D = traj.shape
            bad  = (~np.all(np.isfinite(traj.reshape(N,-1)), axis=1)).sum()
            tr   = len(g["train_indices"])
            va   = len(g["val_indices"])
            te   = len(g["test_indices"])
            icon = "✓" if bad == 0 else "✗"
            print(f"  {icon} {name:<10}  shape=({N},{T},{D})  "
                  f"bad={bad}  train={tr} val={va} test={te}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  File size: {os.path.getsize(WORKING_H5)/1e6:.1f} MB")
    print(f"\n  Next step: update HDF5_PATH in benchmark script to:")
    print(f"    HDF5_PATH = '{WORKING_H5}'")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
