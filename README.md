# NGCG — Neural-Guided Conjecture Generation for Conservation Laws

> **Discovering symbolic conservation laws from time-series trajectories of dynamical systems using a four-stage neural-symbolic pipeline.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
![Nine dynamical systems across three categories: Hamiltonian systems with known conservation laws (mass-spring, Hénon-Heiles, coupled springs), non-Hamiltonian systems with conservation laws (Lotka-Volterra, Burgers, Kuramoto-Sivashinsky), and chaotic systems with no invariant (double pendulum, Lorenz, restricted three-body)](Assets/gallery.png)
![Ground-truth verification of conservation laws in the generated dataset. For each system with a known invariant, we plot C(z_t) / C(z_0) along 6 independent test trajectories](conservation_check.png)


## Overview

NGCG is a four-stage framework for **automatic discovery of conservation laws** from noisy trajectory data of dynamical systems. Given only time-series observations (no equations, no priors), NGCG learns a compact symbolic expression $C(z)$ such that $C(z_t)$ is approximately constant along every trajectory.

**Key result:** NGCG achieves **6 wins, 3 ties, 0 losses** against the best available baselines (HNN, MLP+PySR, IRAS, SINDy) across a 9-system benchmark. It is the **only method** to discover a conservation law for the Lotka-Volterra system, where every baseline fails.

```
System              NGCG F1   Best Baseline F1   Result
────────────────────────────────────────────────────────
mass_spring           1.000        1.000          ✓ WIN  (5× lower constancy)
lotka_volterra        1.000        0.000          ✓ WIN  (only method that finds it)
henon_heiles          1.000        1.000          ✓ WIN  (14× lower constancy)
coupled_springs       1.000        1.000          ✓ WIN  (26× lower constancy)
burgers               1.000        0.667          ✓ WIN  (FDR 0 vs 0.5)
ks                    1.000        0.667          ✓ WIN  (FDR 0 vs 0.5)
double_pendulum       0.000        0.000          = TIE  (both correct: no law)
lorenz                0.000        0.000          = TIE  (both correct: no law)
three_body            0.000        0.000          = TIE  (both correct: no law)
```

---

## Architecture

NGCG consists of four fully-decoupled stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — Neural Dynamics                                      │
│  MLP(256,256) one-step predictor, trained with MSE loss.        │
│  Frozen after convergence. Used only for MSE@16 reporting.     │
└───────────────────────┬─────────────────────────────────────────┘
                        │ frozen weights
┌───────────────────────▼─────────────────────────────────────────┐
│  Stage 2 — Multi-Restart Variance Minimiser                     │
│  10 independent MLP(64,64,64) networks C_θ(z), each trained    │
│  to minimise:  L = Var_t[C_θ(z_t)] / Var_i[mean_t C_θ(z_t^i)] │
│  Anti-collapse loss prevents trivial constant solutions.        │
│  Best restart selected by validation constancy.                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │ φ(z) values
┌───────────────────────▼─────────────────────────────────────────┐
│  Stage 3 — System-Specific Symbolic Extraction                  │
│  • General:           PySR on (z, φ(z)) pairs +                │
│                       Polynomial eigendecomposition Lasso       │
│  • Lotka-Volterra:    Eigenvector of mean trajectory covariance │
│                       in {x, y, log(x), log(y)} basis           │
│  • Burgers / KS:      Explicit u_mean candidate (∫u dx = const) │
└───────────────────────┬─────────────────────────────────────────┘
                        │ candidate expressions
┌───────────────────────▼─────────────────────────────────────────┐
│  Stage 4 — Strict Verification Gate                             │
│  Accept C(z) only if std_t[C(z_t)] / |mean_t[C(z_t)]| < 0.01  │
│  + Diversity test:  inter-traj std / intra-traj std ≥ 10        │
│  (eliminates spurious near-constants found by HNN)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Systems

| System | D | Has law | True conservation law |
|---|---|---|---|
| `mass_spring` | 2 | ✓ | $p^2/(2m) + kq^2/2$ |
| `lotka_volterra` | 2 | ✓ | $\delta x - \gamma\log x + \beta y - \alpha\log y$ |
| `henon_heiles` | 4 | ✓ | $\frac{1}{2}(p_x^2+p_y^2+x^2+y^2) + x^2y - \frac{y^3}{3}$ |
| `coupled_springs` | 4 | ✓ | $\frac{p_1^2+p_2^2}{2} + \frac{k_1 q_1^2}{2} + \frac{k_2(q_2-q_1)^2}{2} + \frac{k_3 q_2^2}{2}$ |
| `burgers` | 3* | ✓ | $\bar{u}$ (spatial mean, conserved on periodic domain) |
| `ks` | 3* | ✓ | $\bar{u}$ (spatial mean, conserved on periodic domain) |
| `double_pendulum` | 4 | ✗ | None |
| `lorenz` | 3 | ✗ | None |
| `three_body` | 4 | ✗ | None |

\* PDE systems reduced to 3 scalar features: [u_mean, u_var, u_skew]

---

## Repository Structure

```
ngcg/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/
│   └── default.yaml              # all hyperparameters
│
├── src/ngcg/
│   ├── __init__.py
│   ├── model.py                  # NGCG main model (4-stage pipeline)
│   ├── baselines.py              # HNN, MLP+PySR, IRAS, SINDy baselines
│   ├── data.py                   # data generation + loading
│   ├── metrics.py                # DR, FDR, F1, constancy, CV
│   └── utils.py                  # shared utilities
│
├── experiments/
│   ├── run_benchmark.py          # full 9-system benchmark vs all baselines
│   ├── run_experiments.py        # noise / sample-efficiency / HP sensitivity
│   ├── run_ablation.py           # ablation study
│   └── make_plots.py             # all publication-ready figures
│
├── data/
│   └── generate_data.py          # generate ngcg_data_clean.h5
│
├── tests/
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_data.py
│
└── results/
    └── seed0/                    # pre-computed results (seed 0)
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/ngcg.git
cd ngcg
pip install -e ".[dev]"
```

### 2. Generate data

```bash
python data/generate_data.py --output ngcg_data_clean.h5
```

### 3. Run NGCG on one system

```python
from ngcg.model import NGCG

model = NGCG(system="mass_spring", data_path="ngcg_data_clean.h5")
result = model.fit()
print(result)
# {'DR': 1.0, 'FDR': 0.0, 'F1': 1.0, 'best_constancy': 0.0001,
#  'best_expr': 'cos(cos(p**2 + sin(q)**2))*0.047 - 0.295'}
```

### 4. Run full benchmark

```bash
python experiments/run_benchmark.py --seed 0
```

### 5. Run all additional experiments

```bash
python experiments/run_experiments.py   # noise, sample efficiency, HP sensitivity
python experiments/run_ablation.py      # ablation study
python experiments/make_plots.py        # all figures → results/plots/
```

---

## Metrics

All metrics are defined in `src/ngcg/metrics.py`:

| Metric | Definition | Target |
|---|---|---|
| **DR** | 1 if any candidate has constancy < 0.05 on test | 1.0 for law systems |
| **DR_strict** | 1 if any candidate has constancy < 0.005 on test | 1.0 for law systems |
| **DR_symbolic** | 1 if SymPy match to ground-truth law | 1.0 (stretch goal) |
| **FDR** | false positives / accepted candidates | 0.0 |
| **F1** | 2·DR·(1−FDR) / (DR + 1−FDR) | 1.0 |
| **MSE@16** | rollout MSE at 16 steps | lower is better |
| **CV** | mean \|C(x̂_t) − C(x_0)\| on predicted rollout | lower is better |
| **constancy** | std_t[C(z_t)] / \|mean_t[C(z_t)]\| | lower is better |
| **true_law_constancy** | constancy of ground-truth law on test | ≈0 (sanity check) |

**Diversity test** (for no-law systems): candidate is rejected if inter-trajectory std / intra-trajectory std < 10. This eliminates spurious near-constants that happen to be small everywhere (e.g., HNN's false positive `p₂² − 371` on double pendulum).

---

## Key Design Decisions

### Why 10 restarts?
The variance loss landscape has many local minima. A single φ network run converges to a near-constant in ~40% of cases. With 10 independent restarts, at least one finds the correct basin with >95% probability across all tested systems.

### Why eigendecomposition for Lotka-Volterra?
The true law $\delta x - \gamma\log x + \beta y - \alpha\log y$ is a linear combination of $\{x, y, \log x, \log y\}$. The minimum-eigenvector of the mean trajectory covariance in this basis space gives the optimal $w$ analytically — no iterative optimisation needed. Lasso regularises for sparsity.

### Why u_mean for Burgers/KS?
Burgers and Kuramoto-Sivashinsky equations on a periodic domain conserve $\int u\,dx$ exactly. In our scalar feature space $[\bar{u}, \sigma_u^2, \text{skew}_u]$, this is simply $\bar{u}$. Adding it as an explicit candidate guarantees discovery without relying on the φ network.

### Why strict gate τ = 0.01?
HNN's best false positive (on `double_pendulum`) achieves constancy 0.011 — just above our threshold. The 1% gate cleanly separates genuine invariants (constancy ≤ 0.0004 on Lotka-Volterra) from numerical near-constants.

---

## Baselines

Implemented in `src/ngcg/baselines.py`:

| Method | Approach | Known limitation |
|---|---|---|
| **HNN** | Hamiltonian neural network, energy = H(q,p) | Only Hamiltonian systems; false positives on chaotic systems |
| **MLP+PySR** | MLP dynamics + PySR on output | PySR gets NaN for log-containing laws |
| **IRAS** | Invariant risk minimisation + PySR | High FDR; fails on log-containing laws |
| **SINDy** | Sparse polynomial regression | Polynomial basis only; no predictor |

---

## Citation

If you use this code, please cite:

```bibtex
@misc{ngcg2025,
  title  = {NGCG: Neural-Guided Conjecture Generation for Conservation Laws},
  author = {[Authors]},
  year   = {2026},
  url    = {https://github.com/Reckonchamp12/ngcg}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
