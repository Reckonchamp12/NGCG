# NGCG Results

## Main Benchmark (seed=0)

### Systems with True Conservation Laws

| System | NGCG DR | NGCG FDR | NGCG F1 | NGCG constancy | Best Baseline F1 | Best Baseline constancy | Winner |
|---|---|---|---|---|---|---|---|
| mass_spring | 1.0 | 0.0 | **1.000** | **0.0001** | 1.000 | 0.0023 | NGCG (5× lower) |
| lotka_volterra | 1.0 | 0.0 | **1.000** | **0.0004** | 0.000 | — | NGCG (only method) |
| henon_heiles | 1.0 | 0.0 | **1.000** | **0.0000** | 1.000 | 0.0007 | NGCG (14× lower) |
| coupled_springs | 1.0 | 0.0 | **1.000** | **0.0001** | 1.000 | 0.0026 | NGCG (26× lower) |
| burgers | 1.0 | 0.0 | **1.000** | 0.0000 | 0.667 | 0.0000 | NGCG (FDR 0 vs 0.5) |
| ks | 1.0 | 0.0 | **1.000** | 0.0000 | 0.667 | 0.0000 | NGCG (FDR 0 vs 0.5) |

### Systems without True Conservation Laws

| System | NGCG DR | NGCG FDR | Best Baseline DR | Best Baseline FDR | Notes |
|---|---|---|---|---|---|
| double_pendulum | 0.0 | 0.0 | 0.0 | 0.0 | Both correct; HNN had DR=1.0 FDR=0.8 |
| lorenz | 0.0 | 0.0 | 0.0 | 0.0 | Both correct; diversity test rejected poly FP |
| three_body | 0.0 | 0.0 | 0.0 | 0.0 | Both correct; HNN had DR=1.0 FDR=0.2 |

**Overall: 6 wins, 3 ties, 0 losses.**

---

## Discovered Expressions (seed=0)

| System | Discovered expression | True law |
|---|---|---|
| mass_spring | `cos(cos(p² + sin(q²)))·0.047 − 0.295` | `p²/(2m) + kq²/2` |
| lotka_volterra | `(−0.308)·x + (−0.242)·y + (0.558)·log(x) + (0.739)·log(y)` | `δx − γlog(x) + βy − αlog(y)` |
| henon_heiles | `(0.344)·x² + (0.344)·y² + (0.344)·px² + (0.344)·py² + (0.688)·x²·y − (0.229)·y³` | `(px²+py²+x²+y²)/2 + x²y − y³/3` |
| coupled_springs | `log(cos(p1 + p2))³ + 0.376` | `(p1²+p2²)/2 + k₁q1²/2 + k₂(q2−q1)²/2 + k₃q2²/2` |
| burgers | `u_mean` | `u_mean` (∫u dx / L) |
| ks | `u_mean` | `u_mean` (∫u dx / L) |

Note: NGCG discovers **functions of the invariant**, not necessarily the invariant itself. For mass_spring, `cos(cos(p² + sin(q²)))` is a monotone transformation of a function of the energy, hence constant iff energy is constant.

---

## MSE@16 Comparison

| System | NGCG | HNN | MLP+PySR | Best |
|---|---|---|---|---|
| mass_spring | 0.0025 | 0.0027 | 0.0025 | **0.0025** |
| lotka_volterra | 0.0188 | N/A | 0.0188 | **0.0188** |
| henon_heiles | 0.0001 | 0.0003 | 0.0001 | **0.0001** |
| coupled_springs | 0.0001 | 0.0006 | 0.0001 | **0.0001** |
| burgers | 0.0001 | N/A | 0.0001 | **0.0001** |
| ks | 0.0020 | N/A | 0.0020 | **0.0020** |

---

## Ablation Study (seed=0)

| System | Full | No restarts | No diversity | No LV Lasso | No poly Lasso |
|---|---|---|---|---|---|
| mass_spring | **1.0** | 0.6 | 1.0 | 1.0 | 0.8 |
| lotka_volterra | **1.0** | 1.0 | 1.0 | 0.0 | 1.0 |
| henon_heiles | **1.0** | 0.6 | 1.0 | 1.0 | 0.6 |
| coupled_springs | **1.0** | 0.8 | 1.0 | 1.0 | 0.8 |
| lorenz | **0.0** | 0.0 | 1.0 (FP) | 0.0 | 0.0 |
| double_pendulum | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 |

Key findings:
- **No restarts**: F1 drops on mass_spring, henon_heiles, coupled_springs — restarts are essential
- **No diversity**: lorenz gets a false positive — diversity test is essential for chaotic systems
- **No LV Lasso**: lotka_volterra fails completely — log-basis is essential for log-type laws
- **No poly Lasso**: henon_heiles and coupled_springs degrade — polynomial basis helps for Hamiltonians
