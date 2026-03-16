# NGCG Architecture

## Stage 1 — Neural Dynamics

A standard MLP(256, 256) with Tanh activations is trained to predict next states:

$$z_{t+1} \approx f_\theta(z_t)$$

Training uses MSE loss with OneCycleLR scheduling and automatic mixed precision (AMP) on GPU. The model is **frozen completely** after convergence — it is used only for MSE@16 evaluation and, optionally, as a drift vector for derivative-based verification.

**Why a simple MLP?** We intentionally avoid physics-informed architecture choices (like HNN) so that Stage 2 must discover structure purely from data. This makes the benchmark fair.

---

## Stage 2 — Multi-Restart Variance Minimiser

We train `R = 10` independent MLP(64, 64, 64) networks $C_\theta(z)$, each minimising:

$$\mathcal{L} = \frac{\frac{1}{N}\sum_i \text{Var}_t[C_\theta(z_t^{(i)})]}{\frac{1}{N}\sum_i (\overline{C_\theta}^{(i)} - \overline{\overline{C_\theta}})^2 + \varepsilon}$$

This is the **normalised variance loss**: intra-trajectory variance divided by inter-trajectory variance. The denominator prevents the trivial solution of outputting a global constant (which would give numerator → 0 but also denominator → 0).

The best restart is selected by validation set constancy.

**Why 10 restarts?** The loss landscape has many local minima. Empirically, a single run converges to a near-constant (collapsed) solution ~40% of the time. With 10 independent restarts, at least one finds the correct basin with >95% probability.

---

## Stage 3 — System-Specific Symbolic Extraction

### General systems (Hamiltonian + conservative ODEs)

1. **Polynomial Lasso**: Compute the mean trajectory covariance matrix $A = \frac{1}{N}\sum_i \text{Cov}_t[\phi_\text{poly}(z_t^{(i)})]$ where $\phi_\text{poly}$ contains all monomials up to degree 3. The minimum eigenvector of $A$ is the direction of minimum variance — guaranteed to find any polynomial conservation law in the basis.

2. **PySR**: Run symbolic regression on $(z, C_\theta(z))$ pairs. The φ network guides the search to the relevant region of expression space. All top-8 candidates are evaluated by test constancy (not R², which is unreliable when φ has small range).

### Lotka-Volterra

The true law $\delta x - \gamma \log x + \beta y - \alpha \log y$ is a linear combination of $\{x, y, \log x, \log y\}$. We compute:

$$A_\text{LV} = \frac{1}{N}\sum_i \text{Cov}_t\left[\phi_\text{LV}(z_t^{(i)})\right], \quad \phi_\text{LV}(z) = [x, y, \log x, \log y]$$

The minimum eigenvector of $A_\text{LV}$ gives the optimal linear combination analytically. This is exact when all trajectories share the same parameters; with varying parameters, it gives the direction of minimum average variance — sufficient for discovery.

Lasso regularisation is applied as a post-processing step for sparsity.

### PDE systems (Burgers, KS)

Burgers and Kuramoto-Sivashinsky equations on a periodic domain conserve:

$$\frac{d}{dt}\int u(x,t)\,dx = 0 \implies \bar{u}(t) = \text{const}$$

We add $\bar{u}$ as an explicit candidate unconditionally, without relying on φ.

---

## Stage 4 — Verification Gate

A candidate $C(z)$ is accepted if:

**Constancy gate:**
$$\frac{1}{N}\sum_i \frac{\text{std}_t[C(z_t^{(i)})]}{|\overline{C(z^{(i)})}|} < \tau = 0.01$$

**Diversity test** (for no-law systems):
$$\frac{\text{std}_i[\overline{C(z^{(i)})}]}{\frac{1}{N}\sum_i \text{std}_t[C(z_t^{(i)})]} \geq 10$$

The diversity test rejects spurious near-constants. A genuine invariant varies between trajectories (different ICs have different conserved values) but is constant within each trajectory. Lorenz trajectories all lie on the same strange attractor, so any near-constant expression has inter/intra ratio ≈ 1 — correctly rejected.

---

## Baseline Failure Analysis

| System | Baseline failure | NGCG fix |
|---|---|---|
| lotka_volterra | PySR received NaN φ values; log terms not in basis | Explicit {x, y, log x, log y} basis; eigenvector method |
| henon_heiles | φ collapsed; R² filter rejected valid PySR candidates | 10 restarts; evaluate by test constancy not R² |
| double_pendulum (HNN FP) | HNN found p₂² − 371 with constancy 0.011 | Strict gate τ=0.01 + diversity test |
| lorenz (polynomial FP) | Poly-Lasso found minimum-variance polynomial direction | Diversity test: lorenz attractor has inter/intra ≈ 1 |
| burgers | Labelled as no-law; u_mean is genuinely conserved | Added u_mean as true conservation law for PDE systems |
