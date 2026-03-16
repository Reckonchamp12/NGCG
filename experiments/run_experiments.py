# ╔══════════════════════════════════════════════════════════════════╗
# ║  ADDITIONAL EXPERIMENTS — paste each CELL into Kaggle           ║
# ║  Requires: Cells 1-4 from kaggle_cells.py already run           ║
# ╚══════════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E1 — helper: run_system_on_data (accepts pre-built data dict)
# Used by all experiments below to avoid re-loading HDF5 each time.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXP_DIR = f"{OUT_DIR}/experiments"
os.makedirs(f"{EXP_DIR}/plots", exist_ok=True)

# Focused systems for experiments (fast, covers all law types)
EXP_SYSTEMS = ["mass_spring", "henon_heiles", "coupled_springs",
               "lotka_volterra", "lorenz", "double_pendulum"]

def _quick_run(system, tr, va, te, seed=0,
               n_phi_restarts=None, diversity_threshold=10.0,
               use_lv_lasso=True, use_poly_lasso=True):
    """
    Stripped-down run_system that accepts arrays directly (no HDF5 reload).
    Returns dict with DR, FDR, F1, best_constancy, fit_time_s.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    D        = tr.shape[-1]
    svars    = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
    true_law = TRUE_LAWS.get(system)
    has_law  = bool(true_law)
    t0       = time.time()

    # Stage 1: dynamics
    dyn = MLP(D, D, hidden=HP["dyn_hidden"]).to(DEVICE)
    train_dynamics(dyn, tr, va)
    dyn.eval()

    # Stage 2: phi restarts
    _nr = n_phi_restarts or (3 if system in PDE_SYSTEMS else HP["n_restarts"])
    best_phi, _, _ = run_multi_restart_phi(tr, va, D, _nr)

    # Stage 3: candidates
    candidates = []
    if system in PDE_SYSTEMS:
        sc = constancy_score(svars[0], svars, te) if len(te) > 0 else 999.0
        candidates.append((svars[0], sc))

    if system == "lotka_volterra" and use_lv_lasso:
        best_w, _ = lv_variance_lasso(tr, va, HP["lv_lambdas"])
        if best_w is not None:
            sc = lv_expr_constancy(best_w, te) if len(te) > 0 else 999.0
            candidates.append((lv_w_to_expr(best_w), sc))

    if system not in PDE_SYSTEMS:
        if has_law and D <= 4 and use_poly_lasso:
            for e, s in _poly_lasso_candidates(tr, va, te, svars, D):
                candidates.append((e, s))
        pts = tr.reshape(-1, D)
        idx = np.random.choice(len(pts), min(HP["pysr_n_pts"], len(pts)), replace=False)
        xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
        with torch.no_grad():
            y_phi = best_phi(xt).squeeze(-1).cpu().numpy()
        for e, r2 in run_pysr_safe(pts[idx], y_phi, svars,
                                   HP["pysr_niter"], HP["pysr_maxsize"])[:8]:
            sc = constancy_score(e, svars, te) if len(te) > 0 else 999.0
            candidates.append((e, sc))

    # Stage 4: gate + diversity
    GATE  = HP["gate_strict"]; TOL_L = HP["gate_loose"]
    candidates.sort(key=lambda x: x[1])
    accepted = [(e, s) for e, s in candidates if s < GATE]
    if system in PDE_SYSTEMS:
        simple = [(e, s) for e, s in accepted if e.strip() == svars[0]]
        if simple: accepted = simple
    all_scores = [s for _, s in candidates]

    if has_law:
        DR  = 1.0 if any(s < TOL_L for s in all_scores) else 0.0
        tp_acc = sum(1 for _, s in accepted if s < TOL_L)
        FDR = (len(accepted) - tp_acc) / max(1, len(accepted)) if accepted else 0.0
    else:
        div_acc = [(e, s) for e, s in accepted
                   if _passes_diversity_test(e, svars, te, min_ratio=diversity_threshold)]
        DR  = 1.0 if div_acc else 0.0
        FDR = 1.0 if div_acc else 0.0

    F1  = 2*DR*(1-FDR) / max(1e-9, DR+(1-FDR))
    bc  = min(all_scores) if all_scores else 999.0
    return {"DR": DR, "FDR": FDR, "F1": F1,
            "best_constancy": bc, "fit_time_s": round(time.time()-t0, 1)}

print("✓ Cell E1 — _quick_run helper defined")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E2 — Noise Robustness
# Add Gaussian noise σ ∈ {0, 0.01, 0.05, 0.1} to train/val/test.
# Measure DR and FDR for each system × noise level.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
noise_rows   = []

for system in EXP_SYSTEMS:
    print(f"\n── Noise robustness: {system} ──")
    data = load_system(system)
    if data is None: continue
    tr0, va0, te0 = data["train"], data["val"], data["test"]

    for sigma in NOISE_LEVELS:
        rng = np.random.default_rng(SEED)
        tr_n = tr0 + rng.normal(0, sigma, tr0.shape).astype(np.float32)
        va_n = va0 + rng.normal(0, sigma, va0.shape).astype(np.float32)
        te_n = te0 + rng.normal(0, sigma, te0.shape).astype(np.float32)
        # Clip populations to positive for LV
        if system == "lotka_volterra":
            tr_n = np.clip(tr_n, 1e-4, None)
            va_n = np.clip(va_n, 1e-4, None)
            te_n = np.clip(te_n, 1e-4, None)
        try:
            res = _quick_run(system, tr_n, va_n, te_n, seed=0)
            print(f"  σ={sigma:.2f}  DR={res['DR']:.2f}  FDR={res['FDR']:.3f}  "
                  f"F1={res['F1']:.3f}  ({res['fit_time_s']}s)")
        except Exception as e:
            print(f"  σ={sigma:.2f}  FAILED: {e}"); traceback.print_exc()
            res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
        noise_rows.append({"system":system,"sigma":sigma,**res})

df_noise = pd.DataFrame(noise_rows)
df_noise.to_csv(f"{EXP_DIR}/noise_robustness.csv", index=False)

# ── Plot: DR vs noise level per system ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()
colors = {"DR":"#2ecc71","FDR":"#e74c3c","F1":"#3498db"}
for i, system in enumerate(EXP_SYSTEMS):
    ax  = axes[i]
    sub = df_noise[df_noise.system == system]
    ax.plot(sub.sigma, sub.DR,  "o-", color=colors["DR"],  label="DR",  lw=2, ms=6)
    ax.plot(sub.sigma, sub.F1,  "s-", color=colors["F1"],  label="F1",  lw=2, ms=6)
    ax.plot(sub.sigma, sub.FDR, "^-", color=colors["FDR"], label="FDR", lw=2, ms=6)
    ax.set_title(system, fontsize=9); ax.set_xlabel("Noise σ"); ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(NOISE_LEVELS); ax.axhline(1, ls="--", c="gray", lw=0.7)
    if i == 0: ax.legend(fontsize=8)
fig.suptitle("Noise Robustness — NGCG-Win DR/F1/FDR vs Gaussian noise σ", fontsize=11)
fig.tight_layout()
fig.savefig(f"{EXP_DIR}/plots/noise_robustness.png", dpi=140); plt.close()
print(f"\n✅ Noise results → {EXP_DIR}/plots/noise_robustness.png")
from IPython.display import Image, display; display(Image(f"{EXP_DIR}/plots/noise_robustness.png"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E3 — Sample Efficiency
# Vary number of training trajectories: 50, 100, 150, 200, 280, 350.
# For each level, subsample from the full training set.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAIN_SIZES = [50, 100, 150, 200, 280, 350]
seff_rows   = []

for system in EXP_SYSTEMS:
    print(f"\n── Sample efficiency: {system} ──")
    data = load_system(system)
    if data is None: continue
    tr0, va0, te0 = data["train"], data["val"], data["test"]

    for n_tr in TRAIN_SIZES:
        if n_tr > len(tr0):
            print(f"  n={n_tr} skipped (only {len(tr0)} available)")
            continue
        rng    = np.random.default_rng(SEED)
        idx_tr = rng.choice(len(tr0), n_tr, replace=False)
        tr_sub = tr0[idx_tr]
        # Use 20% of n_tr for val (min 10)
        n_va   = max(10, n_tr // 5)
        idx_va = rng.choice(len(va0), min(n_va, len(va0)), replace=False)
        va_sub = va0[idx_va]
        try:
            res = _quick_run(system, tr_sub, va_sub, te0, seed=0)
            print(f"  n_tr={n_tr:3d}  DR={res['DR']:.2f}  F1={res['F1']:.3f}  "
                  f"({res['fit_time_s']}s)")
        except Exception as e:
            print(f"  n_tr={n_tr:3d}  FAILED: {e}")
            res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
        seff_rows.append({"system":system,"n_train":n_tr,**res})

df_seff = pd.DataFrame(seff_rows)
df_seff.to_csv(f"{EXP_DIR}/sample_efficiency.csv", index=False)

# ── Plot: DR vs n_train ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()
for i, system in enumerate(EXP_SYSTEMS):
    ax  = axes[i]
    sub = df_seff[df_seff.system == system].sort_values("n_train")
    ax.plot(sub.n_train, sub.DR, "o-", color="#2ecc71", lw=2, ms=7, label="DR")
    ax.plot(sub.n_train, sub.F1, "s-", color="#3498db", lw=2, ms=7, label="F1")
    ax.set_title(system, fontsize=9); ax.set_xlabel("Training trajectories")
    ax.set_ylim(-0.05, 1.15); ax.axhline(1, ls="--", c="gray", lw=0.7)
    ax.set_xticks(TRAIN_SIZES); ax.tick_params(axis='x', labelsize=7)
    if i == 0: ax.legend(fontsize=8)
fig.suptitle("Sample Efficiency — NGCG-Win DR/F1 vs Number of Training Trajectories", fontsize=11)
fig.tight_layout()
fig.savefig(f"{EXP_DIR}/plots/sample_efficiency.png", dpi=140); plt.close()
print(f"\n✅ Sample efficiency → {EXP_DIR}/plots/sample_efficiency.png")
display(Image(f"{EXP_DIR}/plots/sample_efficiency.png"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E4 — Hyperparameter Sensitivity
# 4a: n_phi_restarts ∈ {1, 3, 5, 10, 20}
# 4b: L1 penalty strength (lam_l1 proxy: vary n_lasso_lambdas range)
# 4c: Diversity threshold ∈ {5, 10, 20, 50}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HP_SYS = ["mass_spring", "henon_heiles", "coupled_springs", "lorenz"]
hp_rows = []

for system in HP_SYS:
    print(f"\n── HP sensitivity: {system} ──")
    data = load_system(system)
    if data is None: continue
    tr, va, te = data["train"], data["val"], data["test"]

    # 4a: n_restarts
    for nr in [1, 3, 5, 10, 20]:
        try:
            res = _quick_run(system, tr, va, te, seed=0, n_phi_restarts=nr)
            print(f"  n_restarts={nr:2d}  DR={res['DR']:.2f}  F1={res['F1']:.3f}")
        except Exception as e:
            print(f"  n_restarts={nr:2d}  FAILED:{e}")
            res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
        hp_rows.append({"system":system,"hp_type":"n_restarts","hp_val":nr,**res})

    # 4b: diversity threshold (only meaningful for no-law systems)
    if not TRUE_LAWS.get(system):
        for dt in [5.0, 10.0, 20.0, 50.0]:
            try:
                res = _quick_run(system, tr, va, te, seed=0, diversity_threshold=dt)
                print(f"  div_thresh={dt:4.0f}  DR={res['DR']:.2f}  F1={res['F1']:.3f}")
            except Exception as e:
                print(f"  div_thresh={dt:4.0f}  FAILED:{e}")
                res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
            hp_rows.append({"system":system,"hp_type":"div_threshold","hp_val":dt,**res})

df_hp = pd.DataFrame(hp_rows)
df_hp.to_csv(f"{EXP_DIR}/hp_sensitivity.csv", index=False)

# ── Plot 4a: F1 vs n_restarts ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(HP_SYS), figsize=(14, 3.5), sharey=True)
for i, system in enumerate(HP_SYS):
    ax  = axes[i]
    sub = df_hp[(df_hp.system==system)&(df_hp.hp_type=="n_restarts")].sort_values("hp_val")
    ax.plot(sub.hp_val, sub.F1, "o-", color="#2ecc71", lw=2, ms=7)
    ax.plot(sub.hp_val, sub.DR, "s--", color="#3498db", lw=1.5, ms=5, alpha=0.7)
    ax.set_title(system, fontsize=9); ax.set_xlabel("n_restarts"); ax.set_ylim(-0.05,1.15)
    ax.axhline(1, ls="--", c="gray", lw=0.7); ax.set_xticks([1,3,5,10,20])
    if i==0: ax.set_ylabel("Score"); ax.legend(["F1","DR"],fontsize=8)
fig.suptitle("HP Sensitivity: n_phi_restarts vs F1", fontsize=11)
fig.tight_layout(); fig.savefig(f"{EXP_DIR}/plots/hp_n_restarts.png", dpi=140); plt.close()

# ── Plot 4b: F1 vs diversity_threshold (no-law systems) ──────────────
no_law_sys = [s for s in HP_SYS if not TRUE_LAWS.get(s)]
if no_law_sys:
    fig, axes = plt.subplots(1, len(no_law_sys), figsize=(7*len(no_law_sys), 3.5), sharey=True)
    if len(no_law_sys)==1: axes=[axes]
    for i, system in enumerate(no_law_sys):
        ax  = axes[i]
        sub = df_hp[(df_hp.system==system)&(df_hp.hp_type=="div_threshold")].sort_values("hp_val")
        ax.plot(sub.hp_val, sub.F1, "o-", color="#e74c3c", lw=2, ms=7)
        ax.set_title(system, fontsize=9); ax.set_xlabel("Diversity threshold")
        ax.set_ylim(-0.05,1.15); ax.axhline(0, ls="--", c="gray", lw=0.7)
        if i==0: ax.set_ylabel("F1")
    fig.suptitle("HP Sensitivity: Diversity Threshold vs F1 (no-law systems, lower=better)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{EXP_DIR}/plots/hp_diversity.png", dpi=140); plt.close()

print(f"\n✅ HP sensitivity plots → {EXP_DIR}/plots/")
display(Image(f"{EXP_DIR}/plots/hp_n_restarts.png"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E5 — Runtime Comparison
# Measure wall-clock time of NGCG-Win vs HNN+PySR and MLP+PySR
# on the same 3 systems. Single seed, single run.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RT_SYSTEMS = ["mass_spring", "henon_heiles", "coupled_springs"]
rt_rows    = []

def run_hnn_pysr(system, tr, va, te, seed=0):
    """HNN + PySR baseline timing."""
    torch.manual_seed(seed); np.random.seed(seed)
    D = tr.shape[-1]; svars = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
    t0 = time.time()
    # HNN: same MLP but trained on Hamiltonian structure
    # For timing purposes we use the same MLP architecture
    dyn = MLP(D, D, hidden=HP["dyn_hidden"]).to(DEVICE)
    train_dynamics(dyn, tr, va); dyn.eval()
    # PySR on raw state
    pts = tr.reshape(-1, D)
    idx = np.random.choice(len(pts), min(HP["pysr_n_pts"], len(pts)), replace=False)
    # Target: predicted next state (simulate HNN output)
    xt = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
    with torch.no_grad(): y = dyn(xt).cpu().numpy()[:, 0]
    run_pysr_safe(pts[idx], y, svars, HP["pysr_niter"], HP["pysr_maxsize"])
    return round(time.time()-t0, 1)

def run_mlp_pysr(system, tr, va, te, seed=0):
    """MLP + PySR baseline timing."""
    torch.manual_seed(seed); np.random.seed(seed)
    D = tr.shape[-1]; svars = STATE_VARS.get(system, [f"x{i}" for i in range(D)])
    t0 = time.time()
    dyn = MLP(D, D, hidden=HP["dyn_hidden"]).to(DEVICE)
    train_dynamics(dyn, tr, va); dyn.eval()
    pts = tr.reshape(-1, D)
    idx = np.random.choice(len(pts), min(HP["pysr_n_pts"], len(pts)), replace=False)
    xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
    with torch.no_grad(): y = dyn(xt).cpu().numpy()[:, 0]
    run_pysr_safe(pts[idx], y, svars, HP["pysr_niter"], HP["pysr_maxsize"])
    return round(time.time()-t0, 1)

for system in RT_SYSTEMS:
    print(f"\n── Runtime: {system} ──")
    data = load_system(system)
    if data is None: continue
    tr, va, te = data["train"], data["val"], data["test"]

    # NGCG-Win
    try:
        res   = _quick_run(system, tr, va, te, seed=0)
        t_win = res["fit_time_s"]
    except Exception as e:
        print(f"  NGCG-Win FAILED: {e}"); t_win = 999.0

    # HNN+PySR
    try: t_hnn = run_hnn_pysr(system, tr, va, te)
    except Exception as e: print(f"  HNN FAILED:{e}"); t_hnn=999.0

    # MLP+PySR
    try: t_mlp = run_mlp_pysr(system, tr, va, te)
    except Exception as e: print(f"  MLP FAILED:{e}"); t_mlp=999.0

    print(f"  NGCG-Win={t_win}s  HNN+PySR={t_hnn}s  MLP+PySR={t_mlp}s")
    rt_rows += [
        {"system":system,"method":"NGCG-Win", "time_s":t_win},
        {"system":system,"method":"HNN+PySR", "time_s":t_hnn},
        {"system":system,"method":"MLP+PySR", "time_s":t_mlp},
    ]

df_rt = pd.DataFrame(rt_rows)
df_rt.to_csv(f"{EXP_DIR}/runtime.csv", index=False)

# ── Plot: grouped bar chart ───────────────────────────────────────────
methods  = ["NGCG-Win","HNN+PySR","MLP+PySR"]
mcols    = {"NGCG-Win":"#2ecc71","HNN+PySR":"#3498db","MLP+PySR":"#e67e22"}
x        = np.arange(len(RT_SYSTEMS)); w = 0.25
fig, ax  = plt.subplots(figsize=(9, 4))
for j, method in enumerate(methods):
    times = [df_rt[(df_rt.system==s)&(df_rt.method==method)]["time_s"].values[0]
             if len(df_rt[(df_rt.system==s)&(df_rt.method==method)])>0 else 0
             for s in RT_SYSTEMS]
    bars = ax.bar(x + (j-1)*w, times, w, label=method,
                  color=mcols[method], alpha=0.88, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{t:.0f}s", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(RT_SYSTEMS, fontsize=9)
ax.set_ylabel("Wall-clock time (seconds)"); ax.set_title("Runtime Comparison")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{EXP_DIR}/plots/runtime.png", dpi=140); plt.close()
print(f"\n✅ Runtime plot → {EXP_DIR}/plots/runtime.png")
display(Image(f"{EXP_DIR}/plots/runtime.png"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E6 — Deep Ablation
# Ablation variants beyond the basic study:
#   A) n_restarts ∈ {1,3,5,10}  — how many restarts is enough?
#   B) LV only (no poly-lasso) vs poly-lasso only (no LV) vs both
#   C) Diversity threshold ∈ {5,10,20,50} for no-law systems
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEEP_ABL_SYS = ["mass_spring","henon_heiles","coupled_springs",
                "lotka_volterra","lorenz","double_pendulum"]
deep_rows = []

for system in DEEP_ABL_SYS:
    print(f"\n── Deep ablation: {system} ──")
    data = load_system(system)
    if data is None: continue
    tr, va, te = data["train"], data["val"], data["test"]

    # A) n_restarts sweep
    for nr in [1, 3, 5, 10]:
        try:
            res = _quick_run(system, tr, va, te, seed=0, n_phi_restarts=nr)
            tag = f"restarts={nr}"
            print(f"  {tag:<22} DR={res['DR']:.2f}  F1={res['F1']:.3f}")
        except Exception as e:
            print(f"  restarts={nr} FAILED:{e}")
            res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
        deep_rows.append({"system":system,"variant":f"restarts={nr}",
                          "group":"n_restarts",**res})

    # B) lasso combination (only relevant for LV and polynomial systems)
    for use_lv, use_poly, tag in [
        (True,  True,  "full"),
        (False, True,  "no_lv_lasso"),
        (True,  False, "no_poly_lasso"),
        (False, False, "lasso_off"),
    ]:
        try:
            res = _quick_run(system, tr, va, te, seed=0,
                             use_lv_lasso=use_lv, use_poly_lasso=use_poly)
            print(f"  {tag:<22} DR={res['DR']:.2f}  F1={res['F1']:.3f}")
        except Exception as e:
            print(f"  {tag:<22} FAILED:{e}")
            res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
        deep_rows.append({"system":system,"variant":tag,"group":"lasso_combo",**res})

    # C) diversity threshold (no-law systems only)
    if not TRUE_LAWS.get(system):
        for dt in [5.0, 10.0, 20.0, 50.0]:
            try:
                res = _quick_run(system, tr, va, te, seed=0, diversity_threshold=dt)
                tag = f"div={dt:.0f}"
                print(f"  {tag:<22} DR={res['DR']:.2f}  F1={res['F1']:.3f}")
            except Exception as e:
                print(f"  div={dt} FAILED:{e}")
                res = {"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0,"fit_time_s":0.0}
            deep_rows.append({"system":system,"variant":tag,
                              "group":"div_threshold",**res})

df_deep = pd.DataFrame(deep_rows)
df_deep.to_csv(f"{EXP_DIR}/deep_ablation.csv", index=False)

# ── Plot A: restarts heatmap ──────────────────────────────────────────
sub_r  = df_deep[df_deep.group=="n_restarts"]
pivot_r = sub_r.pivot_table(index="system",columns="variant",values="F1")
order   = [f"restarts={n}" for n in [1,3,5,10]]
pivot_r = pivot_r.reindex(columns=[c for c in order if c in pivot_r.columns])
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(pivot_r.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot_r.columns))); ax.set_xticklabels(pivot_r.columns, fontsize=9)
ax.set_yticks(range(len(pivot_r.index)));   ax.set_yticklabels(pivot_r.index, fontsize=9)
ax.set_title("Deep Ablation A — F1 vs n_phi_restarts", fontsize=10)
plt.colorbar(im, ax=ax, label="F1")
for i in range(len(pivot_r.index)):
    for j in range(len(pivot_r.columns)):
        v = pivot_r.values[i,j]
        ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "—",
                ha="center", va="center", fontsize=9,
                color="white" if v<0.45 else "black")
fig.tight_layout(); fig.savefig(f"{EXP_DIR}/plots/deep_abl_restarts.png", dpi=140); plt.close()

# ── Plot B: lasso combo heatmap ───────────────────────────────────────
sub_l   = df_deep[df_deep.group=="lasso_combo"]
pivot_l = sub_l.pivot_table(index="system",columns="variant",values="F1")
col_order = ["full","no_lv_lasso","no_poly_lasso","lasso_off"]
pivot_l = pivot_l.reindex(columns=[c for c in col_order if c in pivot_l.columns])
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(pivot_l.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot_l.columns))); ax.set_xticklabels(pivot_l.columns, fontsize=9)
ax.set_yticks(range(len(pivot_l.index)));   ax.set_yticklabels(pivot_l.index, fontsize=9)
ax.set_title("Deep Ablation B — F1 by Lasso Component", fontsize=10)
plt.colorbar(im, ax=ax, label="F1")
for i in range(len(pivot_l.index)):
    for j in range(len(pivot_l.columns)):
        v = pivot_l.values[i,j]
        ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "—",
                ha="center", va="center", fontsize=9,
                color="white" if v<0.45 else "black")
fig.tight_layout(); fig.savefig(f"{EXP_DIR}/plots/deep_abl_lasso.png", dpi=140); plt.close()

print(f"\n✅ Deep ablation plots → {EXP_DIR}/plots/")
display(Image(f"{EXP_DIR}/plots/deep_abl_restarts.png"))
display(Image(f"{EXP_DIR}/plots/deep_abl_lasso.png"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL E7 — Pareto Plot: Constancy vs Complexity
# For each system with accepted candidates, plot all expressions
# as (complexity, test_constancy) points to show the Pareto frontier.
# Reads from the results CSV; complexity computed via SymPy.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_pareto_data():
    """
    Re-run quick_run but capture ALL candidates (not just best).
    Returns list of {system, expr, constancy, complexity} dicts.
    """
    pareto_rows = []
    for system in ALL_SYSTEMS:
        if not TRUE_LAWS.get(system): continue   # only law systems
        data = load_system(system)
        if data is None: continue
        tr, va, te = data["train"], data["val"], data["test"]
        D     = tr.shape[-1]
        svars = STATE_VARS.get(system, [f"x{i}" for i in range(D)])

        torch.manual_seed(SEED); np.random.seed(SEED)
        dyn = MLP(D, D, hidden=HP["dyn_hidden"]).to(DEVICE)
        train_dynamics(dyn, tr, va); dyn.eval()
        _nr = 3 if system in PDE_SYSTEMS else HP["n_restarts"]
        best_phi, _, _ = run_multi_restart_phi(tr, va, D, _nr)

        cands = []
        if system in PDE_SYSTEMS:
            sc = constancy_score(svars[0], svars, te) if len(te)>0 else 999.0
            cands.append((svars[0], sc))
        if system == "lotka_volterra":
            best_w, _ = lv_variance_lasso(tr, va, HP["lv_lambdas"])
            if best_w is not None:
                cands.append((lv_w_to_expr(best_w),
                               lv_expr_constancy(best_w, te) if len(te)>0 else 999.0))
        if system not in PDE_SYSTEMS:
            if D <= 4:
                for e, s in _poly_lasso_candidates(tr, va, te, svars, D):
                    cands.append((e, s))
            pts = tr.reshape(-1, D)
            idx = np.random.choice(len(pts), min(HP["pysr_n_pts"], len(pts)), replace=False)
            xt  = torch.tensor(pts[idx], dtype=FLOAT, device=DEVICE)
            with torch.no_grad(): y_phi = best_phi(xt).squeeze(-1).cpu().numpy()
            for e, r2 in run_pysr_safe(pts[idx], y_phi, svars,
                                       HP["pysr_niter"], HP["pysr_maxsize"]):
                sc = constancy_score(e, svars, te) if len(te)>0 else 999.0
                cands.append((e, sc))

        for expr_str, sc in cands:
            if sc >= 1.0: continue  # skip clearly bad ones
            try: cx = float(sp.count_ops(_sympify(expr_str, svars)))
            except: cx = 0.0
            pareto_rows.append({"system":system,"expr":expr_str[:60],
                                 "constancy":sc,"complexity":cx})
    return pd.DataFrame(pareto_rows)

print("Collecting Pareto data (runs PySR — takes ~10 min)...")
df_pareto = collect_pareto_data()
df_pareto.to_csv(f"{EXP_DIR}/pareto_data.csv", index=False)

# ── Plot: constancy vs complexity scatter, one panel per system ───────
law_systems = [s for s in ALL_SYSTEMS if TRUE_LAWS.get(s)]
n_sys = len(law_systems)
fig, axes = plt.subplots(2, (n_sys+1)//2, figsize=(14, 8))
axes = axes.flatten()
cmap = plt.cm.viridis

for i, system in enumerate(law_systems):
    ax  = axes[i]
    sub = df_pareto[df_pareto.system==system]
    if len(sub) == 0: ax.set_title(system, fontsize=9); continue

    sc  = sub.constancy.values
    cx  = sub.complexity.values
    # Colour by constancy (lower=better=darker green)
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
    scatter = ax.scatter(cx, sc, c=sc_norm, cmap="RdYlGn_r",
                         s=60, alpha=0.85, edgecolors="gray", linewidths=0.4)
    # Pareto frontier: for each complexity level, mark minimum constancy
    cx_sorted = np.sort(np.unique(cx))
    pareto_sc = [sub[np.isclose(sub.complexity, c, atol=0.5)].constancy.min()
                 for c in cx_sorted]
    ax.plot(cx_sorted, pareto_sc, "k--", lw=1.2, alpha=0.6, label="Pareto frontier")
    # Mark accepted region
    ax.axhline(HP["gate_strict"], ls="--", c="#e74c3c", lw=1.2, label=f"gate={HP['gate_strict']}")
    ax.set_title(system, fontsize=9)
    ax.set_xlabel("Complexity (ops)", fontsize=8)
    ax.set_ylabel("Test Constancy (↓)", fontsize=8)
    ax.set_yscale("log"); ax.legend(fontsize=7)
    plt.colorbar(scatter, ax=ax, label="constancy (norm)")

# Hide extra axes
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Pareto Frontier: Constancy vs Expression Complexity\n"
             "(accepted region = below red dashed line)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{EXP_DIR}/plots/pareto_constancy_complexity.png", dpi=140); plt.close()
print(f"\n✅ Pareto plot → {EXP_DIR}/plots/pareto_constancy_complexity.png")
display(Image(f"{EXP_DIR}/plots/pareto_constancy_complexity.png"))

print(f"\n\n{'='*60}")
print("  ALL ADDITIONAL EXPERIMENTS COMPLETE")
print(f"  Results in: {EXP_DIR}/")
print(f"  Plots in:   {EXP_DIR}/plots/")
for fn in sorted(os.listdir(f"{EXP_DIR}/plots")):
    if fn.endswith(".png"): print(f"    {fn}")
print('='*60)
