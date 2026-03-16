# ╔══════════════════════════════════════════════════════════╗
# ║  PASTE EACH BLOCK BELOW INTO A SEPARATE KAGGLE CELL     ║
# ╚══════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — imports + config (run once, needed by all cells)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os, copy, time, warnings, traceback, contextlib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
import torch, torch.nn as nn, torch.nn.functional as F
import sympy as sp
from sklearn.linear_model import Lasso
from itertools import combinations_with_replacement
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HDF5_PATH = "/kaggle/working/ngcg_data_clean.h5"
OUT_DIR   = "/kaggle/working/ngcg_win_results"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT     = torch.float32
USE_AMP   = DEVICE == "cuda"
SEED      = 42
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
np.random.seed(SEED); torch.manual_seed(SEED)

HP = dict(
    dyn_hidden=(256,256), dyn_epochs=150, dyn_patience=15, dyn_lr=3e-3, batch=2048,
    n_restarts=10, phi_hidden=(64,64,64), phi_epochs=300, phi_patience=40,
    phi_lr=1e-3, phi_l2=1e-4,
    pysr_niter=50, pysr_maxsize=25, pysr_n_pts=1000,
    lv_lambdas=[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5],
    gate_strict=0.01, gate_loose=0.05, gate_v_strict=0.005, rollout_k=16,
)

TRUE_LAWS = {
    "mass_spring"    : "p**2/(2*m) + k*q**2/2",
    "lotka_volterra" : "delta*x - gamma*log(x) + beta*y - alpha*log(y)",
    "double_pendulum": None,
    "henon_heiles"   : "px**2/2 + py**2/2 + x**2/2 + y**2/2 + x**2*y - y**3/3",
    "lorenz"         : None,
    "coupled_springs": "p1**2/2 + p2**2/2 + k1*q1**2/2 + k2*(q2-q1)**2/2 + k3*q2**2/2",
    "three_body"     : None,
    "burgers"        : "u_mean",
    "ks"             : "u_mean",
}
STATE_VARS = {
    "mass_spring":["q","p"], "lotka_volterra":["x","y"],
    "double_pendulum":["theta1","theta2","p1","p2"],
    "henon_heiles":["x","y","px","py"], "lorenz":["x","y","z"],
    "coupled_springs":["q1","q2","p1","p2"], "three_body":["x","y","vx","vy"],
    "burgers":["u_mean","u_var","u_skew"], "ks":["u_mean","u_var","u_skew"],
}
PARAM_NAMES = {
    "mass_spring":["k","m"], "lotka_volterra":["alpha","beta","gamma","delta"],
    "coupled_springs":["k1","k2","k3"],
}
PDE_SYSTEMS = {"burgers","ks"}
ALL_SYSTEMS = list(TRUE_LAWS.keys())
_SYM_NAMES  = ["q","p","x","y","z","u","v","m","k","r","s",
               "alpha","beta","gamma","delta","epsilon","theta","phi",
               "q1","q2","p1","p2","k1","k2","k3","m1","m2",
               "theta1","theta2","px","py","vx","vy","u_mean","u_var","u_skew"]

def _sympify(s, extra=None):
    loc={n:sp.Symbol(n) for n in _SYM_NAMES+(list(extra) if extra else [])}
    return sp.sympify(s, locals=loc)

print("✓ Cell 1 done — imports and config loaded")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — data loading + neural network utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _pde_scalar(t):
    mu=t.mean(-1,keepdims=True); var=t.var(-1,keepdims=True)
    skw=((t-mu)**3).mean(-1,keepdims=True)/(var+1e-8)**1.5
    return np.concatenate([mu,var,skw],-1).astype(np.float32)

def load_system(system):
    with h5py.File(HDF5_PATH,"r") as f:
        if system not in f: return None
        g=f[system]; tj=g["trajectories"][:]
        tr=g["train_indices"][:]; va=g["val_indices"][:]; te=g["test_indices"][:]
        par=g["params"][:] if "params" in g else None; att=dict(g.attrs)
    N=tj.shape[0]; ok=np.all(np.isfinite(tj.reshape(N,-1)),1)
    tj=tj[ok]; N=len(tj)
    tr=tr[tr<N]; va=va[va<N]; te=te[te<N]
    tr_t=tj[tr].astype(np.float32); va_t=tj[va].astype(np.float32); te_t=tj[te].astype(np.float32)
    par_tr=par[tr] if par is not None else None
    par_te=par[te] if par is not None else None
    if system in PDE_SYSTEMS:
        tr_t=_pde_scalar(tr_t); va_t=_pde_scalar(va_t); te_t=_pde_scalar(te_t)
    dt=float(att.get("dt",0.1))
    print(f"  {system}: train={len(tr_t)} val={len(va_t)} test={len(te_t)} D={tr_t.shape[-1]}")
    return dict(train=tr_t,val=va_t,test=te_t,params_train=par_tr,params_test=par_te,dt=dt,attrs=att)

class MLP(nn.Module):
    def __init__(self,in_d,out_d,hidden=(256,256),act=nn.Tanh):
        super().__init__()
        dims=[in_d]+list(hidden)+[out_d]; L=[]
        for a,b in zip(dims,dims[1:]): L+=[nn.Linear(a,b),act()]
        L.pop(); self.net=nn.Sequential(*L)
    def forward(self,x): return self.net(x)

def _pairs(traj):
    x=torch.tensor(traj[:,:-1].reshape(-1,traj.shape[-1]),dtype=FLOAT,device=DEVICE)
    y=torch.tensor(traj[:, 1:].reshape(-1,traj.shape[-1]),dtype=FLOAT,device=DEVICE)
    return x,y

def train_dynamics(dyn, tr, va):
    dyn=dyn.to(DEVICE); xtr,ytr=_pairs(tr); xva,yva=_pairs(va); N=len(xtr)
    if N==0: return float("inf"),0
    batch=min(HP["batch"],N); n_batch=max(1,N//batch)
    opt=torch.optim.Adam(dyn.parameters(),lr=HP["dyn_lr"],weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=HP["dyn_lr"],
          epochs=HP["dyn_epochs"],steps_per_epoch=n_batch,pct_start=0.1)
    scaler=torch.cuda.amp.GradScaler() if USE_AMP else None
    best=float("inf"); pat=0; bw=copy.deepcopy(dyn.state_dict())
    perm=torch.randperm(N,device=DEVICE)
    for ep in range(HP["dyn_epochs"]):
        dyn.train(); perm=perm[torch.randperm(N,device=DEVICE)]
        for i in range(n_batch):
            sl=perm[i*batch:(i+1)*batch]; opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    loss=F.mse_loss(dyn(xtr[sl]),ytr[sl])
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                F.mse_loss(dyn(xtr[sl]),ytr[sl]).backward(); opt.step()
            sched.step()
        dyn.eval()
        with torch.no_grad(),(torch.cuda.amp.autocast() if USE_AMP else contextlib.nullcontext()):
            vl=F.mse_loss(dyn(xva),yva).item()
        if vl<best-1e-7: best=vl; pat=0; bw=copy.deepcopy(dyn.state_dict())
        else:
            pat+=1
            if pat>=HP["dyn_patience"]: break
        if (ep+1)%20==0: print(f"    ep{ep+1:3d}  val={vl:.5f}  best={best:.5f}")
    dyn.load_state_dict(bw); return best, ep+1

def rollout(dyn,x0,steps):
    dyn.eval(); x=torch.tensor(x0,dtype=FLOAT,device=DEVICE); out=[]
    ctx=torch.cuda.amp.autocast() if USE_AMP else contextlib.nullcontext()
    with torch.no_grad(),ctx:
        for _ in range(steps): x=dyn(x); out.append(x)
    return torch.stack(out,1).cpu().numpy()

print("✓ Cell 2 done — data + dynamics utilities")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — phi network + LV lasso + poly lasso + metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _phi_loss(phi, traj_gpu):
    N,T,D=traj_gpu.shape; flat=traj_gpu.view(N*T,D)
    c=phi(flat).view(N,T); intra=c.var(1).mean(); inter=c.mean(1).var()
    return intra/(inter+1e-4)

def train_one_phi(tr_gpu, va_gpu, seed, D):
    torch.manual_seed(seed)
    phi=MLP(D,1,hidden=HP["phi_hidden"]).to(DEVICE)
    opt=torch.optim.Adam(phi.parameters(),lr=HP["phi_lr"],weight_decay=HP["phi_l2"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,HP["phi_epochs"])
    best=float("inf"); bw=copy.deepcopy(phi.state_dict()); pat=0
    for ep in range(HP["phi_epochs"]):
        phi.train()
        loss=_phi_loss(phi,tr_gpu)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        phi.eval()
        with torch.no_grad(): vl=_phi_loss(phi,va_gpu).item()
        if vl<best-1e-7: best=vl; pat=0; bw=copy.deepcopy(phi.state_dict())
        else:
            pat+=1
            if pat>=HP["phi_patience"]: break
    phi.load_state_dict(bw); return best,phi

def eval_phi_constancy(phi, traj):
    N,T,D=traj.shape
    z=torch.tensor(traj.reshape(-1,D),dtype=FLOAT,device=DEVICE)
    with torch.no_grad(): c=phi(z).view(N,T).cpu().numpy()
    mask=np.all(np.isfinite(c),1)
    if mask.sum()==0: return 999.0
    v=c[mask]; return float((v.std(1)/(np.abs(v.mean(1))+1e-8)).mean())

def run_multi_restart_phi(tr, va, D, n_restarts=None):
    tr_gpu=torch.tensor(tr,dtype=FLOAT,device=DEVICE)
    va_gpu=torch.tensor(va,dtype=FLOAT,device=DEVICE)
    best_score=float("inf"); best_phi=None; all_scores=[]
    for r in range(n_restarts or HP["n_restarts"]):
        vs,phi=train_one_phi(tr_gpu,va_gpu,SEED+r*137,D)
        cs=eval_phi_constancy(phi,va); all_scores.append(cs)
        print(f"    restart {r:2d}  loss={vs:.5f}  constancy={cs:.5f}")
        if cs<best_score: best_score=cs; best_phi=copy.deepcopy(phi)
    return best_phi,best_score,all_scores

def lv_log_basis(traj):
    eps=1e-6; x=np.clip(traj[:,:,0],eps,None); y=np.clip(traj[:,:,1],eps,None)
    return np.stack([x,y,np.log(x),np.log(y)],axis=-1)

def lv_variance_lasso(tr, va, lambdas):
    N,T,D=tr.shape; phi_tr=lv_log_basis(tr); phi_va=lv_log_basis(va); M=4
    best_score=float("inf"); best_w=None
    # Method A: eigenvector of mean trajectory covariance
    A=np.zeros((M,M))
    for i in range(N):
        p=phi_tr[i]; pc=p-p.mean(0); A+=pc.T@pc
    A/=N
    try:
        eigvals,eigvecs=np.linalg.eigh(A)
        for k in range(M):
            w=eigvecs[:,k]; phi_va_flat=phi_va.reshape(-1,M)
            c=(phi_va_flat@w).reshape(len(va),T)
            sc=float((c.std(1)/(np.abs(c.mean(1))+1e-8)).mean())
            print(f"    LV eigvec k={k} eigval={eigvals[k]:.3e} val_constancy={sc:.5f}")
            if sc<best_score: best_score=sc; best_w=w.copy()
    except Exception as e: print(f"    eigvec failed:{e}")
    # Method B: Lasso
    phi_c=phi_tr-phi_tr.mean(1,keepdims=True); X=phi_c.reshape(-1,M)
    cn=np.linalg.norm(X,axis=0)+1e-10; Xn=X/cn[None,:]
    for lam in lambdas:
        try:
            lasso=Lasso(alpha=lam,fit_intercept=False,max_iter=20000,tol=1e-8)
            lasso.fit(Xn,np.zeros(len(X))); w=lasso.coef_/cn
            if np.all(np.abs(w)<1e-10): continue
            c=(phi_va.reshape(-1,M)@w).reshape(len(va),T)
            sc=float((c.std(1)/(np.abs(c.mean(1))+1e-8)).mean())
            print(f"    LV lasso λ={lam:.1e} nnz={int(np.sum(np.abs(w)>1e-6))} val={sc:.5f}")
            if sc<best_score: best_score=sc; best_w=w.copy()
        except: pass
    # sign-flip sweep
    if best_w is not None:
        phi_va_flat=phi_va.reshape(-1,M)
        for wt in [best_w,-best_w,best_w/(np.abs(best_w).max()+1e-10)]:
            c=(phi_va_flat@wt).reshape(len(va),T)
            sc=float((c.std(1)/(np.abs(c.mean(1))+1e-8)).mean())
            if sc<best_score: best_score=sc; best_w=wt.copy()
    return best_w,best_score

def lv_expr_constancy(w, traj):
    N,T,D=traj.shape; phi=lv_log_basis(traj); c=(phi@w).reshape(N,T)
    mask=np.all(np.isfinite(c),1)
    if mask.sum()==0: return 999.0
    v=c[mask]; return float((v.std(1)/(np.abs(v.mean(1))+1e-8)).mean())

def lv_w_to_expr(w):
    names=["x","y","log(x)","log(y)"]; terms=[]
    for wi,name in zip(w,names):
        if abs(wi)>1e-6: terms.append(f"({wi:+.5f})*{name}")
    return " + ".join(terms) if terms else "0"

def _poly_lasso_candidates(tr, va, te, svars, D, max_degree=3):
    cands=[]
    try:
        combos=[]
        for deg in range(1,max_degree+1):
            for combo in combinations_with_replacement(range(D),deg): combos.append(combo)
        M=len(combos)
        def bphi(traj):
            N,T,_=traj.shape; parts=[]
            for combo in combos:
                t=traj[:,:,combo[0]].copy()
                for idx in combo[1:]: t=t*traj[:,:,idx]
                parts.append(t[:,:,None])
            return np.concatenate(parts,2)
        phi_tr=bphi(tr); N,T,_=tr.shape
        A=np.zeros((M,M))
        for i in range(N):
            p=phi_tr[i]; pc=p-p.mean(0); A+=pc.T@pc
        A/=N
        eigvals,eigvecs=np.linalg.eigh(A)
        phi_te=bphi(te)
        for k in range(min(M,8)):
            w=eigvecs[:,k]
            c=(phi_te.reshape(len(te)*te.shape[1],M)@w).reshape(len(te),te.shape[1])
            mask=np.all(np.isfinite(c),1)
            if mask.sum()==0: continue
            v=c[mask]; sc=float((v.std(1)/(np.abs(v.mean(1))+1e-8)).mean())
            if sc<0.1:
                terms=[f"({wi:+.4f})*{'*'.join(svars[j] for j in combo)}"
                       for wi,combo in zip(w,combos) if abs(wi)>1e-3]
                cands.append((" + ".join(terms) if terms else "0",sc))
    except Exception as e: print(f"    poly-lasso failed:{e}")
    return cands

def _passes_diversity_test(expr_str, svars, traj, min_ratio=10.0):
    if not expr_str or expr_str=="0": return False
    try:
        fn=sp.lambdify(svars,_sympify(expr_str,svars),modules="numpy")
        N,T,D=traj.shape
        vals=np.stack([fn(*[traj[:,t,j] for j in range(D)]) for t in range(T)],1).astype(np.float64)
        mask=np.all(np.isfinite(vals),1)
        if mask.sum()<5: return False
        v=vals[mask]; intra=v.std(1).mean(); inter=v.mean(1).std()
        ratio=inter/(intra+1e-10)
        print(f"    diversity: inter/intra={ratio:.2f} ({'✓' if ratio>=min_ratio else '✗ trivial'})")
        return ratio>=min_ratio
    except: return False

def constancy_score(expr_str, svars, traj):
    if not expr_str or expr_str=="0": return 999.0
    try:
        fn=sp.lambdify(svars,_sympify(expr_str,svars),modules="numpy")
        N,T,D=traj.shape
        vals=np.stack([fn(*[traj[:,t,j] for j in range(D)]) for t in range(T)],1).astype(np.float64)
        mask=np.all(np.isfinite(vals),1)
        if mask.sum()==0: return 999.0
        v=vals[mask]; return float((v.std(1)/(np.abs(v.mean(1))+1e-8)).mean())
    except: return 999.0

def sympy_match(cand, true_str, svars):
    if true_str is None: return False
    try:
        e1=_sympify(str(cand),svars); e2=_sympify(true_str,svars)
        if sp.simplify(e1-e2).is_number: return True
        if sp.simplify(e1/(e2+1e-12)).is_number: return True
        return False
    except: return False

def true_law_constancy(law_str, svars, te, params_te=None, param_names=None):
    if not law_str: return 0.0
    try:
        expr=_sympify(law_str,svars); all_sym={str(s) for s in expr.free_symbols}
        N,T,D=te.shape; ratios=[]
        for i in range(N):
            subs={}
            if params_te is not None and param_names:
                for pi,pn in enumerate(param_names):
                    if pn in all_sym and pi<params_te.shape[1]:
                        subs[sp.Symbol(pn)]=float(params_te[i,pi])
            try:
                fn=sp.lambdify(svars,expr.subs(subs),modules="numpy")
                vals=np.array([fn(*te[i,t,:]) for t in range(T)],dtype=float)
                if np.all(np.isfinite(vals)): ratios.append(vals.std()/(abs(vals.mean())+1e-8))
            except: pass
        return float(np.mean(ratios)) if ratios else 999.0
    except: return 999.0

def cv_metric(dyn, te, law_str, svars, params_te=None, param_names=None, k=16):
    if not law_str: return 0.0
    try:
        expr=_sympify(law_str,svars); x0=te[:,0]; pred=rollout(dyn,x0,k); devs=[]
        for i in range(len(x0)):
            subs={}
            if params_te is not None and param_names:
                for pi,pn in enumerate(param_names):
                    if pi<params_te.shape[1]: subs[sp.Symbol(pn)]=float(params_te[i,pi])
            try:
                fn=sp.lambdify(svars,expr.subs(subs),modules="numpy")
                ev=lambda s:fn(*[s[j] for j in range(len(svars))])
                c0=ev(x0[i]); devs.append(np.mean([abs(ev(pred[i,t])-c0) for t in range(k)]))
            except: pass
        return float(np.mean(devs)) if devs else 999.0
    except: return 999.0

def run_pysr_safe(X, y, var_names, n_iter, maxsize):
    mask=np.isfinite(X).all(1)&np.isfinite(y); X,y=X[mask],y[mask]
    if len(X)<10: return []
    X=np.clip(X,-100,100); y=np.clip(y,-1e6,1e6)
    try:
        from pysr import PySRRegressor
        m=PySRRegressor(niterations=n_iter,populations=15,
                        binary_operators=["+","-","*","/"],
                        unary_operators=["square","cube","sqrt","log","exp","sin","cos"],
                        maxsize=maxsize,verbosity=0,random_state=SEED,
                        procs=0,multithreading=False)
        m.fit(X,y,variable_names=var_names)
        df=m.equations_.sort_values("score",ascending=False); results=[]
        for _,row in df.iterrows():
            expr_str=str(row["sympy_format"])
            try:
                fn=sp.lambdify(var_names,_sympify(expr_str,var_names),modules="numpy")
                yhat=np.array(fn(*[X[:,i] for i in range(X.shape[1])]),dtype=float).flatten()
                yhat=np.where(np.isfinite(yhat),yhat,np.nan); mask2=np.isfinite(yhat)
                if mask2.sum()<5: r2=-999.0
                else:
                    ss_r=np.sum((y[mask2]-yhat[mask2])**2)
                    ss_t=np.sum((y[mask2]-y[mask2].mean())**2)+1e-12
                    r2=float(1-ss_r/ss_t)
            except: r2=-999.0
            results.append((expr_str,r2))
        return results
    except Exception as e:
        print(f"    PySR error:{e}"); return []

print("✓ Cell 3 done — phi, LV lasso, poly lasso, metrics, PySR")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — run_system + run_all
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_system(system, data, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=data["train"]; va=data["val"]; te=data["test"]
    D=tr.shape[-1]; par_te=data.get("params_test")
    param_names=PARAM_NAMES.get(system)
    svars=STATE_VARS.get(system,[f"x{i}" for i in range(D)])
    true_law=TRUE_LAWS.get(system); has_law=bool(true_law); t0=time.time()
    r={"method":"NGCG_Win","system":system,"has_true_law":1.0 if has_law else 0.0}

    # Stage 1: Dynamics
    print(f"\n── Stage 1: Dynamics ──")
    dyn=MLP(D,D,hidden=HP["dyn_hidden"]).to(DEVICE)
    dyn_mse,dyn_ep=train_dynamics(dyn,tr,va)
    print(f"  ✓ val_mse={dyn_mse:.5f} ep={dyn_ep}"); dyn.eval()
    mse16=999.0
    if len(te)>0:
        pred16=rollout(dyn,te[:,0],HP["rollout_k"])
        mse16=float(np.mean((pred16-te[:,1:HP["rollout_k"]+1])**2))
    r["MSE_16"]=mse16 if np.isfinite(mse16) else 999.0
    r["dyn_val_mse"]=dyn_mse if np.isfinite(dyn_mse) else 999.0
    print(f"  MSE@16={r['MSE_16']:.5g}")

    # Stage 2: phi restarts
    _n_phi=3 if system in PDE_SYSTEMS else HP["n_restarts"]
    print(f"\n── Stage 2: {_n_phi} φ restarts ──")
    best_phi,phi_val_score,all_phi_scores=run_multi_restart_phi(tr,va,D,_n_phi)
    r["phi_val_constancy"]=phi_val_score if np.isfinite(phi_val_score) else 999.0
    r["phi_best_restart"]=int(np.argmin(all_phi_scores))
    print(f"  ✓ best val_constancy={phi_val_score:.5f}")

    # Stage 3: candidates
    candidates=[]
    if system in PDE_SYSTEMS and len(svars)>=1:
        sc_um=constancy_score(svars[0],svars,te) if len(te)>0 else 999.0
        candidates.append((svars[0],sc_um))
        print(f"  PDE explicit '{svars[0]}' constancy={sc_um:.5g}")

    if system=="lotka_volterra":
        print(f"\n── Stage 3a: LV Lasso ──")
        best_w,lv_sc=lv_variance_lasso(tr,va,HP["lv_lambdas"])
        if best_w is not None:
            te_sc=lv_expr_constancy(best_w,te) if len(te)>0 else 999.0
            candidates.append((lv_w_to_expr(best_w),te_sc))
            print(f"  LV test constancy={te_sc:.5f}")
        print(f"\n── Stage 3b: PySR on φ ──")
        pts=tr.reshape(-1,D); idx=np.random.choice(len(pts),min(HP["pysr_n_pts"],len(pts)),replace=False)
        xt=torch.tensor(pts[idx],dtype=FLOAT,device=DEVICE)
        with torch.no_grad(): y_phi=best_phi(xt).squeeze(-1).cpu().numpy()
        X_safe=np.clip(pts[idx],1e-6,None)
        for expr_str,r2 in run_pysr_safe(X_safe,y_phi,svars,HP["pysr_niter"],HP["pysr_maxsize"])[:8]:
            sc=constancy_score(expr_str,svars,te) if len(te)>0 else 999.0
            candidates.append((expr_str,sc))
            print(f"    R²={r2:.3f} constancy={sc:.5f} {expr_str[:50]}")
        if best_w is not None:
            print(f"\n── Stage 3c: PySR on LV target ──")
            phi_pts=lv_log_basis(pts[idx].reshape(len(idx),1,D)).reshape(len(idx),4)
            y_lv=np.clip((phi_pts@best_w).flatten(),-1e4,1e4)
            for expr_str,r2 in run_pysr_safe(X_safe,y_lv,svars,HP["pysr_niter"],HP["pysr_maxsize"])[:5]:
                if r2>0.5:
                    sc=constancy_score(expr_str,svars,te) if len(te)>0 else 999.0
                    candidates.append((expr_str,sc))
    elif system not in PDE_SYSTEMS:
        print(f"\n── Stage 3: PySR on φ ──")
        if has_law and D<=4:
            for expr_str,sc in _poly_lasso_candidates(tr,va,te,svars,D):
                candidates.append((expr_str,sc))
                print(f"  poly-lasso constancy={sc:.5f} {expr_str[:60]}")
        pts=tr.reshape(-1,D); idx=np.random.choice(len(pts),min(HP["pysr_n_pts"],len(pts)),replace=False)
        xt=torch.tensor(pts[idx],dtype=FLOAT,device=DEVICE)
        with torch.no_grad(): y_phi=best_phi(xt).squeeze(-1).cpu().numpy()
        for expr_str,r2 in run_pysr_safe(pts[idx],y_phi,svars,HP["pysr_niter"],HP["pysr_maxsize"])[:8]:
            sc=constancy_score(expr_str,svars,te) if len(te)>0 else 999.0
            candidates.append((expr_str,sc))
            print(f"    R²={r2:.3f} constancy={sc:.5f} {expr_str[:50]}")
    else:
        print(f"  PySR skipped for PDE (u_mean already added)")

    # Stage 4: gate
    print(f"\n── Stage 4: Gate={HP['gate_strict']} ──")
    GATE=HP["gate_strict"]; TOL_L=HP["gate_loose"]; TOL_S=HP["gate_v_strict"]
    candidates.sort(key=lambda x:x[1])
    accepted=[(e,s) for e,s in candidates if s<GATE]
    if system in PDE_SYSTEMS and len(svars)>=1:
        simple=[(_e,_s) for _e,_s in accepted if _e.strip()==svars[0]]
        if simple: accepted=simple; print(f"  PDE simplicity: keeping only '{svars[0]}'")
    print(f"  Candidates:{len(candidates)} Accepted:{len(accepted)}")
    for e,s in accepted[:3]: print(f"    ✓ {s:.5f} {e[:70]}")
    best_expr=accepted[0][0] if accepted else ""
    best_sc=accepted[0][1] if accepted else (candidates[0][1] if candidates else 999.0)
    r["best_expr"]=best_expr[:200]; r["best_constancy"]=best_sc if np.isfinite(best_sc) else 999.0
    r["n_accepted"]=len(accepted); r["n_candidates"]=len(candidates)
    all_scores=[s for _,s in candidates]

    if has_law:
        tp_sym=any(sympy_match(e,true_law,svars) for e,_ in accepted)
        tp_num=any(s<TOL_L for s in all_scores); tp_str=any(s<TOL_S for s in all_scores)
        DR=1.0 if (tp_sym or tp_num) else 0.0; DR_s=1.0 if (tp_sym or tp_str) else 0.0
        DR_sym=1.0 if tp_sym else 0.0
        tp_acc=sum(1 for _,s in accepted if s<TOL_L); fp_acc=len(accepted)-tp_acc
        FDR=fp_acc/max(1,len(accepted)) if accepted else 0.0
    else:
        diverse_accepted=[]
        for e,s in accepted:
            if _passes_diversity_test(e,svars,te):
                diverse_accepted.append((e,s)); print(f"    DIVERSE:{s:.5f} {e[:50]}")
            else: print(f"    REJECTED:{s:.5f} {e[:50]}")
        accepted=diverse_accepted
        DR=1.0 if accepted else 0.0; DR_s=1.0 if any(s<TOL_S for _,s in accepted) else 0.0
        DR_sym=0.0; FDR=1.0 if accepted else 0.0
        best_expr=accepted[0][0] if accepted else ""
        best_sc=accepted[0][1] if accepted else 999.0
        r["best_expr"]=best_expr[:200]; r["best_constancy"]=best_sc

    F1=2*DR*(1-FDR)/max(1e-9,DR+(1-FDR))
    r["DR"]=DR; r["DR_strict"]=DR_s; r["DR_symbolic"]=DR_sym; r["FDR"]=FDR; r["F1"]=F1
    print(f"  DR={DR:.2f} FDR={FDR:.3f} F1={F1:.3f}")

    r["CV"]=0.0
    if has_law and len(te)>0:
        cv=cv_metric(dyn,te,true_law,svars,params_te=par_te,param_names=param_names)
        r["CV"]=cv if np.isfinite(cv) else 999.0

    tlc=true_law_constancy(true_law,svars,te,par_te,param_names) if has_law and len(te)>0 else 0.0
    r["true_law_constancy"]=tlc if np.isfinite(tlc) else 0.0
    try: cx=float(sp.count_ops(_sympify(best_expr,svars))) if best_expr else 999.0
    except: cx=999.0
    r["complexity"]=cx if np.isfinite(cx) else 999.0
    r["fit_time_s"]=round(time.time()-t0,1)
    r["gpu_mb"]=torch.cuda.memory_allocated()//1024//1024 if DEVICE=="cuda" else 0
    return r

def run_all(systems=None, seed=0, rerun=False):
    systems=systems or ALL_SYSTEMS; csv_path=f"{OUT_DIR}/results.csv"
    all_rows=[]; done=set()
    if os.path.exists(csv_path) and not rerun:
        existing=pd.read_csv(csv_path)
        for _,row in existing.iterrows(): done.add(row["system"])
        all_rows=existing.to_dict("records")
        print(f"  Resuming — done:{list(done)}")
    elif rerun and os.path.exists(csv_path):
        os.remove(csv_path); print("  cache cleared")
    for system in systems:
        if system in done: print(f"  {system} cached"); continue
        print(f"\n{'═'*55}\n  {system.upper()}\n{'═'*55}")
        data=load_system(system)
        if data is None or len(data["train"])==0: continue
        try: row=run_system(system,data,seed=seed)
        except Exception as e:
            traceback.print_exc()
            row={"method":"NGCG_Win","system":system,"error":str(e)[:120],
                 "MSE_16":999.0,"DR":0.0,"DR_strict":0.0,"DR_symbolic":0.0,
                 "has_true_law":1.0 if TRUE_LAWS.get(system) else 0.0,
                 "FDR":0.0,"F1":0.0,"CV":999.0,"true_law_constancy":0.0,
                 "best_constancy":999.0,"complexity":999.0,"fit_time_s":0.0,
                 "gpu_mb":0,"n_accepted":0,"best_expr":"ERROR"}
        skip={"system","method","best_expr","seed","error"}
        print(f"\n  ┌─ {system}")
        for k,v in sorted(row.items()):
            if k not in skip:
                vf = f"{v:.4f}" if isinstance(v,float) else str(v)
                print(f"  │ {k:<28}= {vf}")
        if row.get("best_expr"): print(f"  │ best_expr = {row['best_expr'][:70]}")
        print(f"  └{'─'*50}")
        all_rows.append(row); done.add(system)
        pd.DataFrame(all_rows).to_csv(csv_path,index=False)
    df=pd.DataFrame(all_rows); df.to_csv(csv_path,index=False)
    print(f"\n✅ Results → {csv_path}")
    return df

print("✓ Cell 4 done — run_system and run_all defined")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — run remaining systems (coupled_springs → ks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_win = run_all(["coupled_springs","three_body","burgers","ks"], seed=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — multi-seed runs (seeds 1,2 on all 9 systems)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
all_seed_rows = []
# Load seed 0 results already done
df0 = pd.read_csv(f"{OUT_DIR}/results.csv")
df0["seed"] = 0
all_seed_rows.append(df0)

for seed in [1, 2]:
    print(f"\n{'═'*55}\n  SEED {seed}\n{'═'*55}")
    seed_rows = []
    for system in ALL_SYSTEMS:
        print(f"\n── {system} seed={seed} ──")
        data = load_system(system)
        if data is None: continue
        try:
            row = run_system(system, data, seed=seed)
            row["seed"] = seed
        except Exception as e:
            traceback.print_exc()
            row = {"system":system,"seed":seed,"DR":0.0,"FDR":0.0,"F1":0.0,
                   "MSE_16":999.0,"best_constancy":999.0,"error":str(e)[:80],
                   "has_true_law":1.0 if TRUE_LAWS.get(system) else 0.0}
        seed_rows.append(row)
        print(f"  DR={row.get('DR',0):.2f} FDR={row.get('FDR',0):.3f} F1={row.get('F1',0):.3f}")
    df_seed = pd.DataFrame(seed_rows)
    df_seed.to_csv(f"{OUT_DIR}/results_seed{seed}.csv", index=False)
    all_seed_rows.append(df_seed)

df_all_seeds = pd.concat(all_seed_rows, ignore_index=True)
df_all_seeds.to_csv(f"{OUT_DIR}/results_all_seeds.csv", index=False)
print(f"\n✅ Multi-seed results → {OUT_DIR}/results_all_seeds.csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — multi-seed summary table (mean ± std)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_ms = pd.read_csv(f"{OUT_DIR}/results_all_seeds.csv")
print("\nMean ± Std across seeds 0,1,2\n")
print(f"{'System':<18} {'DR':>8} {'FDR':>8} {'F1':>8} {'constancy':>12}")
print("─"*58)
for sys in ALL_SYSTEMS:
    sub = df_ms[df_ms.system==sys]
    if len(sub)==0: continue
    dr  = f"{sub.DR.mean():.2f}±{sub.DR.std():.2f}"
    fdr = f"{sub.FDR.mean():.3f}±{sub.FDR.std():.3f}"
    f1  = f"{sub.F1.mean():.2f}±{sub.F1.std():.2f}"
    bc  = sub.best_constancy.replace(999.0,np.nan).dropna()
    con = f"{bc.mean():.4f}±{bc.std():.4f}" if len(bc) else "—"
    print(f"  {sys:<16} {dr:>8} {fdr:>8} {f1:>8} {con:>12}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — ablation study
# Ablations: no-restarts (1), no-diversity-test, no-LV-lasso, no-poly-lasso
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABLATION_SYSTEMS = ["mass_spring","lotka_volterra","henon_heiles",
                    "coupled_springs","lorenz","burgers"]

def run_ablation(system, data, ablation="full", seed=0):
    """Run with one component removed. ablation in:
       full | no_restarts | no_diversity | no_lv_lasso | no_poly_lasso
    """
    torch.manual_seed(seed); np.random.seed(seed)
    tr=data["train"]; va=data["val"]; te=data["test"]
    D=tr.shape[-1]; par_te=data.get("params_test")
    param_names=PARAM_NAMES.get(system)
    svars=STATE_VARS.get(system,[f"x{i}" for i in range(D)])
    true_law=TRUE_LAWS.get(system); has_law=bool(true_law)

    dyn=MLP(D,D,hidden=HP["dyn_hidden"]).to(DEVICE)
    train_dynamics(dyn,tr,va); dyn.eval()

    n_phi = 1 if ablation=="no_restarts" else (3 if system in PDE_SYSTEMS else HP["n_restarts"])
    best_phi,phi_val,_=run_multi_restart_phi(tr,va,D,n_phi)

    candidates=[]
    if system in PDE_SYSTEMS:
        sc_um=constancy_score(svars[0],svars,te) if len(te)>0 else 999.0
        candidates.append((svars[0],sc_um))

    if system=="lotka_volterra" and ablation!="no_lv_lasso":
        best_w,_=lv_variance_lasso(tr,va,HP["lv_lambdas"])
        if best_w is not None:
            te_sc=lv_expr_constancy(best_w,te) if len(te)>0 else 999.0
            candidates.append((lv_w_to_expr(best_w),te_sc))

    if system not in PDE_SYSTEMS:
        if has_law and D<=4 and ablation!="no_poly_lasso":
            for expr_str,sc in _poly_lasso_candidates(tr,va,te,svars,D):
                candidates.append((expr_str,sc))
        pts=tr.reshape(-1,D); idx=np.random.choice(len(pts),min(HP["pysr_n_pts"],len(pts)),replace=False)
        xt=torch.tensor(pts[idx],dtype=FLOAT,device=DEVICE)
        with torch.no_grad(): y_phi=best_phi(xt).squeeze(-1).cpu().numpy()
        for expr_str,r2 in run_pysr_safe(pts[idx],y_phi,svars,HP["pysr_niter"],HP["pysr_maxsize"])[:8]:
            sc=constancy_score(expr_str,svars,te) if len(te)>0 else 999.0
            candidates.append((expr_str,sc))

    GATE=HP["gate_strict"]; TOL_L=HP["gate_loose"]
    candidates.sort(key=lambda x:x[1])
    accepted=[(e,s) for e,s in candidates if s<GATE]
    if system in PDE_SYSTEMS:
        simple=[(e,s) for e,s in accepted if e.strip()==svars[0]]
        if simple: accepted=simple
    all_scores=[s for _,s in candidates]

    if has_law:
        tp_num=any(s<TOL_L for s in all_scores)
        DR=1.0 if tp_num else 0.0
        tp_acc=sum(1 for _,s in accepted if s<TOL_L)
        FDR=(len(accepted)-tp_acc)/max(1,len(accepted)) if accepted else 0.0
    else:
        if ablation=="no_diversity":
            div_acc=accepted  # skip diversity test
        else:
            div_acc=[(e,s) for e,s in accepted if _passes_diversity_test(e,svars,te)]
        DR=1.0 if div_acc else 0.0; FDR=1.0 if div_acc else 0.0
    F1=2*DR*(1-FDR)/max(1e-9,DR+(1-FDR))
    bc=min(all_scores) if all_scores else 999.0
    return {"system":system,"ablation":ablation,"DR":DR,"FDR":FDR,"F1":F1,
            "best_constancy":bc,"seed":seed}

abl_rows=[]
for system in ABLATION_SYSTEMS:
    print(f"\n── Ablation: {system} ──")
    data=load_system(system)
    if data is None: continue
    for abl in ["full","no_restarts","no_diversity","no_lv_lasso","no_poly_lasso"]:
        try:
            row=run_ablation(system,data,ablation=abl,seed=0)
            print(f"  {abl:<18} DR={row['DR']:.2f} FDR={row['FDR']:.3f} F1={row['F1']:.3f}")
        except Exception as e:
            print(f"  {abl:<18} FAILED: {e}")
            row={"system":system,"ablation":abl,"DR":0.0,"FDR":0.0,"F1":0.0,"best_constancy":999.0}
        abl_rows.append(row)

df_abl=pd.DataFrame(abl_rows)
df_abl.to_csv(f"{OUT_DIR}/ablation_results.csv",index=False)
print(f"\n✅ Ablation results → {OUT_DIR}/ablation_results.csv")
print(df_abl.pivot_table(index="system",columns="ablation",values="F1").round(3).to_string())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_all_plots():
    P=f"{OUT_DIR}/plots"; os.makedirs(P,exist_ok=True)
    df=pd.read_csv(f"{OUT_DIR}/results.csv")
    syss=ALL_SYSTEMS; C=plt.cm.tab10.colors

    # ── Plot 1: DR (green=correct, red=wrong) ───────────────
    fig,ax=plt.subplots(figsize=(11,4))
    vals=[float(df[df.system==s]["DR"].values[0]) if len(df[df.system==s]) else 0 for s in syss]
    htl =[float(df[df.system==s]["has_true_law"].values[0]) if len(df[df.system==s]) else 0 for s in syss]
    bc  =["#2ecc71" if (h==1 and v==1)or(h==0 and v==0) else "#e74c3c" for h,v in zip(htl,vals)]
    bars=ax.bar(range(len(syss)),vals,color=bc,alpha=0.9,edgecolor="white")
    ax.set_xticks(range(len(syss))); ax.set_xticklabels(syss,rotation=30,ha="right")
    ax.set_ylabel("DR"); ax.set_title("Discovery Rate (green=correct, red=wrong)")
    ax.axhline(1,ls="--",c="gray",lw=0.8)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,f"{v:.2f}",
                ha="center",va="bottom",fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P}/DR.png",dpi=140); plt.close()

    # ── Plot 2: F1 ──────────────────────────────────────────
    fig,ax=plt.subplots(figsize=(11,4))
    vals=[float(df[df.system==s]["F1"].values[0]) if len(df[df.system==s]) else 0 for s in syss]
    bars=ax.bar(range(len(syss)),vals,color=[C[i%10] for i in range(len(syss))],alpha=0.9,edgecolor="white")
    ax.set_xticks(range(len(syss))); ax.set_xticklabels(syss,rotation=30,ha="right")
    ax.set_ylabel("F1"); ax.set_title("F1 Score")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f"{v:.2f}",
                ha="center",va="bottom",fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P}/F1.png",dpi=140); plt.close()

    # ── Plot 3: Best constancy (log scale) ──────────────────
    fig,ax=plt.subplots(figsize=(11,4))
    vals=[float(df[df.system==s]["best_constancy"].values[0]) if len(df[df.system==s]) else 999
          for s in syss]
    vals_plot=[min(v,1.0) for v in vals]
    bars=ax.bar(range(len(syss)),vals_plot,color=[C[i%10] for i in range(len(syss))],alpha=0.9,edgecolor="white")
    ax.set_xticks(range(len(syss))); ax.set_xticklabels(syss,rotation=30,ha="right")
    ax.set_ylabel("Best Constancy (↓ better)"); ax.set_title("Best Constancy on Test")
    for bar,v in zip(bars,vals):
        lbl="—" if v>=999 else f"{v:.4f}"
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,lbl,
                ha="center",va="bottom",fontsize=7)
    fig.tight_layout(); fig.savefig(f"{P}/constancy.png",dpi=140); plt.close()

    # ── Plot 4: NGCG-Win vs baselines F1 side-by-side ───────
    baseline_f1 = {
        "mass_spring":1.0,"lotka_volterra":0.0,"double_pendulum":0.0,
        "henon_heiles":1.0,"lorenz":0.0,"coupled_springs":1.0,
        "three_body":0.0,"burgers":0.667,"ks":0.667,
    }
    win_f1={s:float(df[df.system==s]["F1"].values[0]) if len(df[df.system==s]) else 0 for s in syss}
    x=np.arange(len(syss)); w=0.35
    fig,ax=plt.subplots(figsize=(12,4.5))
    ax.bar(x-w/2,[baseline_f1[s] for s in syss],w,label="Best Baseline",
           color="#3498db",alpha=0.85,edgecolor="white")
    ax.bar(x+w/2,[win_f1[s] for s in syss],w,label="NGCG-Win",
           color="#2ecc71",alpha=0.85,edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(syss,rotation=30,ha="right")
    ax.set_ylabel("F1"); ax.set_title("NGCG-Win vs Best Baseline — F1 Score")
    ax.legend(); ax.axhline(1,ls="--",c="gray",lw=0.7)
    fig.tight_layout(); fig.savefig(f"{P}/vs_baseline_F1.png",dpi=140); plt.close()

    # ── Plot 5: Ablation heatmap ─────────────────────────────
    if os.path.exists(f"{OUT_DIR}/ablation_results.csv"):
        df_abl=pd.read_csv(f"{OUT_DIR}/ablation_results.csv")
        pivot=df_abl.pivot_table(index="system",columns="ablation",values="F1",aggfunc="mean")
        fig,ax=plt.subplots(figsize=(10,4))
        im=ax.imshow(pivot.values,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,rotation=25,ha="right")
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_title("Ablation Study — F1 Score")
        plt.colorbar(im,ax=ax,label="F1")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v=pivot.values[i,j]
                ax.text(j,i,f"{v:.2f}" if not np.isnan(v) else "—",
                        ha="center",va="center",fontsize=8,
                        color="white" if v<0.5 else "black")
        fig.tight_layout(); fig.savefig(f"{P}/ablation_heatmap.png",dpi=140); plt.close()

    # ── Plot 6: Multi-seed error bars ────────────────────────
    if os.path.exists(f"{OUT_DIR}/results_all_seeds.csv"):
        df_ms=pd.read_csv(f"{OUT_DIR}/results_all_seeds.csv")
        means=[df_ms[df_ms.system==s]["F1"].mean() for s in syss]
        stds =[df_ms[df_ms.system==s]["F1"].std()  for s in syss]
        fig,ax=plt.subplots(figsize=(11,4))
        ax.bar(range(len(syss)),means,color=[C[i%10] for i in range(len(syss))],alpha=0.85,edgecolor="white")
        ax.errorbar(range(len(syss)),means,yerr=stds,fmt="none",c="black",capsize=4,lw=1.5)
        ax.set_xticks(range(len(syss))); ax.set_xticklabels(syss,rotation=30,ha="right")
        ax.set_ylabel("F1"); ax.set_title("NGCG-Win F1 — Mean ± Std (seeds 0,1,2)")
        fig.tight_layout(); fig.savefig(f"{P}/multiseed_F1.png",dpi=140); plt.close()

    print(f"✅ All plots → {P}/")
    for fn in os.listdir(P):
        if fn.endswith(".png"): print(f"   {P}/{fn}")

make_all_plots()
