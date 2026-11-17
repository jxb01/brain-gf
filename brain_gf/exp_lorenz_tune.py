"""Two questions after the partial-observation result:
   1. is the ESN just badly tuned?  sweep n_res / spectral radius
   2. does feeding LAGS into the op bank (memory + nonlinearity) beat the linear AR?
"""
import numpy as np
from brain_gf.nest_ops import OpBrain

F = 8.0; N_DIM = 10; KEEP = 3
def rhs(x): return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
def simulate(steps=60000, dt=0.01, every=5, seed=0):
    rng = np.random.default_rng(seed); x = F + 0.1*rng.standard_normal(N_DIM); out=[]
    for t in range(steps):
        k1=rhs(x); k2=rhs(x+dt/2*k1); k3=rhs(x+dt/2*k2); k4=rhs(x+dt*k3)
        x = x + dt/6*(k1+2*k2+2*k3+k4)
        if t % every == 0: out.append(x.copy())
    return np.array(out)

def r2(y,p): return 1.0-((y-p)**2).sum()/((y-y.mean(0))**2).sum()
def ridge(Fm,Y,lam=1e-3): return np.linalg.solve(Fm.T@Fm+lam*np.eye(Fm.shape[1]), Fm.T@Y)

class ESN:
    def __init__(s, n_in, n_res, spectral, seed=0):
        rng=np.random.default_rng(seed)
        s.Win=rng.uniform(-1,1,(n_in,n_res))
        W=rng.standard_normal((n_res,n_res)); W[rng.random((n_res,n_res))>0.1]=0.0
        s.W=W*(spectral/max(np.max(np.abs(np.linalg.eigvals(W))),1e-9)); s.n=n_res
    def run(s,Z):
        h=np.zeros(s.n); o=[]
        for z in Z:
            h=np.tanh(z@s.Win+s.W@h); o.append(np.concatenate([h,z]))
        return np.array(o)

def lag(Z,k):
    pad=np.repeat(Z[:1],k,0); Zp=np.vstack([pad,Z])
    return np.stack([Zp[k-j:k-j+len(Z)] for j in range(k+1)],1).reshape(len(Z),-1)

X=simulate()[2000:]; Z=X[:,:KEEP]; split=int(0.7*len(Z))

print("ESN sweep (partial obs, 5-step test R2)")
print(f"{'n_res':>6} {'spectral':>9} {'1-step':>8} {'5-step':>8}")
for nr in (300, 800):
    for sp in (0.5, 0.9, 1.2):
        Fm=ESN(KEEP,nr,sp).run(Z)
        out=[]
        for h in (1,5):
            Yt=Z[h:]; Ft=Fm[:len(Z)-h]
            W=ridge(Ft[:split],Yt[:split]); out.append(r2(Yt[split:],Ft[split:]@W))
        print(f"{nr:>6} {sp:>9} {out[0]:>+8.3f} {out[1]:>+8.3f}")

print()
print("op bank fed with lags (memory + nonlinearity), 5-step")
print(f"{'input':<18} {'1-step':>8} {'5-step':>8} {'dim':>6}")
for k in (0, 2, 5, 10):
    Zm = Z if k == 0 else lag(Z, k)
    b = OpBrain(1, Zm.shape[1], 8, 128, 16, KEEP, agg="concat", combine="concat",
                seed=0, static_mask=True)
    Fm = np.stack([b.forward(z)[1] for z in Zm])
    out=[]
    for h in (1,5):
        Yt=Z[h:]; Ft=Fm[:len(Z)-h]
        W=ridge(Ft[:split],Yt[:split]); out.append(r2(Yt[split:],Ft[split:]@W))
    tag = ("raw x_t" if k == 0 else f"lag k={k}")
    print(f"{tag:<18} {out[0]:>+8.3f} {out[1]:>+8.3f} {Fm.shape[1]:>6}")
print()
print("reference: linear AR lag k=20 = +1.000 / +0.787")
