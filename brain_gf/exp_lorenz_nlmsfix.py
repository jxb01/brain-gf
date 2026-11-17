"""Fix the online learner on real-scale data: per-feature running standardisation.

Ridge survives heterogeneous feature scales thanks to its lambda; NLMS has no
regulariser, so one feature at scale 400 blows the whole readout up.
"""
import numpy as np
from brain_gf.nest_ops import OpBrain

F = 8.0; N_DIM = 10
def rhs(x): return (np.roll(x,-1) - np.roll(x,2))*np.roll(x,1) - x + F
def sim(steps=60000, dt=0.01, every=5, seed=0):
    rng=np.random.default_rng(seed); x=F+0.1*rng.standard_normal(N_DIM); o=[]
    for t in range(steps):
        k1=rhs(x); k2=rhs(x+dt/2*k1); k3=rhs(x+dt/2*k2); k4=rhs(x+dt*k3)
        x=x+dt/6*(k1+2*k2+2*k3+k4)
        if t%every==0: o.append(x.copy())
    return np.array(o)
def r2(y,p): return 1.0-((y-p)**2).sum()/((y-y.mean(0))**2).sum()

X=sim()[2000:]
mu,sd=X[:int(0.7*len(X))].mean(0),X[:int(0.7*len(X))].std(0)
Z=(X-mu)/sd; split=int(0.7*len(Z))

b = OpBrain(1, N_DIM, 8, 128, 16, N_DIM, agg="concat", combine="concat",
             seed=0, static_mask=True)
W = np.zeros((b.W.shape[0], N_DIM))
m = np.zeros(b.W.shape[0]); v = np.ones(b.W.shape[0])
lr = 0.05
for i in range(split - 1):
    _pred, f = b.forward(Z[i])
    m = 0.99*m + 0.01*f
    v = 0.99*v + 0.01*(f - m)**2
    fn = np.clip((f - m)/np.sqrt(v + 1e-6), -10, 10)
    e = Z[i+1] - fn @ W
    W = W + lr*np.outer(fn, e)/(float(fn @ fn) + 1e-8)

preds=[]
for i in range(split, len(Z)-1):
    _p, f = b.forward(Z[i])
    fn = np.clip((f - m)/np.sqrt(v + 1e-6), -10, 10)
    preds.append(fn @ W)
print(f"NLMS + per-feature running standardisation (1 pass, lr={lr}):")
print(f"  1-step test R2 = {r2(Z[split+1:], np.array(preds)):+.3f}")
print(f"  reference: ridge on the same features = +0.996, ESN = +0.983, AR lag5 = +1.000")
