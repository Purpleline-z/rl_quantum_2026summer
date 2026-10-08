"""Does a label-free graph over ALL images (kNN in SimCLR space + temporal chain) help predict held-out decisive judgments?
Linear Bradley-Terry head on (smoothed) frozen features; pair-group-disjoint repeated 5-fold CV. Diagnostic only."""
import numpy as np, pandas as pd, scipy.sparse as sp, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]  # repository root
WT = str(ROOT)
REP=f"{WT}/code/active_learning_studies/image_representation_analysis/results/representation_exploration"
man=pd.read_csv(f"{REP}/manifest.csv"); man["key"]=man.path.str.replace(r"^.*/(STO_ideal_[^/]+/.*|Trajectories/.*)$",r"\1",regex=True)
idx={k:i for i,k in enumerate(man.key)}; n=len(man)
X=np.load(f"{REP}/cache/rheed_simclr_resnet18.npy").astype(np.float64); X=(X-X.mean(0))/ (X.std(0)+1e-8)
d=pd.read_csv(f"{WT}/data/original data/Quantum Label Data - Pairwise_Comparisonv1.8.csv")
d=d[d.Winner.astype(str).isin(["1","2"]) & ~d.Reconstruction_Type.isin(["Twinned(2 x 1)","Other"])].copy()
d["a"]=d.Image1_Path.map(idx).astype(int); d["b"]=d.Image2_Path.map(idx).astype(int); d["y"]=(d.Winner.astype(str)=="1").astype(int)
d["grp"]=d.a.astype(str)+"|"+d.b.astype(str); print("decisive rows",len(d),"groups",d.grp.nunique())
def adj(edges):
    r=[e[0] for e in edges]+[e[1] for e in edges]; c=[e[1] for e in edges]+[e[0] for e in edges]
    A=sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(n,n)); A.data[:]=1; return A
Z=X/np.linalg.norm(X,axis=1,keepdims=True); S=Z@Z.T; np.fill_diagonal(S,-9)
def knn_adj(k): nb=np.argsort(-S,1)[:,:k]; return adj([(i,j) for i in range(n) for j in nb[i]])
tr=man[man.source!="ideal"].copy(); tr["folder"]=tr.key.str.split("/").str[1]; tr["seq"]=tr.key.str.extract(r"/(\d+)_RR")[0].astype(int)
te=[]
for f,g in tr.groupby("folder"):
    ids=[idx[k] for k in g.sort_values("seq").key]; te+=list(zip(ids[:-1],ids[1:]))
A_t=adj(te)
def sgc(A,K,selfw=1.0):
    A=A+selfw*sp.eye(n); dg=np.asarray(A.sum(1)).ravel(); Dn=sp.diags(dg**-0.5); An=Dn@A@Dn; H=X.copy()
    for _ in range(K): H=An@H
    return H
def pca(H,m=32): return PCA(m,random_state=0).fit_transform(H)
feats={"raw (no graph)":pca(X)}
Ak=knn_adj(10)
for name,A in [("kNN10",Ak),("kNN10+temporal",((Ak+A_t)>0).astype(float))]:
    for K in (1,2,4): feats[f"SGC {name} K={K}"]=pca(sgc(A,K))
# spectral embedding of kNN10+temporal (label-free)
A=((Ak+A_t)>0).astype(float); dg=np.asarray(A.sum(1)).ravel(); L=sp.eye(n)-sp.diags(dg**-.5)@A@sp.diags(dg**-.5)
w,V=np.linalg.eigh(L.toarray()); feats["spectral(32) kNN10+temporal"]=V[:,1:33]
def cv(F,C,reps=20,per_type=True):
    aucs=[];lls=[]
    for r in range(reps):
        rng=np.random.default_rng(r); gl=d.grp.unique(); perm=dict(zip(gl,rng.permutation(len(gl))%5)); fold=d.grp.map(perm).values
        P=np.full(len(d),np.nan)
        for f in range(5):
            trn=np.where(fold!=f)[0]; tst=np.where(fold==f)[0]
            for t in d.Reconstruction_Type.unique():
                tt=[i for i in trn if d.Reconstruction_Type.iloc[i]==t]; ts=[i for i in tst if d.Reconstruction_Type.iloc[i]==t]
                if not ts: continue
                D=F[d.a.values[tt]]-F[d.b.values[tt]]; y=d.y.values[tt]
                m=LogisticRegression(C=C,fit_intercept=False,max_iter=2000).fit(np.vstack([D,-D]),np.r_[y,1-y])
                P[ts]=m.predict_proba(F[d.a.values[ts]]-F[d.b.values[ts]])[:,1]
        aucs.append(roc_auc_score(d.y,P)); lls.append(log_loss(d.y,np.clip(P,1e-6,1-1e-6)))
    return np.mean(aucs),np.std(aucs),np.mean(lls)
print(f"{'features':36s} {'C':>6s} {'AUC':>6s} {'sd':>5s} {'logloss':>8s}   (chance AUC .5, logloss .693)")
for nm,F in feats.items():
    F=F/F.std()
    for C in (0.001,0.01,0.1):
        a,s,l=cv(F,C); print(f"{nm:36s} {C:6.3f} {a:6.3f} {s:5.3f} {l:8.3f}",flush=True)
