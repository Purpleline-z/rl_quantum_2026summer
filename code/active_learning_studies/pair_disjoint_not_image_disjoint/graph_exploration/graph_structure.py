import sys, re, numpy as np, pandas as pd, scipy.sparse as sp, scipy.sparse.csgraph as cg
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]  # repository root
WT = str(ROOT)
REP=f"{WT}/code/active_learning_studies/image_representation_analysis/results/representation_exploration"
man=pd.read_csv(f"{REP}/manifest.csv")
man["key"]=man.path.str.replace(r"^.*/(STO_ideal_[^/]+/.*|Trajectories/.*)$",r"\1",regex=True)
X=np.load(f"{REP}/cache/rheed_simclr_resnet18.npy"); idx={k:i for i,k in enumerate(man.key)}
n=len(man); print("nodes",n,"ideal",(man.source=="ideal").sum(),"traj",(man.source!="ideal").sum())
d=pd.read_csv(f"{WT}/data/original data/Quantum Label Data - Pairwise_Comparisonv1.8.csv")
d=d[~d.Reconstruction_Type.isin(["Twinned(2 x 1)","Other"])]
d["a"]=d.Image1_Path.map(idx); d["b"]=d.Image2_Path.map(idx); assert d.a.notna().all() and d.b.notna().all()
pairs=sorted({tuple(sorted(p)) for p in zip(d.a.astype(int),d.b.astype(int))}); print("pair groups",len(pairs))
def adj(edges,n=n):
    r=[e[0] for e in edges]+[e[1] for e in edges]; c=[e[1] for e in edges]+[e[0] for e in edges]
    return sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(n,n))
def fiedler_cc(A, nodes=None):
    # algebraic connectivity of the largest component and component count
    k,l=cg.connected_components(A); big=np.bincount(l).argmax(); sub=np.where(l==big)[0]
    L=cg.laplacian(A[sub][:,sub].astype(float),normed=False).toarray(); w=np.linalg.eigvalsh(L)
    return k,len(sub),w[1]
# 1 comparison graph
A_cmp=adj(pairs); k,m,f=fiedler_cc(A_cmp)
touched=int((A_cmp.sum(1)>0).sum()); print(f"comparison graph: touched nodes {touched}, components among touched {k-(n-touched)}, largest {m}, fiedler(largest)={f:.3f}")
# 2 decisive per-type directed graph
dec=d[d.Winner.astype(str).isin(["1","2"])].copy()
dec["w"]=np.where(dec.Winner.astype(str)=="1",dec.a,dec.b).astype(int); dec["l"]=np.where(dec.Winner.astype(str)=="1",dec.b,dec.a).astype(int)
print("\nper-type decisive directed graph (loser->winner)")
for t,g in list(dec.groupby("Reconstruction_Type"))+[("ALL",dec)]:
    ed=list(zip(g.l,g.w)); nodes=set(g.l)|set(g.w); A=adj(ed); k,l=cg.connected_components(A)
    comp=Counter(l[list(nodes)]); both=len(set(g.l)&set(g.w))
    print(f"{t:14s} edges {len(g):3d} nodes {len(nodes):3d} comps {len(comp):3d} largest {max(comp.values())} nodes-with-both-win-and-loss {both}")
# 3 temporal adjacency
tr=man[man.source!="ideal"].copy(); tr["folder"]=tr.key.str.split("/").str[1]; tr["seq"]=tr.key.str.extract(r"/(\d+)_RR")[0].astype(int)
tedges=[]
for f,g in tr.groupby("folder"):
    g=g.sort_values("seq"); ids=[idx[k] for k in g.key]; tedges+=list(zip(ids[:-1],ids[1:]))
    print(f"folder {f}: {len(g)} frames, seq {g.seq.min()}..{g.seq.max()}")
A_t=adj(tedges)
# 4 kNN graph
def knn(X,k,metric="cos"):
    Z=X/np.linalg.norm(X,axis=1,keepdims=True); S=Z@Z.T; np.fill_diagonal(S,-9); nb=np.argsort(-S,1)[:,:k]
    return [(i,j) for i in range(len(X)) for j in nb[i]], S
for k in (5,10):
    ke,S=knn(X,k); A_k=adj(ke)
    # homophily on ideal nodes: fraction of an ideal node's kNN (among ideal nodes) w/ same type
    ideal=np.where(man.source=="ideal")[0]; lab=man.label.values
    Si=S[np.ix_(ideal,ideal)]; nb=np.argsort(-Si,1)[:,:k]; hom=np.mean([np.mean(lab[ideal][nb[r]]==lab[ideal][r]) for r in range(len(ideal))])
    # ideal neighbours of trajectory images
    print(f"\nkNN k={k}: kNN-only components {cg.connected_components(A_k)[0]}; ideal-ideal same-type neighbour rate {hom:.2f}")
    for name,A in [("kNN",A_k),("kNN+temporal",A_k+A_t),("kNN+temporal+comparison",A_k+A_t+A_cmp),("temporal+comparison",A_t+A_cmp),("kNN+comparison",A_k+A_cmp)]:
        kk,mm,ff=fiedler_cc((A>0).astype(float)); print(f"  {name:26s} components {kk:4d} largest {mm:4d} fiedler {ff:.4f}")
    # are labelled pairs near each other in feature space? rank of b among a's neighbours
    ranks=[ (S[a]>S[a,b]).sum() for a,b in pairs]; print(f"  labelled pair: median rank of partner among all 1277 images = {np.median(ranks):.0f}; fraction within top-{k} = {np.mean(np.array(ranks)<k):.2f}")
    rnd=np.random.default_rng(0).integers(0,n,(2000,2)); rr=[(S[a]>S[a,b]).sum() for a,b in rnd if a!=b]; print(f"  random pair: median rank {np.median(rr):.0f}")
