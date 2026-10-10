import numpy as np, pandas as pd
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]  # repository root
WT = str(ROOT)
REP=f"{WT}/code/active_learning_studies/image_representation_analysis/results/representation_exploration"
man=pd.read_csv(f"{REP}/manifest.csv"); X=np.load(f"{REP}/cache/rheed_simclr_resnet18.npy")
Z=X/np.linalg.norm(X,axis=1,keepdims=True); ideal=np.where(man.source=="ideal")[0]; lab=man.label.values
S=Z[ideal]@Z[ideal].T; np.fill_diagonal(S,-9); nb=np.argsort(-S,1)[:,:5]
c=Counter()
for r,i in enumerate(ideal):
    for j in nb[r]:
        a,b=lab[i],lab[ideal[j]]
        if a!=b: c[tuple(sorted((a,b)))]+=1
print("cross-type 5-NN edges among ideal images:",dict(c.most_common()), "of", len(ideal)*5)
print(man[man.source=="ideal"].label.value_counts().to_dict())
