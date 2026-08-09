#!/usr/bin/env python3
"""Label-free PCA/t-SNE comparison of raw, ImageNet, and RHEED-SimCLR features."""
from __future__ import annotations
import argparse, inspect, json, random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from project_paths import CODE_ROOT, RESULT_ROOT, resolve_data_root

HERE = Path(__file__).resolve().parent; DATA = resolve_data_root() / "original data"
OUT = RESULT_ROOT / "pca_tsne_dataset_exploration"; SEED = 42
IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
IDEAL_FOLDERS = {"(1 x 1)": "STO_ideal_1x1", "Twinned(2 x 1)": "STO_ideal_Twinned2x1", "c(6 x 2)": "STO_ideal_c6x2", "RT13": "STO_ideal_RT13", "HTR": "STO_ideal_HTR"}
COLORS = {"(1 x 1)": "#1f77b4", "Twinned(2 x 1)": "#ff7f0e", "c(6 x 2)": "#2ca02c", "RT13": "#d62728", "HTR": "#9467bd"}

def seed_everything(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def build_manifest(data_root: Path = DATA) -> pd.DataFrame:
    """One stable, label-safe order: ideals followed by unlabeled trajectories."""
    rows = []
    for label, folder in IDEAL_FOLDERS.items():
        rows += [{"path": str(p.resolve()), "source": "ideal", "label": label, "is_labeled": True} for p in image_files(data_root / folder)]
    rows += [{"path": str(p.resolve()), "source": "trajectory", "label": "", "is_labeled": False} for p in image_files(data_root / "Trajectories")]
    frame = pd.DataFrame(rows, columns=["path", "source", "label", "is_labeled"])
    if frame.empty: raise FileNotFoundError(f"No supported images under {data_root}")
    missing = set(IDEAL_FOLDERS) - set(frame.loc[frame.is_labeled, "label"])
    if missing: raise ValueError(f"Missing labelled ideal folders/images for: {sorted(missing)}")
    return frame

def raw_pixels(path: str) -> np.ndarray:
    with Image.open(path) as image: return np.asarray(image.convert("L").resize((224, 224)), dtype=np.float32).reshape(-1) / 255.0

class ImagePathDataset(Dataset):
    def __init__(self, paths: Iterable[str], transform: Callable): self.paths, self.transform = list(paths), transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image: return self.transform(image.convert("L"))

def simclr_transform(): return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Lambda(lambda x: x.repeat(3, 1, 1)), T.Normalize([.5]*3, [.25]*3)])
def grayscale_to_rgb(image: Image.Image): return image.convert("L").convert("RGB")
class Encoder(nn.Sequential):
    def forward(self, x): return super().forward(x).flatten(1)

@dataclass
class EncoderProvenance:
    name: str; transform: str; checkpoint: str | None = None; checkpoint_keys: str | None = None; loaded_tensors: int | None = None; expected_tensors: int | None = None

def freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def make_simclr_encoder(checkpoint: Path = CODE_ROOT / "classifier2" / "simclr_resnet18_encoder.pth"):
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    encoder = Encoder(*list(models.resnet18(weights=None).children())[:-1]); state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    if not isinstance(state, dict): raise ValueError("SimCLR checkpoint is not a state dictionary")
    cleaned = {k.replace("encoder.", ""): v for k, v in state.items() if not k.startswith("projector.")}; target = encoder.state_dict()
    matched = {k: v for k, v in cleaned.items() if k in target and target[k].shape == v.shape}; encoder.load_state_dict(cleaned, strict=False)
    if not matched: raise ValueError("No compatible encoder tensors were loaded from SimCLR checkpoint")
    return freeze(encoder), EncoderProvenance("rheed_simclr_resnet18", "grayscale -> RGB repeat; resize 224; Normalize(mean=.5, std=.25)", str(checkpoint), "encoder. prefix removed; projector. keys excluded", len(matched), len(target))

def make_imagenet_encoder():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    try: model = models.resnet18(weights=weights)
    except Exception as exc: raise RuntimeError(f"ImageNet weights unavailable ({exc}); no random-weight substitute was used.") from exc
    return freeze(Encoder(*list(model.children())[:-1])), lambda image: weights.transforms()(grayscale_to_rgb(image)), EncoderProvenance("imagenet_resnet18", "grayscale -> RGB repeat; official ResNet18_Weights.IMAGENET1K_V1 transforms", "torchvision::ResNet18_Weights.IMAGENET1K_V1")

def extract_encoder_embeddings(paths, model, transform, device, batch_size):
    loader = DataLoader(ImagePathDataset(paths, transform), batch_size=batch_size, shuffle=False, num_workers=0); values = []; model = model.to(device).eval()
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("Encoder must be frozen during extraction")
    with torch.no_grad():
        for batch in loader: values.append(model(batch.to(device)).cpu().numpy())
    result = np.concatenate(values)
    if result.ndim != 2 or result.shape[1] != 512 or not np.isfinite(result).all(): raise ValueError("Encoder extraction did not produce finite [N, 512] embeddings")
    return result

def cache_path(name): return OUT / "cache" / f"{name}.npy"
def extract_representation(name, manifest, device, batch_size, reuse_cache):
    path = cache_path(name)
    if reuse_cache and path.exists(): return np.load(path), {"representation": name, "cache": str(path), "reused": True}
    paths = manifest.path.tolist()
    if name == "raw_pixels": features = np.stack([raw_pixels(p) for p in paths]); provenance = {"representation": name, "feature_dimension": 224*224, "transform": "grayscale resize 224; flatten; divide by 255"}
    elif name == "rheed_simclr_resnet18":
        model, info = make_simclr_encoder(); features = extract_encoder_embeddings(paths, model, simclr_transform(), device, batch_size); provenance = asdict(info)
    elif name == "imagenet_resnet18":
        try: model, transform, info = make_imagenet_encoder(); features = extract_encoder_embeddings(paths, model, transform, device, batch_size); provenance = asdict(info)
        except RuntimeError as exc:
            status = OUT / "metrics" / "imagenet_resnet18_skipped_status.txt"; status.parent.mkdir(parents=True, exist_ok=True); status.write_text(str(exc) + "\n"); return None, {"representation": name, "status": "skipped", "reason": str(exc)}
    else: raise ValueError(f"Unknown representation: {name}")
    if not np.isfinite(features).all(): raise ValueError(f"{name} has non-finite features; refusing to continue")
    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, features); provenance.update({"cache": str(path), "reused": False, "shape": list(features.shape)}); return features, provenance

def tsne_projection(values):
    dims = min(50, values.shape[0]-1, values.shape[1]); reduced = PCA(n_components=dims, random_state=SEED).fit_transform(values); perplexity = min(30, max(5, (values.shape[0]-1)//3))
    kwargs = dict(n_components=2, random_state=SEED, init="pca", learning_rate="auto", perplexity=perplexity); kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1500
    return TSNE(**kwargs).fit_transform(reduced)

def separation_metrics(standardized, pca_xy, manifest, learned):
    labelled = manifest.is_labeled.to_numpy(bool); x, xy, y = standardized[labelled], pca_xy[labelled], manifest.loc[labelled, "label"].to_numpy(); classes, counts = np.unique(y, return_counts=True)
    metrics = {"n_ideal": len(y), "per_class_count": {str(k): int(v) for k, v in zip(classes, counts)}}; folds_n = min(5, int(counts.min()))
    if folds_n >= 2:
        folds = StratifiedKFold(n_splits=folds_n, shuffle=True, random_state=SEED); min_train = min(len(train) for train, _ in folds.split(x, y)); k = min(5, min_train)
        metrics.update(knn_cv_accuracy=float(cross_val_score(KNeighborsClassifier(n_neighbors=k), x, y, cv=folds).mean()), knn_k=k, cv_folds=folds_n)
    if len(classes) < len(y): metrics.update(silhouette_full=float(silhouette_score(x,y)), silhouette_pca_2d=float(silhouette_score(xy,y)), calinski_harabasz_full=float(calinski_harabasz_score(x,y)), calinski_harabasz_pca_2d=float(calinski_harabasz_score(xy,y)))
    centroids = {c: x[y==c].mean(0) for c in classes}; matrix = pd.DataFrame([[float(np.linalg.norm(centroids[a]-centroids[b])) for b in classes] for a in classes], index=classes, columns=classes)
    metrics["mean_within_class_distance"] = float(np.mean([np.linalg.norm(row-centroids[label]) for row,label in zip(x,y)])); metrics["mean_between_centroid_distance"] = float(np.mean([np.linalg.norm(centroids[a]-centroids[b]) for i,a in enumerate(classes) for b in classes[i+1:]]))
    if learned:
        unit, cunit = normalize(x), {c: normalize(centroids[c][None])[0] for c in classes}; metrics["mean_within_class_cosine_distance"] = float(np.mean([1-np.dot(row,cunit[label]) for row,label in zip(unit,y)])); metrics["mean_between_centroid_cosine_distance"] = float(np.mean([1-np.dot(cunit[a],cunit[b]) for i,a in enumerate(classes) for b in classes[i+1:]]))
    return metrics, matrix

def save_coordinates(name, manifest, pca_xy, tsne_xy):
    directory = OUT / "coordinates"; directory.mkdir(parents=True, exist_ok=True)
    for method, xy in (("pca", pca_xy), ("tsne", tsne_xy)):
        table = manifest.copy(); table["x"], table["y"] = xy[:,0], xy[:,1]; table.to_csv(directory / f"{name}_{method}.csv", index=False)
def plot(name, method, xy, manifest, subtitle):
    fig, ax = plt.subplots(figsize=(7,5.5)); unlabelled = ~manifest.is_labeled.to_numpy(bool); ax.scatter(xy[unlabelled,0],xy[unlabelled,1],s=5,c="lightgray",alpha=.45,label="Unlabeled trajectories",zorder=1)
    for label in IDEAL_FOLDERS:
        mask=manifest.label.eq(label).to_numpy(); ax.scatter(xy[mask,0],xy[mask,1],s=28,c=COLORS[label],edgecolors="white",linewidths=.35,label=f"{label} (n={mask.sum()})",zorder=2)
    ax.set(title=f"{name}: {method.upper()}\n{subtitle}",xlabel=f"{method.upper()} 1",ylabel=f"{method.upper()} 2"); ax.set_aspect("equal",adjustable="datalim"); ax.legend(fontsize=8); fig.tight_layout(); output=OUT/"figures"; output.mkdir(parents=True,exist_ok=True); fig.savefig(output/f"{name}_{method}.png",dpi=180); fig.savefig(output/f"{name}_{method}.pdf"); plt.close(fig)
def write_report(metadata, metric_rows):
    directory=OUT/"metrics"; directory.mkdir(parents=True,exist_ok=True); table=pd.DataFrame(metric_rows); table.to_csv(directory/"representation_metrics.csv",index=False)
    if table.empty: rendered = "No representation completed."
    else:
        columns = list(table.columns); rendered = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n" + "\n".join("| " + " | ".join(str(row[c]) for c in columns) + " |" for _, row in table.iterrows())
    text=["# PCA and t-SNE representation-baseline analysis","","## Population",f"The manifest contains {metadata['n_images']} images: {metadata['n_ideal']} labelled ideal images from five reconstruction folders and {metadata['n_trajectory']} unlabeled trajectory images.","","## Exploratory metrics (ideal images only)",rendered,"","## Interpretation limits","Raw pixels are a no-learned-representation baseline; ImageNet ResNet-18 is an off-domain label-free baseline; RHEED SimCLR is domain-specific and label-free. PCA/t-SNE separation is supporting exploratory evidence only, not proof of classification performance. t-SNE can visually exaggerate clusters, so conclusions should use the fixed protocol and full-space metrics. No pairwise labels or active-learning outcomes were used during feature extraction."]
    (OUT/"report.md").write_text("\n".join(text),encoding="utf-8")
def run(representations,reuse_cache,batch_size,device_name,data_root=None):
    seed_everything(); OUT.mkdir(parents=True,exist_ok=True); manifest=build_manifest(DATA if data_root is None else data_root); manifest.to_csv(OUT/"manifest.csv",index=False); device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else ("cpu" if device_name=="auto" else device_name)); metadata={"seed":SEED,"device":str(device),"n_images":len(manifest),"n_ideal":int(manifest.is_labeled.sum()),"n_trajectory":int((~manifest.is_labeled).sum()),"ideal_folders":IDEAL_FOLDERS,"representations":[]}; metric_rows=[]
    for name in representations:
        features,provenance=extract_representation(name,manifest,device,batch_size,reuse_cache); metadata["representations"].append(provenance)
        if features is None: continue
        if name!="raw_pixels": features=normalize(features)
        standardized=StandardScaler().fit_transform(features)
        if not np.isfinite(standardized).all(): raise ValueError(f"Non-finite standardized {name} values")
        pca=PCA(n_components=2,random_state=SEED); pca_xy=pca.fit_transform(standardized); tsne_xy=tsne_projection(standardized); save_coordinates(name,manifest,pca_xy,tsne_xy); plot(name,"pca",pca_xy,manifest,f"explained variance={pca.explained_variance_ratio_.sum():.3f}"); plot(name,"tsne",tsne_xy,manifest,"fixed t-SNE protocol")
        metrics,centroids=separation_metrics(standardized,pca_xy,manifest,name!="raw_pixels"); metrics["representation"]=name; metrics["pca_explained_variance_2d"]=float(pca.explained_variance_ratio_.sum()); metric_rows.append(metrics); (OUT/"metrics").mkdir(parents=True,exist_ok=True); centroids.to_csv(OUT/"metrics"/f"{name}_centroid_distance_matrix.csv")
    (OUT/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); write_report(metadata,metric_rows)
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--all",action="store_true"); parser.add_argument("--representation",choices=["raw_pixels","imagenet_resnet18","rheed_simclr_resnet18"]); parser.add_argument("--reuse-cache",action="store_true"); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--device",default="auto"); parser.add_argument("--data-root"); args=parser.parse_args(); selected=["raw_pixels","imagenet_resnet18","rheed_simclr_resnet18"] if args.all else ([args.representation] if args.representation else None)
    if not selected: parser.error("Choose --all or --representation")
    run(selected,args.reuse_cache,args.batch_size,args.device,resolve_data_root(args.data_root) / "original data")
if __name__=="__main__": main()
