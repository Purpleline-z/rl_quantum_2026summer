from __future__ import annotations
import tempfile
from pathlib import Path
import sys, unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).parent.parent))
import explore_pca_tsne as module
class ExplorationTests(unittest.TestCase):
    def test_raw_pixel_shape_and_range(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.bmp"; Image.fromarray(np.full((5,7),128,dtype=np.uint8)).save(p); v=module.raw_pixels(str(p)); self.assertEqual(v.shape,(224*224,)); self.assertTrue(np.all((v>=0)&(v<=1)))
    def test_manifest_has_ideals_and_unlabelled_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for folder in module.IDEAL_FOLDERS.values(): (root/folder).mkdir(parents=True); Image.fromarray(np.zeros((3,3),dtype=np.uint8)).save(root/folder/"x.bmp")
            (root/"Trajectories"/"day").mkdir(parents=True); Image.fromarray(np.zeros((3,3),dtype=np.uint8)).save(root/"Trajectories"/"day"/"u.bmp"); m=module.build_manifest(root); self.assertEqual(set(m[m.is_labeled].label),set(module.IDEAL_FOLDERS)); self.assertEqual(int((~m.is_labeled).sum()),1)
    def test_metrics_exclude_trajectory_labels(self):
        labels=list(module.IDEAL_FOLDERS)*2; m=pd.DataFrame({"is_labeled":[True]*len(labels)+[False],"label":labels+[""]}); values=np.arange(len(m)*4,dtype=float).reshape(len(m),4); metrics,_=module.separation_metrics(values,values[:,:2],m,False); self.assertEqual(metrics["n_ideal"],len(labels)); self.assertNotIn("",metrics["per_class_count"])
    def test_tsne_is_reproducible(self):
        values=np.random.default_rng(1).normal(size=(20,6)); self.assertTrue(np.allclose(module.tsne_projection(values),module.tsne_projection(values)))
    def test_frozen_resnet_encoder_outputs_512_features(self):
        model=module.freeze(module.Encoder(*list(module.models.resnet18(weights=None).children())[:-1]))
        with torch.no_grad(): output=model(torch.zeros(1,3,224,224))
        self.assertEqual(tuple(output.shape),(1,512)); self.assertFalse(any(p.requires_grad for p in model.parameters()))
    def test_missing_imagenet_weights_writes_explicit_skip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); image=root/"x.bmp"; Image.fromarray(np.zeros((3,3),dtype=np.uint8)).save(image); old_out,old_loader=module.OUT,module.make_imagenet_encoder
            try:
                module.OUT=root/"out"; module.make_imagenet_encoder=lambda: (_ for _ in ()).throw(RuntimeError("weights unavailable"))
                features,meta=module.extract_representation("imagenet_resnet18",pd.DataFrame({"path":[str(image)]}),torch.device("cpu"),1,False)
                self.assertIsNone(features); self.assertEqual(meta["status"],"skipped"); self.assertTrue((module.OUT/"metrics"/"imagenet_resnet18_skipped_status.txt").exists())
            finally: module.OUT,module.make_imagenet_encoder=old_out,old_loader
    def test_tiny_cpu_smoke_with_balanced_ideals_and_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); old_data,old_out=module.DATA,module.OUT
            try:
                for value,folder in enumerate(module.IDEAL_FOLDERS.values()):
                    (root/folder).mkdir(parents=True); Image.fromarray(np.full((8,8),value*25,dtype=np.uint8)).save(root/folder/"x.bmp")
                (root/"Trajectories"/"day").mkdir(parents=True); Image.fromarray(np.full((8,8),200,dtype=np.uint8)).save(root/"Trajectories"/"day"/"u.bmp")
                module.DATA=root; module.OUT=root/"out"; module.run(["raw_pixels"],False,2,"cpu",data_root=root); self.assertTrue((module.OUT/"report.md").exists()); self.assertTrue((module.OUT/"coordinates"/"raw_pixels_tsne.csv").exists())
            finally: module.DATA,module.OUT=old_data,old_out
if __name__=="__main__": unittest.main()
