#!/usr/bin/env python3
"""Aggregate completed per-seed simulator preliminary JSON results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulator_study_common import PROTOCOL, atomic_json_write

BASELINE="real_only_classifier2_compatible_baseline"; TREATMENT="synthetic_pretraining_then_real_finetuning"; DIAGNOSTIC="synthetic_only_real_transfer_diagnostic"
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--drive-output",required=True,type=Path); args=parser.parse_args(); root=args.drive_output
    rows=[]
    for arm in (BASELINE,DIAGNOSTIC,TREATMENT):
        for seed in PROTOCOL["paired_training_seeds"]:
            path=root/"per_seed_results"/arm/f"seed_{seed}_completed_result.json"
            if not path.is_file(): raise RuntimeError(f"Missing completed result: {path}")
            value=json.loads(path.read_text()); rows.append({"seed":seed,"arm":arm,"outer_test_accuracy":value["outer_test_accuracy"],"outer_test_macro_recall":value["outer_test_macro_recall"]})
    values=pd.DataFrame(rows); baseline=values[values.arm==BASELINE].set_index("seed"); treatment=values[values.arm==TREATMENT].set_index("seed")
    paired=(treatment[["outer_test_accuracy","outer_test_macro_recall"]]-baseline[["outer_test_accuracy","outer_test_macro_recall"]]).reset_index()
    rng=np.random.default_rng(20260817); bootstrap=[]
    for _ in range(10000): bootstrap.append(paired.iloc[rng.integers(0,len(paired),len(paired))][["outer_test_accuracy","outer_test_macro_recall"]].mean().to_dict())
    boot=pd.DataFrame(bootstrap); summary=values.groupby("arm")[["outer_test_accuracy","outer_test_macro_recall"]].agg(["mean","std"]).reset_index(); out=root/"aggregated_results"; out.mkdir(parents=True,exist_ok=True)
    values.to_csv(out/"all_completed_per_seed_results.csv",index=False); paired.to_csv(out/"paired_treatment_minus_real_only_differences.csv",index=False); summary.to_csv(out/"arm_summary_mean_and_sample_sd.csv",index=False)
    atomic_json_write({"paired_difference_mean":paired[["outer_test_accuracy","outer_test_macro_recall"]].mean().to_dict(),"paired_bootstrap_95_percent_ci":{metric:[float(boot[metric].quantile(.025)),float(boot[metric].quantile(.975))] for metric in boot},"interpretation_rule":"Do not claim benefit unless paired real outer-test differences are consistently positive across seeds."},out/"paired_bootstrap_summary.json")
    figures=out/"figures"; figures.mkdir(exist_ok=True)
    order=[BASELINE,DIAGNOSTIC,TREATMENT]; means=[values[values.arm==arm].outer_test_accuracy.mean() for arm in order]; sds=[values[values.arm==arm].outer_test_accuracy.std() for arm in order]
    plt.figure(figsize=(9,4)); plt.bar(range(len(order)),means,yerr=sds,capsize=4); plt.xticks(range(len(order)),["real-only", "synthetic-only\ntransfer", "synthetic→real\nfine-tune"]); plt.ylabel("Real outer-test accuracy"); plt.ylim(0,1); plt.tight_layout(); plt.savefig(figures/"real_outer_test_accuracy_by_training_arm.png",dpi=180); plt.close()
    plt.figure(figsize=(7,4)); plt.axhline(0,color="black",linewidth=1); plt.scatter(paired.seed,paired.outer_test_accuracy,s=50); plt.xlabel("Paired seed"); plt.ylabel("Treatment minus real-only accuracy"); plt.tight_layout(); plt.savefig(figures/"paired_outer_test_accuracy_difference_by_seed.png",dpi=180); plt.close()
    print(out)
if __name__=="__main__": main()
