#!/usr/bin/env python3
"""Create a small qualitative synthetic-vs-real contact sheet; it is not a metric."""
from __future__ import annotations
import argparse, io, zipfile
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
from simulator_study_common import PROTOCOL, archive_member_root, load_partition_table, resolve_data_root

def tile(canvas, image, x, y, label):
    image=image.convert("L"); image.thumbnail((210,158)); canvas.paste(image,(x,y)); ImageDraw.Draw(canvas).text((x,y+160),label,fill="white")
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--synthetic-archive",required=True,type=Path); parser.add_argument("--drive-output",required=True,type=Path); parser.add_argument("--data-root")
    args=parser.parse_args(); output=args.drive_output/"preflight_audits"/"real_and_synthetic_contact_sheet.png"; output.parent.mkdir(parents=True,exist_ok=True); canvas=Image.new("L",(1350,420),0)
    with zipfile.ZipFile(args.synthetic_archive) as bundle:
        root=archive_member_root(args.synthetic_archive); rows=pd.read_csv(io.BytesIO(bundle.read(f"{root}/metadata.csv")))
        for column,label in enumerate(("twinned_2x1","c_6x2","rt13")):
            row=rows[rows.reconstruction_label==label].iloc[0]; tile(canvas,Image.open(io.BytesIO(bundle.read(f"{root}/{row.image_path}"))),column*225,0,"synthetic "+label)
    partition=load_partition_table(args.drive_output); resolve_data_root(args.data_root)
    for column,label in enumerate(PROTOCOL["real_evaluation_types"]):
        row=partition[(partition.partition=="outer_test")&(partition.label==label)].iloc[0]; tile(canvas,Image.open(row.image_id),675+column*225,0,"real "+label)
    canvas.save(output); print(output)
if __name__=="__main__": main()
