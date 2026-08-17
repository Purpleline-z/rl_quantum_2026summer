#!/usr/bin/env python3
"""Run only the simulator preliminary behavior tests and write a Drive marker."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from simulator_study_common import atomic_json_write
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--drive-output",required=True,type=Path); args=parser.parse_args()
    test=Path(__file__).parents[2]/"active_learning_program"/"code_behavior_tests"/"test_simulator_augmented_classifier2_preliminary.py"
    run=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(test.parent),"-p",test.name],capture_output=True,text=True)
    atomic_json_write({"returncode":run.returncode,"stdout":run.stdout,"stderr":run.stderr},args.drive_output/"preflight_audits"/"simulator_study_behavior_test_result.json")
    print(run.stdout); print(run.stderr)
    if run.returncode: raise SystemExit(run.returncode)
if __name__=="__main__": main()
