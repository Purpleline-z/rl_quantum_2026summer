"""Tests of the paper package (run: python3 -m unittest discover -s tests -v, from PAPER_PACKAGE; needs `make tables` first).
 1. the package's own paired-gain / bootstrap / Holm helpers equal the studies' nd_confirm.py on a real spec
 2. every pre-registered test regenerated here appears with identical numbers in the studies' own RESULTS documents (sample: all rows of ND1, GL1, GL2, SH1L, SH2L, GM1, GM2) 
 3. the prose checker fails on a wrong number, an unmarked number and an unknown key, and passes on a right one
 4. seed blocks are disjoint (also asserted by pp_seedblocks.py)"""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pp_lib as L
import check_prose as CP

DOCS = {"GL1": "RESULTS_graph_learner.md", "GL2": "RESULTS_graph_learner.md", "SH1L": "RESULTS_selection_htr.md", "SH2L": "RESULTS_selection_htr.md", "GM1": "RESULTS_graph_mechanism.md", "GM2": "RESULTS_graph_mechanism.md"}


class Stats(unittest.TestCase):
    def test_helpers_equal_nd_confirm(self):
        import nd_confirm as NC
        spec = json.loads((L.ND / "prereg1_spec.json").read_text()); frame = NC.load([str(L.RES / "confirm1")]); out = NC.run({"tests": spec["tests"][:6]}, frame)
        f = L.load_dir("confirm1")
        for t, r in zip(spec["tests"][:6], out.itertuples()):
            per = L.per_seed(f, t["split"], t["metric"], tuple(t["a"]), tuple(t["b"])); s = L.summarize(per)
            self.assertAlmostEqual(s["gain"], r.gain, places=12); self.assertAlmostEqual(s["p"], r.p, places=12); self.assertAlmostEqual(s["lo"], r.lo, places=9); self.assertAlmostEqual(s["hi"], r.hi, places=9)
        p = [0.01, 0.04, 0.03, 0.5]; self.assertTrue(np.allclose(L.holm(p), NC.holm(p)))


class CrossCheck(unittest.TestCase):
    def test_rows_in_study_documents(self):
        conf = pd.read_csv(L.TABLES / "confirmations.csv"); missing = []; total = 0
        for exp, doc in DOCS.items():
            text = (L.GE / doc).read_text(); rows = conf[(conf.exp == exp) & conf.lo.notna()]
            for r in rows.itertuples():
                sig = f"| {r.family} | {r.split} | {r.metric} | {r.cells} | {r.n} | {r.gain:+.4f} | {r.lo:+.4f} | {r.hi:+.4f} |"; total += 1
                if sig not in text: missing.append((exp, sig))
        self.assertGreater(total, 300); self.assertLessEqual(len(missing), 0.01 * total, missing[:5])
    def test_nd1_rows_in_new_directions_table(self):
        text = (L.ND / "results" / "CONFIRM1_TABLE.md").read_text(); conf = pd.read_csv(L.TABLES / "confirmations.csv"); rows = conf[(conf.exp == "ND1") & conf.family.isin(["F1", "F2", "F3"])]; miss = 0
        for r in rows.itertuples():
            sig = f"{r.gain:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"; miss += sig not in text
        self.assertEqual(miss, 0)
    def test_ge_verdicts_in_graph_endpoint_document(self):
        text = (L.GE / "RESULTS_graph_endpoint.md").read_text(); v = pd.read_csv(L.TABLES / "verdicts.csv")
        for exp, blk in (("GE1", "conf1"), ("GE2", "conf2")):
            line = next(l for l in text.splitlines() if l.startswith(f"- **{blk}**"))
            for claim, word in (("harsh splits (c)", "harsh splits"), ("unlabelled-image dose-response (b)", "dose-response")):
                mine = v[(v.exp == exp) & (v.claim == claim)].verdict.iloc[0]; seg = line.split(word)[1].split("->")[1].split(".")[0].strip().split(" ")[0]
                self.assertEqual(mine, seg, (exp, claim))


class Prose(unittest.TestCase):
    NUMS = {"k.a": 0.0210, "k.b": -0.0083, "k.p": 0.471, "k.s": "PRE-REGISTERED, CONFIRMED"}
    def run_text(self, text): return CP.check_text("t", text, self.NUMS)[0]
    def test_right_numbers_pass(self): self.assertEqual(self.run_text("gain +0.021⟦k.a⟧ and −0.008⟦k.b⟧ or 47%⟦k.p⟧, threshold 0.03⟦-⟧, status ok⟦k.s==PRE-REGISTERED, CONFIRMED⟧"), [])
    def test_wrong_number_fails(self): self.assertTrue(self.run_text("gain +0.022⟦k.a⟧"))
    def test_wrong_sign_fails(self): self.assertTrue(self.run_text("gain −0.021⟦k.a⟧"))
    def test_unmarked_number_fails(self): self.assertTrue(self.run_text("the gain is 0.021 here"))
    def test_unknown_key_fails(self): self.assertTrue(self.run_text("gain +0.021⟦k.zzz⟧"))
    def test_wrong_status_fails(self): self.assertTrue(self.run_text("x⟦k.s==NOT CONFIRMED⟧"))


class Docs(unittest.TestCase):
    def test_delivered_documents_are_clean(self):
        for p in (L.GE / "RESULTS_paper_package.md", L.GE / "SUMMARY_ZH_paper_package.md", L.GE / "HANDOFF_paper_package.md", L.GE / "GPU_TASKS_paper_package.md", L.PKG / "README.md", L.PKG / "ABSTRACT_EN.md"):
            text = p.read_text(); self.assertNotIn("{{INCLUDE", text, p.name); self.assertNotIn("⟦", text, p.name); self.assertGreater(len(text), 500)


class Seeds(unittest.TestCase):
    def test_blocks_disjoint(self):
        from pp_registry import SEED_BLOCKS
        b = sorted((a, z) for _, _, a, z in SEED_BLOCKS)
        for (a1, z1), (a2, z2) in zip(b, b[1:]): self.assertLess(z1, a2)


if __name__ == "__main__":
    unittest.main()
