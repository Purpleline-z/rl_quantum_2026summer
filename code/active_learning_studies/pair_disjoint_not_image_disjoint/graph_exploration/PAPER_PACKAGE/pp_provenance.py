"""Pre-registration order check from git history: for every pre-registration file the commit that ADDED it (and its time) must precede the first commit that added any file of the confirmation result
directories of its study.  Also records, per study, the branch and the tip commit that was imported into this package.  Needs the full history of the branches (git fetch origin); if the history is not
available it writes 'n/a' rows and the check is reported as not verifiable (the make target does not fail on that, but check_prose.py prints a warning)."""
import subprocess
from datetime import datetime

import pandas as pd

import pp_lib as L
from pp_lib import ND, Numbers, STUDIES as STUDIES_DIR
from pp_registry import STUDIES

REPO = L.PKG.parents[4]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def first_add(path):
    """commit hash + ISO time of the earliest commit (any branch) that added `path` (file or directory)."""
    out = git("log", "--all", "--diff-filter=A", "--format=%h %cI", "--", path).splitlines()
    return (out[-1].split(" ", 1) if out else (None, None))


def main():
    rows = []; ok_all = True
    for sid, st in STUDIES.items():
        for pfile, pcommit, rel, dirs in st["prereg"]:
            prepath = "code/active_learning_studies/" + (rel if rel.startswith("new_directions") else "pair_disjoint_not_image_disjoint/" + rel)
            h, t = first_add(prepath)
            for d in dirs:
                dpath = f"code/active_learning_studies/new_directions/results/{d}"; rh, rt_ = first_add(dpath)
                if rt_ is None or t is None: rows.append(dict(study=sid, prereg=pfile, prereg_commit_registry=pcommit, prereg_commit_git=h, prereg_time=t, result_dir=d, first_result_commit=rh, first_result_time=rt_, prereg_before_results="n/a (not in history)")); continue
                before = datetime.fromisoformat(t) < datetime.fromisoformat(rt_); ok_all &= before
                rows.append(dict(study=sid, prereg=pfile, prereg_commit_registry=pcommit, prereg_commit_git=h, prereg_time=t, result_dir=d, first_result_commit=rh, first_result_time=rt_, prereg_before_results=str(before)))
    t = pd.DataFrame(rows); t.to_csv(L.TABLES / "prereg_order.csv", index=False); L.write(L.TABLES / "prereg_order.md", L.to_md(t, "{}"))
    prov = pd.DataFrame([dict(study=sid, name=s["name"], branch=s["branch"], tip_commit_at_import=s["tip"], docs="; ".join(s["docs"])) for sid, s in STUDIES.items()]); prov.to_csv(L.TABLES / "provenance.csv", index=False); L.write(L.TABLES / "provenance.md", L.to_md(prov, "{}"))
    bad = t[t.prereg_before_results == "False"]; nv = int((t.prereg_before_results == "n/a (not in history)").sum()); print(f"{len(t)} checks: {int((t.prereg_before_results == 'True').sum())} verified (pre-registration committed before the first result file), {len(bad)} violated, {nv} not verifiable")
    N = Numbers("prov"); N.put("prov.total", len(t)); N.put("prov.verified", int((t.prereg_before_results == "True").sum())); N.put("prov.violated", len(bad)); N.put("prov.not_verifiable", nv); N.save()
    assert len(bad) == 0, bad


if __name__ == "__main__":
    main()
