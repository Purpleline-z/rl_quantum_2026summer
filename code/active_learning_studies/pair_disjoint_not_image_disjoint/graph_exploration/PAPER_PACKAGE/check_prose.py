"""Verifies every number in the prose against the generated tables; exits with status 1 on any mismatch.

Prose sources (prose/*.src.md) are written with a marker right after every data number:   +0.021⟦cl.L5.A.auc.gain⟧   47%⟦af.anc_all_min.near<=0.001⟧   (the marker names a key of tables/numbers_*.json).
  - a numeric literal followed by ⟦key⟧ must equal the stored value rounded to the literal's number of decimals (sign respected; a '%' literal is compared with 100 x value);
  - ⟦key==TEXT⟧ / ⟦key~~PREFIX⟧ checks a stored string (a claim status) for equality / prefix;
  - ⟦-⟧ marks a number that is not data (thresholds such as 0.03, the 95 % level, seed ranges);
  - any decimal number or percentage WITHOUT a marker is an error ('unverified number'), as is an unknown key.
After a clean run the markers are stripped and the readable documents are written to build/ (and copied to their delivery locations by the Makefile).
  python3 check_prose.py [--no-write]"""
import re
import sys

import pp_lib as L

MARK = re.compile(r"⟦([^⟧]*)⟧")
NUM_BEFORE = re.compile(r"([+\-−±]?\d[\d,]*(?:\.\d+)?%?)$")
BARE = re.compile(r"(?<![\w.])\d+\.\d+%?|(?<![\w.])\d+%")


def strip_code(text):
    return re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)


def check_text(name, text, numbers):
    errors = []; checked = 0
    body = strip_code(text)
    for m in MARK.finditer(body):
        key = m.group(1).strip(); line = body.count("\n", 0, m.start()) + 1
        if key == "-": continue
        sm = re.match(r"(.+?)(==|~~)(.*)$", key, flags=re.S)
        if sm:
            k, op, want = sm.group(1).strip(), sm.group(2), sm.group(3)
            if k not in numbers: errors.append(f"{name}:{line}: unknown key {k}"); continue
            have = str(numbers[k]); ok = have == want if op == "==" else have.startswith(want)
            if not ok: errors.append(f"{name}:{line}: {k} is '{have}', prose says '{want}' ({op})")
            checked += 1; continue
        if key not in numbers: errors.append(f"{name}:{line}: unknown key {key}"); continue
        before = body[:m.start()]; nm = NUM_BEFORE.search(before)
        if not nm: errors.append(f"{name}:{line}: marker {key} is not preceded by a number"); continue
        lit = nm.group(1); pct = lit.endswith("%"); raw = lit.rstrip("%").replace(",", "").replace("−", "-").replace("±", ""); decimals = len(raw.split(".")[1]) if "." in raw else 0
        val = numbers[key]
        if val is None: errors.append(f"{name}:{line}: {key} is None"); continue
        want = float(val) * (100 if pct else 1); got = float(raw)
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if abs(want - got) > tol or (raw.startswith("+") and want < 0) or (raw.startswith("-") and want > 0 and abs(want) > tol):
            errors.append(f"{name}:{line}: {key} = {want:.6g}{'%' if pct else ''}, prose says {lit}")
        checked += 1
    unmarked = MARK.sub("\x00", body)
    for line_no, line in enumerate(unmarked.split("\n"), 1):
        line = re.sub(r"^#+ \d+(\.\d+)*\.? ", "# ", line)          # section numbers in headings are not data
        for m in BARE.finditer(line):
            if line[m.end():m.end() + 1] == "\x00": continue
            errors.append(f"{name}:{line_no}: unverified number '{m.group(0)}' (add a ⟦key⟧ marker or ⟦-⟧)")
    return errors, checked


def main():
    numbers = L.all_numbers(); write = "--no-write" not in sys.argv; allerr = []; total = 0
    for f in sorted(L.PROSE.glob("*.src.md")):
        text = f.read_text(); err, n = check_text(f.name, text, numbers); allerr += err; total += n
        if write and not err: L.write(L.BUILD / f.name.replace(".src.md", ".md"), MARK.sub("", text))
    for e in allerr: print("MISMATCH", e)
    print(f"{total} marked numbers checked in {len(list(L.PROSE.glob('*.src.md')))} documents; {len(allerr)} problems")
    sys.exit(1 if allerr else 0)


if __name__ == "__main__":
    main()
