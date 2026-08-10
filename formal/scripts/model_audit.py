#!/usr/bin/env python3
"""Model-code drift guard (formal/README.md's mapping table).

The Lean model in formal/ is audited against specific regions of
sunfish.py.  This script hashes those regions and FAILS when they
drift, making silent divergence between code and model impossible: any
change to an audited region must land in the same commit as the
re-audit (update formal/README.md's fidelity section and refresh the
hashes here with `--update`).

Mechanical, not a proof: the guard pins WHAT was audited, the README
records what the audit established, and the Lean files carry the
theorems.  This stands in until the leanpy/lean-surfaces track makes
the code-model correspondence itself checked.

Audited regions (one hash per object, whitespace-normalized so pure
reformatting of surrounding code does not fire):
  - Searcher.bound       (the search: every mapping-table row)
  - Searcher.search      (the MTD-bi driver: Driver.lean)
  - Position.rotate      (RotateNegatesScore)
  - Position.move        (ValGame.score_identity)
  - Position.value       (KingCaptureValHigh / HighValIsKingCapture)
  - Position.gen_moves   (Game.moves, CaptureFirst's list)
  - Position.king_capture (the substitution/in-check scan, kp = 0 note)
  - constants            (MATE_LOWER, MATE_UPPER, QS, QS_A,
                          EVAL_ROUGHNESS, TABLE_SIZE)

Run from the repo root:  python formal/scripts/model_audit.py
Refresh after a re-audit: python formal/scripts/model_audit.py --update
"""
import ast
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUNFISH = ROOT / "sunfish.py"

CONSTANTS = ["MATE_LOWER", "MATE_UPPER", "QS", "QS_A", "EVAL_ROUGHNESS", "TABLE_SIZE"]

EXPECTED = {
    "Position.gen_moves": "14d69d763fe2185d",
    "Position.king_capture": "077e364f886a1826",
    "Position.move": "c95ddc3e690012a8",
    "Position.rotate": "cb12fe4a160ae663",
    "Position.value": "339f53cfaa228d42",
    "Searcher.bound": "cae8ee304652daca",
    "Searcher.search": "f9aa8c81b84ff44b",
    "constants": "02227a9fd04eb181",
}


def normalize(src: str) -> str:
    return "\n".join(line.rstrip() for line in src.strip().splitlines())


def digest(src: str) -> str:
    return hashlib.sha256(normalize(src).encode()).hexdigest()[:16]


def extract_regions():
    src = SUNFISH.read_text()
    tree = ast.parse(src)
    regions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("Position", "Searcher"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    key = f"{node.name}.{item.name}"
                    if key in ("Searcher.bound", "Searcher.search", "Position.rotate",
                               "Position.move", "Position.value", "Position.gen_moves",
                               "Position.king_capture"):
                        regions[key] = ast.get_source_segment(src, item)
    consts = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in CONSTANTS for n in names):
                consts.append(ast.get_source_segment(src, node))
    regions["constants"] = "\n".join(consts)
    missing = {"Searcher.bound", "Searcher.search", "Position.rotate", "Position.move",
               "Position.value", "Position.gen_moves", "Position.king_capture",
               "constants"} - set(regions)
    if missing:
        raise SystemExit(f"model_audit: audited region(s) not found in sunfish.py: {sorted(missing)}")
    return regions


# Distinctive source anchors the Lean model cites BY NAME rather than by line
# number.  Hashing catches code drift but is blind to a stale line number in a
# Lean comment, so the model was drifting in a way the guard structurally could
# not see (Killer.lean cited lines 339/356-357/366 for code that had moved to
# 391/422-423).  Citing anchors instead makes the class self-maintaining: rename
# or delete one of these and the check fires.
ANCHORS = [
    "def king_capture",
    "killer = self.tp_move.get(pos)",
    "if killer and pos.value(killer) >= val_lower:",
    "yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)",
    "yield None, pos.score",
    "score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)",
    "if depth <= 1 and pos.score + val < gamma:",
    "yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)",
    "best, live = -MATE_UPPER, False",
    "if depth and not live and all(",
    "pos.rotate(nullmove=True).king_capture()",
    "self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)",
    "lower, upper = 1 - MATE_UPPER, MATE_UPPER",
    "if depth > 0 and pos in self.history:",
]

# Raw "line N" citations in the Lean sources are fragile: they rot silently.
# We ratchet rather than ban outright -- the count may fall, never rise.
LINE_CITATION_BUDGET = 149


def check_anchors(src):
    missing = [a for a in ANCHORS if a not in src]
    return missing


LINE_CITE_RE = re.compile(r"lines? \d{2,4}(\s*[-–]\s*\d{2,4})?")


def count_line_citations():
    total = 0
    for f in sorted((ROOT / "formal" / "Sunfish").glob("*.lean")):
        total += len(LINE_CITE_RE.findall(f.read_text()))
    return total


def main():
    regions = extract_regions()
    actual = {k: digest(v) for k, v in sorted(regions.items())}
    if "--update" in sys.argv:
        me = pathlib.Path(__file__)
        text = me.read_text()
        block = "EXPECTED = {\n" + "".join(
            f'    "{k}": "{v}",\n' for k, v in actual.items()) + "}"
        text = re.sub(r"EXPECTED = \{.*?\}", block, text, count=1, flags=re.S)
        text = re.sub(r"LINE_CITATION_BUDGET = .*",
                      f"LINE_CITATION_BUDGET = {count_line_citations()}", text, count=1)
        me.write_text(text)
        print("model_audit: EXPECTED hashes refreshed:")
        for k, v in actual.items():
            print(f"  {k}: {v}")
        return 0
    src = SUNFISH.read_text()
    missing = check_anchors(src)
    if missing:
        print("model_audit: cited ANCHOR(S) no longer present in sunfish.py:")
        for a in missing:
            print(f"  {a!r}")
        print("The Lean model cites these by name (formal/Sunfish/*.lean).")
        print("Update the citation and the model together, then re-run.")
        return 1
    cites = count_line_citations()
    if LINE_CITATION_BUDGET is not None and cites > LINE_CITATION_BUDGET:
        print(f"model_audit: raw line-number citations rose {LINE_CITATION_BUDGET} -> {cites}.")
        print("Line numbers rot silently; cite a distinctive source anchor instead")
        print("(and add it to ANCHORS so the guard checks it).")
        return 1
    drifted = {k: v for k, v in actual.items() if EXPECTED.get(k) != v}
    if not EXPECTED:
        print("model_audit: no EXPECTED hashes recorded; run with --update first")
        return 1
    if drifted:
        print("model_audit: AUDITED REGION(S) DRIFTED without a same-commit re-audit:")
        for k, v in sorted(drifted.items()):
            print(f"  {k}: expected {EXPECTED.get(k, '<none>')}, found {v}")
        print("The formal model in formal/ was audited against these regions")
        print("(formal/README.md, 'Model fidelity').  Re-audit the model against")
        print("the change, update the README in the SAME commit, then refresh:")
        print("    python formal/scripts/model_audit.py --update")
        return 1
    print("model_audit: all audited regions match the recorded audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
