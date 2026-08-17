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
  - constants            (MATE_LOWER, MATE_UPPER, QS, QS_A, LMR,
                          EVAL_ROUGHNESS, NULL_MARGIN, TABLE_SIZE)

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

CONSTANTS = ["MATE_LOWER", "MATE_UPPER", "QS", "QS_A", "LMR", "EVAL_ROUGHNESS",
             "NULL_MARGIN", "TABLE_SIZE"]

EXPECTED = {
    "Position.gen_moves": "3453dbe008109d3d",
    "Position.king_capture": "077e364f886a1826",
    "Position.move": "69bb2460cd611c9e",
    "Position.rotate": "cb12fe4a160ae663",
    "Position.value": "11d52eaa8a661352",
    "Searcher.bound": "98cfd0badd6f7ed4",
    "Searcher.search": "f2951f9448855273",
    "constants": "62b96e206341a2fb",
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
    "base = QS if depth == 0 else -MATE_UPPER",
    "margin = max(depth - 1, 0) * QS_A",
    "val_lower = max(base, min(MATE_LOWER, gamma - pos.score - margin)) if depth <= 3 else base",
    "if (not root and 2 < depth < 6 and abs(pos.score) < 750",
    "guard = depth >= 6 and abs(pos.score) < 750 and any(c in pos.board for c in \"RBNQ\")",
    "target = pos.score + NULL_MARGIN",
    "d -= -self.bound(nullpos, 1 - target, depth - 7) >= target",
    "yield None, pos.score",
    "score = cap if (cap := pos.score + EVAL_ROUGHNESS) < gamma else min(cap,",
    "-self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))",
    "proof = score >= gamma and pos.king_capture()",
    "if killer and (val := pos.value(killer)) >= val_lower:",
    "yield killer, MATE_UPPER if val >= MATE_LOWER else val",
    "values = sorted(((v, m) for m in pos.gen_moves() if (v := pos.value(m)) >= base), reverse=True)",
    "n = sum(v >= val_lower for v, m in values)",
    "yield from ((m, MATE_UPPER if v >= MATE_LOWER else v) for v, m in values[:n])",
    "if n < len(values): yield None, min(MATE_LOWER - 1, pos.score + values[n][0] + margin)",
    "if move is not None and score < MATE_LOWER:",
    "cap = (MATE_UPPER if depth > 3 else",
    "min(MATE_LOWER - 1, pos.score + val + margin))",
    "move_depth = d - 1 - (not root and guard and val < LMR)",
    "score = min(cap, 0 if root and child in self.history else -self.bound(child, 1 - gamma, move_depth))",
    "best, live = -MATE_UPPER, False",
    "if depth and not live and all(",
    "pos.rotate(nullmove=True).king_capture()",
    "mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)",
    "self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)",
    "lower, upper = 1 - MATE_UPPER, MATE_UPPER",
    "if pst[\"K\"] is not king:",
    "self.tp_score.clear()",
    # The docstring is a model claim, so it is pinned like code: the two
    # exact clauses, the reservation sentence that says WHY they are exact,
    # and the zone map at the constants that names each landmark.
    "It is assumed 1 - MATE_UPPER < gamma <= MATE_UPPER.",
    "- our own king already captured: r = -MATE_UPPER.",
    "if the opponent king capturable: r = MATE_UPPER",
    "tokens the fold compares for equality, never scores",
    "so an exact MATE_UPPER proves a",
    "Only a searched real move sets",
    "every move in tp_move is legal.",
    "RESERVED TOKENS, never an evaluation",
    "band admission edges",
    "mate DISTANCE, strictly between the two",
    "Two jobs, deliberately one number",
]

# Raw "line N" citations in the Lean sources are fragile: they rot silently.
# We ratchet rather than ban outright -- the count may fall, never rise.
LINE_CITATION_BUDGET = 135


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
