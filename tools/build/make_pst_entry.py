"""Build the PST 4k entry: our engine + classic's tables, no NNUE.

Not a sum of two measurements -- an actual file, packed by the real
packer, that must play with no other file beside it. `3208 + 579` was
composed, and lzma shares one dictionary across the whole stream, so the
incremental cost of the tables inside THIS source is not the cost of the
same tables inside classic.

Transformations, all mechanical:
  - drop the .sfnn loader, the SWAR constants, nn_cp and _ext
  - paste classic's pst/K_MID/K_END literals in their place
  - drop the accumulator fields (acc, pf, kb) from Position and every
    site that threads them, so score == ps and the eval is the table
usage: build_pst_entry.py OUT.py
"""
import re
import sys

import pathlib

# Derive the repo root from THIS file, never a hard-coded path: the
# absolute one broke CI on every machine but the author's, and inside a
# worktree it silently regenerated from the OTHER checkout -- so
# check_entry.sh verified a file it had never read.
REPO = str(pathlib.Path(__file__).resolve().parents[2])
OUT = sys.argv[1]
src = open(REPO + "/nnue_4k/sunfish_nnue.py").read()
classic = open(REPO + "/sunfish.py").read()

# ---- classic's evaluation tables, verbatim -------------------------------
m = re.search(r"\npiece = \{.*?\n(?=\n*###)", classic, re.S)
if not m:
    m = re.search(r"\npst = \{.*?\n(?=\n*###)", classic, re.S)
assert m, "classic tables not found"
tables = m.group(0)
# classic pads its tables at import; keep whatever padding code follows
pad = re.search(r"\npst = \{.*?\n\}", classic, re.S)
assert pad, "pst literal not found"

# ---- excise the NNUE region ---------------------------------------------
i = src.find("import json as _json")
j = src.find("###############################################################################\n"
             "# Piece-Square tables")
assert i > 0 and j > i, "NNUE region not found"
src = src[:i] + tables.strip("\n") + "\n\n" + src[j:]

# the packed engine's own table section now duplicates classic's: drop it
k = src.find("###############################################################################\n"
             "# Piece-Square tables")
k2 = src.find("###############################################################################", k + 10)
assert k2 > k
src = src[:k] + src[k2:]

# ---- strip the accumulator from Position --------------------------------
src = src.replace(
    'class Position(namedtuple("Position", "board score ps wc bc ep kp acc pf kb")):',
    'class Position(namedtuple("Position", "board score wc bc ep kp")):', 1)

src = src.replace("""        return Position(
            self.board[::-1].swapcase(), -self.score, -self.ps, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
            self.acc, self.pf ^ 1, self.kb,
        )""",
"""        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )""", 1)

src = src.replace("""        ps = self.ps + self.value(move)""",
                  """        score = self.score + self.value(move)""", 1)
src = re.sub(r"\n *# Every board mutation below.*?acc -= row\[q\]\[j\]\n", "\n", src, flags=re.S)
src = re.sub(r"\n *if B > 1:\n *# Mover's own bucket.*?kb = nb \* B \+ kb % B if self\.pf == 0 else kb - ob \+ nb\n",
             "\n", src, flags=re.S)
src = re.sub(r"\n *acc \+= row\[\"R\"\]\[kp\] - row\[\"R\"\]\[r\]", "", src)
src = re.sub(r"\n *acc \+= row\[prom\]\[j\] - row\[\"P\"\]\[j\]", "", src)
src = re.sub(r"\n *acc -= row\[\"p\"\]\[j \+ S\]", "", src)
src = re.sub(r"\n *# A king move across a bucket boundary.*?acc \+= row\[c\]\[s\]\n", "\n", src, flags=re.S)
src = src.replace("""        # We rotate the returned position, so it's ready for the next player
        pf = self.pf ^ 1
        return Position(board[::-1].swapcase(), -ps + nn_cp(acc, pf, board), -ps,
                        bc, wc, 119 - ep if ep else 0, 119 - kp if kp else 0,
                        acc, pf, kb)""",
"""        # We rotate the returned position, so it's ready for the next player
        return Position(board[::-1].swapcase(), -score, bc, wc,
                        119 - ep if ep else 0, 119 - kp if kp else 0)""", 1)

src = src.replace('''def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position (and its accumulator) from scratch. `board` is
    already in the side-to-move's orientation."""
    ps = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
             for i, p in enumerate(board) if p.isalpha())
    kb = 0
    if B > 1:
        own, opp = kbucket(board.index("K")), kbucket(119 - board.index("k"))
        kb = own * B + opp if pf == 0 else opp * B + own
    acc = ACC_BASE
    row = ROWS[pf][kb]
    for i, p in enumerate(board):
        if p in _PIECES:
            acc += row[p][i]
    return Position(board, ps + nn_cp(acc, pf, board), ps, wc, bc, ep, kp,
                    acc, pf, kb)''',
'''def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position from scratch; `board` is in the mover's orientation."""
    score = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
                for i, p in enumerate(board) if p.isalpha())
    return Position(board, score, wc, bc, ep, kp)''', 1)

# ps was the packed engine's clip-free score; with no net they are the same
src = src.replace("pos.ps", "pos.score").replace("self.ps", "self.score")
# ---- classic's tables need classic's king-table phase rule ---------------
# The packed engine switches to K_END on a bare king. That is right for ITS
# K_END -- a trained bare-king mop-up table -- but this entry pastes CLASSIC's
# K_END, the centralization gradient classic calls out as "important to win
# KRK/KQK endings", and classic keys it on queens-off. Over 37,374 positions
# from real games the two rules disagree on 62.1%. Measured +52.3 +/- 21.1
# fixed-node, +30.5 +/- 24.4 timed head-to-head, and it SAVES 11 packed bytes.
#
# Swapping the table also invalidates the incrementally carried `pos.score`,
# which was accumulated under the old table and which rotate() then sign-flips
# every ply -- an oscillating phantom tempo on every stand-pat and futility
# margin for the whole search. Rebuild the root under the table this search
# will actually use (+3 bytes; net -8, entry 3483 -> 3475).
old = ('        # Bare-king endings: swap in the centralization gradient (packed\'s\n'
       '        # own measured condition; classic keys on queens-off instead).\n'
       '        # Both directions every search: table state must never outlive the\n'
       '        # condition.\n'
       '        bare = sum(c.isupper() for c in pos.board) == 1 or sum(c.islower() for c in pos.board) == 1\n'
       '        pst["K"] = K_END if bare else K_MID\n')
new = ('        # Classic\'s K_END is a centralization gradient, and classic keys it\n'
       '        # on queens-off. Both directions every search: table state must\n'
       '        # never outlive the condition.\n'
       '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'
       '        # The carried score was accumulated under the OTHER table.\n'
       '        pos = self.root = from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp)\n')
assert src.count(old) == 1, "king-table block not found verbatim in sunfish_nnue.py"
src = src.replace(old, new, 1)

open(OUT, "w").write(src)
try:
    compile(src, OUT, "exec")
    print("built %s (%d source bytes) -- compiles" % (OUT, len(src)))
except SyntaxError as e:
    print("SYNTAX ERROR line %d: %s" % (e.lineno, src.split("\n")[e.lineno - 1][:80]))
    sys.exit(1)
