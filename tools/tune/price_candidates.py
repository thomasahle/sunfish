"""Price the fitted eval candidates: real file, real packer, bytes off disk.

`3475 - 94` was not a number and neither is `3378 + 395`. lzma carries one
dictionary across the whole stream, so a table's cost depends on the source it
sits in and on the OTHER tables beside it -- a second table set that resembles
the first compresses far better than its own size suggests. Every row below is
one real entry source through `tools/build/pack.sh`, packed, measured, and run
alone in an empty directory.

The fitted piece VALUES move too (R 479 -> 472, Q 929 -> 923, ...), so the
codec's hard-coded value line is regenerated per candidate rather than reused.

usage: price_candidates.py FITS.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval4k"))
import codec  # noqa: E402
import measure  # noqa: E402
import splice  # noqa: E402

FITS = json.load(open(sys.argv[1]))
ROOT = measure.ROOT
BASE = open(os.path.join(ROOT, splice.ENTRY)).read()
PIECES = "PNBRQK"

# The landed root block: the queens-off king swap plus the fresh rebuild. Every
# tapered candidate replaces exactly this, because this is where the engine
# already re-derives its table once per search.
ROOT_OLD = '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'


def valline(vals):
    return "piece = {%s}\n" % ", ".join('"%s": %d' % (p, vals[p]) for p in PIECES)


def region(tables, name="pst"):
    """codec.emit with this candidate's own piece values, under a chosen name."""
    raw = {p: tables[p] for p in PIECES}
    vals = {p: tables["_value_" + p] for p in PIECES}
    src = codec.emit(vals, raw)
    # swap in the fitted values, and rename the target dict if asked
    src = src.replace(codec.emit(vals, raw).split("\n")[0] + "\n", valline(vals), 1)
    if name != "pst":
        src = src.replace("pst = {}", name + " = {}").replace("pst[_k]", name + "[_k]")
        src = src.replace('K_MID, K_END = pst["K"]', 'K_MID, K_END = %s["K"]' % name)
    return src


def price(label, src, note=""):
    packed, size = measure.pack(src, label.replace(" ", "_"))
    mv = measure.standalone(packed)
    print("  %-26s %5d bytes  %+5d vs entry  %4d spare   %s  %s"
          % (label, size, size - BASESIZE, 4096 - size, mv, note))
    return size


_, BASESIZE = measure.pack(BASE, "base")
print("entry as landed: %d bytes, %d spare\n" % (BASESIZE, 4096 - BASESIZE))
print("candidate prices (built, not composed):")

# ---- A: flat refit -- same shape, different numbers -------------------------
A = FITS["flat"]["tables"]
srcA = splice.splice(BASE, region(A))
sizeA = price("A flat refit", srcA)

# ---- B: queens seam -- a second full table set, selected where K already is --
B = FITS["qseam"]["tables"]
regB = region(B["qon"]) + "\n" + region(B["qoff"], "pst2").replace(
    'K_MID, K_END = pst2["K"], tuple(piece["K"] + 70\n'
    "   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n", "")
# pst2 needs its own decoder integer but not a second piece line
regB = regB.replace(valline({p: B["qoff"]["_value_" + p] for p in PIECES}), "piece2 = {%s}\n" % ", ".join(
    '"%s": %d' % (p, B["qoff"]["_value_" + p]) for p in PIECES), 1)
regB = regB.replace("pst2[_k] = tuple", "pst2[_k] = tuple").replace(
    "for _i in range(64)]\n _v //= 210 ** 64\n pst2", "for _i in range(64)]\n _v //= 210 ** 64\n pst2")
regB = regB.replace("piece[_k] for _i in range(64)] \n", "piece[_k] for _i in range(64)]\n")
# the second table block must read piece2, not piece
head, sep, tail = regB.partition("piece2 = {")
tail = tail.replace("piece[_k]", "piece2[_k]")
regB = head + sep + tail
srcB = splice.splice(BASE, regB).replace(
    ROOT_OLD,
    '        pst.update(pst2 if "Q" not in pos.board or "q" not in pos.board else pst1)\n'
    '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n')
srcB = srcB.replace("pst = {}\n", "pst = {}\npst1 = {}\n", 1)
srcB = srcB.replace(' pst[_k] = tuple([0] * 20', ' pst[_k] = pst1[_k] = tuple([0] * 20', 1)
sizeB = price("B queens-seam", srcB)

# ---- C: continuous phase blend ----------------------------------------------
C = FITS["phase"]["tables"]
regC = region(C["mg"]) + "\n" + region(C["eg"], "pstE").replace(
    'K_MID, K_END = pstE["K"], tuple(piece["K"] + 70\n'
    "   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n", "")
regC = regC.replace(valline({p: C["eg"]["_value_" + p] for p in PIECES}), "pieceE = {%s}\n" % ", ".join(
    '"%s": %d' % (p, C["eg"]["_value_" + p]) for p in PIECES), 1)
head, sep, tail = regC.partition("pieceE = {")
tail = tail.replace("piece[_k]", "pieceE[_k]")
regC = head + sep + tail
srcC = splice.splice(BASE, regC).replace(
    ROOT_OLD,
    '        _ph = min(24, sum({"N": 1, "B": 1, "R": 2, "Q": 4}.get(c.upper(), 0)\n'
    "                          for c in pos.board if c.isalpha()))\n"
    '        for _k in "PNBRQ":\n'
    "            pst[_k] = tuple((a * _ph + b * (24 - _ph)) // 24\n"
    "                            for a, b in zip(pstM[_k], pstE[_k]))\n"
    '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n')
srcC = srcC.replace("pst = {}\n", "pst = {}\npstM = {}\n", 1)
srcC = srcC.replace(' pst[_k] = tuple([0] * 20', ' pst[_k] = pstM[_k] = tuple([0] * 20', 1)
sizeC = price("C phase blend", srcC)

print("\nheld-out loss (from the fit, NOT evidence of strength):")
print("  classic %.6f | A %.6f | B %.6f | C %.6f"
      % (FITS["classic_heldout"], FITS["flat"]["heldout"],
         FITS["qseam"]["heldout"], FITS["phase"]["heldout"]))
for k, sz in (("flat", sizeA), ("qseam", sizeB), ("phase", sizeC)):
    d = sz - BASESIZE
    print("  %-6s %+5d bytes for %+.2f%% held-out loss" % (
        k, d, 100 * (FITS[k]["heldout"] - FITS["classic_heldout"]) / FITS["classic_heldout"]))
np.save(os.path.join(os.path.dirname(sys.argv[1]), "sizes.npy"),
        np.array([BASESIZE, sizeA, sizeB, sizeC]))
