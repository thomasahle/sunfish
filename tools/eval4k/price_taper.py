"""Price a SECOND eval table set selected at the root -- on the landed generator.

Not a proposal to land: a price. Every number here is one real entry source
through `tools/build/pack.sh`, packed, measured off disk, and run alone in an
empty directory. No Elo is claimed anywhere.

RE-ANCHORED 2026-08-13. The previous version spliced at

    bare = sum(c.isupper() for c in pos.board) == 1 or ...
    pst["K"] = K_END if bare else K_MID

which no longer exists: the `kend` fix replaced the bare-king test with
classic's queens-off rule. The script failed its own `assert old in src`, so
its "+74 bytes" style numbers could not be reproduced from it. The anchor is
now the landed line, and two things about the landed root change the price in
our favour:

  1. **The queens-off king rule IS the phase seam.** The engine already tests
     queens-off once per search to pick K_MID/K_END, so selecting a whole
     second table set on the SAME boolean costs one `pst.update` and no new
     condition. This is the cheap taper.
  2. **The stale-score rebuild is already there.** The landed root follows the
     swap with `pos = self.root = from_board(...)`, which was previously part
     of the taper's own cost. A taper now inherits it for zero bytes.

`K` is never tapered. Its two tables are the landed kend fix, and mirroring or
re-quantising it perturbs a measured +30.5 Elo fix -- so `K` rides in the exact
block (`exact="K"`) and the root keeps its own K_MID/K_END line untouched.

The eg data here is FILLER, so these are shape prices, not data prices:

  same      eg == mg              perfectly correlated -- MACHINERY ONLY
  perturbed mg + N(0, 12cp)       widens the value range, so it adds levels
  shuffled  mg's values permuted  the SAME multiset, zero correlation

FILLER IS NOT AN UPPER BOUND, which is what the previous version of this file
claimed ("a real eg table would share structure with mg and lzma would find
some of it"). Measured against the real fitted qseam tables through this same
builder, filler comes in 60-75 bytes CHEAP: fitted values are less round than
classic's hand-made ones, and that costs more than correlation saves -- the
same effect that made a plain refit +63 bytes rather than free. Read these as
a FLOOR for a fitted version.

`shuffled` is also consistently cheaper than `perturbed`, for the same reason
in miniature: permuting reuses the exact value multiset, while adding noise
widens lo..hi and buys extra levels in the mixed-radix pack.

usage: price_taper.py            # the standard sweep
       price_taper.py STEP [--mirror] [--exact]
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec  # noqa: E402
import measure  # noqa: E402
import splice  # noqa: E402

ROOT = measure.ROOT
PIECES = "PNBRQK"
SET = "PNBRQ"   # K is never in a second set; the kend fix owns it

# The landed root region. Unique in the file, and asserted before every splice.
ROOT_OLD = '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'

# Seam form: the existing queens-off boolean selects the whole table set.
ROOT_SEAM = '''        q = "Q" in pos.board and "q" in pos.board
        pst.update(pst1 if q else EGT)
        pst["K"] = K_MID if q else K_END
'''

# Blend form: a continuous 24-point phase, the more expressive alternative.
ROOT_BLEND = '''        ph = min(24, sum({"N": 1, "B": 1, "R": 2, "Q": 4}.get(c.upper(), 0)
                         for c in pos.board if c.isalpha()))
        for k in "PNBRQ":
            pst[k] = tuple(m + (e - m) * (24 - ph) // 24 for m, e in zip(pst1[k], EGT[k]))
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
'''


def fillers(raw, seed=20260813):
    """Three eg table sets bracketing the correlation a real fit would have."""
    rng = random.Random(seed)
    out = {"same": {k: list(v) for k, v in raw.items()}}
    out["perturbed"] = {k: [int(round(x + rng.gauss(0, 12))) for x in v] for k, v in raw.items()}
    out["shuffled"] = {k: rng.sample(list(v), len(v)) for k, v in raw.items()}
    return out


def block(tabs, dest, step, half, valname="piece"):
    """One extra decode block writing `dest`, reading its own piece-value dict.

    Variable names are NOT renamed between blocks: each block re-initialises
    `_v` and runs to completion at import time, so reuse is safe and it keeps
    the second block compressing against the first. (The old script renamed
    every temporary, which cost bytes for nothing.)
    """
    src = codec._block(tabs, SET, step, half, dest, init=True)
    if valname != "piece":
        src = src.replace("piece[_k]", valname + "[_k]")
    return src


def evalregion(raw, piece, egsets, step, half, kexact=True, valline=True):
    """The whole spliced eval region: mg (+ exact K) then one block per eg set.

    egsets is a list of (dest, tables, values) -- one entry for a taper, three
    for a 4-bucket skeleton. `values` may be None to share the mg piece dict,
    which is what a filler skeleton wants and what a real fit does NOT get:
    a second fitted set has its own piece values and pays for its own line.
    """
    src = codec.emit(piece, raw, step, half, exact="K" if kexact else "")
    # keep an unaliased copy of the mg set so the root can restore it: the
    # decode writes `pst` in place and `pst` is a module global reused across
    # searches, so without `pst1` a single endgame search would clobber it.
    old = " pst[_k] = tuple([0] * 20"
    assert src.count(old) >= 1
    src = src.replace(old, " pst[_k] = pst1[_k] = tuple([0] * 20", 1)
    src = src.replace("pst = {}\n", "pst = {}\npst1 = {}\n", 1)
    for i, entry in enumerate(egsets):
        dest, tabs = entry[0], entry[1]
        vals = entry[2] if len(entry) > 2 else None
        vn = "piece" if (vals is None and not valline) else "piece%d" % (i + 2)
        if vn != "piece":
            v = vals if vals is not None else piece
            src += "%s = {%s}\n" % (vn, ", ".join('"%s": %d' % (k, int(v[k])) for k in PIECES))
        src += block(tabs, dest, step, half, vn)
    return src


def build(base, egsets, rootcode, step, half, kexact=True, valline=True):
    piece, raw = splice.classic_tables()
    src = splice.splice(base, evalregion(raw, piece, egsets, step, half, kexact, valline))
    assert ROOT_OLD in src, "landed queens-off root line not found -- re-anchor again"
    return src.replace(ROOT_OLD, rootcode, 1)


def main():
    base = open(os.path.join(ROOT, splice.ENTRY)).read()
    _, cur = measure.pack(base, "cur")
    print("entry as landed: %d bytes, %d spare" % (cur, 4096 - cur))
    print("eg data is FILLER -- these are SHAPE prices, not data prices. No Elo claimed.\n")
    _, raw = splice.classic_tables()
    fill = fillers(raw)

    if len(sys.argv) > 1:
        rows = [(int(sys.argv[1]), "--mirror" in sys.argv, "--exact" not in sys.argv)]
    else:
        rows = [(1, False, True), (8, False, True), (8, True, True), (8, True, False)]

    # ONE-SET REFERENCES. The marginal cost of a second table set is only
    # readable against the same encoding with one set, never against the
    # landed 3378 entry -- quantisation alone already moves the base by ~160.
    piece, raw = splice.classic_tables()
    print("one-set references (classic tables, root unchanged):")
    ref = {}
    for step, half, kexact in rows:
        enc = "step %d%s%s" % (step, ", mirrored" if half else "", ", K exact" if kexact else "")
        src = splice.splice(base, codec.emit(piece, raw, step, half, exact="K" if kexact else ""))
        _, size = measure.pack(src, "ref")
        ref[(step, half, kexact)] = size
        print("  %-24s %6d bytes %7d spare %+8d vs entry" % (enc, size, 4096 - size, size - cur))
    print()

    print("%-9s %-24s %-10s %6s %7s %8s %9s  %s"
          % ("root", "encoding", "eg filler", "bytes", "spare", "vs entry", "vs 1-set", "standalone"))
    last = None
    for name, rootcode in (("seam", ROOT_SEAM), ("blend", ROOT_BLEND)):
        for step, half, kexact in rows:
            enc = "step %d%s%s" % (step, ", mirrored" if half else "", ", K exact" if kexact else "")
            for fname in ("same", "perturbed", "shuffled"):
                src = build(base, [("EGT", fill[fname])], rootcode, step, half, kexact)
                packed, size = measure.pack(src, "taper")
                bm = measure.standalone(packed)
                print("%-9s %-24s %-10s %6d %7d %+8d %+9d  %s"
                      % (name, enc, fname, size, 4096 - size, size - cur,
                         size - ref[(step, half, kexact)], bm))
                last = src
        print()
    # Decode cost is measured once, on the largest build: the TCEC startup
    # budget is 60 s and two exact table sets are the worst case here.
    print("decode of the largest build: %.2f ms (startup budget 60 s)"
          % measure.decode_time(last, "taper"))


if __name__ == "__main__":
    main()
