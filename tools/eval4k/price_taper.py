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

# RE-ANCHORED AGAIN 2026-08-17. `pend` landed (61b1a51) and rewrote this exact
# seam a second time: the single `pst["K"] = K_MID if ... else K_END` line the
# 2026-08-13 re-anchor targeted is gone, replaced by a three-line block that
# hoists the queens-off boolean into `end` and switches the PAWN table too. The
# script failed its own assert, so nothing in it could be reproduced -- the
# same failure mode, in the same file, that the 2026-08-13 header describes.
#
# Two consequences beyond the anchor text:
#   1. Every root form below MUST keep the pawn seam. pend is a landed,
#      confirmed +21.31 and a taper that silently drops it would be measured
#      against a baseline it no longer shares.
#   2. `splice` replaces the WHOLE eval region, which now contains the
#      P_MID/P_END derivation as well -- so `evalregion` has to emit it, or
#      every spliced build dies at the root with NameError. Priced builds are
#      run standalone by measure.py, so that would have been caught, but it is
#      cheaper to be right than to be caught.
ROOT_OLD = ('        end = "Q" not in pos.board or "q" not in pos.board\n'
            '        pst["K"] = K_END if end else K_MID\n'
            '        pst["P"] = P_END if end else P_MID\n')

# Seam form: the existing queens-off boolean selects the whole table set.
ROOT_SEAM = '''        end = "Q" not in pos.board or "q" not in pos.board
        pst.update(EGT if end else pst1)
        pst["K"] = K_END if end else K_MID
        pst["P"] = P_END if end else P_MID
'''

# Blend form: a continuous 24-point phase, the more expressive alternative.
# The phase expression is the one `ktap` uses (str.count per piece letter),
# not the per-character genexp the 2026-08-13 version wrote: same 24-point
# scale, and it is the form actually being screened, so the price is the
# price of a thing that exists.
ROOT_BLEND = '''        end = "Q" not in pos.board or "q" not in pos.board
        ph = min(24, sum(pos.board.count(c) * w
                         for c, w in zip("NnBbRrQq", (1, 1, 1, 1, 2, 2, 4, 4))))
        for k in "PNBRQ":
            pst[k] = tuple(e + (m - e) * ph // 24 for m, e in zip(pst1[k], EGT[k]))
        pst["K"] = K_END if end else K_MID
'''

# The pend derivation. codec.emit does not produce it (it predates the
# landing), so the spliced region has to carry it or the root breaks.
PEND_SRC = ('P_MID, P_END = pst["P"], tuple(x and x + (8 - i // 10) ** 2 * 2\n'
            "   for i, x in enumerate(pst[\"P\"]))\n")


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
    # pend rides in front of the K_MID/K_END line, exactly as the entry has it.
    kline = 'K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n'
    assert src.count(kline) == 1, "codec.emit no longer ends with the K_MID line"
    src = src.replace(kline, PEND_SRC + kline, 1)
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


def deltaregion(raw, piece, egtabs, step, half, dstep):
    """mg through the normal codec, then eg as a COARSE DELTA on top of mg.

    THE IDEA THIS PRICES. A second table set costs what its own entropy costs,
    and a fitted endgame table is NOT independent of the midgame one -- a rook
    is worth about the same on d4 in both. Storing eg as `mg + delta` at a
    coarse quantisation makes most of the 320 numbers zero, and a mixed-radix
    pack over few levels is where this codec is cheapest: 320 values at 5
    levels is 743 bits where 320 values at 210 levels is 2,468.

    `x and x + y` is pend's own trick, and it is load-bearing rather than
    cute: the padded tables carry 0 in the 40 border squares, no real square
    can be 0 (every entry includes its piece value, min 100), so it adds the
    delta on the board and leaves the border alone. Getting this wrong would
    put nonzero scores on squares the engine cannot reach -- harmless in this
    engine because padding is never indexed, and therefore exactly the kind of
    wrong that survives a smoke test.
    """
    src = codec.emit(piece, raw, step, half, exact="K" if kexact_default else "")
    kline = 'K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n'
    src = src.replace(kline, PEND_SRC + kline, 1)
    src = src.replace(" pst[_k] = tuple([0] * 20", " pst[_k] = pst1[_k] = tuple([0] * 20", 1)
    src = src.replace("pst = {}\n", "pst = {}\npst1 = {}\n", 1)

    # The delta is taken against the DECODED mg table, not the raw one: at
    # step > 1 the shipped mg is already rounded, and a delta against the
    # unrounded values would encode the mg quantisation error a second time.
    ns = {}
    exec(src, ns)
    mg = ns["pst1"]
    deltas = {}
    for k in SET:
        want = [egtabs[k][r * 8 + f] + piece[k] for r in range(8) for f in range(8)]
        have = [mg[k][20 + r * 10 + 1 + f] for r in range(8) for f in range(8)]
        deltas[k] = [int(round((w - h) / dstep)) for w, h in zip(want, have)]
    lo = min(min(v) for v in deltas.values())
    hi = max(max(v) for v in deltas.values())
    lvl = hi - lo + 1
    vals = [x - lo for k in SET for x in deltas[k]]
    src += codec.DEC % codec.enc(codec.mixed(vals, lvl))
    src += "EGT = {}\n"
    src += 'for _k in "%s":\n' % SET
    src += " _t = [(_v // %d ** _i %% %d + %d) * %d for _i in range(64)]\n" % (lvl, lvl, lo, dstep)
    src += " _v //= %d ** 64\n" % lvl
    src += (" EGT[_k] = tuple(x and x + y for x, y in zip(pst1[_k], [0] * 20 + sum("
            "([0] + _t[_i * 8:_i * 8 + 8] + [0] for _i in range(8)), []) + [0] * 20))\n")
    return src, lvl


kexact_default = True


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
    piece, raw = splice.classic_tables()
    fill = fillers(raw)

    if "--delta" in sys.argv:
        print("DELTA FORM: eg stored as mg + a coarse delta, mg at step 1, K exact.")
        print("%-9s %-12s %6s %6s %7s %8s %9s  %s"
              % ("root", "delta step", "levels", "bytes", "spare", "vs entry", "vs 1-set", "standalone"))
        src0 = splice.splice(base, codec.emit(piece, raw, 1, False, exact="K")
                             .replace('K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n',
                                      PEND_SRC + 'K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n', 1))
        _, ref1 = measure.pack(src0, "ref1")
        print("  one-set reference (step 1, K exact): %d bytes" % ref1)
        for name, rootcode in (("seam", ROOT_SEAM), ("blend", ROOT_BLEND)):
            for fname in ("same", "perturbed", "shuffled"):
                for dstep in (1, 4, 8, 16, 25):
                    region, lvl = deltaregion(raw, piece, fill[fname], 1, False, dstep)
                    src = splice.splice(base, region)
                    assert ROOT_OLD in src, "landed pend seam not found -- re-anchor again"
                    src = src.replace(ROOT_OLD, rootcode, 1)
                    packed, size = measure.pack(src, "delta")
                    bm = measure.standalone(packed)
                    print("%-9s %-12s %6d %6d %7d %+8d %+9d  %s"
                          % (name, "%s/%d" % (fname, dstep), lvl, size,
                             4096 - size, size - cur, size - ref1, bm))
            print()
        return

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
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
