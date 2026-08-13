"""Price the tapered (mg/eg) PST *shape* on the real packer.

Not a proposal to land: a price. The ledger priced tapering at "~300-400 B
for the second table plus ~100 B of accumulator threading" and declined it.
Two things change that arithmetic:

  1. Through the startup decoder a 384-value table costs 395 B exact / 248 B
     at step 8 / 134 B mirrored at step 8 -- not ~350-400 B of literal.
  2. The second accumulator is NOT needed.  `search()` already swaps
     `pst["K"]` between K_MID and K_END at the root, so the engine already
     has a per-search table rebuild; tapering is the same mechanism with a
     phase instead of a boolean, and the tables stay fixed for the whole
     search exactly as the comment there requires.

The eg table used here is a PERTURBED COPY of classic's, i.e. data with the
same distribution but no correlation to the mg table.  That makes the data
figure an UPPER BOUND: a real eg table would share structure with mg and
lzma would find some of it.

usage: price_taper.py [step] [--mirror]
"""
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec  # noqa: E402
import measure  # noqa: E402
import splice  # noqa: E402

ROOT = measure.ROOT
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HALF = "--mirror" in sys.argv

PIECE, RAW = splice.classic_tables(open(os.path.join(ROOT, splice.ENTRY)).read())
rng = random.Random(20260813)
EG = {k: [v[rng.randrange(64)] for _ in range(64)] for k, v in RAW.items()}


def emit_two():
    """Two tables through one codec pass, plus the phase blend at the root."""
    mg = codec.emit(PIECE, RAW, STEP, HALF)
    eg = codec.emit(PIECE, EG, STEP, HALF)
    # keep only the payload+decode of the second, renaming pst -> EGT
    eg = eg[eg.index("_v=0"):eg.index('K_MID, K_END')]
    eg = eg.replace("_v", "_w").replace("_c", "_e").replace("_d", "_f")
    eg = eg.replace("_k", "_g").replace("_t", "_u").replace("_h", "_x")
    eg = eg.replace("_r", "_y").replace("_q", "_z").replace("_i", "_j")
    eg = eg.replace("pst = {}", "EGT = {}").replace("pst[_g]", "EGT[_g]")
    src = mg[: mg.index("K_MID, K_END")] + eg
    src += "MGT = dict(pst)\n"
    src += 'PH = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}\n'
    return src


TAPER_SEARCH = '''        # Phase-blended tables, rebuilt once per search (same mechanism as
        # the bare-king swap this replaces, with a phase instead of a bool).
        ph = min(24, sum(PH[c.upper()] for c in pos.board if c.isalpha()))
        for k in "PNBRQK":
            pst[k] = tuple(m + (e - m) * (24 - ph) // 24 for m, e in zip(MGT[k], EGT[k]))
        # The carried score was accumulated under the PREVIOUS tables; under
        # new tables it is stale by up to 134 cp (measured, KRK). Rebuild it.
        history[-1] = pos = self.root = from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp)
'''


def build():
    src = open(os.path.join(ROOT, splice.ENTRY)).read()
    src = splice.splice(src, emit_two())
    old = ("        bare = sum(c.isupper() for c in pos.board) == 1 "
           "or sum(c.islower() for c in pos.board) == 1\n"
           '        pst["K"] = K_END if bare else K_MID\n')
    assert old in src, "bare-king swap not found"
    src = src.replace(old, TAPER_SEARCH, 1)
    # the search reads pos before the swap; it already does (`pos = self.root =
    # history[-1]` sits above), so no reordering is needed.
    return src


def main():
    base = open(os.path.join(ROOT, splice.ENTRY)).read()
    _, cur = measure.pack(base, "cur")
    src = build()
    packed, size = measure.pack(src, "taper")
    print("current entry           %d bytes (%d spare)" % (cur, 4096 - cur))
    print("tapered, step %-2d %-7s %d bytes (%d spare)  delta %+d" %
          (STEP, "mirrored" if HALF else "full", size, 4096 - size, size - cur))
    print("decode %.2f ms" % measure.decode_time(src, "taper"))
    print("standalone:", measure.standalone(packed))
    open("/tmp/taper.packed", "wb").write(open(packed, "rb").read())
    os.chmod("/tmp/taper.packed", 0o755)
    print("artifact kept at /tmp/taper.packed")


if __name__ == "__main__":
    main()
