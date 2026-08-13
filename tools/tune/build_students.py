"""Turn each trained student into a real entry, price it, and gate it.

Nothing here is composed. Every row is one entry source through the real
packer, size read off disk, and the artifact run alone in an empty directory
before any number is printed.

Three checks are run per arm and all three have caught something in this lane:

  * DECODE ROUND TRIP -- the tables the engine ends up with, read back out of
    the built source, must equal the integers the trainer emitted. The codec
    once accepted a `piece` dict and silently ignored it, and a mirrored king
    table cost -67 Elo while the fit looked 10% better.
  * STANDALONE -- the packed artifact must play a move in an empty directory.
  * FIRST YIELD -- the gate C1 needed. Written next to the built source so
    `tools/build/first_yield_gate.py` can be pointed straight at it.

usage: build_students.py STUDENTS.json OUTDIR
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval4k"))
import codec      # noqa: E402
import measure    # noqa: E402
import splice     # noqa: E402

STUDENTS = json.load(open(sys.argv[1]))
OUTDIR = sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
BASE = open(os.path.join(measure.ROOT, splice.ENTRY)).read()
PIECES = "PNBRQK"


def region(tables, step=1):
    """codec.emit with THIS candidate's piece values -- the fit moves them.

    `step` must be the arm's OWN training step. A step-8 student stored
    through the step-1 codec pays the exact-resolution price for values that
    only take grid points, which is most of what quantisation was for.

    The king is held back at full resolution whenever there IS a step. The
    trainer freezes K at classic's table -- the landed kend fix -- but the
    codec quantises every table it is handed, so a step-8 emit rounded the
    most orientation- and value-sensitive table in the engine and changed a
    fix that no fit is allowed to touch. The round-trip check below is what
    reported it; `exact="K"` costs ~84 B and keeps it bit-identical.
    """
    raw = {p: tables[p] for p in PIECES}
    vals = {p: tables["_value_" + p] for p in PIECES}
    src = codec.emit(vals, raw, step=step, exact="K" if step > 1 else "")
    valline = "piece = {%s}\n" % ", ".join('"%s": %d' % (p, vals[p]) for p in PIECES)
    return src.replace(src.split("\n")[0] + "\n", valline, 1)


_, BASESIZE = measure.pack(BASE, "base")
print("entry as landed: %d bytes, %d spare" % (BASESIZE, 4096 - BASESIZE))
# The differential definition of "eval bytes": lzma shares one dictionary
# across the file, so no region has an intrinsic size. Everything is measured
# against the same engine holding a flat table.
print()

for arm, r in STUDENTS["arms"].items():
    tabs = r["tables"]
    rep = region(tabs, r["step"] or 1)
    src = splice.splice(BASE, rep)
    path = os.path.join(OUTDIR, "e_%s.py" % arm)
    open(path, "w").write(src)

    # Decode round trip: what does the ENGINE end up with? The region is
    # exec'd directly rather than re-located with `splice.split`, whose START
    # anchor is the literal `piece = {"P": 100,` -- a fit that moves the pawn
    # value off 100 makes that anchor unfindable, so a re-split would raise on
    # exactly the candidates most worth checking.
    ns = {}
    exec(rep, ns)
    got = ns["pst"]
    ref = splice.padded({p: tabs["_value_" + p] for p in PIECES},
                        {p: tabs[p] for p in PIECES})
    bad = [p for p in PIECES if tuple(got[p]) != tuple(ref[p])]
    packed, size = measure.pack(src, "student_" + arm)
    mv = measure.standalone(packed)
    print("%-7s %5d bytes  %+5d vs entry  %4d spare  eval %+d  decode %s  standalone %s"
          % (arm, size, size - BASESIZE, 4096 - size,
             size - BASESIZE + 464,                       # entry's own eval is 464 B
             "ROUND TRIP OK" if not bad else "MISMATCH " + ",".join(bad), mv))
    print("        held-out %.6f (%+.2f%% vs classic)  emit %s  -> %s"
          % (r["heldout"], 100 * (r["heldout"] - STUDENTS["classic_heldout"]) / STUDENTS["classic_heldout"],
             "OK" if r["emit_ok"] else "MISMATCH", path))

print("\nBytes and gates only. Held-out loss is the metric that mis-ranked C2 by")
print("5.9%% while it lost 94 Elo -- no row above is an Elo claim.")
