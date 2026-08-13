"""Price the eval GRID under a byte BUDGET, not against a byte minimum.

The objective changed: the eval is to occupy 1024-1500 bytes, leaving ~2500 for
the engine. So the question is no longer "what is the cheapest table that does
not lose" but "what is the most capacity that DECODES inside 1024-1500 bytes
into (piece,square)-shaped tables the O(1) incremental eval can already read".

Everything here is measured by BUILDING: one real entry source per row through
tools/build/pack.sh, size off disk, run alone in an empty directory.

WHAT "EVAL BYTES" MEANS HERE
----------------------------
lzma carries one dictionary across the whole file, so no region has an
intrinsic size. The only honest definition is differential, against a build
that is identical except that it holds NO table data:

    eval bytes(X) = packed(entry with scheme X) - packed(entry with ZERO stub)

The zero stub still defines `piece`, `pst`, `K_MID` and `K_END` -- it is the
same engine with a flat evaluation -- so the difference is the table data plus
its decoder plus its root selector, which is exactly the thing being budgeted.
It is a difference of two real builds, never a slice of one.

THE GRID
--------
Sets are selected ONCE AT THE ROOT and fixed for the search, which is the only
form the incremental score permits (see price_kbucket.py). Two partitions,
composable into a product:

    seam    queens-off, the engine's existing boolean -- ~50 B of machinery
    wings   king-wing buckets, a position-global property -- ~32 B/bucket

`K` stays exact and unbucketed in every row: its two tables are the landed
kend fix, worth a measured +30.5 Elo, and a screen must not bundle it.

Filler data throughout, and filler is a FLOOR: real fitted tables measured
60-75 B/set MORE, because fitted values are less round than classic's.

usage: price_grid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec  # noqa: E402
import measure  # noqa: E402
import price_taper as pt  # noqa: E402
import splice  # noqa: E402

BUDGET = (1024, 1500)

# A build with the same engine and NO table data: the zero of the eval axis.
ZERO_STUB = '''piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {_k: tuple([0] * 20 + sum(([0] + [piece[_k]] * 8 + [0] for _i in range(8)), [])
                 + [0] * 20) for _k in "PNBRQK"}
K_MID, K_END = pst["K"], tuple(piece["K"] + 70
   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))
'''

# (label, seam, nwings) -> number of sets is (2 if seam else 1) * nwings
CONFIGS = [
    ("1 flat", False, 1),
    ("2 seam", True, 1),
    ("2 wings", False, 2),
    ("4 wings", False, 4),
    ("4 seam x wings2", True, 2),
    ("8 seam x wings4", True, 4),
]

# (step, half) -- capacity per set rises to the left, bytes per set to the right
ENCODINGS = [(8, True), (4, True), (2, True), (8, False), (4, False), (1, False)]


def selector(seam, nk, nsets):
    """Root code choosing one of `nsets` table sets from position-global facts.

    Set 0 is always `pst1`, the set the decoder already wrote into `pst`, so
    the common case costs a dict update of five keys and nothing else.
    """
    names = ", ".join(["pst1"] + ["T%d" % i for i in range(2, nsets + 1)])
    terms = []
    if seam:
        terms.append('(0 if "Q" in pos.board and "q" in pos.board else %d)' % nk)
    if nk == 2:
        # same-wing / opposite-wing: colour-symmetric, which a SHARED pst wants
        terms.append('((pos.board.index("K") % 10 > 5) != (pos.board.index("k") % 10 > 5))')
    elif nk == 4:
        terms.append('(pos.board.index("K") % 10 > 5) * 2 + (pos.board.index("k") % 10 > 5)')
    if not terms:
        return None
    idx = "\n                   + ".join(terms)
    return ("        pst.update((%s)[%s])\n" % (names, idx)
            + '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n')


def probe_selector(nsets):
    """A set-count BYTE PROBE, not a final partition: the seam x wings index
    folded onto `nsets` buckets. The semantically-final selectors are priced in
    the main grid above; this one exists so the set-count axis can be swept at
    counts that are not a product of 2 and 4, and its byte cost is the same
    handful of characters either way."""
    names = ", ".join(["pst1"] + ["T%d" % i for i in range(2, nsets + 1)])
    return ('        pst.update((%s)[((0 if "Q" in pos.board and "q" in pos.board else 4)\n'
            '                   + (pos.board.index("K") %% 10 > 5) * 2\n'
            '                   + (pos.board.index("k") %% 10 > 5)) %% %d])\n'
            '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'
            % (names, nsets))


def phase_selector(nsets, cuts):
    """The RECOMMENDED partition: phase quantiles, not king wings.

    King wings are catastrophically unbalanced in our data -- 80.4% of
    positions have the white king on the king side, so the both-queenside
    corner of a 4-wing product holds 981 training positions and the 8-set
    product's worst bucket holds 48. Phase quantiles are far better balanced,
    though NOT equal-count: phase is a coarse integer whose histogram is lumpy
    (2,300 training positions at phase 4 alone), so whole phase values cannot be
    split and the worst bucket still runs about half the average one.

    `cuts` is a 25-character string mapping phase 0..24 to a bucket index --
    cheaper than a chain of comparisons and it makes the quantile boundaries
    data, not code.
    """
    names = ", ".join(["pst1"] + ["T%d" % i for i in range(2, nsets + 1)])
    return ('        pst.update((%s)[int("%s"[min(24, sum(\n'
            '            {"N": 1, "B": 1, "R": 2, "Q": 4}.get(c.upper(), 0)\n'
            '            for c in pos.board if c.isalpha()))])])\n'
            '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'
            % (names, cuts))


def main():
    base = open(os.path.join(pt.ROOT, splice.ENTRY)).read()
    _, cur = measure.pack(base, "cur")
    _, zero = measure.pack(splice.splice(base, ZERO_STUB), "zero")
    piece, raw = splice.classic_tables()
    print("entry as landed: %d bytes | zero-eval stub: %d bytes" % (cur, zero))
    print("so the LANDED eval occupies %d bytes; the budget is %d-%d.\n"
          % (cur - zero, BUDGET[0], BUDGET[1]))
    print("filler data -- a FLOOR, fitted costs +60-75 B/set. No Elo claimed.\n")

    print("%-17s %-14s %5s %7s %8s %7s %9s  %s"
          % ("sets", "encoding", "sets", "params", "bytes", "spare", "EVAL B", "in budget"))
    rows = []
    for label, seam, nk in CONFIGS:
        nsets = (2 if seam else 1) * nk
        for step, half in ENCODINGS:
            egs = [("T%d" % (i + 2), pt.fillers(raw, 20260813 + i)["perturbed"])
                   for i in range(nsets - 1)]
            root = selector(seam, nk, nsets)
            if root is None:
                src = splice.splice(base, codec.emit(piece, raw, step, half, exact="K"))
            else:
                src = pt.build(base, egs, root, step, half, True, valline=False)
            packed, size = measure.pack(src, "grid")
            bm = measure.standalone(packed)
            assert bm.startswith("bestmove ") and len(bm.split()) == 2, bm
            ev = size - zero
            # free parameters: 5 tables/set, 32 values if mirrored else 64
            params = nsets * 5 * (32 if half else 64)
            fit = "IN BUDGET" if BUDGET[0] <= ev <= BUDGET[1] else (
                "over 4096" if size > 4096 else ("under" if ev < BUDGET[0] else "over budget"))
            print("%-17s step %d%-9s %5d %7d %8d %7d %9d  %s"
                  % (label, step, ", mirrored" if half else "", nsets, params,
                     size, 4096 - size, ev, fit))
            rows.append((label, step, half, nsets, params, size, ev))
        print()

    inb = [r for r in rows if BUDGET[0] <= r[6] <= BUDGET[1] and r[5] <= 4096]
    print("rows landing inside the %d-%d B eval budget: %d" % (BUDGET[0], BUDGET[1], len(inb)))
    for r in sorted(inb, key=lambda r: -r[4]):
        print("  %-17s step %d%-10s %5d sets %6d params %6d eval B  total %d"
              % (r[0], r[1], ", mirrored" if r[2] else "", r[3], r[4], r[6], r[5]))
    sweep(base, cur, zero)


def sweep(base, cur, zero):
    """How many sets actually fit, priced on FITTED-SHAPED data.

    The classic-derived filler above is a floor, and correcting it by the
    measured "+60-75 B per fitted set" would be composed arithmetic -- exactly
    what this lane does not do. So instead of a correction, a better BUILD: take
    the real fitted tables from `fits.json` and permute each one independently
    per set. That preserves the fitted value MULTISET -- the roundness that
    makes fitted data expensive -- while making every set distinct, and it needs
    no new fit. Still not a candidate's true price (a real fit correlates its
    sets, which lzma would find), but it brackets from the other side.
    """
    import json
    import random
    P = "PNBRQK"
    fj = os.path.join(pt.ROOT, "tools/tune/candidates/fits.json")
    if not os.path.exists(fj):
        return
    T = json.load(open(fj))["flat"]["tables"]
    vals = {p: T["_value_" + p] for p in P}
    fitted = {p: [int(v) for v in T[p]] for p in P}
    print("\nHOW MANY SETS FIT, on fitted-shaped data (permuted real fitted tables)")
    print("engine (zero-eval stub) = %d B, so the eval ceiling is %d B\n" % (zero, 4096 - zero))
    print("%-16s %5s %7s %8s %7s %9s  %s"
          % ("encoding", "sets", "params", "bytes", "spare", "EVAL B", "verdict"))
    for step, half in ((8, True), (4, True), (2, True)):
        for nsets in (1, 2, 3, 4, 5, 6, 7, 8):
            rng = random.Random(4096 + nsets)
            egs = [("T%d" % (i + 2), {p: rng.sample(fitted[p], 64) for p in P})
                   for i in range(nsets - 1)]
            root = probe_selector(nsets) if nsets > 1 else None
            if root is None:
                src = splice.splice(base, codec.emit(vals, fitted, step, half, exact="K"))
            else:
                src = pt.build(base, egs, root, step, half, True, valline=False)
                src = src.replace(codec.emit(vals, fitted, step, half, exact="K").split("\n")[0],
                                  "piece = {%s}" % ", ".join('"%s": %d' % (p, vals[p]) for p in P), 1)
            packed, size = measure.pack(src, "sweep")
            bm = measure.standalone(packed)
            assert bm.startswith("bestmove "), bm
            ev = size - zero
            v = ("OVER 4096" if size > 4096 else
                 "in budget" if BUDGET[0] <= ev <= BUDGET[1] else
                 "under budget")
            print("%-16s %5d %7d %8d %7d %9d  %s"
                  % ("step %d%s" % (step, ", mirr" if half else ""), nsets,
                     nsets * 5 * (32 if half else 64), size, 4096 - size, ev, v))
        print()
    recommended(base, zero, vals, fitted)


def recommended(base, zero, vals, fitted):
    """The recommended partition priced end to end: phase quantiles, mirrored
    step 8, K exact, on fitted-shaped data. Cut strings come from the real
    phase quantiles of the labelled set (see tools/tune/bucket_census.py)."""
    import random
    P = "PNBRQK"
    # phase 0..24 -> bucket, at the equal-count quantiles of set20260813
    # from tools/tune/bucket_census.py, TRAIN counts of set20260813
    CUTS = {
        4: "0000111222223333333333333",
        6: "0000122333444444455555555",
        7: "0001123344455555556666666",
        8: "0001233445556666666777777",
    }
    print("RECOMMENDED PARTITION: phase quantiles (balanced by construction)")
    print("%-8s %5s %7s %8s %7s %9s  %s"
          % ("sets", "enc", "params", "bytes", "spare", "EVAL B", "verdict"))
    for nsets in (4, 6, 7, 8):
        rng = random.Random(777 + nsets)
        egs = [("T%d" % (i + 2), {p: rng.sample(fitted[p], 64) for p in P})
               for i in range(nsets - 1)]
        src = pt.build(base, egs, phase_selector(nsets, CUTS[nsets]), 8, True, True, valline=False)
        src = src.replace(codec.emit(vals, fitted, 8, True, exact="K").split("\n")[0],
                          "piece = {%s}" % ", ".join('"%s": %d' % (p, vals[p]) for p in P), 1)
        packed, size = measure.pack(src, "rec")
        bm = measure.standalone(packed)
        assert bm.startswith("bestmove "), bm
        ev = size - zero
        v = ("OVER 4096" if size > 4096 else
             "IN BUDGET" if BUDGET[0] <= ev <= BUDGET[1] else "under budget")
        print("%-8d %5s %7d %8d %7d %9d  %s   %s"
              % (nsets, "st8m", nsets * 160, size, 4096 - size, ev, v, bm))


if __name__ == "__main__":
    main()
