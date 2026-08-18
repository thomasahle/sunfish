#!/usr/bin/env python3
"""Reweight a polyglot book from ``attribute.py`` statistics.

THE RULE, in full, because a reweighting rule that is not written down is not a
measurement instrument.  For each node (position key) of the SOURCE book, with
entries ``i = 1..m``, source weights ``w_i`` and prior ``p_i = w_i / sum(w)``:

    n_i  = games credited to move i          (from the stats file)
    s_i  = wins + 0.5 * draws, mover's POV   (from the stats file)

    theta_i = (s_i + alpha * mu) / (n_i + alpha),    mu = 0.5
    what_i  proportional to  p_i * theta_i

``theta_i`` is the Dirichlet/Beta posterior mean of the move's score rate under
a Beta(alpha/2, alpha/2) prior centred on an even score.  Shrinkage is toward
**mu = 0.5, a fixed unestimated constant**, not toward the node's own observed
mean: the node mean would be estimated from the same games it is used to judge,
and a level shift common to every sibling cancels in the renormalisation
anyway.  The choice is deliberately the conservative one.

ALPHA IS CALIBRATED, NOT PICKED.  Take two siblings, one scoring 100% and one
scoring 0% over ``n`` games each -- the loudest signal ``n`` games can produce.
Their weight ratio moves by exactly ``1 + 2n/alpha``.  Setting

    alpha = 2 * N_min

makes that factor exactly **2x at n = N_min**: N_min games of a maximally
separated result buys one doubling, and a realistic 55/45 split at n = N_min
buys about 7%.  The default ``alpha = 60`` is ``N_min = 30``, which is the
per-cell sample size a few-hundred-game tournament can actually deliver.

Then two guards:

* **Exploration floor.**  No entry may end below ``--floor`` (default 2%) of its
  node's mass, applied by water-filling.  A book that prunes a line to zero can
  never learn that it was wrong, and the floor is also what bounds the shift:
  the largest weight ratio the rule can ever produce at a node is
  ``(1 - (m-1)*f) / f``.  Where the floor is infeasible (``m * f > 1``) it
  degrades to ``1/m``, i.e. uniform.
* **Identity where there is no information.**  A node with no credited games at
  all, or whose computed multipliers are all 1, is copied through
  **byte-for-byte**.  Empty stats therefore reproduce the source book bit-for-bit.

Quantisation to the polyglot ushort is per node:
``W_i = clip(round(what_i * S), 1, 65535)`` with ``S = max(sum(w_i), --scale)``,
so a source book of all-1 weights (``polyglot make-book -uniform``) gets the
headroom it needs to express a tilt at all.

Everything else in the file -- key order, entry order within a key, the ``learn``
field -- is preserved exactly, so a diff of two books is a diff of weights.

Usage::

    tools/book/rebuild.py --book src.bin --stats stats.json --out v1.bin
"""

import argparse
import hashlib
import json
import struct
import sys

ENTRY = struct.Struct(">QHHI")
VERSION = 1
MAX_USHORT = 65535
PROMO = " nbrq"
SQUARES = ["abcdefgh"[s & 7] + "12345678"[s >> 3] for s in range(64)]


def read_raw(path):
    """Return the book as a list of (key, raw_move, weight, learn), in file order."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % ENTRY.size: raise SystemExit("%s: %d bytes is not a multiple of 16" % (path, len(data)))
    return [ENTRY.unpack_from(data, i) for i in range(0, len(data), ENTRY.size)]


def write_raw(path, entries):
    """Write entries to a polyglot book.  Keys must be non-decreasing (the format's index)."""
    keys = [e[0] for e in entries]
    if any(b < a for a, b in zip(keys, keys[1:])): raise SystemExit("entries are not sorted by key")
    with open(path, "wb") as f:
        for key, mv, w, learn in entries:
            f.write(ENTRY.pack(key, mv, max(0, min(MAX_USHORT, int(w))), learn))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_move_uci(raw):
    """Polyglot's move encoding -> UCI, in the book's own (king-takes-rook) spelling."""
    to_sq, from_sq, promo = raw & 0x3F, (raw >> 6) & 0x3F, (raw >> 12) & 0x7
    return SQUARES[from_sq] + SQUARES[to_sq] + (PROMO[promo] if promo else "")


def apply_floor(w, floor):
    """Water-fill the normalised list `w` so every element is at least `floor`.

    Elements pushed up to the floor are pinned there; the rest are rescaled to
    fill what is left.  Repeating until nothing new falls below the floor is
    what makes it a fixed point rather than one pass.
    """
    f = min(floor, 1.0 / len(w))
    if f <= 0: return list(w)
    w, pinned = list(w), set()
    for _ in range(len(w)):
        newly = [i for i, x in enumerate(w) if i not in pinned and x < f]
        if not newly: break
        pinned |= set(newly)
        for i in newly: w[i] = f
        free = [i for i in range(len(w)) if i not in pinned]
        mass = sum(w[i] for i in free)
        if free and mass > 0:
            k = (1.0 - f * len(pinned)) / mass
            for i in free: w[i] *= k
    return w


def reweight_node(entries, moves, alpha, floor, scale, mu=0.5):
    """entries: [(key, raw_move, weight, learn)] for ONE key.  moves: {uci_spelling: (n, s)}.

    Returns the new weight list, or None when the node must pass through unchanged.
    """
    total_games = sum(n for n, _ in moves.values())
    if total_games == 0: return None
    src = [float(e[2]) for e in entries]
    tot = sum(src)
    if tot <= 0: return None
    prior = [x / tot for x in src]
    theta = []
    for e in entries:
        n, s = moves.get(raw_move_uci(e[1]), (0, 0.0))
        theta.append((s + alpha * mu) / (n + alpha))
    post = [p * t for p, t in zip(prior, theta)]
    z = sum(post)
    if z <= 0: return None
    post = apply_floor([x / z for x in post], floor)
    # No information moved: pass the node through byte-for-byte rather than requantise it.
    if all(abs(x - p) <= 1e-9 * max(p, 1e-12) for x, p in zip(post, prior)): return None
    s_node = max(tot, float(scale))
    return [max(1, min(MAX_USHORT, int(round(x * s_node)))) for x in post]


def rebuild(src_entries, stats, alpha, floor, scale, min_games=0):
    """Returns (new_entries, per-node report rows)."""
    by_key = {}
    for node in stats["nodes"]:
        # Join on polyglot's own move spelling; `uci` is the human one and differs for castling.
        by_key[int(node["key"], 16)] = ({m.get("raw", m["uci"]): (m["games"], m["score"]) for m in node["moves"]},
                                        node)
    out, report, matched, i = [], [], set(), 0
    while i < len(src_entries):
        j = i
        while j < len(src_entries) and src_entries[j][0] == src_entries[i][0]: j += 1
        group = src_entries[i:j]
        moves, node = by_key.get(group[0][0], ({}, None))
        if node is not None: matched.add(group[0][0])
        new = None
        if sum(n for n, _ in moves.values()) >= max(1, min_games):
            new = reweight_node(group, moves, alpha, floor, scale)
        if new is None:
            out.extend(group)
        else:
            out.extend((k, mv, nw, ln) for (k, mv, _, ln), nw in zip(group, new))
            tot_new = float(sum(new)) or 1.0
            tot_old = float(sum(e[2] for e in group)) or 1.0
            report.append({
                "key": "%016x" % group[0][0], "fen": node.get("fen") if node else None,
                "games": sum(n for n, _ in moves.values()),
                "moves": [{"uci": raw_move_uci(e[1]), "games": moves.get(raw_move_uci(e[1]), (0, 0.0))[0],
                           "score": moves.get(raw_move_uci(e[1]), (0, 0.0))[1],
                           "p_old": e[2] / tot_old, "p_new": nw / tot_new}
                          for e, nw in zip(group, new)],
            })
        i = j
    return out, report, len(by_key) - len(matched)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True, help="source polyglot .bin")
    ap.add_argument("--stats", required=True, help="attribute.py JSON ('-' for an empty/identity rebuild)")
    ap.add_argument("--out", required=True, help="destination .bin")
    ap.add_argument("--alpha", type=float, default=60.0, help="Dirichlet pseudo-count; 2*N_min (default 60 = N_min 30)")
    ap.add_argument("--floor", type=float, default=0.02, help="exploration floor as a share of node mass (default 0.02)")
    ap.add_argument("--scale", type=int, default=1000, help="per-node quantisation headroom (default 1000)")
    ap.add_argument("--min-games", type=int, default=0, help="nodes with fewer credited games pass through unchanged")
    ap.add_argument("--report", help="write a JSON report of every node that moved")
    args = ap.parse_args(argv)

    src = read_raw(args.book)
    if args.stats == "-":
        stats = {"nodes": []}
    else:
        with open(args.stats) as f: stats = json.load(f)
    if stats["nodes"] and stats.get("meta", {}).get("book_sha256") not in (None, sha256(args.book)):
        print("warning: stats were attributed against a different book (%s)" % stats["meta"]["book_sha256"][:16],
              file=sys.stderr)
    out, report, unmatched = rebuild(src, stats, args.alpha, args.floor, args.scale, args.min_games)
    if unmatched: print("warning: %d stats nodes are not in this book" % unmatched, file=sys.stderr)
    write_raw(args.out, out)
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"meta": {"tool": "rebuild.py", "version": VERSION, "book": args.book,
                                "book_sha256": sha256(args.book), "stats": args.stats,
                                "alpha": args.alpha, "floor": args.floor, "scale": args.scale,
                                "min_games": args.min_games, "out": args.out, "out_sha256": sha256(args.out),
                                "entries": len(out), "nodes_changed": len(report),
                                "stats_nodes_not_in_book": unmatched},
                       "nodes": report}, f, indent=1)
    print("%s: %d entries, %d nodes reweighted, sha256 %s" % (args.out, len(out), len(report), sha256(args.out)),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
