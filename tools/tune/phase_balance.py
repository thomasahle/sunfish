"""Census and phase-balanced selection of training positions.

WHY THIS EXISTS. Two 384-parameter programmes died on the same instrument --
C2 at -93.8 (Stockfish depth 8) and d1 at -76.0 (our own search at 160k nodes)
-- and after the second one the cause was measured rather than guessed: the
POSITION MIX, not the teacher. In `set20260813` the four phase bands hold
43.1 / 30.6 / 11.3 / 15.0 percent of the positions, classic's loss against our
own teacher is ALREADY LOWEST at phase 18-24 (0.007962, its best band), and a
global least-squares fit therefore spends that band -- the one with no headroom
-- to buy the deep-endgame majority. d1 came out 7.47% WORSE than classic at
phase 18-24 on all 12 splits, and lost 76 Elo.

This tool builds the set where that trade is not available, by construction.

TWO SOURCES, and the second one is the whole point.

  a PGN directory   Walk the games, apply the sampling rule, harvest FENs.
  a labelled .npz   Take the positions (and their LABELS) straight out of a set
                    that has already been labelled.

The .npz source is what makes the phase-mix experiment cost nothing. A flat
draw fits inside `distill160k` -- every position in it is already labelled by
our own search at 160,000 nodes -- so re-balancing needs no teacher run at all,
and the resulting fit differs from d1 in the MIX and in absolutely nothing
else: same teacher, same labels, same features, same split rule, same model,
same encoding, same gates, same screen.

TWO MODES, and the order matters.

  census   Report the SUPPLY per band. Census is not selection: it says what a
           target mix could possibly be, and it is what you are allowed to look
           at BEFORE the target is written down.

  select   Draw the PRE-REGISTERED target. Refuses to run unless the target is
           passed explicitly, so the mix can never be quietly reverse-engineered
           from whatever happened to be available.

DETERMINISM. Selection is keyed on `sha256(seed + fen)` and nothing else -- not
on file order, not on a shuffle, not on how many games a PGN happened to hold.
Re-running on the same source gives the same set. This is the same keying the
trainer's split uses, and for the same reason: an index-based draw silently
stops being reproducible the moment the row count moves.

usage: phase_balance.py census SOURCE [EXCLUDE.npz ...]
       phase_balance.py select SOURCE OUT_STEM TARGET [EXCLUDE.npz ...]

SOURCE is a directory of *.pgn or a labelled .npz.
TARGET is "flat:N" or "0-5=N,6-11=N,12-17=N,18-24=N".
OUT_STEM gets .fen + .json, plus .npz when the source carried labels.
"""
import glob
import hashlib
import json
import os
import sys

import chess
import chess.pgn
import numpy as np

# The trainer's phase weights, verbatim (4ku's). Any divergence here and the
# bands this set is balanced across stop being the bands the loss is reported
# in, which is the whole point of the exercise.
PHASE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
BANDS = ((0, 5), (6, 11), (12, 17), (18, 24))
SEED = 20260814
# Sampling rule for the PGN source, byte-identical to texel_data.py.
PLY_MIN, PLY_STRIDE, PIECES_MIN = 10, 7, 6


def phase_of_fen(fen):
    return min(sum(PHASE[p.symbol().upper()]
                   for p in chess.Board(fen).piece_map().values()), 24)


def band_of(ph):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= ph <= hi: return i
    raise AssertionError("phase %r outside every band" % ph)


def key(fen):
    """The draw order for a position -- a function of the FEN alone."""
    return hashlib.sha256((str(SEED) + fen).encode()).hexdigest()


def excluded(paths):
    out = set()
    for p in paths:
        d = np.load(p, allow_pickle=False)
        out |= {str(f) for f in d["fens"]}
        d.close()
    return out


def harvest_pgn(pgndir, exclude):
    """Every position the sampling rule admits, deduped, minus the spent ones.

    The whole corpus is walked. texel_data.py stops at NPOS*3 candidates, which
    is fine when the draw is uniform and fatal when it is stratified: the bands
    are not uniformly distributed through a game, so an early stop truncates
    the rarest band hardest -- exactly the band this set exists to protect.
    """
    pgns = sorted(glob.glob(os.path.join(pgndir, "*.pgn")))
    assert pgns, "no *.pgn in %s" % pgndir
    seen, rows, dropped = set(), [], {"dup": 0, "spent": 0}
    for path in pgns:
        n_games = 0
        with open(path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None: break
                n_games += 1
                board = g.board()
                for ply, mv in enumerate(g.mainline_moves()):
                    board.push(mv)
                    if ply < PLY_MIN or ply % PLY_STRIDE or board.is_check(): continue
                    if len(board.piece_map()) < PIECES_MIN: continue
                    fen = board.fen()
                    if fen in seen: dropped["dup"] += 1; continue
                    seen.add(fen)
                    if fen in exclude: dropped["spent"] += 1; continue
                    rows.append((fen, None))
        print("  %-46s %5d games" % (os.path.basename(path), n_games), flush=True)
    return rows, dropped, None


def harvest_npz(path, exclude):
    """Positions out of an already-labelled set, carrying their row index."""
    d = np.load(path, allow_pickle=False)
    fens = [str(f) for f in d["fens"]]
    rows, dropped = [], {"dup": 0, "spent": 0}
    seen = set()
    for i, fen in enumerate(fens):
        if fen in seen: dropped["dup"] += 1; continue
        seen.add(fen)
        if fen in exclude: dropped["spent"] += 1; continue
        rows.append((fen, i))
    print("  %-46s %5d rows" % (os.path.basename(path), len(fens)), flush=True)
    return rows, dropped, d


def bucket(rows):
    by_band = [dict() for _ in BANDS]
    for fen, idx in rows:
        by_band[band_of(phase_of_fen(fen))][fen] = idx
    return by_band


def report(by_band, dropped, exclude):
    tot = sum(len(b) for b in by_band)
    print("\nSUPPLY after dedup and after removing %d already-spent positions"
          % dropped["spent"])
    print("  (%d duplicate FENs collapsed; exclusion list held %d)"
          % (dropped["dup"], len(exclude)))
    print("\n  band      available    share")
    for (lo, hi), b in zip(BANDS, by_band):
        print("  %-8s %9d  %6.1f%%" % ("%d-%d" % (lo, hi), len(b),
                                       100 * len(b) / tot if tot else 0))
    print("  %-8s %9d" % ("TOTAL", tot))
    print("  a FLAT draw is capped by the thinnest band: %d per band, %d total"
          % (min(len(b) for b in by_band), 4 * min(len(b) for b in by_band)))
    return tot


def parse_target(spec):
    if spec.startswith("flat:"):
        return [int(spec.split(":", 1)[1])] * len(BANDS)
    want = dict(kv.split("=") for kv in spec.split(","))
    return [int(want["%d-%d" % (lo, hi)]) for lo, hi in BANDS]


def main():
    mode, source = sys.argv[1], sys.argv[2]
    argv = sys.argv[3:]
    out = tgt = None
    if mode == "select":
        out, tgt, argv = argv[0], argv[1], argv[2:]
    excl_paths = argv or []

    exclude = excluded(excl_paths)
    print("excluding %d FENs from %s"
          % (len(exclude), ", ".join(map(os.path.basename, excl_paths)) or "nothing"))
    print("source: %s" % source, flush=True)
    rows, dropped, src = (harvest_npz(source, exclude) if source.endswith(".npz")
                          else harvest_pgn(source, exclude))
    by_band = bucket(rows)
    report(by_band, dropped, exclude)

    if mode == "census":
        print("\nCENSUS ONLY. Nothing selected. Write the target mix down before"
              " running `select`.")
        return

    want = parse_target(tgt)
    short = [("%d-%d" % BANDS[i], want[i], len(by_band[i]))
             for i in range(len(BANDS)) if len(by_band[i]) < want[i]]
    if short:
        # Silently returning a smaller band would produce a set that is not the
        # pre-registered mix while looking exactly like one.
        raise SystemExit("TARGET NOT SATISFIABLE: " + "; ".join(
            "band %s wants %d, supply has %d" % s for s in short))

    picked = []
    for i, b in enumerate(by_band):
        picked += sorted(b, key=lambda f: key(f))[:want[i]]
    picked.sort(key=key)                      # one deterministic order overall

    open(out + ".fen", "w").write("".join(f + "\n" for f in picked))
    meta = {
        "seed": SEED, "target": tgt,
        "per_band": dict(zip(["%d-%d" % b for b in BANDS], want)),
        "selected": len(picked),
        "source": os.path.basename(source),
        "excluded_sets": [os.path.basename(p) for p in excl_paths],
        "excluded_fens": len(exclude),
        "sampling": ("rows of a labelled set" if source.endswith(".npz") else
                     "ply>=%d, every %dth ply, not in check, >=%d pieces"
                     % (PLY_MIN, PLY_STRIDE, PIECES_MIN)),
        "order": "sha256(seed+fen), lowest first, per band then overall",
        "phase_weights": PHASE, "bands": ["%d-%d" % b for b in BANDS],
        "sha256_of_selection": hashlib.sha256("\n".join(picked).encode()).hexdigest(),
    }

    # When the source carried labels, carry them through. Re-labelling
    # positions that are already labelled by the same teacher would be pure
    # cost, and would introduce a second labelling run as a difference between
    # this set and the one it is being compared against.
    if src is not None:
        idx = {f: i for f, i in ((f, by_band[band_of(phase_of_fen(f))][f]) for f in picked)}
        order = np.array([idx[f] for f in picked])
        arrays = {k: src[k][order] for k in ("X", "y", "fens")}
        assert [str(f) for f in arrays["fens"]] == picked, "row gather lost the order"
        m = json.loads(str(src["meta"]))
        m.update({"rebalanced": meta, "kept": len(picked),
                  "parent": os.path.basename(source)})
        np.savez_compressed(out + ".npz", meta=np.array(json.dumps(m, indent=1)), **arrays)
        print("\nwrote %s.npz: X %s, labels carried from %s (no relabelling)"
              % (out, arrays["X"].shape, os.path.basename(source)))
        src.close()

    json.dump(meta, open(out + ".json", "w"), indent=1)
    print("wrote %s.fen: %d positions, %s" % (out, len(picked), meta["per_band"]))
    print("selection sha256 %s" % meta["sha256_of_selection"][:16])


if __name__ == "__main__":
    main()
