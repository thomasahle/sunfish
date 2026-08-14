#!/usr/bin/env python3
"""Measure sunfish.py's node rate as a function of piece count, once.

The virtual-clock surrogate (vmatch.py) converts a time budget into a NODE
budget, so it needs one number the C twin cannot supply: how many nodes the
*Python* engine would have searched in that time.  That number is a property
of the reference engine and the host, not of the experiment, so it is
measured ONCE and shipped as data with provenance (npsmodel.json).

Two subcommands, because the two halves need different interpreters:

  positions   python3 + python-chess + sunfish_c.  Plays one deterministic
              twin self-game and samples the FENs it passes through, so the
              profile's positions are drawn from the distribution the
              surrogate actually meets (opening -> middlegame -> endgame),
              not from a hand-picked list.
  measure     pypy3 + sunfish.py.  Times a fixed node budget at each FEN.
  fit         python3.  Fits the piecewise-linear profile and writes
              npsmodel.json.

ESTIMATOR: the MAX over reps, not the mean.  A rep can only be slowed by
host load, never sped up, so the maximum is the least-disturbed sample of
the quantity we want ("how fast does this engine run when nothing is in the
way").  Means on a laptop that is simultaneously running matches measure the
laptop, not the engine.

WARMUP is a rep that is thrown away: pypy's JIT needs one pass over the
search before its numbers mean anything, and every move of a real game is
warm.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


# ---------------------------------------------------------------- positions --
def cmd_positions(args):
    import chess
    binary = os.path.join(HERE, "sunfish_c")
    tables = os.path.join(HERE, "tables_classic.txt")
    proc = subprocess.Popen([binary, tables], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)

    def ask(fen4, moves):
        pos = "position fen %s" % fen4
        if moves:
            pos += " moves " + " ".join(moves)
        proc.stdin.write(pos + "\n")
        proc.stdin.flush()
        while proc.stdout.readline().strip() != "ok":
            pass
        proc.stdin.write("go nodes %d\n" % args.nodes)
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("twin died")
            if line.startswith("bestmove"):
                return line.split()[1]

    # Several self-games, not one: a single deterministic game ends in a
    # repetition draw long before the piece count gets low, so the profile
    # would have no endgame knots at all (measured: one startpos game covers
    # 32 down to 18 pieces and stops).
    starts = [chess.STARTING_FEN]
    if args.starts and os.path.exists(args.starts):
        for line in open(args.starts):
            line = line.strip()
            if line and "/" in line.split()[0]:
                starts.append(line)
    samples = []
    for start in starts[:args.games]:
        board = chess.Board(start)
        fen4 = " ".join(board.fen().split()[:4])
        moves = []
        while len(moves) < args.plies and board.outcome(claim_draw=True) is None:
            if len(moves) % args.every == 0:
                samples.append((board.fen(), len(board.piece_map())))
            mv = ask(fen4, moves)
            if mv == "(none)":
                break
            board.push(chess.Move.from_uci(mv))
            moves.append(mv)
    proc.stdin.write("quit\n")
    proc.stdin.flush()

    # One position per distinct piece count, densest coverage first, then
    # trimmed to --count by keeping the widest spread.
    by_pieces = {}
    for fen, n in samples:
        by_pieces.setdefault(n, fen)
    keys = sorted(by_pieces, reverse=True)
    if len(keys) > args.count:
        idx = [round(i * (len(keys) - 1) / (args.count - 1)) for i in range(args.count)]
        keys = [keys[i] for i in sorted(set(idx))]
    out = [{"pieces": k, "fen": by_pieces[k]} for k in keys]
    json.dump(out, open(args.out, "w"), indent=1)
    print("%d positions, piece counts %s -> %s"
          % (len(out), out[0]["pieces"], out[-1]["pieces"]), file=sys.stderr)


# ------------------------------------------------------------------ measure --
def cmd_measure(args):
    """Run under pypy3.  Times `--nodes` nodes of sunfish.py at each FEN."""
    sys.path.insert(0, ROOT)
    import sunfish
    import sunfish_ui.uci as uci
    # uci.py injects its engine at run() time rather than importing one, so a
    # direct from_fen() call has to do the injection itself.
    uci.sunfish = sunfish
    from sunfish_ui.uci import WHITE, from_fen, get_color

    positions = json.load(open(args.positions))
    results = []
    for entry in positions:
        fen = entry["fen"]
        pos = from_fen(*fen.split()[:6])
        hist = [pos] if get_color(pos) == WHITE else [pos.rotate(), pos]
        times = []
        for rep in range(args.reps + 1):          # rep 0 is the warmup
            searcher = sunfish.Searcher()
            t0 = time.perf_counter()
            nodes = 0
            for depth, gamma, score, move in searcher.search(hist):
                nodes = searcher.nodes
                if nodes >= args.nodes:
                    break
            dt = time.perf_counter() - t0
            if rep:
                times.append(nodes / dt)
        results.append({"pieces": entry["pieces"], "fen": fen,
                        "nps": max(times), "reps": [round(t) for t in times]})
        print("pieces %2d  nps %8.0f  (reps %s)"
              % (entry["pieces"], max(times), [round(t) for t in times]),
              file=sys.stderr, flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


# ---------------------------------------------------------------------- fit --
def cmd_fit(args):
    """LINEAR in piece count, from the measured samples.

    Deliberately the simplest model that respects the one real effect: fewer
    pieces means fewer moves generated per node, so nps rises into the
    endgame.  It is a LINE and not an interpolation through every knot on
    purpose -- the per-position scatter is dominated by tactical density,
    which is not a function of piece count, so interpolating the knots would
    fit noise and then apply it to positions that never produced it.  The
    residual scatter is recorded, not hidden: it is the honest width of the
    node budget the surrogate hands the twin.
    """
    samples = json.load(open(args.samples))
    xs = [s["pieces"] for s in samples]
    ys = [s["nps"] for s in samples]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rms = (sum(r * r for r in resid) / n) ** 0.5
    sst = sum((y - my) ** 2 for y in ys)
    model = {
        "model": "linear-in-piece-count",
        "intercept": round(a, 1),
        "slope_per_piece": round(b, 2),
        "clamp_pieces": [min(xs), max(xs)],
        "knots": [[x, round(y, 1)] for x, y in sorted(zip(xs, ys))],
        "residual_rms_nps": round(rms, 1),
        "residual_rms_frac": round(rms / my, 3),
        "r2": round(1 - sum(r * r for r in resid) / sst, 3),
        "provenance": {
            "engine": "sunfish.py at %s" % args.rev,
            "interpreter": args.interpreter,
            "host": args.host,
            "node_budget_per_rep": args.nodes,
            "reps": args.reps,
            "estimator": "max over reps (least-disturbed sample)",
            "positions": "twin self-game sample, one per distinct piece count",
            "measured": args.date,
            "caveat": "a FRESH Searcher() per rep, so BOTH tables start "
                      "empty.  An in-game move is colder than that only in "
                      "tp_score (sunfish.py:516 clears it per search) and "
                      "WARMER in tp_move, which persists all game -- so this "
                      "profile understates in-game nps and the surrogate "
                      "searches slightly fewer nodes per virtual second than "
                      "pypy would.  It is the same understatement on both "
                      "arms of any A/B, so it compresses absolute depth "
                      "without biasing a comparison.",
        },
    }
    json.dump(model, open(args.out, "w"), indent=1)
    print("nps ~ %.0f %+.1f*pieces   R2 %.3f   residual RMS %.0f (%.0f%%)"
          % (a, b, model["r2"], rms, 100 * rms / my))


def nps_for(model, pieces):
    """Evaluate the shipped profile.  Used by vmatch.py and the tests.

    Clamped to the measured piece range: the model does not extrapolate past
    data it never saw (a bare king has no measurement behind it).
    """
    lo, hi = model["clamp_pieces"]
    p = min(max(pieces, lo), hi)
    return model["intercept"] + model["slope_per_piece"] * p


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("positions")
    p.add_argument("--nodes", type=int, default=20000)
    p.add_argument("--plies", type=int, default=140)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--starts", default=os.path.join(ROOT, "tests", "files", "chessathome_openings.fen"))
    p.add_argument("--out", default=os.path.join(HERE, "npspositions.json"))
    p.set_defaults(fn=cmd_positions)

    p = sub.add_parser("measure")
    p.add_argument("--positions", default=os.path.join(HERE, "npspositions.json"))
    p.add_argument("--nodes", type=int, default=25000)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=os.path.join(HERE, "npssamples.json"))
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("fit")
    p.add_argument("--samples", default=os.path.join(HERE, "npssamples.json"))
    p.add_argument("--out", default=os.path.join(HERE, "npsmodel.json"))
    p.add_argument("--rev", default="unknown")
    p.add_argument("--interpreter", default="pypy3")
    p.add_argument("--host", default="unknown")
    p.add_argument("--date", default="unknown")
    p.add_argument("--nodes", type=int, default=25000)
    p.add_argument("--reps", type=int, default=3)
    p.set_defaults(fn=cmd_fit)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
