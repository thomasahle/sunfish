"""Distillation labels: the value OUR OWN SEARCH converges to, not Stockfish's.

Why this replaces `texel_data.py`'s labeller, on the record of this lane:

  * Static SF-depth-8 loss is ANTI-CORRELATED with strength here. C2 fit that
    objective 5.9% better than classic on held-out data and played -93.8 +/-
    32.7 over 405 games. The held-out metric mis-ranks, so it cannot be the
    thing we optimise.
  * SF's centipawn is not our centipawn. A fit with free piece values that
    regresses our tables onto another engine's scale absorbs the scale
    mismatch into the parameters we ship.
  * The search's converged value IS the quantity the engine maximises. A leaf
    table that predicts it makes the leaf agree with the root, which is the
    only agreement that shows up in games.

The label is the score of the LAST COMPLETED DEPTH -- the same commit rule
`main()` uses for the move it plays (a mid-depth fail-high can come from a
deep fail-low dive probe at an absurd gamma). The MTD bracket is tracked here
exactly as `Searcher.search` tracks it, so the number is the engine's own, not
a re-derivation.

THE TT MUST NOT CARRY. `texel_data.py` learned this from Stockfish: the same
FEN scored -14 in one slot and -22 in another because the hash carried across
positions, and both modes were run-to-run reproducible, which is why it was
invisible. `Searcher.search` clears `tp_score` but NOT `tp_move`, and move
ordering changes the tree, so a fresh `Searcher` is built per position and
`--shuffle-control` re-labels a subset in a different order and asserts the
labels are bit-identical.

usage: distill_label.py FENS.txt OUT.jsonl NODES [SHARD] [NSHARDS]
       distill_label.py FENS.txt OUT.jsonl --sweep N1,N2,... [SHARD] [NSHARDS]
"""
import hashlib
import json
import os
import pathlib
import platform
import sys
import time

REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/nnue_4k")
sys.path.insert(0, REPO)
import pst_entry as E                                              # noqa: E402
import sunfish_ui.uci as uci                                       # noqa: E402

# The driver is imported for from_fen ONLY, and it reads its engine from a
# module global that run() would normally set. Set it explicitly: an unset
# `sunfish` here would fall through to a *different* engine's board layout,
# which is the same class of bug as the stale-driver incident.
uci.sunfish = E


def parse_args(argv):
    fens_path, out_path = argv[1], argv[2]
    if argv[3] == "--sweep":
        nodes = [int(x) for x in argv[4].split(",")]
        rest = argv[5:]
    else:
        nodes = [int(argv[3])]
        rest = argv[4:]
    shard = int(rest[0]) if rest else 0
    nshards = int(rest[1]) if len(rest) > 1 else 1
    return fens_path, out_path, nodes, shard, nshards


def label(fen, max_nodes):
    """Return (white-POV cp, depth completed, nodes, first_yield_nodes, flag).

    `flag` is "" on a normal label, or a word naming why the position has no
    ordinary value (mate/stalemate seen at the root, or no depth completed
    inside the budget).
    """
    board, color, castling, enpas = fen.split()[:4]
    pos = uci.from_fen(board, color, castling, enpas, 0, 1)
    s = E.Searcher()                       # fresh: no tp_move carry-over
    s.node_cap = max_nodes
    s.deadline = 1 << 63                   # nodes are the only budget here
    lower, upper = 1 - E.MATE_UPPER, E.MATE_UPPER
    cur = 1
    done_score, done_depth, flag = None, 0, "nodepth"
    first_yield = None
    try:
        for depth, gamma, score, move in s.search([pos]):
            if depth != cur:
                # previous depth converged; its bracket is the committed value
                if lower > 1 - E.MATE_UPPER or upper < E.MATE_UPPER:
                    lo = lower if lower > 1 - E.MATE_UPPER else upper
                    hi = upper if upper < E.MATE_UPPER else lower
                    done_score, done_depth, flag = (lo + hi) // 2, cur, ""
                lower, upper, cur = 1 - E.MATE_UPPER, E.MATE_UPPER, depth
            if score >= gamma:
                lower = max(lower, score)
                if first_yield is None:
                    # A root fail-high WITH a move is the first playable
                    # answer -- exactly what main() turns into `cand`, and
                    # what its absence turns into `bestmove (none)`.
                    if move is not None: first_yield = s.nodes
            else:
                upper = min(upper, score)
    except E.Stop:
        pass
    if done_score is not None and abs(done_score) >= E.MATE_LOWER:
        flag = "mate"
    # SF-labelled sets store WHITE POV; keep the convention so the two sets
    # are directly comparable position for position.
    if done_score is not None and color == "b":
        done_score = -done_score
    return done_score, done_depth, s.nodes, first_yield, flag


def main():
    fens_path, out_path, node_list, shard, nshards = parse_args(sys.argv)
    fens = [ln.strip() for ln in open(fens_path) if ln.strip()]
    fens = fens[shard::nshards]
    src = open(REPO + "/nnue_4k/pst_entry.py", "rb").read()
    meta = {
        "teacher": "nnue_4k/pst_entry.py",
        "teacher_sha256": hashlib.sha256(src).hexdigest(),
        "teacher_version": E.version,
        "nodes": node_list,
        "interpreter": platform.python_implementation() + " " + platform.python_version(),
        "host": platform.node().split(".")[0],
        "shard": [shard, nshards],
        "positions": len(fens),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tt_per_position": "fresh Searcher, no tp_move carry-over",
        "label": "score of last completed depth, MTD bracket midpoint, white POV",
    }
    out = open(out_path, "w")
    out.write(json.dumps({"meta": meta}) + "\n")
    out.flush()
    t0 = time.time()
    for i, fen in enumerate(fens):
        rec = {"fen": fen}
        for n in node_list:
            cp, d, used, fy, flag = label(fen, n)
            rec["n%d" % n] = {"cp": cp, "depth": d, "nodes": used,
                              "first_yield": fy, "flag": flag}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if i and i % 50 == 0:
            r = (i + 1) / (time.time() - t0)
            print("  %d/%d  %.2f pos/s  ETA %.1f min"
                  % (i + 1, len(fens), r, (len(fens) - i - 1) / r / 60), flush=True)
    out.close()
    print("wrote %s: %d positions, %.1f min"
          % (out_path, len(fens), (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
