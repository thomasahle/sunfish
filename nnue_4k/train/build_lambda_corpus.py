#!/usr/bin/env python3
"""Build the SHARED corpus for the lambda sweep: positions + OUTCOMES + cp.

One corpus, three arms.  The whole point of the lambda experiment is that
only the label blend moves, so every arm must see byte-identical positions;
that is why this is a single build emitting both label channels rather than
three builds.

WHY IT HAD TO EXIST.  `cache_repl8M.pkl` is `FEATS/OFFS/PSTC/Y/KB` and, in
data.py's own words, "carries no FENs" -- no outcome, and no key to join one
back with.  Our archive PGNs are the mirror image: `[Result]` is there but the
movetext carries `{book}` comments only, because our packed engines emit no
`info` lines (the adjudication-inert finding, cutting the other way).  So
outcomes come from the PGN and cp comes from a fresh twin search.

REUSED FROM tools/tune/texel_data.py, which parsed our match PGNs all
campaign -- ideas and lessons, not a fork:
  * python-chess for replay, a FEN set for dedup, sparse sampling to
    decorrelate consecutive plies;
  * THE TT LESSON, which is the important one: `ucinewgame` + `isready`
    before every position.  Without it the transposition table carries over
    and the same FEN scores differently depending on what preceded it (that
    file measured -14 vs -22 on one position at depth 8, and 83 -> 97 on
    another).  A label must be a function of (fen, depth, engine) alone.

FILTERS are the reference recipe's, read off bmdanielsson/nnue-trainer
(ideas only, no code): drop the opening book plies, then keep a position only
if it is not in check and the move played from it is not a capture, not a
promotion, not en-passant, and does not give check.

usage:
  build_lambda_corpus.py scan  OUT.npz PGNGLOB [MAXPOS]     # parse+filter+dedup
  build_lambda_corpus.py label OUT.npz ENGINE TABLES [DEPTH] [NPROC]
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import chess
import chess.pgn
import numpy as np

BOOK_PLIES = 8          # drop the opening book; our books are 8-ply
SAMPLE_EVERY = 3        # decorrelate consecutive plies (texel_data used 7)
MIN_PIECES = 6          # skip dead-drawn shells


def fenkey(fen):
    """House split key: sha256 of the position, so a split is keyed on the
    POSITION and never on its row number (distill_train.py's rule)."""
    return hashlib.sha256(fen.encode()).hexdigest()[:16]


def scan(out, pgnglob, maxpos):
    paths = sorted(glob.glob(pgnglob))
    assert paths, "no PGNs matched %s" % pgnglob
    n_games = n_plies = n_book = n_noisy = n_small = n_dup = 0
    seen, rows = set(), []
    t0 = time.time()
    for path in paths:
        with open(path, errors="replace") as f:
            while True:
                try:
                    g = chess.pgn.read_game(f)
                except Exception:
                    break
                if g is None:
                    break
                res = g.headers.get("Result", "*")
                if res not in ("1-0", "0-1", "1/2-1/2"):
                    continue                      # unfinished/void: no label
                n_games += 1
                wpov = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[res]
                board = g.board()
                for ply, mv in enumerate(g.mainline_moves()):
                    n_plies += 1
                    if ply < BOOK_PLIES:
                        n_book += 1
                        board.push(mv)
                        continue
                    # the reference recipe's quiet filter, applied to the
                    # position BEFORE the move is played
                    if (board.is_check() or board.is_capture(mv)
                            or mv.promotion or board.is_en_passant(mv)
                            or board.gives_check(mv)):
                        n_noisy += 1
                        board.push(mv)
                        continue
                    if len(board.piece_map()) < MIN_PIECES:
                        n_small += 1
                        board.push(mv)
                        continue
                    if ply % SAMPLE_EVERY == 0:
                        fen = board.fen()
                        k = fenkey(fen)
                        if k in seen:
                            n_dup += 1
                        else:
                            seen.add(k)
                            # OUTCOME IS SIDE-TO-MOVE RELATIVE, like the cp
                            # label will be.  Mixing frames is the kind of
                            # sign bug the lambda unit test exists to catch.
                            stm = wpov if board.turn == chess.WHITE else 1.0 - wpov
                            rows.append((fen, stm, k))
                    board.push(mv)
                if maxpos and len(rows) >= maxpos:
                    break
        print("  %-52s games %6d  kept %8d  %.0fs"
              % (os.path.basename(path)[:52], n_games, len(rows), time.time() - t0),
              flush=True)
        if maxpos and len(rows) >= maxpos:
            break

    print("\nFUNNEL")
    print("  games with a result      %9d" % n_games)
    print("  plies seen               %9d" % n_plies)
    print("  - book plies (<%d)        %9d" % (BOOK_PLIES, n_book))
    print("  - noisy (capture/promo/ep/check/gives-check) %9d" % n_noisy)
    print("  - under %d pieces         %9d" % (MIN_PIECES, n_small))
    print("  - duplicate positions    %9d" % n_dup)
    print("  = KEPT (unique, quiet)   %9d" % len(rows))
    if n_plies:
        print("  kept/ply %.3f%%   dedup ratio %.3f"
              % (100.0 * len(rows) / n_plies, n_dup / max(1, n_dup + len(rows))))
    np.savez_compressed(out,
                        fens=np.array([r[0] for r in rows]),
                        outcome=np.array([r[1] for r in rows], dtype=np.float32),
                        fenhash=np.array([r[2] for r in rows]),
                        meta=json.dumps({"pgnglob": pgnglob, "games": n_games,
                                         "plies": n_plies, "book_plies": BOOK_PLIES,
                                         "sample_every": SAMPLE_EVERY,
                                         "min_pieces": MIN_PIECES,
                                         "n_book": n_book, "n_noisy": n_noisy,
                                         "n_small": n_small, "n_dup": n_dup,
                                         "kept": len(rows)}))
    print("\nwrote %s" % out)


def label(out, engine, tables, depth, nproc):
    """Twin-label every position at fixed depth.  cp is SIDE-TO-MOVE
    relative, matching the outcome channel."""
    d = np.load(out, allow_pickle=True)
    fens = list(d["fens"])
    print("labelling %d positions at depth %d with %d workers"
          % (len(fens), depth, nproc), flush=True)
    chunks = [fens[i::nproc] for i in range(nproc)]
    idx = [list(range(i, len(fens), nproc)) for i in range(nproc)]
    procs = []
    for w in range(nproc):
        p = subprocess.Popen([sys.executable, __file__, "_worker", engine, tables,
                              str(depth)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True, bufsize=1)
        p.stdin.write("\n".join(chunks[w]) + "\n")
        p.stdin.close()
        procs.append(p)
    cp = np.zeros(len(fens), dtype=np.float32)
    for w, p in enumerate(procs):
        for j, line in enumerate(p.stdout):
            cp[idx[w][j]] = float(line.strip())
        p.wait()
    sha = hashlib.sha256(open(engine, "rb").read()).hexdigest()[:24]
    np.savez_compressed(out, fens=d["fens"], outcome=d["outcome"],
                        fenhash=d["fenhash"], cp=cp,
                        meta=json.dumps({**json.loads(str(d["meta"])),
                                         "label_depth": depth,
                                         "labeller_sha": sha,
                                         "labeller": os.path.basename(engine)}))
    print("labelled; cp mean %.1f  min %.0f  max %.0f" % (cp.mean(), cp.min(), cp.max()))


def _worker(engine, tables, depth):
    e = subprocess.Popen([engine, tables], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    e.stdin.write("uci\n")
    e.stdin.flush()
    for fen in sys.stdin:
        fen = fen.strip()
        if not fen:
            continue
        # THE TT LESSON (texel_data.py): clear between positions or the label
        # depends on what preceded it.
        e.stdin.write("ucinewgame\nisready\n")
        e.stdin.flush()
        e.stdin.write("position fen %s\ngo depth %d\n" % (fen, depth))
        e.stdin.flush()
        score = 0
        while True:
            ln = e.stdout.readline()
            if not ln:
                break
            if ln.startswith("info") and " score " in ln:
                try:
                    score = int(ln.split(" score ")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if ln.startswith("done"):
                break
        print(score, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "scan":
        scan(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif mode == "label":
        label(sys.argv[2], sys.argv[3], sys.argv[4],
              int(sys.argv[5]) if len(sys.argv) > 5 else 8,
              int(sys.argv[6]) if len(sys.argv) > 6 else 32)
    elif mode == "_worker":
        _worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
