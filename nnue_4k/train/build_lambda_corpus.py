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
  build_lambda_corpus.py scan     OUT.npz PGNGLOB [MAXPOS]  # parse+filter+dedup
  build_lambda_corpus.py scan-sac OUT.npz PGNGLOB [MAXPOS]  # the SEE<0 half
  build_lambda_corpus.py label    OUT.npz ENGINE TABLES [DEPTH] [NPROC]
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
CPMAX = 1000            # registered option (a): drop |cp| above this


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


def scan_sac(out, pgnglob, maxpos):
    """The SACRIFICE half of the owner's filter 4, for the self-play corpus.

    `scan` above implements the reference recipe's quiet filter, which skips
    every position whose played move is a capture -- sacrifices included.  The
    owner's filter 4 is a RELAXATION of exactly that line: keep a capture when
    SEE < 0, because those are the tactically sharp positions a dead-linear
    eval family keeps failing on.

    This emits ONLY the sacrifices, so the existing labelled corpus can be
    kept as-is and merged, rather than re-labelling 737k positions that would
    come back byte-identical (same twin, same depth).  The keep condition
    mirrors the dump-side worker in build_pool.py: a capture whose SEE is
    negative, promotions excluded.  In-check and check-giving positions are
    ALLOWED here, as they are on the dump side, so the two corpora's
    sacrifice classes are defined the same way.
    """
    from build_pool import see as see_of          # the fuzz-validated swap list
    paths = sorted(glob.glob(pgnglob))
    assert paths, "no PGNs matched %s" % pgnglob
    n_games = n_plies = n_book = n_notcap = n_seeok = n_small = n_dup = 0
    n_promo = 0
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
                    continue
                n_games += 1
                wpov = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[res]
                board = g.board()
                for ply, mv in enumerate(g.mainline_moves()):
                    n_plies += 1
                    if ply < BOOK_PLIES:
                        n_book += 1
                        board.push(mv)
                        continue
                    if not board.is_capture(mv):
                        n_notcap += 1
                        board.push(mv)
                        continue
                    if mv.promotion:
                        n_promo += 1
                        board.push(mv)
                        continue
                    if see_of(chess, board, mv) >= 0:
                        n_seeok += 1
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
                            stm = wpov if board.turn == chess.WHITE else 1.0 - wpov
                            rows.append((fen, stm, k))
                    board.push(mv)
                if maxpos and len(rows) >= maxpos:
                    break
        print("  %-52s games %6d  sacs %8d  %.0fs"
              % (os.path.basename(path)[:52], n_games, len(rows), time.time() - t0),
              flush=True)
        if maxpos and len(rows) >= maxpos:
            break

    print("\nSACRIFICE FUNNEL")
    print("  games with a result      %9d" % n_games)
    print("  plies seen               %9d" % n_plies)
    print("  - book plies (<%d)        %9d" % (BOOK_PLIES, n_book))
    print("  - not a capture          %9d" % n_notcap)
    print("  - promotion capture      %9d" % n_promo)
    print("  - capture with SEE >= 0  %9d" % n_seeok)
    print("  - under %d pieces         %9d" % (MIN_PIECES, n_small))
    print("  - duplicate positions    %9d" % n_dup)
    print("  = KEPT (unique sacs)     %9d" % len(rows))
    np.savez_compressed(out,
                        fens=np.array([r[0] for r in rows]),
                        outcome=np.array([r[1] for r in rows], dtype=np.float32),
                        fenhash=np.array([r[2] for r in rows]),
                        meta=json.dumps({"pgnglob": pgnglob, "games": n_games,
                                         "plies": n_plies, "book_plies": BOOK_PLIES,
                                         "sample_every": SAMPLE_EVERY,
                                         "min_pieces": MIN_PIECES,
                                         "class": "sacrifice (capture, SEE < 0)",
                                         "n_notcap": n_notcap, "n_promo": n_promo,
                                         "n_see_ge_0": n_seeok,
                                         "n_small": n_small, "n_dup": n_dup,
                                         "kept": len(rows)}))
    print("\nwrote %s" % out)


def scan_dump(out, dumppath, maxpos):
    """FENs ONLY from a lichess_db_eval-style dump -- the SF evals are read and
    DISCARDED on purpose.

    This builds the distribution-isolating arm: the legacy corpus's POSITIONS
    with our twin's labels, so that against the self-play corpus (same twin,
    same depth, same filter) the ONLY moved variable is where the positions
    came from.  Keeping the dump's own deep SF evals would move the label
    source too and reproduce the confound the experiment exists to remove.

    No outcome channel: a dump position has no game attached.  That is fine --
    the comparison arm is lambda=1, which never reads the outcome channel."""
    # zstd BINARY, not the python module: installing packages into the owner's
    # box env is off-limits by standing rule, and `zstd` is already present.
    seen, rows = set(), []
    n_lines = 0
    if not os.path.exists(dumppath):
        raise SystemExit("dump not found: %s (paths are relative to CWD, and the "
                         "dumps live one level above train/)" % dumppath)
    proc = subprocess.Popen(["zstd", "-dc", dumppath], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    with proc.stdout as r:
        for line in r:
            n_lines += 1
            try:
                fen = json.loads(line)["fen"]
            except Exception:
                continue
            if len(fen.split()) == 4:
                fen = fen + " 0 1"          # dump omits the clocks
            k = fenkey(fen)
            if k in seen:
                continue
            seen.add(k)
            rows.append((fen, 0.5, k))      # outcome unused at lambda=1
            if maxpos and len(rows) >= maxpos:
                break
    # NEVER WRITE AN EMPTY CORPUS SILENTLY.  The first version shelled out to a
    # relative path that did not exist, got zero bytes on stdout, and cheerfully
    # wrote a 0-position npz -- the failure mode this project keeps paying for.
    if proc.poll() is None:
        proc.terminate()
    err = (proc.stderr.read() or "").strip() if proc.stderr else ""
    if n_lines == 0 or not rows:
        raise SystemExit("dump scan read %d lines and kept %d positions -- refusing "
                         "to write an empty corpus.  zstd said: %s"
                         % (n_lines, len(rows), err[:200] or "(nothing)"))
    print("dump scan: %d lines -> %d unique positions" % (n_lines, len(rows)))
    np.savez_compressed(out,
                        fens=np.array([r[0] for r in rows]),
                        outcome=np.array([r[1] for r in rows], dtype=np.float32),
                        fenhash=np.array([r[2] for r in rows]),
                        meta=json.dumps({"source": os.path.basename(dumppath),
                                         "lines": n_lines, "kept": len(rows),
                                         "labels": "twin (dump evals DISCARDED)",
                                         "outcome": "absent (lam=1 only)"}))
    print("wrote %s" % out)


def label(out, engine, tables, depth, nproc):
    """Twin-label every position at fixed depth.  cp is SIDE-TO-MOVE
    relative, matching the outcome channel.

    WORKERS TALK THROUGH FILES, NOT PIPES.  The first version of this
    streamed each worker's FENs into its stdin and read results back from
    stdout, and it DEADLOCKED at full scale: ~1.7 MB of FENs does not fit
    the 64 KB pipe buffer, so the parent blocked in `pipe_write` on worker 0
    while worker 0 -- whose own stdout had filled, unread -- blocked in
    `print`.  Only one worker was ever spawned and the run produced nothing
    for seven hours.  The 300-position smoke passed because 5 KB fits the
    buffer, which is exactly why a smoke must be sized to the failure mode
    and not to convenience.  Files have no such coupling."""
    d = np.load(out, allow_pickle=True)
    fens = list(d["fens"])
    tmp = out + ".work"
    os.makedirs(tmp, exist_ok=True)
    print("labelling %d positions at depth %d with %d workers (file IO)"
          % (len(fens), depth, nproc), flush=True)
    procs = []
    for w in range(nproc):
        fin, fout = os.path.join(tmp, "in%02d" % w), os.path.join(tmp, "out%02d" % w)
        with open(fin, "w") as f:
            f.write("\n".join(fens[w::nproc]) + "\n")
        procs.append((w, fout, subprocess.Popen(
            [sys.executable, __file__, "_worker", engine, tables, str(depth)],
            stdin=open(fin), stdout=open(fout, "w"))))
    t0 = time.time()
    for w, fout, p in procs:
        p.wait()
    cp = np.zeros(len(fens), dtype=np.float32)
    for w, fout, p in procs:
        vals = [int(x) for x in open(fout).read().split()]
        idx = list(range(w, len(fens), nproc))
        if len(vals) != len(idx):
            raise SystemExit("worker %d returned %d of %d labels -- refusing to "
                             "write a partially-labelled corpus"
                             % (w, len(vals), len(idx)))
        cp[idx] = vals
    # ---- CPMAX FILTER (registered option (a), the house default).
    # WHY: the twin returns MATE scores, not evaluations, for forced lines --
    # measured on a 20k sample, 75 positions (0.38%) sit at |cp| > 10,000 with
    # extremes of -47,998 / +47,968 against a sane core of mean -6.5, sd 223
    # inside +-1000.  In win-prob space a squared loss would let that handful
    # dominate every gradient, and the failure would masquerade as "lambda
    # does not work" rather than "the labels were poisoned".  config's cpmax
    # is 1000 and every prior corpus in this campaign used it.
    # lambda=0 (pure outcome) is invariant to this, and all three arms see the
    # IDENTICAL surviving set, which is the shared-corpus property the
    # experiment rests on.
    # FUTURE READER: clamping instead of dropping may return when teacher-data
    # arms arrive (Leela values saturate differently).  That is a NEW
    # registration, not a reopening of this one.
    keep = np.abs(cp) <= CPMAX
    n_mate = int((np.abs(cp) > 10000).sum())
    n_drop = int((~keep).sum())
    fens, outc = np.asarray(d["fens"])[keep], np.asarray(d["outcome"])[keep]
    fhash, cp = np.asarray(d["fenhash"])[keep], cp[keep]
    print("cpmax filter: dropped %d of %d (%.2f%%), of which %d were mate-class "
          "(|cp|>10000); kept %d" % (n_drop, len(keep), 100.0 * n_drop / len(keep),
                                     n_mate, len(cp)))

    # ---- TWO VAL DRAWS, keyed on the POSITION (the standing law).  Disjoint
    # by construction; the judge is the selector, but training-time validation
    # still gets two independent draws so a val gap can be checked against
    # draw noise -- this campaign measured the draw worth more than any dial.
    hh = np.array([int(h[:8], 16) for h in fhash])
    val_a, val_b = (hh % 20 == 0), (hh % 20 == 1)
    sha = hashlib.sha256(open(engine, "rb").read()).hexdigest()[:24]
    csha = hashlib.sha256(cp.tobytes() + outc.tobytes()).hexdigest()[:16]
    np.savez_compressed(out, fens=fens, outcome=outc, fenhash=fhash, cp=cp,
                        val_a=val_a, val_b=val_b,
                        meta=json.dumps({**json.loads(str(d["meta"])),
                                         "label_depth": depth,
                                         "labeller_sha": sha,
                                         "labeller": os.path.basename(engine),
                                         "cpmax": CPMAX, "cp_dropped": n_drop,
                                         "cp_mate_class": n_mate,
                                         "n_final": int(len(cp)),
                                         "val_a": int(val_a.sum()),
                                         "val_b": int(val_b.sum()),
                                         "corpus_sha": csha}))
    print("labelled+filtered %d in %.0fs; cp mean %.1f sd %.1f; val draws %d / %d; "
          "corpus_sha %s" % (len(cp), time.time() - t0, cp.mean(), cp.std(),
                             val_a.sum(), val_b.sum(), csha))


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
    if mode == "scan-dump":
        scan_dump(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif mode == "scan":
        scan(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif mode == "scan-sac":
        scan_sac(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif mode == "label":
        label(sys.argv[2], sys.argv[3], sys.argv[4],
              int(sys.argv[5]) if len(sys.argv) > 5 else 8,
              int(sys.argv[6]) if len(sys.argv) > 6 else 32)
    elif mode == "_worker":
        _worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
