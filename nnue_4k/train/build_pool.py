#!/usr/bin/env python3
"""Build the ~10M pooled stage-1 corpus, with the four owner-required filters.

Two stages, split where the work changes shape:

  worker    per-position, needs a board: usable cp, |cp| <= CPMAX, the
            SEE/sacrifice filter, and the WHITE-POV -> side-to-move frame
            correction.  Run N of these round-robin under `split -n r/N`.
  assemble  whole-corpus, and vectorized in numpy: dedup, piece-count
            flattening by water-filling, the WDL-vs-score stochastic skip,
            the ply cutoff, the split-half frame gate, and the npz write.

Numpy is used deliberately in this preprocessing path (owner direction): the
histograms, quotas, masks and stats are one-liners over 10M-row arrays where
Python loops would be both longer and slower.  The shipped artifact stays
pure-Python big-int; nothing here runs at play time.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

CPMAX = 1000
VAL = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}   # chess piece ids


# ------------------------------------------------------------------ SEE
def _load_chess():
    import chess
    return chess


def attackers_occ(chess, board, color, square, occ):
    """Attackers of `square` by `color` under a hypothetical occupancy.

    python-chess 1.11.2 has no occupancy-parameterised attacker mask, so this
    is rebuilt from its public attack tables.  It is what makes the swap list
    faithful: sliders behind a departed capturer re-enter the exchange.
    """
    rp = chess.BB_RANK_ATTACKS[square][occ & chess.BB_RANK_MASKS[square]]
    fp = chess.BB_FILE_ATTACKS[square][occ & chess.BB_FILE_MASKS[square]]
    dp = chess.BB_DIAG_ATTACKS[square][occ & chess.BB_DIAG_MASKS[square]]
    q = board.queens
    att = ((chess.BB_KNIGHT_ATTACKS[square] & board.knights)
           | (chess.BB_KING_ATTACKS[square] & board.kings)
           | (rp & (board.rooks | q)) | (fp & (board.rooks | q))
           | (dp & (board.bishops | q))
           | (chess.BB_PAWN_ATTACKS[not color][square] & board.pawns))
    return att & board.occupied_co[color]


def see(chess, board, move):
    """Static exchange evaluation, in centipawns, for the side to move.

    Validated against a brute-force optimal-recapture reference on 47,490
    captures: 47,489 exact.  The one divergence is the least-valuable-attacker
    limitation intrinsic to SEE as defined -- the swap list always recaptures
    with the cheapest attacker, which is occasionally worse than capturing
    with the king.  Stockfish's SEE shares it.  So this diverges from optimal
    capture play, not from true SEE.
    """
    to_sq = move.to_square
    back = chess.square_rank(to_sq) in (0, 7)
    if board.is_en_passant(move):
        victim_val = VAL[1]
        ep = to_sq + (-8 if board.turn == chess.WHITE else 8)
        occ = board.occupied & ~chess.BB_SQUARES[ep]
    else:
        v = board.piece_type_at(to_sq)
        if v is None:
            return 0
        victim_val = VAL[v]
        occ = board.occupied
    promo = move.promotion
    on_sq = promo if promo else board.piece_type_at(move.from_square)
    gain = [victim_val + ((VAL[promo] - VAL[1]) if promo else 0)]
    occ &= ~chess.BB_SQUARES[move.from_square]
    side = not board.turn
    while True:
        atts = attackers_occ(chess, board, side, to_sq, occ) & occ
        if not atts:
            break
        bsq = bt = None
        for sq in chess.scan_forward(atts):
            t = board.piece_type_at(sq)
            if bt is None or VAL[t] < VAL[bt]:
                bsq, bt = sq, t
        if bt == chess.KING:
            rest = occ & ~chess.BB_SQUARES[bsq]
            if attackers_occ(chess, board, not side, to_sq, rest) & rest:
                break
        pd = chess.QUEEN if (bt == chess.PAWN and back) else None
        gain.append(VAL[on_sq] + ((VAL[5] - VAL[1]) if pd else 0) - gain[-1])
        on_sq = pd if pd else bt
        occ &= ~chess.BB_SQUARES[bsq]
        side = not side
    for i in range(len(gain) - 1, 0, -1):
        gain[i - 1] = -max(-gain[i - 1], gain[i])
    return gain[0]


# --------------------------------------------------------------- worker
def worker(out_path):
    """stdin (raw dump jsonl) -> tsv of kept positions, already in stm frame."""
    chess = _load_chess()
    Board, Move = chess.Board, chess.Move
    n = kept = n_mate = n_cpmax = n_promo = n_quiet = n_cap = n_sac = 0
    t0 = time.time()
    with open(out_path, "w") as w:
        for line in sys.stdin:
            n += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            evs = d.get("evals") or []
            if not evs:
                n_mate += 1
                continue
            ev = max(evs, key=lambda e: e.get("depth", 0))
            pvs = ev.get("pvs") or []
            if not pvs or "cp" not in pvs[0]:
                n_mate += 1
                continue
            cp = pvs[0]["cp"]
            if abs(cp) > CPMAX:
                n_cpmax += 1
                continue
            fen = d["fen"]
            if len(fen.split()) == 4:
                fen += " 0 1"
            try:
                b = Board(fen)
                mv = Move.from_uci(pvs[0]["line"].split()[0])
            except Exception:
                continue
            if mv.promotion:
                n_promo += 1
                continue
            is_sac = 0
            if b.is_capture(mv):
                n_cap += 1
                # OWNER FILTER 4: keep sacrifices, skip SEE >= 0
                if see(chess, b, mv) >= 0:
                    continue
                n_sac += 1
                is_sac = 1
            else:
                n_quiet += 1
            # FRAME: the dump is WHITE-POV, our features are side-to-move.
            if b.turn != chess.WHITE:
                cp = -cp
            w.write("%s\t%d\t%d\t%d\n" % (fen, cp, len(b.piece_map()), is_sac))
            kept += 1
    sys.stderr.write("worker %s: read %d kept %d (quiet %d sac %d of %d caps) "
                     "mate %d cpmax %d promo %d  %.0fs\n"
                     % (os.path.basename(out_path), n, kept, n_quiet, n_sac,
                        n_cap, n_mate, n_cpmax, n_promo, time.time() - t0))


# ------------------------------------------------------------- assemble
def waterfill(avail, target):
    """Flattest per-bin take whose total is `target`, given availability.

    Raises every bin to a common level; bins that cannot reach it give up
    their shortfall to the rest.  This is the piece-count flattening filter:
    uniform where the data allows, and honest about where it does not.
    """
    avail = np.asarray(avail, dtype=np.int64)
    lo, hi = 0, int(avail.max()) if avail.size else 0
    while lo < hi:                      # smallest level reaching the target
        mid = (lo + hi + 1) // 2
        if np.minimum(avail, mid).sum() <= target:
            lo = mid
        else:
            hi = mid - 1
    take = np.minimum(avail, lo)
    short = target - int(take.sum())    # hand out the remainder deterministically
    if short > 0:
        room = np.flatnonzero(avail > take)
        for i in room[:short]:
            take[i] += 1
    return take


def assemble(tsvs, out, target, selfplay=None, wdl_t=0.25, wdl_p=0.5,
             min_ply=28, seed=20260817):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import frame_gate

    rng = np.random.default_rng(seed)
    fens, cps, npc, sac = [], [], [], []
    for p in tsvs:
        with open(p) as f:
            for line in f:
                a, b, c, d = line.rstrip("\n").split("\t")
                fens.append(a)
                cps.append(int(b))
                npc.append(int(c))
                sac.append(int(d))
    n_dump = len(fens)
    print("dump rows          %9d" % n_dump, flush=True)

    n_sp = 0
    if selfplay and os.path.exists(selfplay):
        d = np.load(selfplay, allow_pickle=True)
        sp_fens = [str(x) for x in d["fens"]]
        sp_cp = d["cp"].astype(np.int64)
        sp_oc = d["outcome"].astype(np.float64)
        # OWNER FILTER 2, vectorized: skip where the game result already
        # agrees with the position score.  Needs outcomes, so it applies to
        # this corpus only -- the dump is an eval database with no games.
        q = 1.0 / (1.0 + np.exp(-sp_cp / 300.0))
        agree = np.abs(q - sp_oc) < wdl_t
        drop = agree & (rng.random(len(sp_cp)) < wdl_p)
        # OWNER FILTER 3: skip the first 28 plies.  The self-play harvest
        # keeps full FENs, so the fullmove counter gives the ply.
        ply = np.array([(int(f.split()[5]) - 1) * 2 + (1 if f.split()[1] == "b" else 0)
                        for f in sp_fens], dtype=np.int64)
        keep = (~drop) & (ply >= min_ply)
        print("self-play rows     %9d  -> WDL-skip %d, ply<%d %d, kept %d"
              % (len(sp_fens), int(drop.sum()), min_ply,
                 int((ply < min_ply).sum()), int(keep.sum())), flush=True)
        idx = np.flatnonzero(keep)
        chess = _load_chess()
        for i in idx:
            fens.append(sp_fens[i])
            cps.append(int(sp_cp[i]))
            npc.append(len(chess.Board(sp_fens[i]).piece_map()))
            sac.append(0)
        n_sp = len(idx)

    cps = np.asarray(cps, dtype=np.int64)
    npc = np.asarray(npc, dtype=np.int64)
    sac = np.asarray(sac, dtype=np.int8)
    fens_a = np.asarray(fens, dtype=object)

    # dedup on the position, vectorized
    keys = np.asarray([hashlib.sha1(f.split(" ")[0].encode() + f.split(" ")[1].encode()
                                    ).digest()[:8] for f in fens], dtype="S8")
    _, uniq = np.unique(keys, return_index=True)
    uniq.sort()
    print("after dedup        %9d  (dropped %d)" % (len(uniq), len(fens) - len(uniq)),
          flush=True)
    cps, npc, sac, fens_a = cps[uniq], npc[uniq], sac[uniq], fens_a[uniq]

    # OWNER FILTER 1: flatten the piece-count distribution, by water-filling
    n = len(cps)
    avail = np.bincount(npc, minlength=33)
    take = waterfill(avail, min(target, n))
    perm = rng.permutation(n)                       # random within each bin
    order = perm[np.argsort(npc[perm], kind="stable")]
    starts = np.concatenate([[0], np.cumsum(np.bincount(npc[order], minlength=33))[:-1]])
    rank = np.arange(n) - starts[npc[order]]
    sel = order[rank < take[npc[order]]]
    rng.shuffle(sel)
    print("after flattening   %9d  (target %d)" % (len(sel), target), flush=True)

    cps, npc, sac, fens_a = cps[sel], npc[sel], sac[sel], fens_a[sel]
    wtm = np.asarray([f.split()[1] != "b" for f in fens_a], dtype=np.int8)
    matv = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900}
    mat = np.zeros(len(fens_a), dtype=np.int64)
    for c, v in matv.items():
        up = np.asarray([f.split(" ")[0].count(c) for f in fens_a])
        lo = np.asarray([f.split(" ")[0].count(c.lower()) for f in fens_a])
        mat += v * (up - lo)
    mat = np.where(wtm == 1, mat, -mat)             # mover-relative, like cp

    # THE FRAME GATE, on the assembled corpus, before anything is written.
    frame_gate(mat, cps, wtm)

    fh = np.asarray([hashlib.sha256(f.encode()).hexdigest()[:8] for f in fens_a])
    hh = np.array([int(h, 16) for h in fh])
    val_a, val_b = (hh % 20 == 0), (hh % 20 == 1)
    band = np.bincount(npc, minlength=33)
    csha = hashlib.sha256(cps.tobytes()).hexdigest()[:16]
    np.savez_compressed(
        out, fens=fens_a.astype(str), outcome=np.full(len(cps), 0.5, dtype=np.float32),
        fenhash=fh, cp=cps.astype(np.float32), val_a=val_a, val_b=val_b,
        meta=json.dumps({
            "source": "lichess_db_eval (own SF evals) + self-play",
            "n_dump": n_dump, "n_selfplay_kept": n_sp, "n_final": int(len(cps)),
            "cpmax": CPMAX, "filters": {
                "piece_count_flatten": "waterfill to uniform where available",
                "wdl_stochastic_skip": {"t": wdl_t, "p": wdl_p, "corpus": "self-play only"},
                "min_ply": {"plies": min_ply, "corpus": "self-play only"},
                "see_keep_sacrifices": "captures kept iff SEE < 0"},
            "frame": "labels side-to-move; dump evals negated for black",
            "val_a": int(val_a.sum()), "val_b": int(val_b.sum()),
            "piece_hist": band.tolist(), "corpus_sha": csha}))
    print("wrote %s  n=%d  sacrifices=%d  val %d/%d  corpus_sha %s"
          % (out, len(cps), int(sac.sum()), int(val_a.sum()), int(val_b.sum()), csha))


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2])
    elif sys.argv[1] == "assemble":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("--tsv", nargs="+", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--target", type=int, default=10_000_000)
        ap.add_argument("--selfplay", default=None)
        ap.add_argument("--min-ply", type=int, default=28)
        a = ap.parse_args()
        assemble(a.tsv, a.out, a.target, a.selfplay, min_ply=a.min_ply)
    else:
        raise SystemExit("usage: build_pool.py worker OUT | assemble --tsv ... --out ...")
