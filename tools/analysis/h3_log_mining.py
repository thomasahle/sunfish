#!/usr/bin/env python3
"""H3 disambiguation miner. Implements tools/analysis/h3_rules.md EXACTLY.

    nice -n 15 python3 tools/analysis/h3_log_mining.py --pgn-dir DIR --log FASTCHESS_LOG

Log parsing only (the fastchess UCI-communication log of the pyleague ladder).
Refuses to print a verdict if the pre-registered instrument-sanity control or
the P0 policy gate fails. Deterministic markdown on stdout.
"""
import argparse
import math
import re
import statistics
import sys
from collections import defaultdict

import chess
import chess.pgn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import loss_mining  # noqa: E402  (same-directory sibling, rules-frozen)

ENTRY = "sunfish4k"
CLASSIC = "classic"
MIN_WTIME = 5000          # ms; below this the /12 formula and floor interact
HARD_ABORT_F = 0.95
F_GATE = (0.75, 1.10)     # P0: 80% of moves must land here
PSEUDO_SWING = 32         # loss median from the taxonomy
R_LOSS = 34 / 94          # d:LOW rate among middlegame self-detected losses

LINE_RE = re.compile(
    r"^\[Engine\] \[(\d+):(\d+):(\d+\.\d+)\] <\s*(0x[0-9a-f]+)>\s+(\S+) (<---|--->) (.*)$")
NPS_RE = re.compile(r"\bnps (\d+)\b")


def parse_log(path):
    """One pass. Returns (games, classic_nps) where games maps a full UCI
    move tuple -> {ply: (t_ms, wtime_ms, abs_ts)} for the ENTRY's moves."""
    day = 0.0
    last_raw = None
    cur_moves = defaultdict(tuple)     # thread -> moves of last position cmd
    pending_go = {}                    # thread -> (ply, wtime_ms, abs_ts)
    thread_recs = defaultdict(dict)    # thread -> {ply: rec}
    games = {}
    classic_nps = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            hh, mm, ss, thread, name, direction, rest = m.groups()
            raw = int(hh) * 3600 + int(mm) * 60 + float(ss)
            if last_raw is not None and raw < last_raw - 3600:
                day += 86400.0     # midnight rollover
            last_raw = raw
            ts = raw + day
            if name == CLASSIC and direction == "--->" and rest.startswith("info depth"):
                n = NPS_RE.search(rest)
                if n:
                    classic_nps.append((ts, int(n.group(1))))
                continue
            if name != ENTRY:
                continue
            if direction == "<---":
                if rest.startswith("position startpos"):
                    moves = tuple(rest.split(" moves ", 1)[1].split()) \
                        if " moves " in rest else ()
                    prev = cur_moves[thread]
                    if len(moves) < len(prev) or moves[:len(prev)] != prev:
                        # new game on this thread: bank the finished one
                        if thread_recs[thread]:
                            games[cur_moves[thread]] = dict(thread_recs[thread])
                            thread_recs[thread] = {}
                        pending_go.pop(thread, None)
                    cur_moves[thread] = moves
                elif rest.startswith("go ") or rest == "go":
                    times = dict(zip(rest.split()[1::2], map(int, rest.split()[2::2])))
                    ply = len(cur_moves[thread])
                    wtime = times.get("wtime" if ply % 2 == 0 else "btime")
                    if wtime is not None:
                        pending_go[thread] = (ply, wtime, ts)
            elif rest.startswith("bestmove"):
                move = rest.split()[1] if len(rest.split()) > 1 else None
                if thread in pending_go:
                    ply, wtime, t0 = pending_go.pop(thread)
                    thread_recs[thread][ply] = ((ts - t0) * 1000.0, wtime, t0)
                if move and move != "(none)":
                    cur_moves[thread] = cur_moves[thread] + (move,)
    for thread, recs in thread_recs.items():   # bank tail games
        if recs:
            games[cur_moves[thread]] = dict(recs)
    return games, classic_nps


def pgn_entry_games(pgn_path):
    """[(index, game, entry_color, result)] for all entry games, plus UCI lists."""
    out = []
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        index = 0
        while (g := chess.pgn.read_game(f)) is not None:
            index += 1
            h = g.headers
            if ENTRY not in (h.get("White"), h.get("Black")):
                continue
            color = chess.WHITE if h.get("White") == ENTRY else chess.BLACK
            uci = tuple(n.move.uci() for n in g.mainline())
            out.append((index, g, color, h.get("Result", "*"), uci))
    return out


def match_recs(uci, log_games):
    """Exact match, else unique prefix (>=30 plies)."""
    if uci in log_games:
        return log_games[uci]
    cands = [v for k, v in log_games.items()
             if len(k) >= 30 and uci[:len(k)] == k]
    return cands[0] if len(cands) == 1 else None


def window_indices(own, swing_idx):
    return list(range(max(0, swing_idx - 5), swing_idx))


def pseudo_swing_idx(own):
    for i, om in enumerate(own):
        if om.fullmove >= PSEUDO_SWING:
            return i
    return None


def is_dlow(own, idx):
    depths = [o.depth for o in own]
    window = depths[max(0, idx - 5):idx]
    if len(window) < 2:
        return False
    return statistics.mean(window) - statistics.median(depths) <= -1.0


def move_stats(own, indices, recs, color):
    """(F, think_s, depth, ts) per filtered move in `indices`."""
    out = []
    for i in indices:
        om = own[i]
        if om.fullmove < 15 or om.npm_before <= 12:
            continue
        ply = (om.fullmove - 1) * 2 + (0 if color == chess.WHITE else 1)
        rec = recs.get(ply) if recs else None
        if rec is None:
            continue
        t_ms, wtime, ts = rec
        if wtime < MIN_WTIME:
            continue
        think_ms = wtime / 12.0
        out.append((t_ms / think_ms, think_ms / 1000.0, om.depth, ts))
    return out


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def hard_rate(stats):
    return round(sum(1 for f, *_ in stats if f >= HARD_ABORT_F) / len(stats), 3) \
        if stats else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-dir", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    losses, _ = loss_mining.mine(args.pgn_dir)
    failures = loss_mining.check_controls(losses)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    target = {L.index: L for L in losses
              if L.corpus == "pyleague" and L.loser == ENTRY
              and L.phase == "MIDDLEGAME" and L.swing_class == "SELF-DETECTED"
              and L.depth_signal == "LOW"}
    print(f"target d:LOW middlegame losses: {len(target)}", file=sys.stderr)

    log_games, classic_nps = parse_log(args.log)
    print(f"log games: {len(log_games)}, classic nps samples: {len(classic_nps)}",
          file=sys.stderr)

    buckets = defaultdict(list)
    for ts, nps in classic_nps:
        buckets[int(ts // 300)].append(nps)
    bucket_med = {b: statistics.median(v) for b, v in buckets.items()}
    global_med = statistics.median(nps for _, nps in classic_nps)

    W, B, SM, PC, allF = [], [], [], [], []
    matched_w_games = unmatched = 0
    n_pc_games = n_sm_games = 0
    r_alt = {28: [0, 0], 32: [0, 0], 36: [0, 0]}
    for index, game, color, result, uci in pgn_entry_games(f"{args.pgn_dir}/pyleague_games.pgn"):
        recs = match_recs(uci, log_games)
        if recs is None:
            unmatched += 1
            continue
        own, _ = loss_mining.extract_own_moves(game, color)
        if len(own) < loss_mining.MIN_SCORED:
            continue
        allF += [s[0] for s in move_stats(own, range(len(own)), recs, color)]
        lost = (result == "0-1") == (color == chess.WHITE) and result in ("1-0", "0-1")
        if index in target:
            kind, sidx = loss_mining.find_swing(own)
            w_idx = window_indices(own, sidx)
            W += move_stats(own, w_idx, recs, color)
            B += move_stats(own, [i for i in range(sidx) if i not in w_idx], recs, color)
            matched_w_games += 1
        elif not lost and result in ("1-0", "0-1", "1/2-1/2"):
            idx = pseudo_swing_idx(own)
            for ps in r_alt:
                j = next((i for i, om in enumerate(own) if om.fullmove >= ps), None)
                if j is not None and j >= 5:
                    r_alt[ps][1] += 1
                    r_alt[ps][0] += is_dlow(own, j)
            if idx is None or idx < 5:
                continue
            n_pc_games += 1
            stats = move_stats(own, window_indices(own, idx), recs, color)
            PC += stats
            if is_dlow(own, idx):
                n_sm_games += 1
                SM += stats

    # --- gates ---
    p0 = sum(1 for f in allF if F_GATE[0] <= f <= F_GATE[1]) / len(allF) if allF else 0
    confounded = sum(1 for *_, ts in W
                     if bucket_med.get(int(ts // 300), global_med) < 0.8 * global_med)
    hW, hB, hSM, hPC = hard_rate(W), hard_rate(B), hard_rate(SM), hard_rate(PC)

    print(f"P0: {p0:.1%} of {len(allF)} entry moves in F-gate band", file=sys.stderr)
    insufficient = []
    if p0 < 0.80:
        insufficient.append(f"P0 policy gate failed ({p0:.1%} < 80%)")
    if len(W) < 20:
        insufficient.append(f"only {len(W)} matched W moves (<20)")
    if W and confounded > len(W) / 3:
        insufficient.append(f"co-tenancy guard: {confounded}/{len(W)} W moves confounded")
    sane = hPC is not None and hB is not None and abs(hPC - hB) < 0.15
    if not sane:
        print(f"INSTRUMENT SANITY FAILED: H(PC)={hPC} vs H(B)={hB}", file=sys.stderr)
        print("No verdict printed.", file=sys.stderr)
        return 1

    # D0 fit on PC moves
    gap = None
    if len(PC) >= 20 and W and SM:
        xs = [math.log(t) for _, t, _, _ in PC]
        ys = [d for _, _, d, _ in PC]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
            max(1e-9, sum((x - mx) ** 2 for x in xs))
        a = my - b * mx
        res = lambda s: [d - (a + b * math.log(t)) for _, t, d, _ in s]  # noqa: E731
        gap = round(statistics.median(res(W)) - statistics.median(res(SM)), 2)

    r_nonloss = r_alt[32][0] / r_alt[32][1] if r_alt[32][1] else None

    # --- verdict per h3_rules.md ---
    if insufficient:
        verdict = "LOGS-INSUFFICIENT: " + "; ".join(insufficient)
    else:
        a_fires = (hW - hSM >= 0.15 if hSM is not None else False) or \
                  (gap is not None and gap <= -0.5)
        b_rate = r_nonloss is not None and r_nonloss >= 0.75 * R_LOSS
        if a_fires and b_rate:
            verdict = "MIXED"
        elif a_fires:
            verdict = "(a) QSEARCH/TREE EXPLOSION"
        elif b_rate and (hSM is None or hW - hSM < 0.15) and (gap is None or gap > -0.5):
            verdict = "(b) TM COMMIT/ALLOCATION"
        else:
            verdict = "LOGS-INSUFFICIENT: no pre-registered criterion fired"

    print("# H3 log-mining output (rules: tools/analysis/h3_rules.md)\n")
    print(f"- matched W games: {matched_w_games}/{len(target)}; unmatched entry games: {unmatched}")
    print(f"- moves: W={len(W)} B={len(B)} SM={len(SM)} (from {n_sm_games} games) "
          f"PC={len(PC)} (from {n_pc_games} games)")
    print(f"- P0 policy gate: {p0:.1%} in [{F_GATE[0]}, {F_GATE[1]}]; "
          f"median F overall {med([f for f in allF])}")
    print(f"- hard-abort rates: H(W)={hW} H(B)={hB} H(SM)={hSM} H(PC)={hPC}")
    print(f"- median F: W={med([s[0] for s in W])} B={med([s[0] for s in B])} "
          f"SM={med([s[0] for s in SM])} PC={med([s[0] for s in PC])}")
    print(f"- median depth: W={med([s[2] for s in W])} B={med([s[2] for s in B])} "
          f"SM={med([s[2] for s in SM])} PC={med([s[2] for s in PC])}")
    print(f"- median think(s): W={med([s[1] for s in W])} B={med([s[1] for s in B])} "
          f"PC={med([s[1] for s in PC])}")
    print(f"- D0 residual gap (W - SM): {gap} ply")
    print(f"- R0: r_loss={R_LOSS:.3f}, r_nonloss@32={r_nonloss and round(r_nonloss, 3)} "
          f"(@28={r_alt[28][0]}/{r_alt[28][1]}, @36={r_alt[36][0]}/{r_alt[36][1]})")
    print(f"- co-tenancy: {confounded}/{len(W)} W moves in depressed-nps buckets "
          f"(classic global median nps {round(global_med)})")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
