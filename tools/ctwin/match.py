#!/usr/bin/env python3
"""Fixed-node C-vs-C match driver for the ctwin lab instrument.

Plays paired-opening (color-swapped) games between two knob-configs of
sunfish_c at a fixed node budget, with python-chess as the legality and
termination arbiter.  This is a SCREENING instrument in the sense of
docs/TESTING.md rules 12-14: fixed-node results hold search effort
constant; only wall-clock matches on the real engine decide.

Config cells come from a JSON matrix (see battery.json): each cell is a
name plus a dict of runtime knobs passed to sunfish_c as NAME=VALUE argv.
Openings come from an EPD/FEN file (first 4 FEN fields used).

Scoring: trinomial SPRT (Wald bounds, logistic Elo model with draw share
estimated from the data, cutechess-style) plus a normal-approximation
Elo point estimate with a 95% interval.  Engines are deterministic, so
each (opening, colors) pair is one distinct game; the book bounds the
sample and the driver says so when SPRT is still undecided at book end.

Zero tolerance for illegal moves: an illegal or absent bestmove while
legal moves exist aborts the run loudly, naming the game (repo law; a
lab harness that scores around a broken engine measures nothing).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

import chess

HERE = os.path.dirname(os.path.abspath(__file__))


class CEngine:
    def __init__(self, knobs, tables=None, binary=None):
        argv = [binary or os.path.join(HERE, "sunfish_c"),
                tables or os.path.join(HERE, "tables_classic.txt")]
        argv += ["%s=%d" % (k, v) for k, v in sorted(knobs.items())]
        self.argv = argv
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.nodes_played = 0
        self.moves_played = 0

    def send(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def bestmove(self, fen4, moves, nodes):
        pos = "position fen %s" % fen4
        if moves:
            pos += " moves " + " ".join(moves)
        self.send(pos)
        while True:                          # eat the "ok"
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died: %s" % self.argv)
            if line.strip() == "ok":
                break
            if line.startswith("err"):
                raise RuntimeError("engine: %s (%s)" % (line.strip(), pos))
        self.send("go nodes %d" % nodes)
        last_nodes = 0
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died: %s" % self.argv)
            if line.startswith("info") and " nodes " in line:
                try:
                    last_nodes = int(line.split(" nodes ")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if line.startswith("bestmove"):
                self.nodes_played += last_nodes
                self.moves_played += 1
                return line.split()[1]

    def newgame(self):
        self.send("ucinewgame")

    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(white, black, fen4, nodes, max_plies):
    """Returns +1 white win / 0 draw / -1 black win (arbiter: python-chess)."""
    board = chess.Board(fen4 + " 0 1")
    white.newgame()
    black.newgame()
    moves = []
    while True:
        over = board.outcome(claim_draw=True)
        if over is not None:
            w = over.winner
            return 0 if w is None else (1 if w == chess.WHITE else -1)
        if len(moves) >= max_plies:
            return 0                          # length adjudication: draw
        eng = white if board.turn == chess.WHITE else black
        mv = eng.bestmove(fen4, moves, nodes)
        if mv == "(none)":
            raise SystemExit(
                "FATAL: bestmove (none) with legal moves present\n"
                "  engine: %s\n  fen: %s\n  moves: %s"
                % (eng.argv, fen4, " ".join(moves)))
        try:
            m = chess.Move.from_uci(mv)
            legal = m in board.legal_moves
        except ValueError:
            legal = False
        if not legal:
            raise SystemExit(
                "FATAL: illegal move %s\n  engine: %s\n  fen: %s\n  moves: %s"
                % (mv, eng.argv, fen4, " ".join(moves)))
        board.push(m)
        moves.append(mv)


def sprt_llr(w, d, l, elo0, elo1):
    """Trinomial SPRT LLR (logistic model, draw ratio from data)."""
    n = w + d + l
    if n == 0 or w == n or l == n:
        return 0.0
    # score probabilities under an elo hypothesis, with the observed draw
    # share; standard cutechess/fishtest-style trinomial approximation.
    def probs(elo):
        expected = 1.0 / (1.0 + 10.0 ** (-elo / 400.0))
        pd = d / n
        pw = expected - pd / 2.0
        pl = 1.0 - expected - pd / 2.0
        eps = 1e-6
        return max(pw, eps), max(pd, eps), max(pl, eps)
    pw0, pd0, pl0 = probs(elo0)
    pw1, pd1, pl1 = probs(elo1)
    return (w * math.log(pw1 / pw0) + d * math.log(pd1 / pd0)
            + l * math.log(pl1 / pl0))


def elo_estimate(w, d, l):
    n = w + d + l
    if n == 0:
        return 0.0, 0.0
    score = (w + d / 2.0) / n
    eps = 1e-9
    score = min(max(score, eps), 1 - eps)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # normal approx on the score fraction
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * (0 - score) ** 2) / n
    se = math.sqrt(var / n) if n > 1 else 0.5
    lo = min(max(score - 1.96 * se, eps), 1 - eps)
    hi = min(max(score + 1.96 * se, eps), 1 - eps)
    # (LOW, HIGH), in that order.  score -> elo is increasing, so the low
    # score bound is the low Elo bound; returning them the other way round
    # printed every interval backwards ("elo +58.5 [+312.9, -140.3]").
    return elo, (-400.0 * math.log10(1.0 / lo - 1.0),
                 -400.0 * math.log10(1.0 / hi - 1.0))


def load_openings(path, seed):
    fens = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")[0].split()
        if len(fields) >= 4 and "/" in fields[0]:
            fens.append(" ".join(fields[:4]))
    random.Random(seed).shuffle(fens)
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-a", required=True, help="JSON knobs or battery.json cell name (candidate)")
    ap.add_argument("--cell-b", required=True, help="JSON knobs or cell name (baseline)")
    ap.add_argument("--battery", default=os.path.join(HERE, "battery.json"))
    ap.add_argument("--openings", default=os.path.join(HERE, "..", "build", "gate_openings.epd"))
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=334, help="max opening pairs")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--elo0", type=float, default=-10.0)
    ap.add_argument("--elo1", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tables", default=None)
    args = ap.parse_args()

    def cell(spec):
        if spec.strip().startswith("{"):
            return spec, json.loads(spec)
        table = json.load(open(args.battery))["cells"]
        return spec, {k: int(v) for k, v in table[spec].items()}

    name_a, knobs_a = cell(args.cell_a)
    name_b, knobs_b = cell(args.cell_b)
    openings = load_openings(args.openings, args.seed)[:args.rounds]
    upper = math.log((1 - args.beta) / args.alpha)
    lower = math.log(args.beta / (1 - args.alpha))

    ea = CEngine(knobs_a, args.tables)
    eb = CEngine(knobs_b, args.tables)
    w = d = l = 0                      # from A's perspective
    verdict = "book exhausted (SPRT undecided)"
    try:
        for g, fen in enumerate(openings):
            for a_is_white in (True, False):
                r = (play_game(ea, eb, fen, args.nodes, args.max_plies)
                     if a_is_white else
                     -play_game(eb, ea, fen, args.nodes, args.max_plies))
                if r > 0:
                    w += 1
                elif r < 0:
                    l += 1
                else:
                    d += 1
            llr = sprt_llr(w, d, l, args.elo0, args.elo1)
            if (g + 1) % 10 == 0 or llr >= upper or llr <= lower:
                elo, (lo, hi) = elo_estimate(w, d, l)
                print("[%3d pairs] A=%s vs B=%s  +%d =%d -%d  elo %+.1f [%+.1f, %+.1f]  LLR %.2f (%.2f, %.2f)"
                      % (g + 1, name_a, name_b, w, d, l, elo, lo, hi, llr, lower, upper),
                      flush=True)
            if llr >= upper:
                verdict = "H1 accepted (elo >= %g)" % args.elo1
                break
            if llr <= lower:
                verdict = "H0 accepted (elo <= %g)" % args.elo0
                break
    finally:
        ea.quit()
        eb.quit()

    elo, (lo, hi) = elo_estimate(w, d, l)
    n = w + d + l
    print("RESULT A=%s vs B=%s: %d games +%d =%d -%d  elo %+.1f [%+.1f, %+.1f]  %s"
          % (name_a, name_b, n, w, d, l, elo, lo, hi, verdict))
    print("nodes/move: A %.0f  B %.0f"
          % (ea.nodes_played / max(ea.moves_played, 1),
             eb.nodes_played / max(eb.moves_played, 1)))


if __name__ == "__main__":
    main()
