#!/usr/bin/env python3
"""VIRTUAL-CLOCK matches on the C twin: time management, ranked without hours.

The twin is 22x sunfish.py and clockless by design (README: "the clock
management branch of classic's main() is not cloned").  That exclusion is
what makes TM the one workstream the twin could not accelerate -- and TM
matches are exactly the ones that cost real hours, because a 60+0 game takes
two minutes of wall clock no matter how fast the engine is.

This driver removes the wall clock from the loop entirely.  Nothing here
sleeps, and no measured duration enters any decision:

  1. FORMULA.  A tmlib manager turns the VIRTUAL clock into (soft, hard).
  2. NODES.    npsmodel.json turns `hard` into a node budget: how far the
               PYTHON engine would have got in that many seconds.
  3. SEARCH.   The twin runs `go nodes` and emits its whole probe trace --
               one line per MTD probe, with the cumulative node count.
  4. REPLAY.   The trace is walked in Python with `elapsed = nodes / nps`
               substituted for the clock, applying the arm's own stop rule
               (the packed 0.8-at-any-yield break, classic's 2/3 rule, or the
               pool's MTD-bracket soft stop).  That yields the move the real
               engine would have played AND what it would have spent.
  5. CHARGE.   virtual clock -= (spend + overhead); flag if negative; += inc.

WHY REPLAY RATHER THAN A TWIN PATCH.  The stop rules live in the DRIVER in
every shipped engine (uci.py's go_loop, the packed entry's inlined loop), so
mirroring them in the driver here is the faithful place for them -- and it
keeps sunfish.c untouched, which matters because any edit to sunfish.c must
re-pass the full node-identity gate (TESTING.md rule 14) before any number
from the twin counts.  It also buys accuracy the twin cannot give: the twin's
node cap is a yield-boundary rule, while the real engine's deadline aborts
INSIDE bound(), so replay models the wall as a mid-probe abort at exactly
`hard` where the twin alone would have overshot by a whole probe.

DETERMINISM.  Same seed, same book, same model -> identical games.  The only
stochastic input in any shipped manager is classic's opening ramp
(`len(hist) + random()`), which is driven here by a per-game seeded RNG.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

import chess

import npsprofile
import tmlib
from match import elo_estimate, load_openings, sprt_llr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import sunfish                                     # noqa: E402 - for MATE_LOWER


class TracingEngine:
    """sunfish_c, driven for its whole probe trace rather than its bestmove."""

    def __init__(self, tables=None, binary=None, knobs=None):
        argv = [binary or os.path.join(HERE, "sunfish_c"),
                tables or os.path.join(HERE, "tables_classic.txt")]
        argv += ["%s=%d" % (k, v) for k, v in sorted((knobs or {}).items())]
        self.argv = argv
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def _send(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def trace(self, fen4, moves, nodes):
        """Returns (probes, bestmove).  probes: (depth, score, kind, mv, nodes)."""
        pos = "position fen %s" % fen4
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("twin died: %s" % self.argv)
            if line.startswith("err"):
                raise RuntimeError("twin: %s (%s)" % (line.strip(), pos))
            if line.strip() == "ok":
                break
        self._send("go nodes %d" % max(1, int(nodes)))
        probes = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("twin died: %s" % self.argv)
            if line.startswith("bestmove"):
                return probes, line.split()[1]
            if not line.startswith("info "):
                continue
            f = line.split()
            d = int(f[f.index("depth") + 1])
            sc = int(f[f.index("cp") + 1])
            n = int(f[f.index("nodes") + 1])
            kind = ("lower" if "lowerbound" in f else
                    "upper" if "upperbound" in f else "exact")
            mv = f[f.index("pv") + 1] if "pv" in f else None
            probes.append((d, sc, kind, mv, n))

    def newgame(self):
        self._send("ucinewgame")

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def replay(probes, twin_best, budget, nps, packed_rule=True, dynamic=False):
    """The driver loop, with `nodes / nps` substituted for the clock.

    Mirrors nnue_4k/pst_entry.py's inlined go loop (packed_rule=True) and
    sunfish_ui/uci.py's `go_loop` (packed_rule=False, and always for the
    pool's "mtd_converged" rule, which only exists there).

    Returns (move, spend_seconds, info).
    """
    soft, hard, rule, frac = budget
    target = soft if rule == "mtd_converged" else frac * hard
    best = cand = None
    best_score = cand_score = None
    d0, stable_iters = 1, 0
    lower, upper, converged_seen = -math.inf, math.inf, False
    stop, spend = "exhausted", 0.0

    for depth, score, kind, mv, nodes in probes:
        elapsed = nodes / nps
        # THE WALL.  The in-search deadline aborts inside bound(), so a probe
        # that would END past `hard` never finishes: the spend is exactly
        # `hard` and the move is whatever had already been committed.
        if elapsed > hard:
            spend, stop = hard, "deadline"
            break
        if depth > d0:
            best, d0 = cand or best, depth
            if cand_score is not None:
                best_score = cand_score
            # The pool's backstop: a new depth is evidence the previous
            # iteration ended, used only when the bracket mirror did not see
            # the convergence itself.  uci.py:311 reads the clock HERE, at
            # the yield that revealed the new depth -- so the reading is this
            # probe's elapsed, not the previous probe's, and the spend that
            # goes on the clock is this one too.
            if (rule == "mtd_converged" and not converged_seen
                    and best is not None and elapsed > target):
                spend, stop = elapsed, "soft-backstop"
                break
            lower, upper, converged_seen = -math.inf, math.inf, False
        if kind == "exact":
            # Terminal root: verified, exact, nothing to search or play.
            spend, stop = elapsed, "terminal"
            break
        if kind == "lower" and mv:
            cand, cand_score = mv, score
        spend = elapsed
        if rule == "mtd_converged":
            if kind == "lower":
                lower = max(lower, score)
            else:
                upper = min(upper, score)
            if not lower < upper - tmlib.EVAL_ROUGHNESS:
                converged_seen = True
                settled = cand or best
                if settled is not None:
                    changed = best is not None and settled != best
                    stable_iters = 0 if changed else stable_iters + 1
                    tgt = soft
                    if dynamic:
                        # uci.py:379-385, the v1.1 arm.  Mate scores bypass
                        # the drop term: a mate score is a different quantity
                        # and differencing it against centipawns produces a
                        # meaningless (and enormous) drop.
                        mate = any(s is not None and abs(s) >= sunfish.MATE_LOWER
                                   for s in (best_score, cand_score))
                        drop = 0.0
                        if not mate and None not in (best_score, cand_score):
                            drop = max(0, best_score - cand_score)
                        tgt = tmlib.dynamic_target(soft, stable_iters, changed,
                                                   drop, mate)
                    best = settled
                    if cand_score is not None:
                        best_score = cand_score
                    if elapsed > tgt:
                        stop = "soft"
                        break
        elif packed_rule:
            # pst_entry.py: `if (best or cand) and elapsed > think*0.8: break`
            # -- at EVERY yield, depth 1 included.
            if (best or cand) and elapsed > target:
                stop = "yield-frac"
                break
        elif depth > 1 and elapsed > target:
            # uci.py:399 -- the same rule at 2/3, guarded by depth > 1.
            stop = "yield-frac"
            break

    move = best or cand
    fallback = move is None
    if fallback:
        # The engine would have answered its STRUCTURAL BESTMOVE FLOOR here:
        # a legal-but-unsearched move.  The twin only exposes its floor via
        # its own bestmove, which comes from a longer search than we allowed,
        # so this substitution is BETTER than the real engine's blind move.
        # It is counted (info["fallback"]) and it biases every result toward
        # the arm that floors most, i.e. AGAINST the fix under test.
        move = twin_best
    return move, spend, {"stop": stop, "fallback": fallback,
                         "depth": d0, "blind": spend <= tmlib.TM_FLOOR + 1e-9}


def ask(arm, fen4, moves, budget, nps):
    """Search to the WALL and replay.  Exactly one `go` per move, always.

    It is tempting to request fewer nodes than the wall when the stop rule
    will obviously fire earlier -- the pool's wall is 5x its soft target, so
    most of that search is thrown away.  It is also WRONG, and the attempt
    is recorded here so it is not made again: `search_setup()` clears the
    twin's score table at every `go` but deliberately NOT its move table
    (sunfish.c:879 vs 1025), because `sunfish.py:516` clears `tp_score` per
    `search()` and keeps `tp_move` across the whole game.  A speculative
    short search therefore leaves its killers behind and the retry is a
    DIFFERENT search -- caught by a prefix assertion on the first position it
    was tried on.  One `go` per move keeps the twin's killer table warmed by
    exactly the search the engine really ran.
    """
    probes, twin_best = arm.engine.trace(fen4, moves, math.ceil(budget.hard * nps))
    return replay(probes, twin_best, budget, nps,
                  packed_rule=(arm.family == "packed"), dynamic=arm.dynamic)


class Arm:
    """A time manager plus the engine family whose driver rules it shipped with."""

    def __init__(self, spec):
        parts = spec.split(":")
        self.name = spec
        self.manager = parts[0]
        if self.manager not in tmlib.MANAGERS:
            raise SystemExit("unknown manager %r (have %s)"
                             % (self.manager, ", ".join(sorted(tmlib.MANAGERS))))
        self.knobs, self.dynamic = {}, False
        for kv in parts[1:]:
            k, _, v = kv.partition("=")
            truthy = v.lower() in ("1", "true", "yes")
            if k == "dynamic":
                # A DRIVER knob (uci.py's TM_DYNAMIC), not a formula knob: it
                # scales the soft target from search stability, which only
                # exists once a search is running.
                self.dynamic = truthy
            elif k == "phase_m":
                self.knobs[k] = truthy
            elif k == "moves":
                self.knobs[k] = int(float(v))
            else:
                self.knobs[k] = float(v)
        self.family = tmlib.FAMILY[self.manager]

    def budget(self, clock, inc, ply, rng):
        b = tmlib.MANAGERS[self.manager](clock, inc, None, ply=ply, **self.knobs)
        if self.family == "classic":
            # The opening ramp is classic's, not the packed artifact's
            # (grepped: `random` does not occur in nnue_4k/pst_entry.py).
            hard = tmlib.opening_ramp(b.hard, ply, rng)
            if hard != b.hard:
                b = tmlib.Budget(min(b.soft, hard), hard, b.rule, b.frac)
        return b


def play_game(white, black, fen4, base, inc, overhead, model, max_plies, seed,
              jitter=0.0):
    """One virtual-clock game.  Returns (result, telemetry)."""
    board = chess.Board(fen4 + " 0 1")
    white.engine.newgame()
    black.engine.newgame()
    rng = random.Random(seed)
    jit = random.Random(seed ^ 0x5EED)
    clocks = {chess.WHITE: base, chess.BLACK: base}
    moves, tele = [], {chess.WHITE: [], chess.BLACK: []}
    while True:
        over = board.outcome(claim_draw=True)
        if over is not None:
            return (0 if over.winner is None
                    else (1 if over.winner == chess.WHITE else -1)), tele, moves
        if len(moves) >= max_plies:
            return 0, tele, moves
        side = board.turn
        arm = white if side == chess.WHITE else black
        ply = len(moves) + 1                        # uci.py's len(hist)
        nps = npsprofile.nps_for(model, len(board.piece_map()))
        b = arm.budget(clocks[side], inc, ply, rng)
        mv, spend, info = ask(arm, fen4, moves, b, nps)
        charge = overhead + (jit.uniform(-jitter, jitter) if jitter else 0.0)
        clocks[side] -= spend + max(charge, 0.0)
        tele[side].append({"ply": ply, "clock": clocks[side], "spend": spend,
                           "soft": b.soft, "hard": b.hard, **info})
        if clocks[side] < 0:
            return (-1 if side == chess.WHITE else 1), tele, moves
        clocks[side] += inc
        if mv == "(none)":
            raise SystemExit("FATAL: bestmove (none) with legal moves\n"
                             "  arm: %s\n  fen: %s\n  moves: %s"
                             % (arm.name, fen4, " ".join(moves)))
        try:
            m = chess.Move.from_uci(mv)
            legal = m in board.legal_moves
        except ValueError:
            legal = False
        if not legal:
            raise SystemExit("FATAL: illegal move %s\n  arm: %s\n  fen: %s\n"
                             "  moves: %s" % (mv, arm.name, fen4, " ".join(moves)))
        board.push(m)
        moves.append(mv)


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True,
                    help="manager[:knob=v...]  e.g. pool:scale=1.2:phase_m=1")
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--tc", default="60+0")
    ap.add_argument("--openings", default=os.path.join(HERE, "..", "..",
                                                       "tests", "files",
                                                       "chessathome_openings.fen"))
    ap.add_argument("--rounds", type=int, default=100, help="opening pairs")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--overhead", type=float, default=0.05,
                    help="SECONDS charged per move on top of the search. 0.05 "
                         "reproduces a local fastchess arena (it is what makes "
                         "the step arm park at 2.1s rather than 2.2s); 0.2 is "
                         "the lichess deployment.")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="seeded +/- uniform jitter on the overhead, SECONDS. "
                         "0 keeps the run deterministic.")
    ap.add_argument("--model", default=os.path.join(HERE, "npsmodel.json"))
    ap.add_argument("--nps-scale", type=float, default=1.0,
                    help="multiply the whole nps profile; the sensitivity knob")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tmlib.verify(verbose=not args.quiet)
    base, _, inc = args.tc.partition("+")
    base, inc = float(base), float(inc or 0)
    model = json.load(open(args.model))
    if args.nps_scale != 1.0:
        model = dict(model, intercept=model["intercept"] * args.nps_scale,
                     slope_per_piece=model["slope_per_piece"] * args.nps_scale)

    a, b = Arm(args.arm_a), Arm(args.arm_b)
    a.engine = TracingEngine(args.tables)
    b.engine = TracingEngine(args.tables)
    openings = load_openings(args.openings, args.seed)[:args.rounds]
    if not openings:
        raise SystemExit("no openings parsed from %s" % args.openings)
    upper = math.log((1 - args.beta) / args.alpha)
    lower = math.log(args.beta / (1 - args.alpha))

    w = d = l = 0
    tel = {a.name: [], b.name: []}
    flags = {a.name: 0, b.name: 0}
    plies_seen, verdict = [], "book exhausted (SPRT undecided)"
    try:
        for g, fen in enumerate(openings):
            for a_white in (True, False):
                seed = args.seed * 1000003 + g * 2 + a_white
                white, black = (a, b) if a_white else (b, a)
                r, tele, moves = play_game(white, black, fen, base, inc,
                                           args.overhead, model, args.max_plies,
                                           seed, args.jitter)
                plies_seen.append(len(moves))
                for side, arm in ((chess.WHITE, white), (chess.BLACK, black)):
                    tel[arm.name].extend(tele[side])
                    if tele[side] and tele[side][-1]["clock"] < 0:
                        flags[arm.name] += 1
                r = r if a_white else -r
                w, d, l = w + (r > 0), d + (r == 0), l + (r < 0)
            llr = sprt_llr(w, d, l, args.elo0, args.elo1)
            if not args.quiet and ((g + 1) % 10 == 0 or llr >= upper or llr <= lower):
                elo, (lo, hi) = elo_estimate(w, d, l)
                print("[%3d pairs] %s vs %s  +%d =%d -%d  elo %+.1f [%+.1f, %+.1f]"
                      "  LLR %.2f (%.2f, %.2f)"
                      % (g + 1, a.name, b.name, w, d, l, elo, lo, hi, llr,
                         lower, upper), flush=True)
            if llr >= upper:
                verdict = "H1 accepted (elo >= %g)" % args.elo1
                break
            if llr <= lower:
                verdict = "H0 accepted (elo <= %g)" % args.elo0
                break
    finally:
        a.engine.quit()
        b.engine.quit()

    elo, (lo, hi) = elo_estimate(w, d, l)
    out = {"arm_a": a.name, "arm_b": b.name, "tc": args.tc,
           "overhead": args.overhead, "games": w + d + l, "w": w, "d": d, "l": l,
           "elo": elo, "ci": [lo, hi], "verdict": verdict,
           "median_plies": med(plies_seen), "arms": {}}
    print("RESULT %s vs %s @ virtual %s: %d games +%d =%d -%d  elo %+.1f "
          "[%+.1f, %+.1f]  median plies %d  %s"
          % (a.name, b.name, args.tc, w + d + l, w, d, l, elo, lo, hi,
             out["median_plies"], verdict))
    # medclk is the median over EVERY move's clock reading, not the clock at
    # the end of a game; minclk is the deepest the clock ever went, which is
    # where a park shows up as a hard floor.
    print("%-22s %7s %7s %7s %7s %7s %7s %7s %7s"
          % ("arm", "medspd", "maxspd", "medclk", "minclk", "blind%", "floorbk",
             "flags", "<2.4s"))
    for arm in (a, b):
        rows = tel[arm.name]
        sp = [r["spend"] for r in rows]
        ends = [r["clock"] for r in rows]
        stops = {}
        for r in rows:
            stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        out["arms"][arm.name] = {
            "moves": len(rows), "median_spend": med(sp), "max_spend": max(sp),
            "median_clock": med(ends),
            "min_clock": min(ends), "blind": sum(r["blind"] for r in rows),
            "fallback": sum(r["fallback"] for r in rows), "flags": flags[arm.name],
            "under_2v4": sum(1 for r in rows if r["clock"] < 2.4),
            # WHICH RULE ended each search.  A pool arm whose searches mostly
            # end on "soft-backstop" rather than "soft" is telling you the
            # bracket mirror is not seeing the convergence -- the exact defect
            # the mirror was built to avoid, and invisible without this count.
            "stops": dict(sorted(stops.items(), key=lambda kv: -kv[1])),
        }
        s = out["arms"][arm.name]
        print("%-22s %7.3f %7.3f %7.2f %7.2f %7.1f %7d %7d %7d"
              % (arm.name, s["median_spend"], s["max_spend"],
                 med(ends), s["min_clock"], 100.0 * s["blind"] / len(rows),
                 s["fallback"], s["flags"], s["under_2v4"]))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
