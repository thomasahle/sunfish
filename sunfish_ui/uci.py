# Advanced UCI interface

import os, re, time
from random import random
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from functools import partial

print = partial(print, flush=True)

# Driver capability version. BUMP THIS whenever the driver gains a feature an
# engine may depend on, and raise the engine's required minimum in the same
# commit. A stale copy of this file shadowing the repo one (sys.path puts the
# engine's grandparent first) silently voided 425 games, nearly voided the
# node-cap fix, and cost a third debugging session in one night. A capability
# check catches a MISSING feature; only a version catches a STALE one.
#   1: go nodes / max_nodes support
#   2: node_cap enforced inside the search (mid-iteration, not per depth)
DRIVER_VERSION = 2

# Longest a "go ponder"/"go infinite" search may run without hearing
# "stop"/"ponderhit". Those commands carry no time budget, so the only thing
# that ends them is the GUI -- and if that message is ever lost the search
# would otherwise spin forever, which on a shared-core VM means starving
# everything else on the box. Ten minutes is longer than any real pondering
# turn, so this never fires in normal play; it is a liveness backstop.
UNBOUNDED_MAX_SECONDS = 600

# ---------------------------------------------------------------- TIME ----
# TWO TIME MANAGERS, chosen by the TM_MANAGER environment variable:
#
#   smooth (default)  the incumbent curve -- ONE number that is both the
#                     target and the wall. Whatever expression run() computes
#                     below is that manager; this block does not touch it.
#   pool              a whole-game resource POOL, divided into a SOFT limit
#                     (stop starting new iterations) and a HARD one (the
#                     in-search deadline). Thomas's design; see the PR.
#
#   P = max(0, T + (M-1)*I - (M+2)*O)     the pool this game still has
#   A = max(0, T - 2*O)                   what THIS move can safely reach
#   t_soft = min(s * P/M, A/4)            do not START a new depth past here
#   t_hard = min(5 * t_soft, A/2)         the wall, enforced inside bound()
#
# T is the clock, I the increment, O the per-move overhead, M the number of
# moves the pool has to cover, s a scale knob (1.0).
#
# EVERY QUANTITY HERE IS IN SECONDS. uci.py works in seconds throughout (the
# UCI millisecond fields are divided by 1000 as they are parsed) while the
# packed engine runs the same arithmetic in milliseconds; tests/
# test_time_budget.py asserts the two agree under t_ms = 1000*t_s, because
# that confusion has cost this project two incidents. The unit is named at
# every site below for the same reason.
#
# WHY A POOL. A single divisor has to answer two questions at once -- "what is
# this move worth" and "how long may one iteration run" -- and they pull it in
# opposite directions: low enough that a long iteration cannot flag, high
# enough that a normal move gets depth. Separating them lets the routine move
# be paced tightly (P/M) while an unstable one may still run to 5x that. It
# also prices the two things a divisor cannot see: the increment is INCOME (M-1
# further moves will earn it) and the per-move overhead is a TAX (M+2 moves pay
# it -- the +2 buys margin for the last move and for the flag itself).
#
# WHY THE A CLAMPS. A is the clock minus the overhead this move and its
# successor cannot avoid, so A/4 keeps at least three more moves' worth of
# clock behind every soft limit, and A/2 is a wall that CANNOT GO NEGATIVE.
# A negative wall is exactly how lichess.org/EAThUL0P was lost: under a 2s
# clock the old `wtime/2 - 1` cap went negative, the budget collapsed to a
# blind floor, and the engine played ~16 more moves at no search.
TM_MANAGER = os.environ.get("TM_MANAGER", "smooth")
if TM_MANAGER not in ("smooth", "pool"):
    raise SystemExit(f"TM_MANAGER={TM_MANAGER!r}: expected 'smooth' or 'pool'")

# O, SECONDS. 200ms is MEASURED, not chosen: the lichess autopsy of the lost
# 3+0 game puts ~200ms/move between our bestmove and the clock actually
# stopping (network plus process turnaround), and the packed twin's 60+0 drain
# forensics reproduce the same figure from the other side. A knob because it
# is a property of the deployment, not of the engine.
MOVE_OVERHEAD = float(os.environ.get("TM_OVERHEAD_MS", "200")) / 1000
# M when the GUI does not send movestogo. Sudden death has no horizon, so one
# has to be assumed; 40 is the classic choice and the one the /40 sudden-death
# evidence was gathered under.
POOL_MOVES = 40
# s, the soft scale. 1.0 ships; the knob exists so a retune is a run, not a
# patch.
SOFT_SCALE = float(os.environ.get("TM_SOFT_SCALE", "1.0"))
# PHASE-M ARM (TM_PHASE_M=1): let M fall with the move number instead of
# standing at 40 -- Lc0's phase curve, in its cheapest form. Spending rises
# through the middlegame, where depth buys the most.
PHASE_M = os.environ.get("TM_PHASE_M", "0") == "1"
# DYNAMIC-TARGET ARM (TM_DYNAMIC=1, "v1.1"): scale the soft limit by search
# stability. A separate knob so v1 can be screened STATIC first -- if the pool
# itself is a regression, no amount of stability tuning saves it, and mixing
# the two would leave us unable to say which half spoke.
TM_DYNAMIC = os.environ.get("TM_DYNAMIC", "0") == "1"
# Smallest positive budget, SECONDS. A degenerate clock (pool exhausted, or a
# GUI that has already flagged us) must still produce a legal move rather than
# a zero-length search: the packed engine's max(think, .05) in seconds.
TM_FLOOR = 0.05


def pool_budget(wtime, winc, movestogo=None, ply=0, overhead=None,
                phase_m=None, scale=None):
    """(t_soft, t_hard) in SECONDS, from a clock and increment in SECONDS.

    ply is the driver's move counter (len(hist), i.e. plies played), read only
    by the phase-M arm. movestogo, when the GUI sends one, is a real horizon
    and replaces M -- clamped into [1, 50] because GUIs send both 0 and
    absurdities, and because a 200-move horizon would pace us at nothing.
    """
    overhead = MOVE_OVERHEAD if overhead is None else overhead
    phase_m = PHASE_M if phase_m is None else phase_m
    scale = SOFT_SCALE if scale is None else scale
    if movestogo:
        moves = min(50, max(1, movestogo))
    elif phase_m:
        moves = max(20, 46 - ply / 2)
    else:
        moves = POOL_MOVES
    pool = max(0.0, wtime + (moves - 1) * winc - (moves + 2) * overhead)  # SECONDS
    avail = max(0.0, wtime - 2 * overhead)                                # SECONDS
    share = scale * pool / moves                                          # SECONDS
    if movestogo:
        # A known horizon is allowed to spend the clock down: at movestogo 1
        # everything but a safety margin is spendable, and holding back A/4
        # there would be leaving the game unplayed. 0.85 is that margin.
        #
        # UNTESTED IN GAMES, and deliberately loud about it: no staged-TC
        # match has run, and nothing we measure with sends movestogo at all
        # (fastchess does not at these TCs, lichess does not). The known
        # foot-gun is that 5x a HORIZON share is a much bigger bite than 5x a
        # fortieth -- measured through the real driver, `go wtime 60000
        # movestogo 10` spends 18.9 s on one move against the incumbent's
        # 6.1 s, all of it inside the 24.5 s wall. That is defensible with ten
        # moves to a control and 36 s left for nine of them, but it is a
        # BELIEF until a staged-TC match says otherwise.
        soft, wall = min(0.85 * share, 0.85 * avail), 0.85 * avail
    else:
        soft, wall = min(share, avail / 4), avail / 2
    hard = max(min(5 * soft, wall), TM_FLOOR)                             # SECONDS
    return min(max(soft, TM_FLOOR), hard), hard


def dynamic_target(soft, stable_iters=0, changed=False, score_drop=0.0, mate=False):
    """Scale the SOFT limit (SECONDS) by how settled the search looks.

    Three signals, each an extension of the same kind: a search that keeps
    changing its mind, or whose score is falling, is being asked a question
    the routine budget did not pay for.

      * stability: 1.15 at a fresh root, falling 0.08 per iteration that kept
        the same move, floored at 0.65 -- a move that has survived six
        iterations does not need a seventh.
      * a best-move CHANGE on this iteration: 1.35, unconditionally.
      * a score DROP, in centipawns: 1 + drop/200, capped at 1.75.

    MATE SCORES BYPASS the drop term: a mate score is not an evaluation, it is
    a different quantity, and differencing it against a centipawn score
    produces a meaningless (and enormous) "drop".
    """
    factor = min(1.15, max(0.65, 1.15 - 0.08 * stable_iters))
    if changed:
        factor *= 1.35
    if not mate:
        factor *= min(1.75, max(1.0, 1 + score_drop / 200))
    return soft * factor


# The engine is injected by run(), not imported: one interface drives
# classic sunfish, pesto and the NNUE variants, which are separate
# modules. That makes the engine's side of the contract implicit, so
# run() checks it up front and names whatever is missing.
#
# Checking eagerly is not just for the error message. `except X:`
# evaluates X only when an exception reaches the clause, so a missing
# Stop would go unseen for as long as no deadline fired, and would then
# raise AttributeError *while handling* the abort it was meant to catch
# - masking it. The same applies in weaker form to the rest: an engine
# without `pst` works until the first FEN arrives, i.e. until an EPD
# opening book is used, which is exactly how issue #156 stayed hidden.
#
# Optional, and so deliberately absent here: TABLE_SIZE (hasattr-guarded
# at both use sites) and features (its presence selects the NNUE
# scoring path in from_fen).
ENGINE_API = ("MATE_LOWER", "Move", "Position", "Searcher", "Stop",
              "opt_ranges", "parse", "render", "version")

# Required ONLY under TM_MANAGER=pool, and checked eagerly when it is set. The
# pool manager reads the soft limit when an MTD iteration converges, which the
# driver detects by mirroring the engine's own bisection bound -- so it needs
# that bound. Conditional rather than in ENGINE_API because an engine that
# never runs the pool manager has no business being rejected for it.
POOL_ENGINE_API = ("EVAL_ROUGHNESS",)


def check_engine_module(module):
    """Fail now, with a name, rather than deep in a search loop."""
    missing = [attr for attr in ENGINE_API if not hasattr(module, attr)]
    if TM_MANAGER == "pool":
        missing += [f"{a} (TM_MANAGER=pool reads it)"
                    for a in POOL_ENGINE_API if not hasattr(module, a)]
    if not hasattr(module, "features") and not hasattr(module, "pst"):
        # from_fen scores a board from the piece-square tables unless the
        # engine brings its own feature extractor.
        missing.append("pst (or features)")
    if missing:
        raise TypeError(
            f"{getattr(module, '__name__', module)!r} is not a usable sunfish "
            f"engine module: missing {', '.join(missing)}")


def render_move(move, white_pov):
    if move is None:
        return "(none)"
    i, j = move.i, move.j
    if not white_pov:
        i, j = 119 - i, 119 - j
    render = sunfish.render
    return render(i) + render(j) + move.prom.lower()


def parse_move(move_str, white_pov):
    parse = sunfish.parse
    i, j, prom = parse(move_str[:2]), parse(move_str[2:4]), move_str[4:].upper()
    if not white_pov:
        i, j = 119 - i, 119 - j
    return sunfish.Move(i, j, prom)


def stop_softly(searcher, gen):
    # Yield from the search until the in-search deadline aborts it
    try:
        yield from gen
    except sunfish.Stop:
        pass


def go_loop(searcher, hist, stop_event, max_movetime=0, max_depth=0, debug=False,
            max_nodes=0, requested_depth=None, open_ended=False, soft_movetime=None):
    # requested_depth: the depth the GUI actually typed, or None - max_depth
    # defaults to 100 and would otherwise be reported as if it were asked for.
    # open_ended: "go infinite"/"go ponder", where a stop is the terminating
    # condition rather than a truncation. Both feed the abort marker below.
    #
    # soft_movetime (SECONDS, pool manager only) splits the one budget in two.
    # max_movetime stays the HARD limit -- it is what run() armed the in-search
    # deadline with, and nothing about that machinery changes -- while
    # soft_movetime says when to stop STARTING iterations. None keeps the
    # incumbent rule: break at any yield past 2/3 of the single budget.
    if debug:
        print(f"Going movetime={max_movetime}, soft={soft_movetime}, "
              f"depth={max_depth}, nodes={max_nodes}")

    # perf_counter: monotonic and high-resolution everywhere - on Windows
    # time.time() ticks at ~15.6ms, so a fast first iteration measured 0.0
    # elapsed and the nps division crashed the search thread (found by the
    # Windows CI smoke job).
    start = time.perf_counter()
    # best_move is COMMITTED only when a depth completes (its MTD bracket
    # converged): a mid-depth fail-high can come from a deep fail-low
    # probe at an absurd gamma and is only a candidate. Before in-search
    # deadlines existed every stop fell between depths, so the last
    # fail-high was always from a completed depth and this distinction
    # was invisible; the deadline made mid-depth stops - and the Qxc6
    # class of giveaways - possible.
    best_move = cand = None
    last_depth = 1
    # Stability signals for the pool manager's dynamic target: the score that
    # came with the committed move, and how many iterations in a row have
    # agreed on the move. cand_score tracks the current depth's best (the MTD
    # bracket's lower bound) until it commits.
    best_score = cand_score = None
    stable_iters = 0
    target = soft_movetime
    # THE DRIVER'S MIRROR OF THE ENGINE'S MTD BRACKET. search() bisects
    # `while lower < upper - EVAL_ROUGHNESS`, tightening on the answer to each
    # probe, and the driver is handed every (gamma, score) pair -- so the same
    # two numbers can be maintained HERE, without touching the engine, and the
    # driver can tell when an iteration has converged.
    #
    # That is what makes a soft limit possible at all. The loop's only other
    # landmark is `depth > last_depth`, which arrives one FULL PROBE OF THE
    # NEXT DEPTH too late: measured on the packed twin, stopping there spends
    # 2.6s against a 1.29s soft limit at 60+0 and 6.8s against 2.27s at 60+1 --
    # a soft limit that is really a 2-3x multiplier, i.e. the opposite of the
    # design. Convergence is the moment before the next iteration starts, and
    # it is the moment the rule is about.
    lower, upper, converged_seen = float("-inf"), float("inf"), False
    # True once a limit we were GIVEN was actually met. Stays False when the
    # loop ends because a stop arrived, or because the in-search deadline
    # aborted the generator - the two are told apart by stop_event below.
    reached_limit = False
    if max_nodes:
        # stop mid-iteration, not between depths: a per-depth check rewards
        # whichever engine prunes less (bigger last iteration = bigger
        # overshoot). Measured 1.74x vs 1.32x at a 20000 cap.
        searcher.node_cap = max_nodes
    for depth, gamma, score, move in stop_softly(searcher, searcher.search(hist)):
        if depth > last_depth:
            best_move, last_depth = cand or best_move, depth
            if cand_score is not None:
                best_score = cand_score
            # BACKSTOP, and note what it is NOT. An engine may end an iteration
            # on a guard the mirror does not model (the packed twin stops on a
            # crossed bracket or a probe cap); then a new depth starting is the
            # only evidence there was, and the soft limit is read one probe
            # late rather than not at all. But when the mirror DID see the
            # convergence, continuing was a decision -- the iteration now
            # running was deliberately started, and stopping it here would undo
            # the "an iteration that has begun may finish" rule the wall exists
            # to bound.
            if (soft_movetime is not None and not converged_seen
                    and best_move is not None and time.perf_counter() - start > target):
                reached_limit = True
                break
            lower, upper, converged_seen = float("-inf"), float("inf"), False
        # Our max_depth implementation is a bit wasteful.
        # We never know when we've seen the last at a certain depth
        # before we get to the next one
        if depth - 1 >= max_depth:
            # This yield is the first probe of depth max_depth+1, and it ran
            # to completion at the sanest window of the whole search: gamma
            # sits inside the previous depth's converged bracket. A fail-high
            # here is finished, paid-for information, not a mid-dive artifact
            # (those arise on LATER probes of a depth, after the bracket
            # reset) - so play its move. This is exactly what the pv-walk
            # driver played at "go depth N" for years; dropping it cost 42
            # points of the WAC depth-3 floor.
            if score >= gamma and move is not None:
                best_move = render_move(move, white_pov=len(hist) % 2 == 1)
            reached_limit = True
            break
        elapsed = time.perf_counter() - start
        fields = {
            "depth": depth,
            "time": round(1000 * elapsed),
            "nodes": searcher.nodes,
            "nps": round(searcher.nodes / max(elapsed, 1e-6)),
        }
        if score >= gamma:
            if move is None:
                # A root fail-high without a move is a verified terminal
                # (bound()'s contract), so the score is exact, not a lower
                # bound. Report it - GUIs and testers deserve the draw/mate
                # score - and stop: there is nothing to search or play.
                fields["score cp"] = score
                print("info", " ".join(f"{k} {v}" for k, v in fields.items()))
                # Terminal root: the search is COMPLETE, not truncated.
                reached_limit = True
                break
            fields["score cp"] = f"{score} lowerbound"
            cand, cand_score = render_move(move, white_pov=len(hist) % 2 == 1), score
            fields["pv"] = " ".join(pv(searcher, hist[-1], include_scores=False))
        else:
            fields["score cp"] = f"{score} upperbound"
        print("info", " ".join(f"{k} {v}" for k, v in fields.items()))

        if soft_movetime is not None:
            # THE SOFT LIMIT, read where an iteration ends. Mirror the probe's
            # answer into the bracket first (max/min, so a contradictory probe
            # from an unstable search can only tighten it -- the packed twin's
            # rule, and identical to classic's plain assignment on an engine
            # whose values are consistent).
            if score >= gamma:
                lower = max(lower, score)
            else:
                upper = min(upper, score)
            if not lower < upper - sunfish.EVAL_ROUGHNESS:
                converged_seen = True
                # Converged: the next probe would start a new depth, which is
                # exactly what the soft limit forbids past its target. Commit
                # this iteration's answer either way -- it is finished, and
                # committing it here is what the depth transition would do one
                # probe later.
                settled = cand or best_move
                if settled is not None:
                    changed = best_move is not None and settled != best_move
                    stable_iters = 0 if changed else stable_iters + 1
                    target = soft_movetime
                    if TM_DYNAMIC:
                        mate = any(s is not None and abs(s) >= sunfish.MATE_LOWER
                                   for s in (best_score, cand_score))
                        drop = 0.0
                        if not mate and None not in (best_score, cand_score):
                            drop = max(0, best_score - cand_score)
                        target = dynamic_target(soft_movetime, stable_iters, changed, drop, mate)
                    best_move = settled
                    if cand_score is not None:
                        best_score = cand_score
                    if elapsed > target:
                        reached_limit = True
                        break

        # We may not have a move yet at depth = 1
        if depth > 1:
            # The incumbent rule: one budget, two thirds of it, checked at
            # every yield. Under the pool manager the soft limit above owns
            # the decision and this must NOT also fire -- max_movetime is the
            # wall there, and stopping at 2/3 of a wall is not a wall.
            if soft_movetime is None and elapsed > max_movetime * 2 / 3:
                reached_limit = True
                break
            # "go nodes N": equal-thinking matches (fixed-node testing).
            # Checked between probes, so both sides overshoot by at most
            # one MTD probe - symmetric, which is all fixed-node needs.
            if max_nodes and searcher.nodes >= max_nodes:
                reached_limit = True
                break
            if stop_event.is_set():
                break

    # FIXME: If we are in "go infinite" we aren't actually supposed to stop the
    # go-loop before we got stop_event. Unfortunately we currently don't know if
    # we are in "go infinite" since it's simply translated to "go depth 100".

    # Play the committed move (last completed depth); fall back to the
    # current depth's candidate only if no depth ever completed. The pv
    # walk is used for the ponder hint only when it agrees with what we
    # actually play - tp_move[root] can hold a mid-dive artifact.
    # Diagnostic tripline for the unexplained early-return reports (~3/28
    # local once, 0.1-0.2s against a >1s budget; NOT reproduced in 260
    # scripted go/ponderhit/stop cycles).  A timed search that ends far
    # under budget without a stop announces itself -- "info string" is
    # spec-legal, GUIs and lichess-bot log it, so a live occurrence
    # self-documents instead of vanishing.
    elapsed = time.perf_counter() - start
    # Against the SOFT limit when there is one: a pool search that stops at
    # its soft limit is a fifth of the way to its wall by design, and a
    # tripline that fires on every routine move is not a tripline.
    budget = max_movetime if soft_movetime is None else soft_movetime
    if (elapsed < budget / 3 and budget < 10**5
            and not stop_event.is_set() and last_depth < 5):
        print(f"info string EARLY-RETURN-DIAG elapsed={elapsed:.3f} "
              f"budget={budget:.2f} depth={last_depth} "
              f"deadline_in={(searcher.deadline or 0) - time.time():.3f}")

    # A search that ends because "stop"/"quit"/EOF arrived, BEFORE the limit it
    # was given, answers a shallower question than the one it was asked - and
    # said so nowhere. Stopping ASAP is correct UCI; being quiet about it is a
    # silent degrade, which this engine does not do (AGENTS.md).
    #
    # The case that costs people days: a one-shot harness pipes
    # `go depth 8` and `quit` together, stdin is drained eagerly, the stop
    # lands during depth 1, and a DEPTH-2 result comes back wearing a
    # well-formed info line and a well-formed bestmove. Nothing distinguishes
    # it from a finished depth-8 search. See docs/TESTING.md rule 13.
    #
    # Not emitted for "go infinite"/"go ponder": there the stop IS the
    # terminating condition, so there is nothing to warn about.
    if stop_event.is_set() and not reached_limit and not open_ended:
        if requested_depth is not None:
            asked = f"depth {requested_depth}"
        elif max_nodes:
            asked = f"nodes {max_nodes}"
        else:
            asked = f"movetime {max_movetime:.2f}s"
        print(f"info string aborted at depth {last_depth} "
              f"(nodes {searcher.nodes}, {elapsed:.2f}s) before requested {asked}")

    played = best_move or cand
    my_pv = pv(searcher, hist[-1], include_scores=False)
    if played and len(my_pv) > 1 and my_pv[0] == played:
        # Suggest the expected reply for the GUI to let us ponder on
        print("bestmove", played, "ponder", my_pv[1])
    else:
        print("bestmove", played or (my_pv[0] if my_pv else "(none)"))


def mate_loop(
    searcher,
    hist,
    stop_event,
    max_movetime=0,
    max_depth=0,
    find_draw=False,
    debug=False,
):
    start = time.time()
    try:
      for d in range(int(max_depth) + 1):
        if find_draw:
            s0 = searcher.bound(hist[-1], 0, d)
            elapsed = time.perf_counter() - start
            print("info", "depth", d, "score lowerbound cp", s0)
            s1 = searcher.bound(hist[-1], 1, d)
            elapsed = time.perf_counter() - start
            print("info", "depth", d, "score upperbound cp", s1)
            if s0 >= 0 and s1 < 1:
                break
        else:
            score = searcher.bound(hist[-1], sunfish.MATE_LOWER, d, root=True)
            elapsed = time.perf_counter() - start
            print(
                "info depth",
                d,
                "score cp",
                score,
                "time",
                round(1000 * elapsed),
                "pv",
                " ".join(pv(searcher, hist[-1], include_scores=False)),
            )
            if score >= sunfish.MATE_LOWER:
                break
        if elapsed > max_movetime:
            break
        if stop_event.is_set():
            break
    except sunfish.Stop:
        pass
    move = searcher.tp_move.get(hist[-1])
    move_str = render_move(move, white_pov=len(hist) % 2 == 1)
    print("bestmove", move_str)


def perft(pos, depth, debug=False):

    def _perft_count(pos, depth):
        # Check that we didn't get to an illegal position
        if can_kill_king(pos):
            return -1
        if depth == 0:
            return 1
        res = 0
        for move in pos.gen_moves():
            cnt = _perft_count(pos.move(move), depth - 1)
            if cnt != -1:
                res += cnt
        return res

    total = 0
    for move in pos.gen_moves():
        move_uci = render_move(move, get_color(pos) == WHITE)
        cnt = _perft_count(pos.move(move), depth - 1)
        if cnt != -1:
            print(f"{move_uci}: {cnt}")
            total += cnt
    print()
    print("Nodes searched:", total)


def run(sunfish_module, startpos):
    global sunfish
    check_engine_module(sunfish_module)
    sunfish = sunfish_module

    debug = False
    hist = [startpos]
    searcher = sunfish.Searcher()

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Noop future to get started
        go_future = executor.submit(lambda: None)
        do_stop_event = Event()
        # The think time to apply when "ponderhit" arrives
        ponder_think = None

        while True:
            try:
                args = input().split()
                if not args:
                    continue

                elif args[0] in ("stop", "quit"):
                    searcher.deadline = 0
                    # Check done() rather than running(): the future is still
                    # pending if "stop" arrives right after "go", and the stop
                    # must not be lost.
                    if not go_future.done():
                        if debug:
                            print("Stopping go loop...")
                        do_stop_event.set()
                        go_future.result()
                    elif debug:
                        print("Go loop not running...")
                    if args[0] == "quit":
                        break

                elif args[0] == "ponderhit":
                    # The predicted move was played, so our clock starts now:
                    # give the (already running) ponder search its time budget,
                    # enforced by the in-search deadline.
                    if ponder_think and not go_future.done():
                        searcher.deadline = time.time() + ponder_think
                    continue

                # The UCI spec requires us to answer "isready" even while
                # searching.
                elif args[0] == "isready":
                    print("readyok")
                    continue

                elif not go_future.done():
                    # The previous search may have just printed its bestmove,
                    # with the thread still winding down for a few more
                    # microseconds. Commands racing that window (as pondering
                    # GUIs do) must be processed, not dropped. If the search
                    # is genuinely still running, the GUI broke protocol:
                    # stop the search rather than hang.
                    try:
                        go_future.result(timeout=1)
                    except TimeoutError:
                        do_stop_event.set()
                        searcher.deadline = 0
                        go_future.result()

                # Make sure we are really done, and throw any errors that may have
                # happened in the go loop.
                go_future.result(timeout=0)

                if args[0] == "uci":
                    print(f"id name {sunfish.version}")
                    for attr, (lo, hi) in sunfish.opt_ranges.items():
                        default = getattr(sunfish, attr)
                        print(
                            f"option name {attr} type spin default {default} min {lo} max {hi}"
                        )
                    print("option name Ponder type check default false")
                    if hasattr(sunfish, "TABLE_SIZE"):
                        # The standard UCI Hash option is in MB; a sunfish
                        # table entry costs roughly 1KB.
                        print(f"option name Hash type spin default {sunfish.TABLE_SIZE // 1000} min 1 max 32768")
                    print("uciok")

                elif args[0] == "setoption":
                    _, uci_key, _, uci_value = args[1:]
                    if uci_key == "Hash" and hasattr(sunfish, "TABLE_SIZE"):
                        # Standard UCI Hash is in MB, ~1KB per table entry
                        sunfish.TABLE_SIZE = int(uci_value) * 1000
                    # Skip options we don't store, like "Ponder"
                    elif uci_key in sunfish.opt_ranges:
                        setattr(sunfish, uci_key, int(uci_value))

                # Tournament managers reuse the engine process for many games.
                # Start each game with fresh tables: it frees the memory of
                # finished games, and tournament-testing showed stale
                # cross-game entries actually cost strength.
                elif args[0] == "ucinewgame":
                    searcher = sunfish.Searcher()

                elif args[:2] == ["position", "startpos"]:
                    hist = [startpos]
                    for ply, move in enumerate(args[3:]):
                        hist.append(hist[-1].move(parse_move(move, ply % 2 == 0)))

                elif args[:2] == ["position", "fen"]:
                    pos = from_fen(*args[2:8])
                    hist = [pos] if get_color(pos) == WHITE else [pos.rotate(), pos]
                    if len(args) > 8:
                        assert args[8] == "moves"
                        for move in args[9:]:
                            hist.append(hist[-1].move(parse_move(move, len(hist) % 2 == 1)))

                elif args[0] == "go":
                    think = 10**6
                    # The pool manager's soft limit, SECONDS; None means the
                    # incumbent single-budget rule (see go_loop).
                    soft_think = None
                    max_depth = 100
                    loop = go_loop

                    # Per the UCI spec the go arguments are key(-value) tokens
                    # that may come in any order and combination. E.g. a GUI
                    # may send "go wtime 60000 btime 60000" for sudden death,
                    # or "go btime 5000 wtime 5000 movestogo 10".
                    opts = {}
                    tokens = iter(args[1:])
                    for tok in tokens:
                        if tok in ("infinite", "ponder"):
                            opts[tok] = True
                        elif tok == "searchmoves":
                            # All remaining tokens are moves
                            opts[tok] = list(tokens)
                        elif tok in ("wtime", "btime", "winc", "binc",
                                     "movestogo", "depth", "nodes", "mate",
                                     "movetime", "draw", "perft"):
                            opts[tok] = int(next(tokens))

                    if "movetime" in opts:
                        think = opts["movetime"] / 1000

                    elif "wtime" in opts or "btime" in opts:
                        wtime = opts.get("wtime", 0) / 1000
                        btime = opts.get("btime", 0) / 1000
                        winc = opts.get("winc", 0) / 1000
                        binc = opts.get("binc", 0) / 1000
                        # we always consider ourselves white, but uci doesn't
                        if len(hist) % 2 == 0:
                            wtime, winc = btime, binc
                        movestogo = opts.get("movestogo")
                        if TM_MANAGER == "pool":
                            # SECONDS in, SECONDS out. think is the WALL (it
                            # arms the deadline below, exactly as before);
                            # soft_think is where new iterations stop.
                            soft_think, think = pool_budget(wtime, winc, movestogo, ply=len(hist))
                        elif movestogo:
                            # an explicit movestogo is a real constraint
                            think = min(wtime / movestogo + winc, wtime / 2 - 1)
                        else:
                            # Increment-aware budget (11-game production
                            # audit): t/40+inc structurally underspent the
                            # clock -- 2.9s of a 35s clock at +2s, median
                            # depth 7 at EVERY TC from 60+1 to 300+5, zero
                            # time-pressure blunders, and 57% of rating
                            # bleed was sub-150cp depth-ceiling drift.
                            # /12 + 0.9*inc front-loads the middlegame
                            # where depth buys Elo; worst-case spend sims
                            # keep >=4s of clock at all tested TCs, the
                            # wtime/2 cap floors tiny clocks, and the
                            # armed in-search deadline guards the hard
                            # edge.  Validated per TESTING.md rule 5
                            # (multi-TC + per-move curves).
                            think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)
                        # Play the opening quickly: early moves benefit
                        # least from deep search, and banked time is worth
                        # more in the middlegame. Opening ONLY (an unscoped
                        # ramp starved whole games at long TCs; see #95).
                        # The random() varies the depth reached, giving our
                        # deterministic engine some opening variety.
                        if len(hist) < 8:
                            think = min(think, len(hist) + random())
                            if soft_think is not None:
                                soft_think = min(soft_think, think)

                    if "depth" in opts:
                        max_depth = opts["depth"]

                    # A ponder search runs as "infinite", but remembers the
                    # think time computed above so "ponderhit" can apply it.
                    # It runs with think = 10**6, so the loop's soft break
                    # cannot fire and the deadline has to carry the whole
                    # budget: the SOFT limit is the right value for it, and
                    # 2/3*think is exactly that under the incumbent manager.
                    ponder_think = None
                    if "ponder" in opts:
                        ponder_think = soft_think if soft_think is not None else think * 2 / 3
                    if "infinite" in opts or "ponder" in opts:
                        think, soft_think = 10**6, None

                    if "mate" in opts or "draw" in opts:
                        max_depth = opts.get("mate", opts.get("draw"))
                        loop = partial(mate_loop, find_draw="draw" in opts)

                    if "perft" in opts:
                        perft(hist[-1], opts["perft"], debug=debug)
                        continue

                    do_stop_event.clear()
                    # Hard wall-clock cap, checked inside the search itself
                    # (sunfish.py Searcher.bound), so budgets hold even when
                    # single iterations run long on slow hardware.
                    #
                    # "go ponder"/"go infinite" set think = 10**6, which used
                    # to leave the deadline unset entirely. Such a search only
                    # ends on "stop"/"ponderhit", so a lost stop (wedged GUI,
                    # dropped connection) pins a CPU forever. Cap them instead:
                    # UNBOUNDED_MAX_SECONDS is far longer than any real ponder,
                    # so normal play is unaffected and the cap only fires when
                    # the command that should have stopped us is already gone.
                    searcher.deadline = time.time() + min(think, UNBOUNDED_MAX_SECONDS)
                    go_future = executor.submit(
                        loop,
                        searcher,
                        hist,
                        do_stop_event,
                        think,
                        max_depth,
                        debug=debug,
                        # mate_loop has no node budget; fixed-node play is a
                        # go_loop concern only. Same for the abort marker's
                        # two inputs: what the GUI actually asked for, and
                        # whether it asked for anything finite at all.
                        **({"max_nodes": opts["nodes"]}
                           if "nodes" in opts and loop is go_loop else {}),
                        **({"requested_depth": opts.get("depth"),
                            "open_ended": "infinite" in opts or "ponder" in opts,
                            "soft_movetime": soft_think}
                           if loop is go_loop else {}),
                    )

                    # Make sure we get informed if the job fails
                    def callback(fut):
                        fut.result(timeout=0)

                    go_future.add_done_callback(callback)

            except (KeyboardInterrupt, EOFError):
                if go_future.running():
                    if debug:
                        print("Stopping go loop...")
                    do_stop_event.set()
                    go_future.result()
                break


# Old tools stuff

WHITE, BLACK = range(2)


def from_fen(board, color, castling, enpas, _hclock, _fclock):
    board = re.sub(r"\d", (lambda m: "." * int(m.group(0))), board)
    board = list(21 * " " + "  ".join(board.split("/")) + 21 * " ")
    board[9::10] = ["\n"] * 12
    board = "".join(board)
    wc = ("Q" in castling, "K" in castling)
    bc = ("k" in castling, "q" in castling)
    ep = sunfish.parse(enpas) if enpas != "-" else 0
    if hasattr(sunfish, 'from_board'):
        # engines carrying an evaluation accumulator (nnue_4k) build the
        # position, and the accumulator, themselves
        pos = sunfish.from_board(board, wc, bc, ep, 0)
        return pos if color == 'w' else pos.rotate()
    if hasattr(sunfish, 'features'):
        wf, bf = sunfish.features(board)
        pos = sunfish.Position(board, 0, wf, bf, wc, bc, ep, 0)
        pos = pos._replace(score=pos.calculate_score())
    else:
        score = sum(sunfish.pst[c][i] for i, c in enumerate(board) if c.isupper())
        score -= sum(sunfish.pst[c.upper()][119-i] for i, c in enumerate(board) if c.islower())
        pos = sunfish.Position(board, score, wc, bc, ep, 0)
    return pos if color == 'w' else pos.rotate()


def get_color(pos):
    """A slightly hacky way to to get the color from a sunfish position"""
    return BLACK if pos.board.startswith("\n") else WHITE


def can_kill_king(pos):
    # If we just checked for opponent moves capturing the king, we would miss
    # captures in case of illegal castling.
    #MATE_LOWER = 60_000 - 10 * 929
    #return any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
    return any(pos.board[m.j] == 'k' or abs(m.j - pos.kp) < 2 for m in pos.gen_moves())


def pv(searcher, pos, include_scores=True, include_loop=False):
    res = []
    seen_pos = set()
    color = get_color(pos)
    origc = color
    if include_scores:
        res.append(str(pos.score))
    while True:
        if hasattr(pos, "wf"):
            move = searcher.tp_move.get(pos.hash())
        elif hasattr(searcher, "tp_move"):
            move = searcher.tp_move.get(pos)
        elif hasattr(searcher, "tt_new"):
            move = searcher.tt_new[0][pos, True].move
        # The tp may have illegal moves, given lower depths don't detect king killing
        if move is None or can_kill_king(pos.move(move)):
            break
        res.append(render_move(move, get_color(pos) == WHITE))
        pos, color = pos.move(move), 1 - color

        if hasattr(pos, "wf"):
            if pos.hash() in seen_pos:
                if include_loop:
                    res.append("loop")
                break
            seen_pos.add(pos.hash())
        else:
            if pos in seen_pos:
                if include_loop:
                    res.append("loop")
                break
            seen_pos.add(pos)

        if include_scores:
            res.append(str(pos.score if color == origc else -pos.score))
    return res
