# Advanced UCI interface

import re, time
from random import random
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from functools import partial

print = partial(print, flush=True)

# Longest a "go ponder"/"go infinite" search may run without hearing
# "stop"/"ponderhit". Those commands carry no time budget, so the only thing
# that ends them is the GUI -- and if that message is ever lost the search
# would otherwise spin forever, which on a shared-core VM means starving
# everything else on the box. Ten minutes is longer than any real pondering
# turn, so this never fires in normal play; it is a liveness backstop.
UNBOUNDED_MAX_SECONDS = 600

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


def check_engine_module(module):
    """Fail now, with a name, rather than deep in a search loop."""
    missing = [attr for attr in ENGINE_API if not hasattr(module, attr)]
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


def go_loop(searcher, hist, stop_event, max_movetime=0, max_depth=0, debug=False):
    if debug:
        print(f"Going movetime={max_movetime}, depth={max_depth}")

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
    for depth, gamma, score, move in stop_softly(searcher, searcher.search(hist)):
        if depth > last_depth:
            best_move, last_depth = cand or best_move, depth
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
            break
        elapsed = time.perf_counter() - start
        fields = {
            "depth": depth,
            "time": round(1000 * elapsed),
            "nodes": searcher.nodes,
            "nps": round(searcher.nodes / max(elapsed, 1e-6)),
        }
        if score >= gamma:
            fields["score cp"] = f"{score} lowerbound"
            cand = render_move(move, white_pov=len(hist) % 2 == 1)
            fields["pv"] = " ".join(pv(searcher, hist[-1], include_scores=False))
        else:
            fields["score cp"] = f"{score} upperbound"
        print("info", " ".join(f"{k} {v}" for k, v in fields.items()))

        # We may not have a move yet at depth = 1
        if depth > 1:
            if elapsed > max_movetime * 2 / 3:
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
                        searcher.deadline = time.time() + ponder_think * 2 / 3
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
                        # Without movestogo, assume the game lasts another
                        # 40 moves.
                        movestogo = opts.get("movestogo", 40)
                        think = min(wtime / movestogo + winc, wtime / 2 - 1)
                        # Play the opening quickly: early moves benefit
                        # least from deep search, and banked time is worth
                        # more in the middlegame. Opening ONLY (an unscoped
                        # ramp starved whole games at long TCs; see #95).
                        # The random() varies the depth reached, giving our
                        # deterministic engine some opening variety.
                        if len(hist) < 8:
                            think = min(think, len(hist) + random())

                    if "depth" in opts:
                        max_depth = opts["depth"]

                    # A ponder search runs as "infinite", but remembers the
                    # think time computed above so "ponderhit" can apply it.
                    ponder_think = think if "ponder" in opts else None
                    if "infinite" in opts or "ponder" in opts:
                        think = 10**6

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
