"""The gate a mate suite is NOT: does the engine ALWAYS produce a LEGAL move?

LMP passed its mate gate 5-vs-5 on the very build that emitted an illegal
move, because "are mates found" and "is a move always produced" are different
questions. This asks the second one, and checks legality too.

Positions in CHECK are the dangerous class: this engine generates pseudo-legal
moves and has no notion of check, so the single legal reply can sort to the
tail where count-triggered pruning discards it.

Takes EITHER a .py source or a PACKED artifact. It used to launch every engine
as `[sys.executable, ENGINE]`, which runs a packed artifact -- a `#!/bin/bash`
self-extractor -- under python3, where it dies on line 1. The gate then scored
that as 100 chess failures: "no-move=40/30/30, GATE FAILED", on the shipped
entry, with the real cause (the engine never started) nowhere in the output.
An engine that cannot START is now a loud abort, never a chess verdict.

STARVATION is the second way to answer no move, and the three classes above
are blind to it. A build with mirrored eval tables passed this gate 100/100
and then answered `bestmove (none)` in a real game. It generated moves
perfectly; it never got to play one. `bound()` polls its budget every 2048
nodes and raises `Stop`, and `search()` reaches its FIRST yield only after the
depth-1, gamma=0 MTD probe completes. When that probe costs more nodes than
the budget, the driver is handed an empty yield stream and prints `(none)` --
no info line, no move, at any depth. The failing build needed 32,638 nodes for
that first probe against a 20,000-node budget; the shipped entry needs 23.

Two things were wrong with the old sample, and both are fixed below:

  * WRONG POSITIONS. Random-playout positions are wildly unbalanced, so the
    null-window probe at gamma=0 cuts off almost immediately -- median 2 nodes.
    The expensive class is QUIET, BALANCED, DENSE: real opening positions,
    where nothing resolves the window. On 334 of them the shipped entry's
    worst first probe is 582 nodes; every mirrored/fitted candidate exceeds
    2048 somewhere.
  * WRONG QUESTION. Asking "did a move come back" only catches starvation when
    the budget happens to land between the two, so it is a coin flip per
    position. The FIRST-YIELD arm asks the quantity itself, off the engine's
    own `info ... nodes` field, and needs no luck: if the first probe costs
    more than 2048 nodes -- the poll granularity, hence the smallest budget
    any engine can observe -- then SOME budget makes this engine answer
    `(none)`, and it is one arena setting away from forfeiting.

Usage:  legality_gate.py ENGINE [MOVETIME_MS] [--nodes N] [--first-yield N]
"""
import os
import subprocess, sys, chess, random

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
ENGINE = argv[0]
MOVETIME = int(argv[1]) if len(argv) > 1 else 300
# The node budget the build will actually be PLAYED at. The old gate only ever
# sent `go movetime`, so the fixed-node path the arena runs on was never
# exercised by the thing gating it.
NODES = int(opts.get("nodes", 20000))
# bound() polls the budget at `nodes % 2048 == 0`, so 2048 nodes is the
# smallest budget any engine can observe. A first yield costing more than that
# means a starving budget EXISTS; see the module docstring.
FIRST_YIELD = int(opts.get("first-yield", 2048))
OPENINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_openings.epd")
# A packed artifact is executable and not Python source; a generator output is
# .py and may not carry the exec bit.
ARGV = ([sys.executable, ENGINE] if ENGINE.endswith(".py")
        else [os.path.abspath(ENGINE)])
random.seed(20260813)


def positions(n_forced=40, n_check=30, n_plain=30):
    """Three classes. FORCED (in check, <=2 legal replies) is the dangerous one:
    the reproduction was in check with EXACTLY ONE legal move, which sorts to
    the tail where a count-triggered rule discards it. An earlier version of
    this gate sampled in-check positions indiscriminately and PASSED a build
    that demonstrably emits illegal moves -- most in-check positions have
    several escapes, so the pathological case almost never came up."""
    forced, checks, plains = [], [], []
    tries = 0
    while (len(forced) < n_forced or len(checks) < n_check or len(plains) < n_plain) and tries < 40000:
        tries += 1
        b = chess.Board()
        for _ in range(random.randint(4, 70)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(random.choice(ms))
        if b.is_game_over():
            continue
        line = " ".join(m.uci() for m in b.move_stack)
        n_legal = b.legal_moves.count()
        if b.is_check() and n_legal <= 2 and len(forced) < n_forced:
            forced.append((line, b))
        elif b.is_check() and len(checks) < n_check:
            checks.append((line, b))
        elif not b.is_check() and len(plains) < n_plain:
            plains.append((line, b))
    return forced, checks, plains


def ask(line, budget):
    """`budget` is a whole go-argument ("movetime 300" / "nodes 20000"), because
    the two paths starve differently and only one of them was ever tested."""
    p = subprocess.Popen(ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"uci\nisready\nposition startpos moves {line}\ngo {budget}\n")
    p.stdin.flush()
    mv = None
    saw = False
    while True:
        o = p.stdout.readline()
        if not o:
            break
        saw = True
        if o.startswith("bestmove"):
            mv = o.split()[1] if len(o.split()) > 1 else None
            break
    p.kill()
    if not saw:
        # Not a chess answer: the engine produced NOTHING at all. Fail loudly
        # with the launch error rather than logging it as a missing move.
        raise SystemExit("ENGINE DID NOT START: %s\nstderr: %s"
                         % (" ".join(ARGV), p.stderr.read()[-600:].strip()))
    return mv


def first_yield_nodes(fens):
    """Nodes the depth-1 gamma=0 probe costs, per position, off the engine's own
    `info` line -- the exact quantity that decides whether a budget can starve
    it. One process for the whole arm: a per-position spawn costs more than the
    measurement. `go nodes` is huge so the probe always completes, and `stop`
    goes out the moment the number is in hand, so each position costs about one
    poll interval rather than a full search.

    Returns (results, reason_skipped). An engine that emits no info lines --
    every PACKED artifact, whose builtin loop prints only `bestmove` -- cannot
    be measured this way, and that is reported as a SKIP, never as a pass."""
    p = subprocess.Popen(ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write("uci\nisready\n")
    p.stdin.flush()
    out = []
    for k, fen in enumerate(fens):
        # Fresh tables per position, or the measurement depends on scan order.
        p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo nodes 4000000\n")
        p.stdin.flush()
        n = None
        while True:
            o = p.stdout.readline()
            if not o:
                break
            # FIRST line only. `stop` takes a poll interval to land, so later
            # info lines keep arriving; recording them measured the stop
            # latency instead of the probe and inflated the shipped entry's
            # worst position from 582 nodes to 1879, against a 2048 budget.
            if n is None and o.startswith("info depth") and " nodes " in o:
                t = o.split()
                n = int(t[t.index("nodes") + 1])
                p.stdin.write("stop\n")
                p.stdin.flush()
            if o.startswith("bestmove"):
                break
        if n is None:
            if k == 0:
                p.kill()
                return [], ("engine emits no `info ... nodes` lines (packed "
                            "artifact / builtin loop): first-yield cannot be "
                            "measured through UCI")
            # It answered, but produced no probe at all -- that IS starvation.
            n = float("inf")
        out.append((n, fen))
    p.kill()
    return out, None


forced, checks, plains = positions()
fails = []
# BOTH budget paths. The mirrored build failed on `nodes`, which the gate that
# cleared it never sent.
for budget in (f"movetime {MOVETIME}", f"nodes {NODES}"):
    for label, group in (("FORCED", forced), ("IN CHECK", checks), ("quiet", plains)):
        none_ct = illegal_ct = 0
        for line, board in group:
            mv = ask(line, budget)
            if mv in (None, "(none)", "0000"):
                none_ct += 1
                fails.append((f"{label}/{budget}", board.fen(), mv, "NO MOVE"))
                continue
            try:
                m = chess.Move.from_uci(mv)
            except Exception:
                illegal_ct += 1
                fails.append((f"{label}/{budget}", board.fen(), mv, "UNPARSEABLE"))
                continue
            if m not in board.legal_moves:
                illegal_ct += 1
                fails.append((f"{label}/{budget}", board.fen(), mv, "ILLEGAL"))
        print(f"{label:9} go {budget:<16} n={len(group):3}  no-move={none_ct}  illegal={illegal_ct}")

fens = [l.split(";")[0].strip() for l in open(OPENINGS) if l.strip()]
measured, skipped = first_yield_nodes(fens)
starved = sorted((n, f) for n, f in measured if n > FIRST_YIELD)
if skipped:
    print(f"\nFIRST-YIELD  n={len(fens):3}  SKIPPED: {skipped}")
else:
    worst = max(n for n, _ in measured)
    print(f"\nFIRST-YIELD  n={len(measured):3}  budget={FIRST_YIELD}  "
          f"worst={worst}  over-budget={len(starved)}")
    for n, f in starved[-4:]:
        print(f"  {n:>9} nodes to first move   {f}")
    fails += [("FIRST-YIELD", f, f"{n} nodes", "STARVABLE") for n, f in starved]

print()
if fails:
    print(f"GATE FAILED: {len(fails)} bad answers")
    for lab, fen, mv, why in fails[:8]:
        print(f"  [{lab}] {why}: {mv}   {fen}")
elif skipped:
    # A skipped arm is not a passed arm. Say so, or the next lane reads a
    # green line as "starvation was checked" when it was not checked at all.
    print("GATE PASSED (LEGALITY ONLY): every position produced a legal move. "
          "The FIRST-YIELD arm did NOT run -- starvation is UNCHECKED on this "
          "build. Run it on the .py source to cover that.")
else:
    print("GATE PASSED: every position produced a legal move, "
          "and every first move arrived inside the smallest observable budget")
