"""How many nodes does the engine need before it has a move it can play?

This is the gate that would have caught C1 before it played 651 games and
answered `bestmove (none)` in one of them.

THE MECHANISM, exactly. `main()` (and `sunfish_ui/uci.py`) can only print a
move the search handed it, and the search hands one over only on a ROOT
FAIL-HIGH -- a yield with `score >= gamma` and a move. Both stop conditions,
the node cap and the wall-clock deadline, are polled in the same place:

    if self.nodes % 2048 == 0 and self.nodes > self.node_cap: raise Stop
    if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop

so the EARLIEST an abort can land is node 2048. Call the node count of the
first root fail-high with a move the position's FIRST YIELD. A build whose
first yield exceeds 2048 has a budget -- of nodes or of time, this is not a
fixed-node-only hazard -- at which it prints `bestmove (none)`, which the
opponent's GUI scores as an illegal move and the loss of a game. C1 was the
extreme case: it never failed high at all, at any depth, and printed no info
line either.

WHY THE NUMBER AND NOT THE SYMPTOM. The first version of this gate asked the
binary question -- `go nodes 1`, is the answer `(none)` -- and over 505
positions it caught C1 on exactly ONE, the position already known to fail.
A gate whose power comes from carrying its own reproducer catches the bug it
was written for and nothing else. Measuring the first-yield node count instead
turns a 1-in-505 event into a distribution with a margin: a build whose worst
position needs 1,900 of its 2,048 nodes has not failed, but it is one ordering
change away from failing, and that is visible here and nowhere else.

THE POSITION SAMPLE IS THE OTHER HALF. The 100-position legality gate PASSED
C1: its positions come from random playouts, and the failure lives in normal
middlegames from our own games. This gate samples OUR OWN GAME POSITIONS,
phase-stratified, and carries the C1 reproducer as position 0.

Part A (in process, the verdict) measures first yield per position directly.
Part B (subprocess, the control) confirms on a subsample that the number means
what it claims: first yield > 2048 iff `go nodes 1` answers `(none)`. Without
B this gate measures a generator whose relationship to the played move is
assumed; the ledger has enough entries about instruments nobody controlled.

Source entries only (`go nodes` is testing-only and is stripped from the
packed artifact), which is the same surface every fixed-node screen uses.

usage: first_yield_gate.py ENTRY.py [FENS.txt]
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = str(HERE.parents[1])
ENGINE = os.path.abspath(sys.argv[1])
# .fen, not .txt: the repo's .gitignore carries a blanket `*.txt`, which is
# how this project already lost a 15,328-position training set. A gate whose
# positions are untracked is a gate nobody else can run.
FENS = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "first_yield_fens.fen")
WINDOW = 2048            # the engine's own stop-poll granularity
CONTROL_N = 12           # positions re-checked end to end through real UCI
assert ENGINE.endswith(".py"), "source entries only: `go nodes` is stripped from the packed artifact"

sys.path.insert(0, REPO)
import sunfish_ui.uci as uci                                       # noqa: E402

spec = importlib.util.spec_from_file_location("entry_under_test", ENGINE)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
# from_fen reads its engine from a module global that run() would normally
# set. Setting it explicitly is not a nicety: an unset `sunfish` here builds
# the board with a DIFFERENT engine's layout.
uci.sunfish = E


def first_yield(fen, cap=1 << 20):
    """Nodes at the first root fail-high carrying a move; None if there is none."""
    board, color, castling, enpas = fen.split()[:4]
    pos = uci.from_fen(board, color, castling, enpas, 0, 1)
    s = E.Searcher()
    s.node_cap, s.deadline = cap, 1 << 63
    try:
        for depth, gamma, score, move in s.search([pos]):
            if score >= gamma and move is not None:
                return s.nodes
    except E.Stop:
        pass
    return None


def plays_a_move(fen):
    """`go nodes 1` through the real UCI surface: abort lands at node 2048."""
    env = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.Popen([sys.executable, ENGINE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1, env=env)
    # A clock is required: with no time tokens the driver computes a NEGATIVE
    # budget and stops after depth 2 for reasons unrelated to this test.
    p.stdin.write("uci\nisready\nposition fen %s\ngo nodes 1 wtime 3600000 btime 3600000\n" % fen)
    p.stdin.flush()
    mv, saw, driver = None, False, None
    while True:
        o = p.stdout.readline()
        if not o:
            break
        saw = True
        if o.startswith("info string driver"): driver = o.strip()
        if o.startswith("bestmove"):
            mv = o.split()[1] if len(o.split()) > 1 else None
            break
    p.kill()
    if not saw:
        raise SystemExit("ENGINE DID NOT START: %s %s\nstderr: %s"
                         % (sys.executable, ENGINE, p.stderr.read()[-600:].strip()))
    # THE CONTROL'S OWN FAILURE MODE, and it bit this gate on its first run.
    # An entry that resolves no sunfish_ui/ falls through to the BUILTIN uci
    # loop, which knows only `position startpos`: it ignores the FEN, searches
    # the opening position, and answers a legal-looking move. A gate that
    # accepts that answer reports PASS for every build ever written.
    if driver is None or " fen" not in driver:
        raise SystemExit("WRONG DRIVER: %s resolved to %s -- the builtin loop ignores "
                         "`position fen`." % (ENGINE, driver or "the builtin loop (no banner)"))
    return mv not in (None, "(none)", "0000")


fens = [ln.strip() for ln in open(FENS) if ln.strip()]
ys = [(f, first_yield(f)) for f in fens]
over = [(f, y) for f, y in ys if y is None or y > WINDOW]
got = sorted(y for _, y in ys if y is not None)

print("first-yield gate: %s" % os.path.basename(ENGINE))
print("  positions %d   window %d nodes" % (len(fens), WINDOW))
if got:
    q = lambda p: got[min(len(got) - 1, int(p * len(got)))]
    print("  first yield  median %d  p90 %d  p99 %d  MAX %d"
          % (q(.5), q(.9), q(.99), got[-1]))
print("  over window: %d   never yields: %d"
      % (sum(1 for _, y in over if y is not None), sum(1 for _, y in over if y is None)))

# Control: the measured number and the played move must agree. Check the worst
# positions (where they can disagree) plus a fixed slice of ordinary ones.
worst = sorted(ys, key=lambda t: (t[1] is not None, t[1]), reverse=True)[:CONTROL_N // 2]
sample = worst + ys[:CONTROL_N - len(worst)]
bad = [(f, y, plays_a_move(f)) for f, y in sample]
bad = [(f, y, m) for f, y, m in bad if m != (y is not None and y <= WINDOW)]
if bad:
    print("\nCONTROL FAILED: the first-yield number does not predict the played move")
    for f, y, m in bad[:4]:
        print("  first_yield=%s but plays_a_move=%s   %s" % (y, m, f))
    sys.exit(2)
print("  control: %d positions agree with `go nodes 1` through real UCI" % len(sample))

print()
if over:
    print("GATE FAILED: %d position(s) need more than %d nodes for a playable move" % (len(over), WINDOW))
    for f, y in over[:8]:
        print("  %s: %s" % ("never yields" if y is None else "%d nodes" % y, f))
    sys.exit(1)
print("GATE PASSED: every position yields a playable move inside the first %d nodes" % WINDOW)
