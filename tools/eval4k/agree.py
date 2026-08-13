"""Move-agreement between two engine SOURCES at fixed nodes.

An exact re-encoding of the tables needs no match to justify it -- it needs
proof that it is exact, and the proof is behavioural: same positions, same
node budget, same move and same score, every time.  For a deliberately lossy
encoding the same instrument bounds the risk before any games are queued.

Two harness rules, both learned the hard way in one evening on this file:

  * The engine SOURCE is driven, not the packed artifact.  The artifact's
    built-in UCI loop understands `position startpos moves ...` and
    `movetime` and NOTHING else -- it silently ignores `position fen` and
    reads `go depth 5` as "60 s on the clock".  A first version of this
    script used both and would have reported agreement figures for the wrong
    positions at a nondeterministic budget.
  * Fixed NODES, via the sunfish_ui driver.  Wall-clock budgets make the
    comparison nondeterministic, so disagreements could not be attributed.
  * `go nodes N` alone is NOT fixed effort.  The node cap is an ADDITIONAL
    stop on top of the clock, and with no time fields the engine defaults to
    wtime=60000 -> a 1.5 s deadline and a 1.2 s soft break.  On a loaded
    machine that break fires before the cap and the reported score becomes a
    function of the load: A against a byte-identical copy of A scored 60/60
    on moves but only 21/60 on scores.  So send an hour on the clock and let
    the node cap be the only binding limit.  Any future "same score" figure
    below 60/60 must first be checked against the A-vs-A control below.

Self-test: after comparing the pair, the script compares A against a
byte-identical copy of A (which MUST agree everywhere, or the instrument is
measuring the machine) and then against a perturbed A (which MUST disagree,
or the instrument cannot see an eval change at all).

usage: agree.py A.py B.py [NODES] [NPOS]
"""
import atexit
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

import chess

REPO = str(pathlib.Path(__file__).resolve().parents[2])
# Both engines must be STAGED into one directory before they are compared.
# The entry's main() does `sys.path.insert(0, grandparent(__file__))` and then
# imports sunfish_ui: an engine sitting at REPO/nnue_4k/x.py resolves the full
# driver, while the same bytes copied to /var/folders/... find nothing and drop
# into the builtin UCI loop, whose `go` parsing is a different program.  That
# is how a byte-identical copy of the entry "disagreed" with itself on 39 of 60
# scores.  A staging dir one level under REPO gives every arm the same driver.
_STAGE = tempfile.mkdtemp(prefix=".agree-", dir=REPO)
atexit.register(shutil.rmtree, _STAGE, True)


def stage(path, name=None):
    """Copy an engine into the shared staging dir; return the staged path."""
    dst = os.path.join(_STAGE, name or os.path.basename(path))
    shutil.copyfile(path, dst)
    return dst


def positions(n, seed=20260813):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(2, 60)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        if b.is_game_over():
            continue
        out.append(" ".join(m.uci() for m in b.move_stack))
    return out


def ask(engine, lines, nodes):
    cmd = "uci\nisready\n"
    for line in lines:
        # An hour on both clocks so the node cap, not the deadline, stops us.
        cmd += "position startpos moves %s\ngo wtime 3600000 btime 3600000 nodes %d\n" % (line, nodes)
    cmd += "quit\n"
    r = subprocess.run(["pypy3", engine], input=cmd, capture_output=True, text=True, timeout=3600)
    moves, scores, cur, driver = [], [], None, None
    for out in r.stdout.splitlines():
        if out.startswith("info string driver"):
            driver = out
        elif out.startswith("info depth") and " score cp " in out:
            cur = out.split(" score cp ")[1].split()[0]
        elif out.startswith("bestmove"):
            moves.append(out.split()[1] if len(out.split()) > 1 else None)
            scores.append(cur)
            cur = None
    if len(moves) != len(lines):
        raise RuntimeError("%s answered %d of %d: %s" %
                           (engine, len(moves), len(lines), r.stderr[-300:]))
    return moves, scores, driver


def compare(a, b, lines, nodes, label):
    ma, sa, da = ask(a, lines, nodes)
    mb, sb, db = ask(b, lines, nodes)
    # Which UCI driver answered is part of the answer: two arms on different
    # drivers are not comparable at all, whatever their agreement figure says.
    if da != db:
        raise RuntimeError("the two arms resolved DIFFERENT drivers, so the "
                           "comparison is meaningless:\n  A: %s\n  B: %s" % (da, db))
    n = len(lines)
    sm = sum(x == y for x, y in zip(ma, mb))
    ss = sum(x == y for x, y in zip(sa, sb))
    print("%-10s positions %d  nodes %d   same move %d/%d (%.1f%%)   same score %d/%d" %
          (label, n, nodes, sm, n, 100.0 * sm / n, ss, n))
    return sm == n and ss == n


def main():
    import re
    a = stage(sys.argv[1], "a.py")
    b = stage(sys.argv[2], "b.py")
    nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    npos = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    lines = positions(npos)
    ok = compare(a, b, lines, nodes, "A vs B")

    # POSITIVE control: A against a byte-identical copy of itself.  Anything
    # short of 60/60 here is the harness or the machine, never the candidate,
    # and it invalidates the A-vs-B row above.
    same = stage(a, "identical.py")
    if not compare(a, same, lines, nodes, "self"):
        print("CONTROL FAILED: the engine disagreed with a byte-identical "
              "copy of itself -- the comparison is measuring the machine "
              "(load, clock) and the A vs B row above means nothing")
        return False

    # NEGATIVE control: A against A with one pawn square shifted 30 cp
    src = open(a).read()
    m = re.search(r'_v=0\nfor _c in "(.)', src)
    assert m, "control needs the encoded form (no literal to perturb)"
    bad = src.replace(m.group(1), "#" if m.group(1) != "#" else "$", 1)
    p = os.path.join(_STAGE, "perturbed.py")
    open(p, "w").write(bad)
    bad_agrees = compare(a, p, lines, nodes, "control")
    if bad_agrees:
        print("CONTROL FAILED: a perturbed table agreed everywhere -- "
              "this instrument cannot detect an eval change, so its "
              "agreement figures mean nothing")
        return False
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
