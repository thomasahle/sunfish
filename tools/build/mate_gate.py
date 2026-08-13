"""Mate-in-1 gate: does the engine still SEE the forced win?

This is deliberately not the legality gate and does not replace it -- the
ledger's standing warning is that a mate suite passed 5-vs-5 on the very build
that answered `bestmove (none)`. The two ask different questions and an eval
change can fail either: legality is about always producing a move, this is
about the eval not reordering the win out of reach.

Mate-in-1 is checked by PLAYING the move and asking python-chess whether the
result is checkmate, not by reading a score. A score check would pass an engine
that reports `mate 1` and then plays something else, and it would need the
engine's mate encoding to be right, which is a separate claim.

THE DRIVER IS PART OF THE MEASUREMENT, and this gate did not check it. It
feeds `position fen`, which ONLY `sunfish_ui/uci.py` understands: an entry that
resolves no `sunfish_ui/` falls through to the builtin loop, which knows only
`position startpos`, searches the OPENING position and answers an opening move.
Run that way on a variant in a scratch directory, this gate reported
`MISS ILLEGAL g1f3` for the SHIPPED ENTRY on three mates it solves 8/8. Loud
rather than silent, but still a chess verdict on a position the engine never
saw -- the same class as the first-yield gate's first run and `agree.py` before
it. So: source entries only, `PYTHONPATH` set to a checkout that HAS a driver,
and the banner demanded by name.

usage: mate_gate.py ENGINE FENFILE [MOVETIME_MS]
"""
import os
import pathlib
import subprocess
import sys

import chess

ENGINE = os.path.abspath(sys.argv[1])
FENS = sys.argv[2]
MOVETIME = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
assert ENGINE.endswith(".py"), "source entries only: `position fen` is a driver feature"
ARGV = [sys.executable, ENGINE]


def _find_ui(*starts):
    """Nearest ancestor that actually contains a sunfish_ui/ -- dev checkout or
    box arena. Same resolution rule as tools/build/first_yield_gate.py."""
    for start in starts:
        for d in [start] + list(start.parents):
            if (d / "sunfish_ui" / "uci.py").exists():
                return str(d)
    raise SystemExit("no sunfish_ui/ found above %s -- stage the gate beside one"
                     % " or ".join(str(s) for s in starts))


REPO = _find_ui(pathlib.Path(__file__).resolve().parent, pathlib.Path(ENGINE).parent)
ENV = dict(os.environ, PYTHONPATH=REPO)


def ask(fen):
    p = subprocess.Popen(ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1, env=ENV)
    p.stdin.write("uci\nisready\nposition fen %s\ngo movetime %d\n" % (fen, MOVETIME))
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
        raise SystemExit("ENGINE DID NOT START: %s\nstderr: %s"
                         % (" ".join(ARGV), p.stderr.read()[-600:].strip()))
    if driver is None or " fen" not in driver:
        raise SystemExit("WRONG DRIVER: %s resolved to %s -- the builtin loop ignores "
                         "`position fen` and answers from the OPENING position."
                         % (ENGINE, driver or "the builtin loop (no banner)"))
    return mv


solved, fails = 0, []
fens = [ln.strip() for ln in open(FENS) if ln.strip()]
for fen in fens:
    b = chess.Board(fen)
    mv = ask(fen)
    try:
        m = chess.Move.from_uci(mv)
    except Exception:
        fails.append((fen, mv, "UNPARSEABLE"))
        continue
    if m not in b.legal_moves:
        fails.append((fen, mv, "ILLEGAL"))
        continue
    b.push(m)
    if b.is_checkmate():
        solved += 1
    else:
        fails.append((fen, mv, "not mate"))

print("mate gate  %s  %d/%d solved  (movetime %d ms)"
      % (os.path.basename(ENGINE), solved, len(fens), MOVETIME))
for fen, mv, why in fails[:8]:
    print("  MISS %-9s %s   %s" % (why, mv, fen))
# The gate is comparative: two arms of one screen must solve the SAME set. An
# absolute threshold would encode this suite's difficulty, which is not the
# question. Exit code carries the count so a caller can compare arms.
sys.exit(0 if not [f for f in fails if f[2] != "not mate"] else 1)
