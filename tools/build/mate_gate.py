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

Same launch discipline as legality_gate.py: a .py source goes through the
interpreter, anything else is executed directly, and an engine that produces no
output at all is a LOUD ABORT rather than a chess verdict.

usage: mate_gate.py ENGINE FENFILE [MOVETIME_MS]
"""
import os
import subprocess
import sys

import chess

ENGINE = sys.argv[1]
FENS = sys.argv[2]
MOVETIME = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
ARGV = [sys.executable, ENGINE] if ENGINE.endswith(".py") else [os.path.abspath(ENGINE)]


def ask(fen):
    p = subprocess.Popen(ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write("uci\nisready\nposition fen %s\ngo movetime %d\n" % (fen, MOVETIME))
    p.stdin.flush()
    mv, saw = None, False
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
        raise SystemExit("ENGINE DID NOT START: %s\nstderr: %s"
                         % (" ".join(ARGV), p.stderr.read()[-600:].strip()))
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
