"""Mate-CONVERSION gate: can the engine actually FINISH a won king ending?

The mate-in-1 gate (tools/build/mate_gate.py) asks whether the eval reorders
an immediate win out of reach. This gate asks a different question the H2
work made urgent: whether the KING-TABLE SEAM lets the engine convert KQK and
KRK at all. A seam that holds K_MID while any queen is on the board keeps the
ATTACKING king passive in KQK -- the king is a mating piece there, and no
depth of search mates without it. That failure is invisible to mate-in-1
suites (the king is already placed) and to legality gates (every answer is a
legal move); it only shows up over a sequence. So: play the sequence.

Protocol, per position: the engine plays the ATTACKER (white, always) via
`position fen` + `go movetime`; the gate plays the defender with a FIXED,
DETERMINISTIC bare-king heuristic -- take any legal capture (a bare king's
legal capture is winning by definition), else maximise centrality (distance
from the nearest edge), else maximise distance from the attacking king, ties
broken by UCI string. The heuristic is part of the pre-registered instrument
and must never be tuned against an arm. CONVERTED means checkmate within the
position's budget of attacker moves; stalemate, losing the piece, an illegal
or unparseable answer, or an exhausted budget are each a distinct FAIL.

Driver discipline is inherited verbatim from mate_gate.py, because this gate
feeds `position fen` too: source entries only, PYTHONPATH set to a checkout
that HAS sunfish_ui/, and the driver banner demanded by name before the first
move is trusted. One engine process per position, kept alive across moves.

Like mate_gate, the verdict is COMPARATIVE: arms of one screen run the same
suite and the split is the reading. The exit code carries only "any fails".

usage: mate_conversion_gate.py ENGINE FENFILE [MOVETIME_MS]
FENFILE lines: <FEN(6 fields)> <budget: max attacker moves> <tag>
"""
import os
import pathlib
import subprocess
import sys

import chess

ENGINE = os.path.abspath(sys.argv[1])
FENS = sys.argv[2]
MOVETIME = int(sys.argv[3]) if len(sys.argv) > 3 else 500
assert ENGINE.endswith(".py"), "source entries only: `position fen` is a driver feature"
ARGV = [sys.executable, ENGINE]


def _find_ui(*starts):
    """Nearest ancestor that actually contains a sunfish_ui/ -- dev checkout or
    box arena. Same resolution rule as tools/build/mate_gate.py."""
    for start in starts:
        for d in [start] + list(start.parents):
            if (d / "sunfish_ui" / "uci.py").exists():
                return str(d)
    raise SystemExit("no sunfish_ui/ found above %s -- stage the gate beside one"
                     % " or ".join(str(s) for s in starts))


REPO = _find_ui(pathlib.Path(__file__).resolve().parent, pathlib.Path(ENGINE).parent)
ENV = dict(os.environ, PYTHONPATH=REPO)


class Arm:
    """One engine process, banner-checked once, queried per move."""

    def __init__(self):
        self.p = subprocess.Popen(ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1, env=ENV)
        self.banner_ok = False

    def ask(self, fen):
        self.p.stdin.write("position fen %s\ngo movetime %d\n" % (fen, MOVETIME))
        self.p.stdin.flush()
        while True:
            o = self.p.stdout.readline()
            if not o:
                raise SystemExit("ENGINE DIED mid-game: %s\nstderr: %s"
                                 % (" ".join(ARGV), self.p.stderr.read()[-600:].strip()))
            if o.startswith("info string driver"):
                if " fen" not in o:
                    raise SystemExit("WRONG DRIVER: %s resolved to %s -- the builtin "
                                     "loop ignores `position fen`." % (ENGINE, o.strip()))
                self.banner_ok = True
            if o.startswith("bestmove"):
                if not self.banner_ok:
                    raise SystemExit("WRONG DRIVER: %s printed no driver banner -- the "
                                     "builtin loop ignores `position fen` and answers "
                                     "from the OPENING position." % ENGINE)
                return o.split()[1] if len(o.split()) > 1 else None

    def close(self):
        self.p.kill()


def defend(board):
    """The fixed defender. board.turn is the defender; deterministic by
    construction (total order on moves)."""
    them = board.king(not board.turn)

    def key(m):
        to = m.to_square
        f, r = chess.square_file(to), chess.square_rank(to)
        return (board.is_capture(m),               # a bare king's capture wins
                min(f, 7 - f, r, 7 - r),           # centrality: edge distance
                chess.square_distance(to, them),   # stay off the attacking king
                m.uci())
    return max(board.legal_moves, key=key)


def play(fen, budget):
    """Returns ('CONVERTED', moves_used) or ('FAIL', reason)."""
    b = chess.Board(fen)
    arm = Arm()
    try:
        for n in range(1, budget + 1):
            mv = arm.ask(b.fen())
            try:
                m = chess.Move.from_uci(mv)
            except Exception:
                return "FAIL", "UNPARSEABLE %r after %d" % (mv, n - 1)
            if m not in b.legal_moves:
                return "FAIL", "ILLEGAL %s after %d" % (mv, n - 1)
            b.push(m)
            if b.is_checkmate():
                return "CONVERTED", n
            if b.is_stalemate():
                return "FAIL", "STALEMATE after %d" % n
            if b.is_insufficient_material():
                return "FAIL", "PIECE LOST after %d" % n
            b.push(defend(b))
            if b.is_insufficient_material():
                return "FAIL", "PIECE LOST after %d" % n
        return "FAIL", "BUDGET (%d moves) exhausted, final %s" % (budget, b.fen())
    finally:
        arm.close()


lines = [ln.split() for ln in open(FENS)
         if ln.strip() and not ln.lstrip().startswith("#")]
converted, fails = 0, []
for parts in lines:
    fen, budget, tag = " ".join(parts[:6]), int(parts[6]), parts[7]
    verdict, detail = play(fen, budget)
    if verdict == "CONVERTED":
        converted += 1
        print("  CONVERTED %-14s in %2d/%d attacker moves" % (tag, detail, budget))
    else:
        fails.append(tag)
        print("  FAIL      %-14s %s" % (tag, detail))

print("mate conversion  %s  %d/%d converted  (movetime %d ms)"
      % (os.path.basename(ENGINE), converted, len(lines), MOVETIME))
sys.exit(0 if not fails else 1)
