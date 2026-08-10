"""Terminal-correctness bench: the null-move/stalemate bug families.

Unlike stalemate2.fen (deep endgame technique: fortresses, repetition
draws), every assertion here is a correctness invariant of the search
around terminal positions, checkable at shallow depth:

- probe LADDERS (both orders, depths 0/2/4, gammas straddling the
  static score, zero, and the mate band) must leave an ordered table -
  this is the check that reproduces the historical crossings, which
  plain driver runs usually miss;
- full-driver runs must leave an ordered table and a legal root move;
- class-tagged value assertions with solver-certain ground truth:
  mate-now positions score in the mate band, stalemate-now exactly 0,
  parent-of-mate at least the mate band (a mating move exists),
  parent-of-stalemate at least a draw.

Position classes (tools/test_files/terminal_bench.epd): natural
playout terminals and their parents, gate-eligible corner mates (every
pseudo-move valuable-but-illegal), the named historical witnesses, an
ahead-stalemate parametric family, and the classic zugzwang suite
(invariants only).

pst["K"] is pinned to K_MID per case: search() mutates the global for
bare-king mop-up, and a polluted table changes pos.score enough to
defuse the delicately balanced witnesses.

There are no known failures: KNOWN_OPEN_CHANNEL is empty and all 148
positions pass.  The three natural sentinel-masking witnesses that used
to be listed there (a king-capturable child cutting off on a partial
bound, masking the parent's sentinel) turned out to be futility
masking, and were closed by making sub-mate futility yields virtual.
The set and its strict-xfail wiring are kept so that a future witness
can be pinned without re-deriving the harness.

Also here: a runtime audit of the killer (tp_move) invariants, which
are load-bearing for the null fast path in bound() but are otherwise
only modelled formally.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish_tools.uci as uci  # noqa: E402
from sunfish_tools.uci import render_move  # noqa: E402

BENCH = ROOT / "tools" / "test_files" / "terminal_bench.epd"

# Empty: the three natural witnesses of the futility-masking channel
# were fixed by the kcx virtual-futility change.  Kept as the hook for
# pinning any future witness (strict xfail, so a fix cannot go unnoticed).
KNOWN_OPEN_CHANNEL = set()


def load_sunfish():
    import importlib.util

    src = (ROOT / "sunfish.py").read_text()
    probe = ROOT / "tests" / "_sunfish_probe.py"
    probe.write_text(src[: src.index("def main():")])
    spec = importlib.util.spec_from_file_location("sunfish_probe", probe)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sunfish_probe"] = mod
    spec.loader.exec_module(mod)
    uci.sunfish = mod
    return mod


def cases():
    for line in BENCH.read_text().splitlines():
        fen_part, rest = line.split("; class ", 1)
        cls = rest.split(";")[0].strip()
        parts = fen_part.split()
        marks = (
            [pytest.mark.xfail(strict=True, reason="open sentinel-masking channel")]
            if parts[0] in KNOWN_OPEN_CHANNEL
            else []
        )
        yield pytest.param(parts, cls, id=f"{cls}-{parts[0][:24]}", marks=marks)


@pytest.mark.parametrize("parts,cls", cases())
def test_terminal_invariants(parts, cls):
    chess = pytest.importorskip("chess")
    sf = load_sunfish()
    sf.pst["K"] = sf.K_MID
    pos = uci.from_fen(parts[0], parts[1], parts[2], parts[3], "0", "1")

    # 1. Probe ladders, both orders: table must stay ordered.
    gs = [pos.score, pos.score + 1, 1, 0, 31, -29,
          sf.MATE_LOWER - 50, -sf.MATE_LOWER + 50]
    for order in (gs, list(reversed(gs))):
        sf.pst["K"] = sf.K_MID
        s = sf.Searcher()
        s.history = set()
        for d in (0, 2, 4):
            for g in order:
                if -sf.MATE_UPPER < g <= sf.MATE_UPPER:
                    s.bound(pos, g, d)
        assert not any(e.lower > e.upper for e in s.tp_score.values()), "ladder crossing"

    # 2. Driver run: ordered table, converged bracket, legal root move.
    sf.pst["K"] = sf.K_MID
    s = sf.Searcher()
    brackets, last = {}, None
    for depth, gamma, score, move in s.search([pos]):
        lo, up = brackets.get(depth, (-10**9, 10**9))
        if score >= gamma:
            lo = max(lo, score)
        else:
            up = min(up, score)
        brackets[depth] = (lo, up)
        last = depth
        if depth > 4 or s.nodes > 250_000:
            break
    assert not any(e.lower > e.upper for e in s.tp_score.values()), "driver crossing"
    lo, up = brackets.get(min(last, 4), brackets[last])
    root_move = s.tp_move.get(pos)
    if root_move is not None:
        board = chess.Board(" ".join(parts[:4]) + " 0 1")
        u = render_move(root_move, white_pov=(parts[1] == "w"))
        assert chess.Move.from_uci(u) in board.legal_moves, "illegal tp_move"

    # 3. Class-specific ground truth.
    if cls in ("mate-now", "mate-now-corner", "witness-standpat-mate"):
        assert up <= -sf.MATE_LOWER + 2000, f"mated node scored [{lo},{up}]"
    elif cls in ("stalemate-now", "stalemate-now-ahead", "witness-null-stalemate"):
        assert lo <= 0 <= up and up - lo <= 60, f"stalemate scored [{lo},{up}]"
    elif cls == "parent-of-mate":
        assert lo >= sf.MATE_LOWER - 2000, f"mate-in-1 scored [{lo},{up}]"
    elif cls == "parent-of-stalemate":
        assert up >= -30, f"draw-in-hand scored [{lo},{up}]"


def test_killer_invariants_over_corpus():
    """Runtime audit of the tp_move invariants over the whole corpus.

    bound()'s null fast path reads tp_move as a legality certificate:
    it decides capturability in O(1) and lets the terminal arm skip its
    scan.  Three properties carry that, all of them consequences of
    "tp_move stores only real fail-high winners" - a claim the source
    comments and the formal event inventory make about the STORE SITE,
    which is exactly the kind of claim a future second store path would
    silently break.  So check them on the table an actual run leaves:

      1. KillerLegal-pseudo: the stored move is one gen_moves yields.
      2. At a king-capturable node the stored move IS the capture
         (pos.value tops the mate band).
      3. Otherwise the stored move is legal - playing it does not leave
         our own king capturable.

    Probes only, no search(): search() repoints the pst["K"] global for
    bare-king mop-up, which would change pos.score and the mate band
    under the audit.
    """
    sf = load_sunfish()
    sf.pst["K"] = sf.K_MID
    s = sf.Searcher()
    for parts, cls in [(p.values[0], p.values[1]) for p in cases()]:
        pos = uci.from_fen(parts[0], parts[1], parts[2], parts[3], "0", "1")
        s.history = set()
        for d in (0, 2, 4):
            for g in (pos.score, pos.score + 1, 0, 1):
                s.bound(pos, g, d)

    audited = 0
    for pos, move in s.tp_move.items():
        assert move in set(pos.gen_moves()), (pos.board, move, "not generated")
        if pos.king_capture() is not None:
            assert pos.value(move) >= sf.MATE_LOWER, (pos.board, move, "not the capture")
        else:
            assert pos.move(move).king_capture() is None, (pos.board, move, "illegal")
        audited += 1
    assert audited > 500, audited
