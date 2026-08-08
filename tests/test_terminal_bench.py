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

Known failures (strict xfail): three natural positions crossing on
every current build - organic instances of the open sentinel-masking
channel (KingCapturableReportsExact; a king-capturable child soundly
cutting off on a partial bound masks the parent's sentinel).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.uci as uci  # noqa: E402
from tools.uci import render_move  # noqa: E402

BENCH = ROOT / "tools" / "test_files" / "terminal_bench.epd"

# Open-channel witnesses: sentinel masking (next arc). Strict xfail.
KNOWN_OPEN_CHANNEL = {
    "3k2b1/8/n1q4p/n6P/4r3/8/1K5b/8",
    "3k2b1/8/n1q4p/n6P/4r3/8/7b/K7",
    "7k/8/4P3/8/3K4/8/8/6Q1",
}


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
