"""The structural bestmove floor: an emitted move is ALWAYS generated for the
CURRENT root, and one is always emitted while a legal move exists.

Thomas's ruling after seedtimed 2026-08-14 (b8 15/200, b8seed 4/200 games
lost by ILLEGAL MOVE, every one the literal `bestmove (none)`): "We should
never accept illegal moves. 15 is too much, so is 4." Zero, achieved
structurally, not statistically.

The failure mechanism these tests replay deterministically: at 1+0 the
driver budget min(wtime/12, wtime/2 - 1) goes negative, the in-search
deadline is already past when `go` arrives, and `Stop` lands before the
first root fail-high -- so best_move, cand and tp_move[root] are all empty.
The old tail printed "(none)"; the floor plays the first generated move of
the current root that does not leave our king capturable. Pseudo-legal is
NOT enough for the floor itself: a pinned-piece move is an illegal-move
forfeit too, so the fallback filters with can_kill_king/king_capture, and
"(none)" survives only for checkmate/stalemate roots, where no tournament
manager ever asks us to move (a weak legal move can lose the game; only a
non-move can forfeit it).

Legality reference: python-chess, the same oracle legality_gate.py uses.
"""
import importlib.util
import pathlib
import re
import subprocess
import sys
import time
from threading import Event

import chess
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRY = ROOT / "nnue_4k" / "pst_entry.py"
sys.path.insert(0, str(ROOT))


def load_entry(name):
    spec = importlib.util.spec_from_file_location(name, ENTRY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def env():
    entry = load_entry("pst_entry_floor")
    import sunfish_ui.uci as uci
    uci.sunfish = entry            # what uci.run() would do
    return entry, uci


def hist_for(uci, fen):
    """Build the driver's `hist` for a FEN, exactly as `position fen` does."""
    pos = uci.from_fen(*fen.split())
    return [pos] if uci.get_color(pos) == uci.WHITE else [pos.rotate(), pos]


def legal(fen):
    return {m.uci() for m in chess.Board(fen).legal_moves}


# Real FENs from the 19 forfeited games (seedtimed 2026-08-14), plus the
# classes the floor must not trip over: side in check, an absolute pin on
# the only developed piece, and a black-to-move rotation case.
FENS = [
    "rnbqk2r/ppp1ppbp/5np1/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R b KQkq - 0 10",
    "rnbqkb1r/pp1p2pp/4pn2/2p5/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 10",
    "r1bqk2r/3pnppp/p1nb4/1pp1p3/4P3/2P2N2/PPBP1PPP/RNBQ1RK1 w kq - 0 10",
    "rnb1kb1r/ppp1pp1p/5np1/3q4/3P4/2NQ4/PP2PPPP/R1B1KBNR b KQkq - 0 10",
    # white in check (1.e4 e5 2.f4 Qh4+): only g3/Ke2 evasions are legal
    "rnb1kbnr/pppp1ppp/8/4p3/4PP1q/8/PPPP2PP/RNBQKBNR w KQkq - 1 3",
    # absolute pin: Nc3 is pinned by Bb4 after 1.e4 e5 2.Nf3 Nc6 3.Nc3 Bb4
    "r1bqk1nr/pppp1ppp/2n5/4p3/1b2P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
]
MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


class Starved:
    """A searcher whose abort beats the first root fail-high -- the exact
    state the 19 forfeits were emitted from (no info line ever printed)."""
    def __init__(self, entry):
        self._entry = entry
        self.tp_move = {}
        self.nodes, self.deadline = 0, 0

    def search(self, history):
        raise self._entry.Stop
        yield  # pragma: no cover -- makes search() a generator

    def bound(self, pos, gamma, depth, root=False):
        raise self._entry.Stop


def bestmove_of(out):
    lines = [l for l in out.splitlines() if l.startswith("bestmove")]
    assert lines, "no bestmove emitted at all: %r" % out
    return lines[-1].split()[1]


# ---------------------------------------------------------------------------
# the floor itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fen", FENS)
def test_first_legal_move_is_legal(env, fen):
    entry, uci = env
    mv = uci.first_legal_move(hist_for(uci, fen))
    assert mv in legal(fen), "floor played %s, legal set %s" % (mv, legal(fen))


def test_first_legal_move_terminal_roots(env):
    entry, uci = env
    assert uci.first_legal_move(hist_for(uci, MATE_FEN)) is None
    assert uci.first_legal_move(hist_for(uci, STALEMATE_FEN)) is None


# ---------------------------------------------------------------------------
# go_loop: abort-before-commit can no longer answer "(none)"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fen", FENS)
def test_go_loop_starved_emits_legal_move(env, capsys, fen):
    entry, uci = env
    uci.go_loop(Starved(entry), hist_for(uci, fen), Event(), max_movetime=0.01)
    assert bestmove_of(capsys.readouterr().out) in legal(fen)


def test_go_loop_starved_terminal_roots_answer_none(env, capsys):
    """Mate/stalemate at the root: "(none)" is the correct answer -- no
    manager asks for a move there -- and nothing else may be emitted."""
    entry, uci = env
    for fen in (MATE_FEN, STALEMATE_FEN):
        uci.go_loop(Starved(entry), hist_for(uci, fen), Event(), max_movetime=0.01)
        assert bestmove_of(capsys.readouterr().out) == "(none)"


def test_go_loop_never_inherits_across_positions(env, capsys):
    """position A, go (full), position B, go (aborted instantly): the answer
    must be generated for B. A's tp_move entries are keyed by A's positions,
    so the only way B gets an answer here is the floor -- this is the
    derive-never-inherit case (#184 precedent)."""
    entry, uci = env
    fen_a, fen_b = FENS[0], FENS[1]

    class Flaky(entry.Searcher):
        starve = False
        def search(self, history):
            if self.starve:
                raise entry.Stop
            yield from super().search(history)

    searcher = Flaky()
    searcher.deadline = time.time() + 5
    uci.go_loop(searcher, hist_for(uci, fen_a), Event(), max_movetime=0.02)
    assert bestmove_of(capsys.readouterr().out) in legal(fen_a)

    searcher.starve = True     # B's go is aborted before any commit
    uci.go_loop(searcher, hist_for(uci, fen_b), Event(), max_movetime=0.01)
    mv = bestmove_of(capsys.readouterr().out)
    assert mv in legal(fen_b), (
        "answer %r is not a move of the CURRENT position" % mv)


def test_mate_loop_starved_emits_legal_move(env, capsys):
    entry, uci = env
    fen = FENS[0]
    uci.mate_loop(Starved(entry), hist_for(uci, fen), Event(),
                  max_movetime=0.01, max_depth=0)
    assert bestmove_of(capsys.readouterr().out) in legal(fen)


# ---------------------------------------------------------------------------
# the builtin loop (what the packed artifact runs without a driver)
# ---------------------------------------------------------------------------

@pytest.fixture()
def builtin_entry(monkeypatch):
    """A fresh entry module whose main() cannot resolve the sunfish_ui
    driver, so it runs the builtin loop the packed artifact ships."""
    saved = {k: sys.modules.get(k) for k in ("sunfish_ui", "sunfish_ui.uci")}
    sys.modules["sunfish_ui"] = None       # forces ImportError inside main()
    sys.modules.pop("sunfish_ui.uci", None)
    try:
        yield load_entry("pst_entry_builtin")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def run_builtin(entry, monkeypatch, commands):
    feed = iter(commands)
    monkeypatch.setattr("builtins.input", lambda: next(feed))
    entry.main()


def test_builtin_starved_go_emits_legal_move(builtin_entry, monkeypatch, capsys):
    entry = builtin_entry
    monkeypatch.setattr(entry, "Searcher",
                        lambda: Starved(entry))
    moves = "e2e4 e7e5 g1f3".split()
    run_builtin(entry, monkeypatch,
                ["position startpos moves " + " ".join(moves),
                 "go wtime 1 btime 1", "quit"])
    board = chess.Board()
    for m in moves:
        board.push_uci(m)
    mv = bestmove_of(capsys.readouterr().out)
    assert mv in {m.uci() for m in board.legal_moves}


def test_builtin_mate_at_root_answers_none_and_survives(builtin_entry,
                                                        monkeypatch, capsys):
    """Fool's mate root, REAL searcher: the `score >= gamma and move` guard
    must keep a verified-terminal yield from crashing the loop, and the
    floor must answer "(none)", not garbage. isready afterwards proves the
    loop is still alive."""
    entry = builtin_entry
    run_builtin(entry, monkeypatch,
                ["position startpos moves f2f3 e7e5 g2g4 d8h4",
                 "go movetime 60", "isready", "quit"])
    out = capsys.readouterr().out
    assert bestmove_of(out) == "(none)"
    assert "readyok" in out


def test_builtin_instant_go_startpos(builtin_entry, monkeypatch, capsys):
    """The startpos-instant-stop case with the real searcher: whatever path
    wins the race (cand vs floor), the emitted move must be legal."""
    entry = builtin_entry
    run_builtin(entry, monkeypatch,
                ["position startpos", "go wtime 1 btime 1", "quit"])
    mv = bestmove_of(capsys.readouterr().out)
    assert mv in {m.uci() for m in chess.Board().legal_moves}


# ---------------------------------------------------------------------------
# end to end: the seedtimed regime over a real pipe (driver path, ~1s)
# ---------------------------------------------------------------------------

def test_uci_pipe_abort_then_new_position():
    fen_a, fen_b = FENS[0], FENS[1]
    cmds = "\n".join([
        "uci", "isready",
        "position fen " + fen_a,
        "go wtime 1 btime 1 winc 0 binc 0",   # negative budget: instant abort
        "position fen " + fen_b,
        "go wtime 1 btime 1 winc 0 binc 0",
        "quit", ""])
    out = subprocess.run([sys.executable, str(ENTRY)], input=cmds,
                         capture_output=True, text=True, timeout=30,
                         cwd=str(ROOT)).stdout
    moves = re.findall(r"^bestmove (\S+)", out, re.M)
    assert len(moves) == 2, out
    assert moves[0] in legal(fen_a), "go #1 answered %r" % moves[0]
    assert moves[1] in legal(fen_b), "go #2 answered %r, not a move of B" % moves[1]
