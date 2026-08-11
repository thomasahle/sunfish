"""Game-level coverage for both of sunfish's UCI interfaces, and for the
boundary between them.

There are two, and after #156 the boundary between them is sharp:

* ``sunfish_tools/uci.py`` -- the real interface (FEN, ponder, spec-complete
  ``go``). ``sunfish.py`` imports it unconditionally, and it ships in the
  wheel, so every configuration that runs the *source* file uses it;
* the loop inlined in ``sunfish.main()`` -- ``startpos``-only, written for the
  packed 4K artifact, and reachable only by deleting the minifier-hide block
  the way ``build/pack.sh`` does.

Issue #156 came from those two being confusable at runtime. The wheel shipped
``sunfish.py`` alone, ``import tools.uci`` failed, a bare ``except ImportError:
pass`` swallowed it, and an installed engine silently ran the artifact's loop
instead. Driven with ``position fen`` (what fastchess sends when the opening
book is EPD) that loop never updates ``hist``, so it answers from the untouched
initial board -- a White opening move while nominally playing Black. The
reported symptom looked like a Black-side orientation bug; the orientation is
in fact correct in both directions, which the first test here pins down.

So these tests assert, in order: the tiny loop plays legal chess as both
colours; the real interface handles a Black-to-move FEN; and a missing or
broken UCI module is loud rather than a silent downgrade to the tiny loop.
"""

import ast
import pathlib
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import types

import pytest

chess = pytest.importorskip("chess")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish_tools.uci as uci  # noqa: E402

# The FEN from issue #156's reproduction: Black to move, so an engine that
# ignored the command and searched the initial position answers a White move.
BLACK_TO_MOVE_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 1 4"

# Mirrors `sed '/# minifier-hide start/,/# minifier-hide end/d'`: whole lines,
# from marker to marker, at any indentation (the bridge's markers are indented
# inside main()).
HIDDEN = re.compile(
    r"^[^\n]*# minifier-hide start[^\n]*\n.*?^[^\n]*# minifier-hide end[^\n]*\n",
    re.MULTILINE | re.DOTALL,
)


def strip_minifier_hidden(source):
    """What build/pack.sh's `sed '/start/,/end/d'` leaves behind."""
    stripped = HIDDEN.sub("", source)
    assert "minifier-hide" not in stripped
    return stripped


def isolated_python(*args):
    """Argv for a child interpreter that can see only the layout under test.

    The flags carry the whole isolation, and each is load-bearing: ``-E``
    drops PYTHONPATH (and every other PYTHON* variable), ``-s`` the user site
    directory, ``-S`` site-packages -- including its ``.pth`` files -- and
    ``-B`` keeps the scratch layouts free of ``__pycache__`` (``-E`` having
    made PYTHONDONTWRITEBYTECODE a no-op).

    ``-S`` is the one that matters. A sunfish checkout is normally set up with
    an editable install, which leaves a meta-path finder in site-packages
    resolving ``sunfish`` and ``sunfish_tools`` to the checkout from *any*
    working directory and *any* sys.path. Without it, a layout built here
    without ``sunfish_tools/`` still imports the checkout's copy: the engine
    starts, and the loudness tests below pass whatever the bridge does.

    Unlike ``-I``, these flags keep the script's own directory (and, for
    ``-c``, the working directory) on sys.path -- that is where these tests
    put the module when they mean it to be found. Nothing is lost by dropping
    site-packages: sunfish.py and sunfish_tools/uci.py import only the
    standard library.
    """
    return [sys.executable, "-E", "-s", "-S", "-B", *args]


class UciEngine:
    """Minimal UCI driver: one subprocess, line-oriented, with timeouts."""

    def __init__(self, script, cwd):
        self.proc = subprocess.Popen(
            isolated_python(str(script)), cwd=str(cwd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.lines = queue.Queue()
        self.stderr = []
        threading.Thread(target=self._drain_out, daemon=True).start()
        threading.Thread(target=self._drain_err, daemon=True).start()

    def _drain_out(self):
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _drain_err(self):
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())

    def send(self, command):
        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # the engine died; the expecting side reports it

    def expect(self, prefix, timeout=60):
        while True:
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty:
                pytest.fail(f"no {prefix!r} within {timeout}s; stderr:\n"
                            + "\n".join(self.stderr))
            if line is None:
                pytest.fail(f"engine exited before {prefix!r}; stderr:\n"
                            + "\n".join(self.stderr))
            if line.startswith(prefix):
                return line

    def handshake(self):
        self.send("uci")
        self.expect("uciok")
        self.send("isready")
        self.expect("readyok")
        return self

    def bestmove(self, movetime=50):
        self.send(f"go movetime {movetime}")
        return self.expect("bestmove").split()[1]

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - best-effort teardown
            self.proc.kill()


def make_engine_dir(tmp_path, *, tiny=False, uci_module=True, broken_import=False):
    """Lay out an engine in `tmp_path` and return the path to its script.

    tiny          -- strip the minifier-hide blocks, i.e. build the packed
                     artifact's source, whose interface is the inlined loop.
    uci_module    -- copy sunfish_tools/ next to it (the wheel's layout).
    broken_import -- copy it, but with an unsatisfiable import at the top,
                     standing in for a dependency slip inside the module.
    """
    source = (ROOT / "sunfish.py").read_text()
    script = tmp_path / "sunfish.py"
    script.write_text(strip_minifier_hidden(source) if tiny else source)

    if uci_module:
        pkg = tmp_path / "sunfish_tools"
        shutil.copytree(ROOT / "sunfish_tools", pkg,
                        ignore=shutil.ignore_patterns("__pycache__"))
        if broken_import:
            uci = pkg / "uci.py"
            uci.write_text("import sunfish_definitely_absent_module\n" + uci.read_text())
    else:
        assert not broken_import
        assert not (tmp_path / "sunfish_tools").exists()
    return script


def play_game(engine, engine_color, plies, seed):
    """Play `plies` half-moves against a seeded random mover.

    Driven with `position startpos moves ...`, the command form every packed
    artifact tournament and every PGN-book match uses. Every engine reply is
    checked for legality in the real position.
    """
    rng = random.Random(seed)
    board = chess.Board()
    played, mine = [], []
    while len(played) < plies and not board.is_game_over():
        if board.turn == engine_color:
            engine.send("position startpos"
                        + (" moves " + " ".join(played) if played else ""))
            reply = engine.bestmove()
            assert reply != "(none)", f"engine resigned {board.fen()}"
            move = chess.Move.from_uci(reply)
            assert move in board.legal_moves, (
                f"illegal move {reply} after "
                f"{' '.join(played) or '<startpos>'} ({board.fen()})")
            mine.append(reply)
        else:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
        played.append(move.uci())
    return mine


@pytest.mark.parametrize("color,name", [(chess.WHITE, "white"), (chess.BLACK, "black")])
def test_tiny_loop_plays_legal_moves(tmp_path, color, name):
    """The packed artifact's inlined loop, played as a game in both colours.

    Issue #156 was filed as "the built-in loop emits White-oriented moves when
    playing Black". It does not, and both of its coordinate flips are
    load-bearing: inverting `ply % 2 == 1` in the position parser or
    `len(hist) % 2 == 0` in the `go` handler fails this test within two moves.
    """
    script = make_engine_dir(tmp_path, tiny=True, uci_module=False)
    engine = UciEngine(script, tmp_path).handshake()
    try:
        mine = play_game(engine, color, plies=14, seed=20260808)
    finally:
        engine.close()
    assert len(mine) >= 6, f"engine made too few moves as {name}: {mine}"


def test_real_interface_answers_a_black_to_move_fen(tmp_path):
    """The regression #156 actually reported, against the wheel's layout.

    An engine laid out the way `pip install sunfish` lays it out -- sunfish.py
    plus sunfish_tools/ and nothing else -- must handle `position fen` and
    answer with a move that is legal *for that FEN*.
    """
    script = make_engine_dir(tmp_path)
    engine = UciEngine(script, tmp_path).handshake()
    try:
        engine.send("position fen " + BLACK_TO_MOVE_FEN)
        reply = engine.bestmove(movetime=100)
    finally:
        engine.close()

    board = chess.Board(BLACK_TO_MOVE_FEN)
    move = chess.Move.from_uci(reply)
    assert move in board.legal_moves, (
        f"{reply} is not legal for {BLACK_TO_MOVE_FEN}; an engine that ignored "
        "the command and searched the initial position answers a White move")


def run_to_completion(script, cwd, commands="uci\nisready\nquit\n", timeout=60):
    return _run(isolated_python(str(script)), cwd, commands, timeout)


def run_entry_point(cwd, commands="uci\nisready\nquit\n", timeout=60):
    """Invoke main() the way the console script pyproject installs does."""
    entry = "import sys; from sunfish import main; sys.exit(main())"
    return _run(isolated_python("-c", entry), cwd, commands, timeout)


def _run(argv, cwd, commands, timeout):
    return subprocess.run(argv, cwd=str(cwd), input=commands,
                          capture_output=True, text=True, timeout=timeout)


def assert_loud_failure(result, needle):
    """The engine must die with a diagnosable error, not serve the tiny loop."""
    assert result.returncode != 0, (
        "engine started anyway -- a silent downgrade to the startpos-only loop "
        f"is exactly issue #156.\nstdout:\n{result.stdout}")
    assert "uciok" not in result.stdout, (
        f"engine answered uciok without its UCI module:\n{result.stdout}")
    assert needle in result.stderr, (
        f"expected {needle!r} in the traceback, got:\n{result.stderr}")


def test_missing_uci_module_is_loud(tmp_path):
    """sunfish.py without sunfish_tools/ must refuse to start.

    This is the deployment mistake behind #156: an engine shipped without its
    interface used to keep running as the packed loop, so nothing looked wrong
    until a GUI rejected a move -- or, worse, until a whole match had been
    played and had to be voided.
    """
    script = make_engine_dir(tmp_path, uci_module=False)
    assert_loud_failure(run_to_completion(script, tmp_path), "sunfish_tools")


def test_entry_point_raises_when_tooling_is_missing(tmp_path):
    """The console script (`sunfish = "sunfish:main"`) must raise too.

    This is the chosen behaviour, not an accident of how the file is run: the
    bridge in main() is a bare import, so a broken installation is a traceback
    at startup. Nothing catches it, and nothing degrades to the tiny loop.
    """
    make_engine_dir(tmp_path, uci_module=False)
    result = run_entry_point(tmp_path)
    assert_loud_failure(result, "sunfish_tools")
    assert "ModuleNotFoundError" in result.stderr, result.stderr


def test_broken_import_inside_uci_module_is_loud(tmp_path):
    """An ImportError raised *inside* the UCI module must propagate.

    Trivially true now that the bridge is a bare import, and pinned because it
    was not true before: the old `except ImportError: pass` caught this case
    too, so a dependency slip in the UCI module downgraded even a full checkout
    to the startpos-only loop, silently, in matches run from that checkout.
    """
    script = make_engine_dir(tmp_path, broken_import=True)
    assert_loud_failure(run_to_completion(script, tmp_path),
                        "sunfish_definitely_absent_module")


# --------------------------------------------------------------------------
# The engine-module contract.
#
# uci.py never imports the engine: run(module, startpos) injects it, so the
# same interface drives classic sunfish, pesto and the NNUE variants. The
# contract is therefore checked at run() entry rather than declared by an
# import, and these tests hold that check to being complete and eager.
# --------------------------------------------------------------------------


def load_engine():
    """Import sunfish.py without running its UCI interface."""
    module = types.ModuleType("sunfish_under_test")
    module.__file__ = str(ROOT / "sunfish.py")
    exec(compile((ROOT / "sunfish.py").read_text(), "sunfish.py", "exec"),
         module.__dict__)
    return module


def engine_attributes_used_by_uci():
    """Every `sunfish.<attr>` uci.py reads, via AST so comments don't count."""
    tree = ast.parse((ROOT / "sunfish_tools" / "uci.py").read_text())
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "sunfish"}


def test_shipped_engine_satisfies_the_contract():
    uci.check_engine_module(load_engine())


def test_contract_covers_every_attribute_the_interface_reads():
    """No `sunfish.<attr>` may be reached without being required or optional.

    Guards the next person who adds one: an attribute that is neither in
    ENGINE_API nor explicitly optional is a silent AttributeError waiting for
    whichever command happens to reach it.
    """
    optional = {"TABLE_SIZE", "features", "from_board", "pst"}  # hasattr-guarded at use
    unaccounted = engine_attributes_used_by_uci() - set(uci.ENGINE_API) - optional
    assert not unaccounted, (
        f"{sorted(unaccounted)} read off the engine module but not declared in "
        "ENGINE_API (or in this test's optional set, with a hasattr guard)")


def test_contract_lists_nothing_stale():
    assert set(uci.ENGINE_API) <= engine_attributes_used_by_uci()


@pytest.mark.parametrize("attr", ["Stop", "Searcher", "MATE_LOWER", "parse"])
def test_engine_missing_a_required_attribute_is_rejected_at_startup(attr):
    """run() must refuse a non-conforming engine before it does anything.

    `Stop` is the case that motivated this. It is only ever named in `except`
    clauses, and an except expression is evaluated when an exception arrives,
    not when the code is reached -- so an engine without it used to run
    normally until the first deadline abort, and then raise AttributeError
    while handling the very abort it was meant to catch. The old
    `except getattr(sunfish, "Stop", ()): pass` was worse still: it degraded
    silently, losing deadline aborts with no error at all.

    Note what this test does *not* do: no search runs, no exception is raised
    in the engine. The rejection has to happen anyway. That is what eager
    means.
    """
    engine = load_engine()
    delattr(engine, attr)
    with pytest.raises(TypeError) as excinfo:
        uci.run(engine, engine.hist[-1])
    assert attr in str(excinfo.value)
    assert getattr(uci, "sunfish", None) is not engine, \
        "run() adopted a non-conforming engine before checking it"


def test_engine_without_pst_or_features_is_rejected():
    """from_fen needs one or the other to score a board. Without either, the
    engine works until the first `position fen` -- i.e. until someone runs it
    with an EPD opening book, which is how #156 stayed hidden for so long."""
    engine = load_engine()
    delattr(engine, "pst")
    assert not hasattr(engine, "features")
    with pytest.raises(TypeError, match="pst"):
        uci.run(engine, engine.hist[-1])


def test_packed_build_still_has_a_working_loop(tmp_path):
    """The corollary of making the import unconditional: the tiny loop is now
    reachable only through the strip, so the strip must still produce a
    self-contained engine with no import of sunfish_tools left in it."""
    script = make_engine_dir(tmp_path, tiny=True, uci_module=False)
    assert "import sunfish_tools" not in script.read_text()
    result = run_to_completion(
        script, tmp_path,
        commands="uci\nisready\nposition startpos moves e2e4\ngo movetime 100\nquit\n")
    assert result.returncode == 0, result.stderr
    assert "uciok" in result.stdout and "bestmove" in result.stdout, result.stdout
