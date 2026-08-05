"""End-to-end test: the real lichess-bot bridge + sunfish vs a mock lichess server.

This launches the official `lichess-bot <https://github.com/lichess-bot-devs/lichess-bot>`_
process (the same bridge contrib/lichess deploys to a VM), pointed at the
in-process HTTP mock in tests/mock_lichess.py, with ``ponder: true`` -- the
configuration that exposed a real ponder race in tools/uci.py which
lichess-bot's own test suite (test_bot/, a Python-class-level fake with no
pondering coverage) does not catch.  Testing over real HTTP also covers
lib/lichess.py, the ndjson streams, and config parsing.

Three bullet games (60+0) are played against a random mover:

1. with instant (0 ms) opponent replies -- the tightest command timing, where
   the opponent's move arrives while the previous search is still winding down;
2. with 50-200 ms jittered replies -- fast realistic replies;
3. with 2-5 s opponent think times -- which let the engine's ponder search run
   deep before the miss/hit arrives.  This is the game that reproduces the
   production forfeit: an engine that only honors "stop"/"ponderhit" between
   search iterations stalls for many seconds once the ponder search is deep,
   burning its own clock until it flags.

Asserted per game: the game reaches a proper terminal status, the engine never
forfeits on time, every sunfish move arrives within 10 seconds of its turn
starting (wall clock, measured server-side), and every move after the first
arrives within 4 seconds (the first move of each game is special: lichess-bot
hardcodes a 10-second search for it; after that, sunfish budgets roughly
wtime/40 ~ 1.5 s, so 4 s only trips on multi-second engine stalls).

Setup choice: lichess-bot is not pip-installable, so a pinned checkout is
cached under ~/.cache/sunfish-bot-ci (override with LICHESS_BOT_CACHE), with a
dedicated venv holding its requirements.  On GitHub CI, cache that directory
with actions/cache to make reruns fast; the first run bootstraps it
automatically (one shallow clone + pip install, roughly a minute).

The test only runs when BOT_CI=1 is set, so plain `pytest` stays fast and
network-free.  Environment overrides:

- BOT_CI=1              enable this test
- SUNFISH_ENGINE_DIR    directory containing sunfish.py + tools/ to test
                        (default: this repository)
- LICHESS_BOT_CACHE     cache directory for the lichess-bot checkout + venv
- LICHESS_BOT_COMMIT    lichess-bot commit to pin (default below)
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

if not os.environ.get("BOT_CI"):
    pytest.skip("set BOT_CI=1 to run the lichess-bot integration test",
                allow_module_level=True)

if sys.version_info < (3, 11):
    pytest.skip("lichess-bot requires Python >= 3.11",
                allow_module_level=True)

import chess  # noqa: E402

from mock_lichess import MockLichess, TERMINAL_STATUSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = Path(os.environ.get("SUNFISH_ENGINE_DIR", REPO_ROOT))
CACHE_DIR = Path(os.environ.get("LICHESS_BOT_CACHE",
                                Path.home() / ".cache" / "sunfish-bot-ci"))
LICHESS_BOT_URL = "https://github.com/lichess-bot-devs/lichess-bot"
# lichess-bot master as of 2026-08-05 (version 2026.8.2.1).
LICHESS_BOT_COMMIT = os.environ.get(
    "LICHESS_BOT_COMMIT", "bedd1d9e86a8c4c96319490533e4e20fe63d1ac8")

STARTUP_TIMEOUT = 120  # engine config check + connect to the event stream
ACCEPT_TIMEOUT = 60  # challenge event -> accepted -> gameStart
GAME_TIMEOUT = 240  # both clocks are 60s; a finished 60+0 game fits easily
MAX_MOVE_SECONDS = 10.0  # any move (lichess-bot searches 10 s for the first)
MAX_LATER_MOVE_SECONDS = 4.0  # moves after the first (sunfish thinks ~1.5 s)


def _run(cmd, **kwargs):
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, **kwargs)


def _ensure_checkout() -> Path:
    """Clone (or update) the pinned lichess-bot commit into the cache."""
    checkout = CACHE_DIR / "lichess-bot"
    if not (checkout / "lichess-bot.py").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], cwd=checkout)
        _run(["git", "remote", "add", "origin", LICHESS_BOT_URL], cwd=checkout)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                          capture_output=True, text=True).stdout.strip()
    if head != LICHESS_BOT_COMMIT:
        _run(["git", "fetch", "-q", "--depth", "1", "origin", LICHESS_BOT_COMMIT],
             cwd=checkout)
        _run(["git", "checkout", "-q", "-f", LICHESS_BOT_COMMIT], cwd=checkout)
    return checkout


def _ensure_venv(checkout: Path) -> Path:
    """Create a venv with lichess-bot's requirements; returns its python."""
    venv_dir = CACHE_DIR / f"venv-py{sys.version_info[0]}.{sys.version_info[1]}"
    python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    requirements = checkout / "requirements.txt"
    stamp = venv_dir / "installed.stamp"
    wanted = f"{LICHESS_BOT_COMMIT}\n{requirements.read_bytes().hex()}"
    if stamp.exists() and stamp.read_text() == wanted:
        return python
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    _run([sys.executable, "-m", "venv", str(venv_dir)])
    _run([str(python), "-m", "pip", "install", "-q", "-r", str(requirements)])
    stamp.write_text(wanted)
    return python


@pytest.fixture(scope="session")
def lichess_bot():
    """(checkout dir, venv python) for the pinned lichess-bot release."""
    assert (ENGINE_DIR / "sunfish.py").exists(), f"no engine in {ENGINE_DIR}"
    assert (ENGINE_DIR / "tools" / "uci.py").exists(), f"no tools/ in {ENGINE_DIR}"
    checkout = _ensure_checkout()
    return checkout, _ensure_venv(checkout)


def _write_config(path: Path, base_url: str, token: str) -> None:
    path.write_text(f"""\
token: "{token}"
url: "{base_url}"

engine:
  dir: "{ENGINE_DIR}"
  name: "sunfish.py"
  working_dir: "{ENGINE_DIR}"
  protocol: "uci"
  ponder: true
  uci_options:
    # Keep the engine's default table size: lichess-bot hardcodes a 10 s
    # search for the first move of a game, and a small table can FIFO-evict
    # the root move during it, making sunfish answer "bestmove (none)".
    TABLE_SIZE: 1000000

challenge:
  concurrency: 1
  accept_bot: true
  variants:
    - standard
  time_controls:
    - bullet
    - blitz
  modes:
    - casual
    - rated

matchmaking:
  allow_matchmaking: false
""")


def _wait_for(condition, timeout: float, message: str, proc=None) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if proc is not None and proc.poll() is not None:
            pytest.fail(f"lichess-bot exited with code {proc.returncode} "
                        f"while waiting: {message}")
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout}s: {message}")
        time.sleep(0.2)


def _play_game(mock: MockLichess, proc, **challenge_kwargs):
    challenge_id = mock.send_challenge(**challenge_kwargs)

    def accepted():
        assert challenge_id not in mock.declined, \
            f"challenge declined: {mock.declined.get(challenge_id)}"
        return challenge_id in mock.games

    _wait_for(accepted, ACCEPT_TIMEOUT,
              f"challenge {challenge_id} was not accepted", proc)
    game = mock.games[challenge_id]
    _wait_for(game.finished.is_set, GAME_TIMEOUT,
              f"game {challenge_id} did not finish", proc)
    return game


def _check_game(game, label: str) -> None:
    bot_color = "white" if game.bot_color == chess.WHITE else "black"
    assert game.status in TERMINAL_STATUSES, \
        f"{label}: game did not complete (status={game.status})"
    assert game.status in {"mate", "stalemate", "draw", "outoftime"}, \
        f"{label}: unexpected termination {game.status} (winner={game.winner})"
    if game.status == "outoftime":
        assert game.winner == bot_color, \
            f"{label}: engine forfeited on time (winner={game.winner})"
    assert game.bot_move_latencies, f"{label}: engine never moved"
    slowest = max(game.bot_move_latencies)
    assert slowest < MAX_MOVE_SECONDS, \
        f"{label}: slowest engine move took {slowest:.1f}s " \
        f"(limit {MAX_MOVE_SECONDS}s); the engine likely hung"
    later_moves = game.bot_move_latencies[1:]
    if later_moves:
        slowest_later = max(later_moves)
        assert slowest_later < MAX_LATER_MOVE_SECONDS, \
            f"{label}: engine move took {slowest_later:.1f}s after the first " \
            f"move (limit {MAX_LATER_MOVE_SECONDS}s); the engine stalled"


def test_bot_plays_two_games(lichess_bot, tmp_path):
    checkout, python = lichess_bot
    mock = MockLichess()
    mock.start()

    config_path = tmp_path / "config.yml"
    _write_config(config_path, mock.base_url, mock.token)
    log_path = tmp_path / "lichess-bot.log"
    log_file = open(log_path, "wb")

    proc = subprocess.Popen(
        [str(python), "lichess-bot.py", "--config", str(config_path),
         "-v", "--disable_auto_logging"],
        cwd=checkout, stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        _wait_for(lambda: mock.event_stream_connections > 0, STARTUP_TIMEOUT,
                  "lichess-bot never connected to the event stream", proc)

        # Game 1: instant opponent replies (the pondering race), bot is white.
        game1 = _play_game(mock, proc, bot_plays_white=True,
                           clock_limit=60, clock_increment=0,
                           opponent_delay=0.0, seed=1)

        # Game 2: jittered 50-200 ms replies, bot is black.
        game2 = _play_game(mock, proc, bot_plays_white=False,
                           clock_limit=60, clock_increment=0,
                           opponent_delay=(0.05, 0.2), seed=2)

        # Game 3: 2-5 s opponent thinks let the ponder search run deep before
        # the miss/hit arrives; a capped game keeps the runtime bounded.
        game3 = _play_game(mock, proc, bot_plays_white=True,
                           clock_limit=60, clock_increment=0,
                           opponent_delay=(2.0, 5.0), move_cap=30, seed=3)

        _check_game(game1, "game 1 (instant replies)")
        _check_game(game2, "game 2 (jittered replies)")
        _check_game(game3, "game 3 (slow opponent)")
    except Exception:
        log_file.flush()
        tail = log_path.read_text(errors="replace").splitlines()[-60:]
        print(f"\n----- tail of {log_path} -----")
        print("\n".join(tail))
        raise
    finally:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=15)
        log_file.close()
        mock.stop()

    for label, game in (("game 1", game1), ("game 2", game2), ("game 3", game3)):
        print(f"{label}: {game.status} winner={game.winner} "
              f"plies={len(game.board.move_stack)} "
              f"slowest bot move {max(game.bot_move_latencies):.2f}s")
