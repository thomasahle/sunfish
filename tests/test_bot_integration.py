"""End-to-end test: the real lichess-bot bridge + sunfish vs a mock lichess server.

This launches the official `lichess-bot <https://github.com/lichess-bot-devs/lichess-bot>`_
process (the same bridge tools/lichess deploys to a VM), pointed at the
in-process HTTP mock in tests/mock_lichess.py, with ``ponder: true`` -- the
configuration that exposed a real ponder race in sunfish_ui/uci.py which
lichess-bot's own test suite (test_bot/, a Python-class-level fake with no
pondering coverage) does not catch.  Testing over real HTTP also covers
lib/lichess.py, the ndjson streams, and config parsing.

The checkout is pinned and then patched with the bundle's lichess-bot.patch --
the exact tree production runs (both bundles ship the identical patch;
see tools/lichess/setup.sh and nnue_4k/lichess/setup.sh).  The patch
exists because of three production incidents, and each has a regression test
here:

* ``test_bot_plays_three_games``: three bullet games against a random mover
  (instant, jittered, and slow replies) -- move timing, pondering races and
  time-forfeit coverage.
* ``test_overflow_games_are_aborted_not_starved``: a game that lichess starts
  beyond ``challenge.concurrency`` (e.g. an outgoing matchmaking challenge
  accepted at the same time as an incoming one) must be aborted promptly, not
  queued behind the busy game process where it would sit until lichess
  abandons it (production: game ZbT4vfAs, 2026-08-11).
* ``test_event_stream_death_is_loud``: if the process reading the event
  stream dies, the bot must notice the silence, say so in the log, and
  restart -- never idle deaf-but-online for an hour (production: 2026-08-11
  post-restart deafness).
* ``test_restart_resumes_live_game``: killing the bot mid-game and starting
  it again must resume and finish the game (the mid-game-deploy case).

Setup choice: lichess-bot is not pip-installable, so a pinned checkout is
cached under ~/.cache/sunfish-bot-ci (override with LICHESS_BOT_CACHE), with a
dedicated venv holding its requirements.  On GitHub CI, cache that directory
with actions/cache to make reruns fast; the first run bootstraps it
automatically (one shallow clone + pip install, roughly a minute).

The test only runs when BOT_CI=1 is set, so plain `pytest` stays fast and
network-free.  Environment overrides:

- BOT_CI=1              enable this test
- SUNFISH_ENGINE_DIR    directory containing sunfish.py + sunfish_ui/ to test
                        (default: this repository)
- LICHESS_BOT_CACHE     cache directory for the lichess-bot checkout + venv
- LICHESS_BOT_COMMIT    lichess-bot commit to pin (default below)
- LICHESS_BOT_PATCH     patch file applied on top of the pin (default: the
                        deployed bundle's lichess-bot.patch; set to the empty
                        string to test the unpatched upstream commit)
"""

import contextlib
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
# The engine under test is parameterized: default classic; the packed
# engine runs as SUNFISH_ENGINE_FILE=nnue_4k/sunfish_nnue.py (SF_NET
# reaches the engine child through lichess-bot's environment).
ENGINE_FILE = os.environ.get("SUNFISH_ENGINE_FILE", "sunfish.py")
if ENGINE_FILE.endswith("sunfish_nnue.py"):
    os.environ.setdefault(
        "SF_NET", str(REPO_ROOT / "nnue_4k" / "net128.sfnn"))
CACHE_DIR = Path(os.environ.get("LICHESS_BOT_CACHE",
                                Path.home() / ".cache" / "sunfish-bot-ci"))
LICHESS_BOT_URL = "https://github.com/lichess-bot-devs/lichess-bot"
# lichess-bot master as of 2026-08-05 (version 2026.8.2.1).
LICHESS_BOT_COMMIT = os.environ.get(
    "LICHESS_BOT_COMMIT", "bedd1d9e86a8c4c96319490533e4e20fe63d1ac8")
# The production patch: this repository's lichess bundle carries it and
# setup.sh applies it on deployment, so the tested tree is the deployed tree.
# The two bundles carry byte-identical patches (asserted in the config
# tests); the integration run reads the nnue copy.
BUNDLE_DIR = REPO_ROOT / "nnue_4k" / "lichess"
PATCH_FILE = os.environ.get("LICHESS_BOT_PATCH",
                            str(BUNDLE_DIR / "lichess-bot.patch"))

STARTUP_TIMEOUT = 120  # engine config check + connect to the event stream
ACCEPT_TIMEOUT = 60  # challenge event -> accepted -> gameStart
GAME_TIMEOUT = 240  # both clocks are 60s; a finished 60+0 game fits easily
MAX_MOVE_SECONDS = 12.0  # any move: lichess-bot searches a full 10 s for the
# first, and the engine is entitled to all of it; allow IO/runner overhead
MAX_LATER_MOVE_SECONDS = 4.0  # moves after the first (sunfish thinks ~1.5 s)
ABORT_WINDOW = 30.0  # a game the bot cannot play must be aborted this fast
# (lichess abandons an unserved game after 1-2 minutes; production T ~ 15 s)


def _run(cmd, **kwargs):
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, **kwargs)


def _ensure_checkout() -> Path:
    """Clone the pinned lichess-bot commit and apply the production patch."""
    checkout = CACHE_DIR / "lichess-bot"
    if not (checkout / "lichess-bot.py").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], cwd=checkout)
        _run(["git", "remote", "add", "origin", LICHESS_BOT_URL], cwd=checkout)
    have = subprocess.run(["git", "cat-file", "-e", LICHESS_BOT_COMMIT],
                          cwd=checkout, capture_output=True)
    if have.returncode != 0:
        _run(["git", "fetch", "-q", "--depth", "1", "origin", LICHESS_BOT_COMMIT],
             cwd=checkout)
    # Always force the pin and re-apply the patch: the tree must be exactly
    # pin + bundle patch, the tree production deploys.
    _run(["git", "checkout", "-q", "-f", LICHESS_BOT_COMMIT], cwd=checkout)
    if PATCH_FILE:
        _run(["git", "apply", PATCH_FILE], cwd=checkout)
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
    """(checkout dir, venv python) for the pinned+patched lichess-bot."""
    assert (ENGINE_DIR / ENGINE_FILE).exists(), f"no engine in {ENGINE_DIR}"
    assert (ENGINE_DIR / "sunfish_ui" / "uci.py").exists(), \
        f"no sunfish_ui/ in {ENGINE_DIR}"
    checkout = _ensure_checkout()
    return checkout, _ensure_venv(checkout)


def _write_config(path: Path, base_url: str, token: str) -> None:
    path.write_text(f"""\
token: "{token}"
url: "{base_url}"

engine:
  dir: "{ENGINE_DIR}"
  name: "{ENGINE_FILE}"
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

greeting:
  hello: "Hi {{opponent}}! I'm {{me}}. Good luck!"
  goodbye: "Good game!"

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


def _stop_bot(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=15)


def _launch_bot(checkout: Path, python: Path, config_path: Path, log_path: Path):
    """Start a lichess-bot process appending to log_path; returns the Popen."""
    log_file = open(log_path, "ab")
    try:
        return subprocess.Popen(
            [str(python), "lichess-bot.py", "--config", str(config_path),
             "-v", "--disable_auto_logging"],
            cwd=checkout, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True)
    finally:
        log_file.close()  # the child holds its own descriptor


@contextlib.contextmanager
def _running_bot(lichess_bot, tmp_path, mock, wait_for_stream=True):
    """A lichess-bot process against `mock`; yields (proc holder, log path).

    The holder is a one-element list so tests that kill and relaunch the bot
    can swap in the new process and still get cleaned up.
    """
    checkout, python = lichess_bot
    config_path = tmp_path / "config.yml"
    _write_config(config_path, mock.base_url, mock.token)
    log_path = tmp_path / "lichess-bot.log"
    proc = _launch_bot(checkout, python, config_path, log_path)
    holder = [proc]
    try:
        if wait_for_stream:
            _wait_for(lambda: mock.event_stream_connections > 0, STARTUP_TIMEOUT,
                      "lichess-bot never connected to the event stream", proc)
        yield holder, log_path
    except Exception:
        tail = log_path.read_text(errors="replace").splitlines()[-80:]
        print(f"\n----- tail of {log_path} -----")
        print("\n".join(tail))
        raise
    finally:
        _stop_bot(holder[-1])
        mock.stop()


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


def test_bot_plays_three_games(lichess_bot, tmp_path):
    mock = MockLichess()
    mock.start()
    with _running_bot(lichess_bot, tmp_path, mock) as (holder, log_path):
        proc = holder[0]

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
        # The greeting must actually arrive (production 2026-08-11: a >140
        # char greeting was silently dropped -- and, unpatched, its failed
        # POST cancelled the first move).
        assert any(line["text"].startswith("Hi ") for line in game1.chat_lines), \
            "greeting never reached the chat"

    for label, game in (("game 1", game1), ("game 2", game2), ("game 3", game3)):
        print(f"{label}: {game.status} winner={game.winner} "
              f"plies={len(game.board.move_stack)} "
              f"slowest bot move {max(game.bot_move_latencies):.2f}s")


def test_overflow_games_are_aborted_not_starved(lichess_bot, tmp_path):
    """K games at challenge.concurrency=1: one served, the rest aborted fast.

    Reproduces the 2026-08-11 production race: while one accepted game is
    being played, lichess starts more games via bare gameStart events (an
    opponent accepting the bot's outgoing matchmaking challenge does exactly
    this; so does reconnecting with several games live).  Unpatched
    lichess-bot gives one such game the pool's reserve process and quietly
    queues the rest forever -- game ZbT4vfAs never saw a move until lichess
    abandoned it.  The invariant: every started game is either played at
    full strength or promptly, explicitly aborted.
    """
    mock = MockLichess()
    mock.start()
    with _running_bot(lichess_bot, tmp_path, mock) as (holder, log_path):
        proc = holder[0]

        # The served game: a normal incoming challenge, accepted by the bot.
        challenge_id = mock.send_challenge(bot_plays_white=True, clock_limit=60,
                                           clock_increment=0,
                                           opponent_delay=(0.05, 0.15),
                                           move_cap=20, seed=4)
        _wait_for(lambda: challenge_id in mock.games, ACCEPT_TIMEOUT,
                  f"challenge {challenge_id} was not accepted", proc)

        # Two more games start while the first is busy.  Their opponent never
        # moves, so any engine service the bot (wrongly) gives them is visible
        # as a bot move, and they stay abortable throughout.
        overflow = [mock.start_direct_game(bot_plays_white=True, clock_limit=60,
                                           opponent_delay=999.0, seed=5 + i)
                    for i in range(2)]

        deadline = time.monotonic() + ABORT_WINDOW
        for game_id in overflow:
            game = mock.games[game_id]
            _wait_for(lambda g=game: g.status != "started",
                      max(0.1, deadline - time.monotonic()),
                      f"overflow game {game_id} was never aborted: it would "
                      "sit unserved until lichess abandoned it", proc)
            assert game.status == "aborted", \
                f"overflow game {game_id} ended {game.status}, expected a " \
                "prompt abort"
            assert not game.bot_move_latencies, \
                f"overflow game {game_id} received engine moves: the one " \
                "game slot must keep full strength (concurrency stays 1)"

        # The served game is unaffected and finishes cleanly.
        game1 = mock.games[challenge_id]
        _wait_for(game1.finished.is_set, GAME_TIMEOUT,
                  f"game {challenge_id} did not finish", proc)
        _check_game(game1, "served game")


@pytest.mark.skipif(shutil.which("lsof") is None, reason="needs lsof")
def test_event_stream_death_is_loud(lichess_bot, tmp_path):
    """A dead event stream must be noticed, logged, and recovered from.

    Reproduces the 2026-08-11 production deafness: the bot sat connected but
    received no events for an hour while lichess started games against it.
    lichess-bot reads the event stream in a child process; if that process
    dies, the unpatched main loop blocks forever on the control queue --
    online, silent, deaf.  Patched, the silence (a healthy stream pings every
    few seconds) is detected within a minute, logged as an error, and the bot
    restarts itself and plays on.
    """
    mock = MockLichess()
    mock.start()
    with _running_bot(lichess_bot, tmp_path, mock) as (holder, log_path):
        proc = holder[0]
        _wait_for(lambda: mock.event_stream_clients, STARTUP_TIMEOUT,
                  "no event stream connection", proc)
        connections_before = mock.event_stream_connections
        client_port = mock.event_stream_clients[-1][1]

        # Kill the process that owns the event-stream socket (the
        # watch_control_stream child), simulating its silent death.
        out = subprocess.run(["lsof", "-t", "-n", "-i", f"tcp:{client_port}"],
                             capture_output=True, text=True)
        pids = {int(p) for p in out.stdout.split()} - {os.getpid()}
        assert pids, f"could not find the event stream reader (port {client_port})"
        for pid in pids:
            os.kill(pid, signal.SIGKILL)

        # The bot must notice the silence and reconnect (via its restart
        # machinery). 60 s silence limit + 10 s restart pause + startup.
        _wait_for(lambda: mock.event_stream_connections > connections_before,
                  150, "the bot never noticed its event stream was dead: "
                  "it is deaf but looks online", proc)
        # Collapse whitespace: the bot's console handler wraps long lines.
        log_flat = " ".join(log_path.read_text(errors="replace").split())
        assert "event stream is assumed dead" in log_flat, \
            "stream death was not logged loudly"

        # Full recovery: a new challenge is accepted and played to the end.
        game = _play_game(mock, proc, bot_plays_white=True, clock_limit=60,
                          clock_increment=0, opponent_delay=(0.05, 0.15),
                          move_cap=12, seed=7)
        _check_game(game, "post-recovery game")


def test_restart_resumes_live_game(lichess_bot, tmp_path):
    """Kill the bot mid-game; a fresh start must resume and finish the game.

    This is the mid-game-deploy case: systemd restarts the service while a
    game is live.  On reconnect the event stream replays gameStart for
    ongoing games (as lichess does) and the bot must pick the game back up
    rather than let it be abandoned -- and the concurrency guard must treat a
    resumed game as the served game, not overflow.
    """
    checkout, python = lichess_bot
    mock = MockLichess()
    mock.start()
    with _running_bot(lichess_bot, tmp_path, mock) as (holder, log_path):
        proc = holder[0]
        # A blitz clock: the bot must survive ~15-25 s of downtime mid-game.
        challenge_id = mock.send_challenge(bot_plays_white=True,
                                           clock_limit=180, clock_increment=0,
                                           opponent_delay=(0.5, 1.5),
                                           move_cap=40, seed=8)
        _wait_for(lambda: challenge_id in mock.games, ACCEPT_TIMEOUT,
                  f"challenge {challenge_id} was not accepted", proc)
        game = mock.games[challenge_id]
        _wait_for(lambda: len(game.board.move_stack) >= 4, 90,
                  "the game never got going", proc)

        # Hard-kill the whole process group mid-game (crash / deploy).
        moves_before = len(game.bot_move_latencies)
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)

        holder.append(_launch_bot(checkout, python,
                                  tmp_path / "config.yml", log_path))
        proc2 = holder[-1]
        _wait_for(game.finished.is_set, GAME_TIMEOUT,
                  "the restarted bot never resumed the live game", proc2)

        assert game.status in {"mate", "stalemate", "draw", "outoftime"}, \
            f"resumed game ended {game.status} (winner={game.winner})"
        bot_color = "white" if game.bot_color == chess.WHITE else "black"
        if game.status == "outoftime":
            assert game.winner == bot_color, \
                f"the bot lost on time across the restart (winner={game.winner})"
        assert len(game.bot_move_latencies) > moves_before, \
            "the restarted bot never moved in the resumed game"
        # Moves after the resume settle back to normal speed (the first one
        # may include the whole restart in its latency).
        settled = game.bot_move_latencies[moves_before + 1:]
        if settled:
            assert max(settled) < MAX_MOVE_SECONDS, \
                f"post-restart engine moves too slow: {max(settled):.1f}s"
