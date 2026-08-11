"""A lightweight mock of the lichess.org bot API, faithful to the wire protocol.

Serves just enough of the lichess bot API over real HTTP for the official
`lichess-bot <https://github.com/lichess-bot-devs/lichess-bot>`_ bridge to run
unmodified against it (its whole client is ``lib/lichess.py``, which talks to
the endpoints below with a Bearer token and reads ndjson streams).

JSON shapes mirror the OpenAPI schemas in the lichess-org/api repository
(doc/specs/schemas/): GameFullEvent, GameStateEvent, GameEventPlayer,
GameEventInfo, ChallengeJson, GameStatusName.  Clocks are enforced for real,
server side, the way lila does it: times are in milliseconds, the clocks only
start running once both players have moved (ply >= 2), each move deducts the
mover's wall-clock elapsed time and adds the increment, and a player whose
flag falls loses with ``status: outoftime`` and a ``winner`` (or draws if the
opponent has insufficient mating material).

The scripted opponent plays uniformly random legal moves (python-chess) after
a configurable delay -- including zero delay, which exercises the
engine-pondering race where the opponent's reply arrives while the previous
search is still winding down.

Only the standard library and python-chess are required.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import chess
import chess.pgn

logger = logging.getLogger(__name__)

# Status ids as in lila (modules/game/src/main/Status.scala).
STATUS_IDS = {
    "created": 10, "started": 20, "aborted": 25, "mate": 30, "resign": 31,
    "stalemate": 32, "timeout": 33, "draw": 34, "outoftime": 35, "cheat": 36,
    "noStart": 37, "unknownFinish": 38, "variantEnd": 60,
}
TERMINAL_STATUSES = frozenset(STATUS_IDS) - {"created", "started"}

STANDARD_VARIANT = {"key": "standard", "name": "Standard", "short": "Std"}

# Sentinel telling a game-stream connection to close.
_CLOSE = object()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        # Clients dropping keep-alive connections mid-request is routine.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


KEEPALIVE_SECONDS = 2.0  # lila sends a bare newline every few seconds
STREAM_LINGER_SECONDS = 1.0  # keep the game stream open a moment after the end


def speed_of(limit: int, increment: int) -> str:
    """Bucket a clock into a lichess speed, like lila's Speed.apply."""
    total = limit + 40 * increment
    if total < 30:
        return "ultraBullet"
    if total < 180:
        return "bullet"
    if total < 480:
        return "blitz"
    if total < 1500:
        return "rapid"
    return "classical"


def _perf_name(speed: str) -> str:
    return {"ultraBullet": "UltraBullet"}.get(speed, speed.capitalize())


class Game:
    """One game with real server-side clocks and a scripted opponent."""

    def __init__(self, server: "MockLichess", game_id: str, bot_color: chess.Color,
                 clock_limit: int, clock_increment: int,
                 opponent_delay, move_cap: int, rated: bool, seed=None):
        self.server = server
        self.id = game_id
        self.bot_color = bot_color
        self.clock_initial_ms = clock_limit * 1000
        self.clock_increment_ms = clock_increment * 1000
        self.speed = speed_of(clock_limit, clock_increment)
        self.rated = rated
        self.opponent_delay = opponent_delay  # seconds, or (lo, hi) for jitter
        self.move_cap = move_cap
        self.created_at = int(time.time() * 1000)

        self.board = chess.Board()
        self.remaining_ms = {chess.WHITE: float(self.clock_initial_ms),
                             chess.BLACK: float(self.clock_initial_ms)}
        self.turn_started_at = time.monotonic()
        self.status = "started"
        self.winner: str | None = None
        self.lock = threading.RLock()
        self.subscribers: list[queue.Queue] = []
        self.finished = threading.Event()
        self.chat_lines: list[dict] = []
        # Wall-clock seconds from "bot's turn began" to the move POST arriving.
        self.bot_move_latencies: list[float] = []
        self._rng = random.Random(seed)

        self._threads = [
            threading.Thread(target=self._ticker, daemon=True,
                             name=f"ticker-{game_id}"),
            threading.Thread(target=self._opponent, daemon=True,
                             name=f"opponent-{game_id}"),
        ]

    def start(self) -> None:
        with self.lock:
            self.turn_started_at = time.monotonic()
        for t in self._threads:
            t.start()

    # -- clock ------------------------------------------------------------

    def _clock_running(self) -> bool:
        # lila: the clock only starts once both players have made a move.
        return len(self.board.move_stack) >= 2

    # -- json shapes (mirroring lichess-org/api schemas) -------------------

    def _player_json(self, color: chess.Color) -> dict:
        if color == self.bot_color:
            return {"id": self.server.bot_id, "name": self.server.bot_name,
                    "title": "BOT", "rating": 1800, "provisional": False}
        return {"id": self.server.opponent_id, "name": self.server.opponent_name,
                "title": "BOT", "rating": 1800, "provisional": False}

    def state_json(self) -> dict:
        """GameStateEvent (GameStateEvent.yaml). Call with self.lock held."""
        state = {
            "type": "gameState",
            "moves": " ".join(m.uci() for m in self.board.move_stack),
            "wtime": max(0, int(self.remaining_ms[chess.WHITE])),
            "btime": max(0, int(self.remaining_ms[chess.BLACK])),
            "winc": self.clock_increment_ms,
            "binc": self.clock_increment_ms,
            "status": self.status,
        }
        if self.winner:
            state["winner"] = self.winner
        return state

    def full_json(self) -> dict:
        """GameFullEvent (GameFullEvent.yaml). Call with self.lock held."""
        return {
            "type": "gameFull",
            "id": self.id,
            "variant": STANDARD_VARIANT,
            "clock": {"initial": self.clock_initial_ms,
                      "increment": self.clock_increment_ms},
            "speed": self.speed,
            "perf": {"name": _perf_name(self.speed)},
            "rated": self.rated,
            "createdAt": self.created_at,
            "white": self._player_json(chess.WHITE),
            "black": self._player_json(chess.BLACK),
            "initialFen": "startpos",
            "state": self.state_json(),
        }

    def event_info_json(self) -> dict:
        """GameEventInfo (GameEventInfo.yaml), used in gameStart/gameFinish."""
        with self.lock:
            bot_color_name = "white" if self.bot_color == chess.WHITE else "black"
            last_move = self.board.move_stack[-1].uci() if self.board.move_stack else ""
            info = {
                "fullId": self.id + "AbCd",
                "gameId": self.id,
                "id": self.id,
                "fen": self.board.fen(),
                "color": bot_color_name,
                "lastMove": last_move,
                "source": "friend",
                "status": {"id": STATUS_IDS[self.status], "name": self.status},
                "variant": STANDARD_VARIANT,
                "speed": self.speed,
                "perf": self.speed,
                "rated": self.rated,
                "hasMoved": bool(self.board.move_stack),
                "opponent": {"id": self.server.opponent_id,
                             "username": self.server.opponent_name,
                             "rating": 1800},
                "isMyTurn": self.status == "started" and self.board.turn == self.bot_color,
                "secondsLeft": int(self.remaining_ms[self.bot_color] / 1000),
                "compat": {"bot": True, "board": False},
            }
            if self.winner:
                info["winner"] = self.winner
            return info

    def pgn(self) -> str:
        with self.lock:
            game = chess.pgn.Game.from_board(self.board)
            game.headers["Event"] = "Casual Bullet game"
            game.headers["Site"] = self.server.base_url + self.id
            game.headers["White"] = self._player_json(chess.WHITE)["name"]
            game.headers["Black"] = self._player_json(chess.BLACK)["name"]
            game.headers["Result"] = self.board.result(claim_draw=True)
            return str(game)

    # -- game flow ---------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
            if self.status != "started":
                q.put(_CLOSE)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _broadcast_locked(self) -> None:
        state = self.state_json()
        for q in self.subscribers:
            q.put(state)

    def _end_locked(self, status: str, winner: str | None) -> None:
        if self.status != "started":
            return
        self.status = status
        self.winner = winner
        self._broadcast_locked()
        self.server.push_event({"type": "gameFinish", "game": self.event_info_json()})
        self.finished.set()
        # Close game-stream connections shortly after the final gameState,
        # like lila does when a game ends.
        subscribers = list(self.subscribers)

        def close_streams():
            for q in subscribers:
                q.put(_CLOSE)
        threading.Timer(STREAM_LINGER_SECONDS, close_streams).start()
        logger.info("game %s over: %s winner=%s after %d plies",
                    self.id, status, winner, len(self.board.move_stack))

    def _adjudicate_locked(self) -> None:
        """End the game if the position is terminal, as lila auto-adjudicates
        for bot games (bots cannot claim draws; lichess claims for them)."""
        board = self.board
        mover = "black" if board.turn == chess.WHITE else "white"  # who just moved
        if board.is_checkmate():
            self._end_locked("mate", mover)
        elif board.is_stalemate():
            self._end_locked("stalemate", None)
        elif board.is_insufficient_material():
            self._end_locked("draw", None)
        elif (board.can_claim_fifty_moves() or board.can_claim_threefold_repetition()
              or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            self._end_locked("draw", None)
        elif len(board.move_stack) >= self.move_cap:
            self._end_locked("draw", None)

    def make_move(self, uci: str, by_bot: bool):
        """Apply a move. Returns (http_status, response_json)."""
        with self.lock:
            if self.status != "started":
                return 400, {"error": f"Game already over: {self.status}"}
            mover = self.board.turn
            if by_bot and mover != self.bot_color:
                return 400, {"error": "It is not your turn to move"}
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                return 400, {"error": f"Invalid move format: {uci}"}
            if move not in self.board.legal_moves:
                return 400, {"error": f"Illegal move: {uci}"}

            now = time.monotonic()
            if self._clock_running():
                left = self.remaining_ms[mover] - (now - self.turn_started_at) * 1000
                if left <= 0:
                    # Flag fell before the move arrived (the ticker usually
                    # catches this first).
                    self.remaining_ms[mover] = 0
                    self._flag_locked(mover)
                    return 400, {"error": "Game already over: outoftime"}
                self.remaining_ms[mover] = left + self.clock_increment_ms
            if by_bot:
                self.bot_move_latencies.append(now - self.turn_started_at)
            self.board.push(move)
            self.turn_started_at = now
            # Adjudicate first so a game-ending move is broadcast exactly
            # once, with its terminal status and winner (as lila does).
            self._adjudicate_locked()
            if self.status == "started":
                self._broadcast_locked()
            return 200, {"ok": True}

    def _flag_locked(self, flagged: chess.Color) -> None:
        other = not flagged
        # lila: flagging is a draw if the opponent cannot possibly mate.
        if self.board.has_insufficient_material(other):
            self._end_locked("outoftime", None)
        else:
            self._end_locked("outoftime", "white" if other == chess.WHITE else "black")

    def resign(self, by_bot: bool = True):
        with self.lock:
            if self.status != "started":
                return 400, {"error": f"Game already over: {self.status}"}
            loser = self.bot_color if by_bot else not self.bot_color
            self._end_locked("resign", "white" if (not loser) == chess.WHITE else "black")
            return 200, {"ok": True}

    def abort(self):
        with self.lock:
            if self.status != "started":
                return 400, {"error": f"Game already over: {self.status}"}
            if len(self.board.move_stack) >= 2:
                return 400, {"error": "Cannot abort: too many moves played"}
            self._end_locked("aborted", None)
            return 200, {"ok": True}

    def _ticker(self) -> None:
        """Enforce the clocks: forfeit the player to move when their flag falls."""
        while not self.finished.is_set():
            with self.lock:
                if self.status == "started" and self._clock_running():
                    mover = self.board.turn
                    elapsed = (time.monotonic() - self.turn_started_at) * 1000
                    if self.remaining_ms[mover] - elapsed <= 0:
                        self.remaining_ms[mover] = 0
                        self._flag_locked(mover)
            time.sleep(0.05)

    def _next_delay(self) -> float:
        delay = self.opponent_delay
        if callable(delay):
            return delay()
        if isinstance(delay, (tuple, list)):
            lo, hi = delay
            return self._rng.uniform(lo, hi)
        return float(delay)

    def _opponent(self) -> None:
        """Play scripted (uniformly random legal) moves for the opponent."""
        while not self.finished.is_set():
            with self.lock:
                my_turn = (self.status == "started"
                           and self.board.turn != self.bot_color)
            if not my_turn:
                time.sleep(0.005)
                continue
            delay = self._next_delay()
            if delay > 0:
                time.sleep(delay)
            with self.lock:
                if self.status != "started" or self.board.turn == self.bot_color:
                    continue
                move = self._rng.choice(list(self.board.legal_moves))
            status, response = self.make_move(move.uci(), by_bot=False)
            if status != 200:
                logger.debug("opponent move rejected: %s", response)


class MockLichess:
    """The mock lichess.org server. Start it, then drive games from the test:

        server = MockLichess()
        server.start()
        game = server.send_challenge(bot_plays_white=True, opponent_delay=0.0)
        game.finished.wait(timeout=180)
    """

    def __init__(self, token: str = "bot-ci-token", bot_name: str = "sunfish_test_bot",
                 opponent_name: str = "mock_opponent", host: str = "127.0.0.1",
                 port: int = 0):
        self.token = token
        self.bot_name = bot_name
        self.bot_id = bot_name.lower()
        self.opponent_name = opponent_name
        self.opponent_id = opponent_name.lower()
        self.events: queue.Queue = queue.Queue()  # ndjson items for /api/stream/event
        self.games: dict[str, Game] = {}
        self.challenges: dict[str, dict] = {}  # id -> {"json": ..., "params": ...}
        self.declined: dict[str, str] = {}  # challenge id -> decline reason
        self.event_stream_connections = 0
        # (host, port) of every client that connected to /api/stream/event,
        # in connection order. The last entry is the live connection; its
        # port identifies which OS process holds the stream (lichess-bot
        # reads it from a child process).
        self.event_stream_clients: list[tuple] = []
        self._game_counter = 0
        self._closing = threading.Event()

        handler = _make_handler(self)
        self._httpd = _Server((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="mock-lichess")

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._closing.set()
        for game in self.games.values():
            game.finished.set()
            with game.lock:
                for q in game.subscribers:
                    q.put(_CLOSE)
        self._httpd.shutdown()
        self._httpd.server_close()

    def push_event(self, event: dict) -> None:
        self.events.put(event)

    # -- test driver API ---------------------------------------------------

    def send_challenge(self, bot_plays_white: bool = True, clock_limit: int = 60,
                       clock_increment: int = 0, opponent_delay=0.0,
                       move_cap: int = 200, rated: bool = False, seed=None) -> str:
        """Emit a challenge event on the event stream; returns the challenge id.

        When lichess-bot accepts it (POST /api/challenge/{id}/accept), the
        game is created and a gameStart event is emitted.
        """
        self._game_counter += 1
        challenge_id = f"mokGam{self._game_counter:02d}"
        speed = speed_of(clock_limit, clock_increment)
        # ChallengeJson.yaml: the challenger is our scripted opponent; color
        # is the color the challenger asks to play.
        challenger_color = "black" if bot_plays_white else "white"
        challenge_json = {
            "id": challenge_id,
            "url": self.base_url + challenge_id,
            "status": "created",
            "challenger": {"id": self.opponent_id, "name": self.opponent_name,
                           "title": "BOT", "rating": 1800, "provisional": False,
                           "online": True, "lag": 4},
            "destUser": {"id": self.bot_id, "name": self.bot_name,
                         "title": "BOT", "rating": 1800, "provisional": False,
                         "online": True, "lag": 4},
            "variant": STANDARD_VARIANT,
            "rated": rated,
            "speed": speed,
            "timeControl": {"type": "clock", "limit": clock_limit,
                            "increment": clock_increment,
                            "show": f"{clock_limit // 60}+{clock_increment}"},
            "color": challenger_color,
            "finalColor": challenger_color,
            "perf": {"icon": "", "name": _perf_name(speed)},
            "direction": "in",
        }
        self.challenges[challenge_id] = {
            "json": challenge_json,
            "params": {"bot_plays_white": bot_plays_white,
                       "clock_limit": clock_limit,
                       "clock_increment": clock_increment,
                       "opponent_delay": opponent_delay,
                       "move_cap": move_cap, "rated": rated, "seed": seed},
        }
        self.push_event({"type": "challenge", "challenge": challenge_json,
                         "compat": {"bot": True, "board": False}})
        return challenge_id

    def start_direct_game(self, bot_plays_white: bool = True, clock_limit: int = 60,
                          clock_increment: int = 0, opponent_delay=0.0,
                          move_cap: int = 200, rated: bool = False, seed=None) -> str:
        """Start a game without any challenge/accept exchange; returns the game id.

        This is legitimate wire behavior: lichess emits a bare gameStart when
        an *outgoing* challenge (matchmaking) is accepted by the opponent, and
        replays gameStart for ongoing games whenever the event stream
        (re)connects.  The bot never gets to accept or reserve these games --
        exactly the race that overflows challenge.concurrency.
        """
        self._game_counter += 1
        game_id = f"dirGam{self._game_counter:02d}"
        game = Game(self, game_id,
                    bot_color=chess.WHITE if bot_plays_white else chess.BLACK,
                    clock_limit=clock_limit, clock_increment=clock_increment,
                    opponent_delay=opponent_delay, move_cap=move_cap,
                    rated=rated, seed=seed)
        self.games[game_id] = game
        game.start()
        self.push_event({"type": "gameStart", "game": game.event_info_json()})
        return game_id

    # -- internal ----------------------------------------------------------

    def accept_challenge(self, challenge_id: str):
        challenge = self.challenges.get(challenge_id)
        if challenge is None:
            return 404, {"error": "No such challenge"}
        if challenge_id in self.games:
            return 400, {"error": "Challenge already accepted"}
        params = challenge["params"]
        game = Game(self, challenge_id,
                    bot_color=chess.WHITE if params["bot_plays_white"] else chess.BLACK,
                    clock_limit=params["clock_limit"],
                    clock_increment=params["clock_increment"],
                    opponent_delay=params["opponent_delay"],
                    move_cap=params["move_cap"],
                    rated=params["rated"],
                    seed=params["seed"])
        self.games[challenge_id] = game
        game.start()
        self.push_event({"type": "gameStart", "game": game.event_info_json()})
        return 200, {"ok": True}

    def decline_challenge(self, challenge_id: str, reason: str):
        if challenge_id not in self.challenges:
            return 404, {"error": "No such challenge"}
        self.declined[challenge_id] = reason
        return 200, {"ok": True}

    def profile_json(self, user_id: str, name: str) -> dict:
        perfs = {key: {"games": 100, "rating": 1800, "rd": 50, "prog": 0}
                 for key in ("bullet", "blitz", "rapid", "classical", "correspondence")}
        return {"id": user_id, "username": name, "title": "BOT", "online": True,
                "perfs": perfs, "createdAt": 1600000000000,
                "seenAt": int(time.time() * 1000),
                "playTime": {"total": 0, "tv": 0}, "followable": True,
                "following": False, "blocking": False}

    def now_playing_json(self) -> dict:
        games = []
        for game in self.games.values():
            if game.status != "started":
                continue
            info = game.event_info_json()
            games.append({
                "gameId": info["gameId"], "fullId": info["fullId"],
                "color": info["color"], "fen": info["fen"],
                "hasMoved": info["hasMoved"], "isMyTurn": info["isMyTurn"],
                "lastMove": info["lastMove"], "opponent": info["opponent"],
                "perf": info["perf"], "rated": info["rated"],
                "secondsLeft": info["secondsLeft"], "source": info["source"],
                "speed": info["speed"], "variant": info["variant"],
            })
        return {"nowPlaying": games}


def _make_handler(server: MockLichess):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 with chunked transfer encoding for the ndjson streams,
        # like the real lichess.org: urllib3 delivers each chunk as it
        # arrives, so stream lines reach the client promptly (a raw
        # unchunked stream would sit in the client's 512-byte read buffer).
        protocol_version = "HTTP/1.1"
        server_version = "lila-mock"

        def log_message(self, fmt, *args):
            logger.debug("%s -- %s", self.address_string(), fmt % args)

        # -- helpers -------------------------------------------------------

        def _json(self, code: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _text(self, code: int, text: str, content_type: str = "text/plain") -> None:
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            auth = self.headers.get("Authorization", "")
            if auth == f"Bearer {server.token}":
                return True
            self._json(401, {"error": "No such token"})
            return False

        def _body(self) -> str:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length).decode() if length else ""

        def _start_ndjson(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def _write_line(self, item) -> None:
            data = (json.dumps(item) + "\n").encode() if item is not None else b"\n"
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _end_stream(self) -> None:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        # -- streams -------------------------------------------------------

        def _serve_event_stream(self) -> None:
            server.event_stream_connections += 1
            server.event_stream_clients.append(self.client_address)
            self._start_ndjson()
            try:
                # Like lila: a (re)connecting event stream first receives the
                # current state -- a gameStart for every ongoing game.  This
                # is what lets a restarted bot resume its live games.
                for game in list(server.games.values()):
                    if game.status == "started":
                        self._write_line({"type": "gameStart",
                                          "game": game.event_info_json()})
                while not server._closing.is_set():
                    try:
                        event = server.events.get(timeout=KEEPALIVE_SECONDS)
                    except queue.Empty:
                        event = None  # keepalive newline, like lila
                    self._write_line(event)
                self._end_stream()
            except OSError:
                pass  # client went away
            finally:
                self.close_connection = True

        def _serve_game_stream(self, game: Game) -> None:
            q = game.subscribe()
            self._start_ndjson()
            try:
                with game.lock:
                    full = game.full_json()
                self._write_line(full)
                while not server._closing.is_set():
                    try:
                        item = q.get(timeout=KEEPALIVE_SECONDS)
                    except queue.Empty:
                        item = None  # keepalive newline
                    if item is _CLOSE:
                        break
                    self._write_line(item)
                self._end_stream()
            except OSError:
                pass  # client went away
            finally:
                game.unsubscribe(q)
                self.close_connection = True

        # -- routing -------------------------------------------------------

        def do_GET(self) -> None:
            try:
                self._route_get()
            except OSError:
                pass

        def do_POST(self) -> None:
            try:
                self._route_post()
            except OSError:
                pass

        def _route_get(self) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            if not self._authorized():
                return

            if url.path == "/api/stream/event":
                self._serve_event_stream()
            elif parts[:4] == ["api", "bot", "game", "stream"] and len(parts) == 5:
                game = server.games.get(parts[4])
                if game is None:
                    self._json(404, {"error": "No such game"})
                else:
                    self._serve_game_stream(game)
            elif url.path == "/api/account":
                self._json(200, server.profile_json(server.bot_id, server.bot_name))
            elif url.path == "/api/account/playing":
                self._json(200, server.now_playing_json())
            elif url.path == "/api/users/status":
                ids = parse_qs(url.query).get("ids", [""])[0].split(",")
                self._json(200, [{"id": uid, "name": uid, "online": True}
                                 for uid in ids if uid])
            elif parts[:2] == ["api", "user"] and len(parts) == 3:
                self._json(200, server.profile_json(parts[2].lower(), parts[2]))
            elif url.path == "/api/bot/online":
                lines = [json.dumps(server.profile_json(server.opponent_id,
                                                        server.opponent_name))]
                self._text(200, "\n".join(lines) + "\n", "application/x-ndjson")
            elif parts[:2] == ["game", "export"] and len(parts) == 3:
                game = server.games.get(parts[2])
                if game is None:
                    self._json(404, {"error": "No such game"})
                else:
                    self._text(200, game.pgn(), "application/x-chess-pgn")
            else:
                self._json(404, {"error": "Not found"})

        def _route_post(self) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            body = self._body()

            if url.path == "/api/token/test":
                # No auth header needed; the body is the token under test.
                result = {}
                for token in filter(None, body.split(",")):
                    if token == server.token:
                        result[token] = {"userId": server.bot_id,
                                         "scopes": "bot:play",
                                         "expires": int(time.time() * 1000) + 10**9}
                    else:
                        result[token] = None
                self._json(200, result)
                return

            if not self._authorized():
                return

            if parts[:2] == ["api", "challenge"] and len(parts) == 4:
                challenge_id, action = parts[2], parts[3]
                if action == "accept":
                    self._json(*server.accept_challenge(challenge_id))
                elif action == "decline":
                    reason = parse_qs(body).get("reason", ["generic"])[0]
                    self._json(*server.decline_challenge(challenge_id, reason))
                elif action == "cancel":
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "Not found"})
                return

            if parts[:3] == ["api", "bot", "game"] and len(parts) >= 5:
                game = server.games.get(parts[3])
                if game is None:
                    self._json(404, {"error": "No such game"})
                    return
                action = parts[4]
                if action == "move" and len(parts) == 6:
                    self._json(*game.make_move(parts[5], by_bot=True))
                elif action == "resign":
                    self._json(*game.resign(by_bot=True))
                elif action == "abort":
                    self._json(*game.abort())
                elif action == "chat":
                    form = parse_qs(body)
                    game.chat_lines.append({"room": form.get("room", [""])[0],
                                            "text": form.get("text", [""])[0]})
                    self._json(200, {"ok": True})
                elif action == "takeback":
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "Not found"})
                return

            self._json(404, {"error": "Not found"})

    return Handler
