#!/usr/bin/env python3
"""Turn confirmed mistakes from a Lichess bot archive into an EPD suite.

Typical reproducible workflow::

    tools/blunder_scan.py sunfish-engine --pgn-cache sunfish-games.pgn \
        --checkpoint sunfish-games.checkpoint.json \
        --output lichess_blunders.epd

The first run downloads one PGN archive and caches it. Later runs read only
that file unless ``--refresh`` is passed. The optional checkpoint journals
each completed game atomically and resumes only when its input and settings
identity matches exactly. A quick fixed-node Stockfish pass finds candidates
using Lichess's winning-chance and mate-advice rules.
Equal-budget single-PV probes then confirm the judgment, while MultiPV probes
at two larger budgets must agree on every accepted near-best move. Positions
too close to the MultiPV boundary are rejected rather than incompletely
labelled.

The output is ordinary EPD accepted by ``tools/tester.py best``. Its ``bm``
operation contains the certified alternatives, while ``c0`` through ``c2``
record the source game, played move, Lichess judgment, and oracle settings.
Network access and Stockfish are generation-time dependencies only; the
committed EPD is a deterministic test input.
"""
import argparse
import dataclasses
import datetime
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Optional
import urllib.parse
import urllib.request

import chess
import chess.engine
import chess.pgn

MATE_RANK = 100_000
LICHESS_ADVICE_COMMIT = "5b905153c32677034dbb3325ecbd66418a03281e"
LICHESS_EVAL_COMMIT = "34b3363839c511b258fec17b30462868e31d9b5a"
LICHESS_BLUNDER_THRESHOLD = 0.3
CHECKPOINT_SCHEMA = "sunfish-blunder-checkpoint-v1"
USER_AGENT = "sunfish-blunder-corpus/1 (+https://github.com/thomasahle/sunfish)"
STANDARD_PERFS = {"bullet", "blitz", "rapid", "classical", "correspondence"}


@dataclasses.dataclass(frozen=True)
class Candidate:
    fen: str
    played: chess.Move
    game_id: str
    ply: int
    user: str
    opponent: str
    result: str
    time_control: str


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """One mover-perspective engine score, without conflating mate and cp."""

    cp: Optional[int] = None
    mate: Optional[int] = None

    def __post_init__(self):
        if (self.cp is None) == (self.mate is None):
            raise ValueError("an evaluation must contain exactly one score kind")

    def __str__(self):
        return f"{self.cp:+d} cp" if self.cp is not None else f"mate {self.mate:+d}"


@dataclasses.dataclass(frozen=True)
class Blunder:
    candidate: Candidate
    best_moves: tuple[chess.Move, ...]
    best_eval: Evaluation
    played_eval: Evaluation
    oracle: str
    scan_nodes: int
    confirm_nodes: int
    stability_nodes: int
    best_margin: int
    boundary_guard: int
    multipv: int
    source_sha: str
    game_order: str = "archive"
    source_game_count: int = 0

    @property
    def cp_loss(self):
        if self.best_eval.cp is None or self.played_eval.cp is None:
            return None
        return self.best_eval.cp - self.played_eval.cp

    @property
    def chance_loss(self):
        if self.best_eval.cp is None or self.played_eval.cp is None:
            return None
        return (winning_chances(self.best_eval.cp)
                - winning_chances(self.played_eval.cp))


@dataclasses.dataclass
class Funnel:
    examined: int = 0
    scan_blunders: int = 0
    confirmed_blunders: int = 0
    stable_labels: int = 0


def parse_games(text, accept=None, limit=None):
    """Parse accepted games in file order, stopping at ``limit`` when set."""
    games, stream = [], io.StringIO(text)
    while (game := chess.pgn.read_game(stream)) is not None:
        if game.errors:
            raise ValueError(f"malformed PGN game: {game.errors}")
        if accept is None or accept(game):
            games.append(game)
            if limit is not None and len(games) >= limit:
                break
    return games


def is_archive_game(game, include_casual=False):
    """Apply the rated, speed, and Standard-variant archive filters locally."""
    headers = game.headers
    if headers.get("Variant", "Standard").lower() != "standard":
        return False
    event = headers.get("Event", "").lower().split()
    if len(event) != 3 or event[2] != "game" or event[1] not in STANDARD_PERFS:
        return False
    return event[0] == "rated" or (include_casual and event[0] == "casual")


def source_games(text, games, include_casual=False, requested_game_id=None):
    """Validate a single export or locally filter and cap a user archive."""
    if requested_game_id:
        selected = parse_games(text, limit=2)
        if len(selected) != 1:
            raise ValueError(
                f"--game-id expected one PGN game, found {len(selected)}")
        variant = selected[0].headers.get("Variant", "Standard")
        if variant.lower() != "standard":
            raise ValueError(
                f"--game-id only supports Standard chess, found {variant!r}")
        return selected
    return parse_games(
        text,
        accept=lambda game: is_archive_game(game, include_casual),
        limit=games or None,
    )


def game_id(game):
    """Extract the stable Lichess game id from a Site header."""
    path = urllib.parse.urlparse(game.headers.get("Site", "")).path
    return path.strip("/").split("/", 1)[0] or "unknown"


def user_outcome(game, user):
    """Return a completed game's result from ``user``'s point of view."""
    headers = game.headers
    user = user.lower()
    white = headers.get("White", "").lower()
    black = headers.get("Black", "").lower()
    if user not in (white, black):
        return "other"
    result = headers.get("Result")
    if result == "1/2-1/2":
        return "draw"
    if result not in ("1-0", "0-1"):
        return "other"
    user_won = ((white == user and result == "1-0")
                or (black == user and result == "0-1"))
    return "win" if user_won else "loss"


def prioritize_games(games, user, losses_first=False):
    """Keep archive order, optionally grouping losses before wins and draws."""
    if not losses_first:
        return games
    priority = {"loss": 0, "win": 1, "draw": 2, "other": 3}
    return sorted(games, key=lambda game: priority[user_outcome(game, user)])


def candidates(game, user):
    """Yield positions immediately before moves made by ``user``."""
    headers = game.headers
    white = headers.get("White", "").lower()
    black = headers.get("Black", "").lower()
    user = user.lower()
    if user not in (white, black):
        return
    my_color = chess.WHITE if white == user else chess.BLACK
    opponent = headers.get("Black" if my_color else "White", "?")
    board = game.board()
    for ply, played in enumerate(game.mainline_moves(), start=1):
        if board.turn == my_color and board.legal_moves.count() > 1:
            yield Candidate(
                fen=board.fen(),
                played=played,
                game_id=game_id(game),
                ply=ply,
                user=user,
                opponent=opponent,
                result=headers.get("Result", "?"),
                time_control=headers.get("TimeControl", "?"),
            )
        board.push(played)


def evaluation(score, color):
    """Preserve a score's cp/mate kind after converting it to ``color`` POV."""
    pov = score.pov(color)
    if pov.is_mate():
        return Evaluation(mate=pov.mate())
    cp = pov.score()
    if cp is None:
        raise ValueError("engine score is neither centipawn nor mate")
    return Evaluation(cp=cp)


def score_rank(score, color):
    """Sortable mover-POV value used only for MultiPV move-set labelling."""
    return score.pov(color).score(mate_score=MATE_RANK)


def winning_chances(cp):
    """Lichess WinPercent.winningChances, in the source's [-1, +1] range."""
    scaled = 0.00368208 * cp
    if scaled >= 0:
        return 2 / (1 + math.exp(-scaled)) - 1
    return 1 - 2 / (1 + math.exp(scaled))


def lichess_judgement(before, after):
    """Mirror Lichess CpAdvice and MateAdvice for mover-POV evaluations."""
    if before.cp is not None and after.cp is not None:
        delta = winning_chances(before.cp) - winning_chances(after.cp)
        for threshold, judgement in (
                (LICHESS_BLUNDER_THRESHOLD, "Blunder"),
                (0.2, "Mistake"), (0.1, "Inaccuracy")):
            if delta >= threshold:
                return judgement
        return None

    # MateCreated: a centipawn evaluation becomes forced mate against mover.
    if (before.cp is not None and after.mate is not None
            and after.mate < 0):
        if before.cp < -999:
            return "Inaccuracy"
        if before.cp < -700:
            return "Mistake"
        return "Blunder"

    # MateLost: mover's forced mate becomes cp or forced mate for opponent.
    if before.mate is not None and before.mate > 0:
        if after.cp is not None:
            if after.cp > 999:
                return "Inaccuracy"
            if after.cp > 700:
                return "Mistake"
            return "Blunder"
        if after.mate is not None and after.mate < 0:
            return "Blunder"

    # Same-side mate-distance changes, including delays, receive no advice.
    return None


def acceptable_moves(ranked, legal_count, margin, boundary_guard=0):
    """Return a safely separated near-best set, or empty when ambiguous."""
    if not ranked:
        return ()
    ranked = sorted(ranked, key=lambda item: item[1], reverse=True)
    if len({move for move, _ in ranked}) != len(ranked):
        return ()
    floor = ranked[0][1] - margin
    accepted = tuple(move for move, score in ranked if score >= floor)
    if len(ranked) < legal_count and ranked[-1][1] >= floor:
        return ()
    rejected = [score for _, score in ranked if score < floor]
    if rejected and rejected[0] > floor - boundary_guard:
        return ()
    return accepted


def fen_key(fen):
    """Rule-relevant static state, including the fifty-move clock."""
    return " ".join(fen.split()[:5])


def deduplicate(blunders):
    """Keep the first confirmed example for each EPD rule state."""
    selected = {}
    for blunder in blunders:
        key = fen_key(blunder.candidate.fen)
        if key not in selected:
            selected[key] = blunder
    return sorted(selected.values(), key=lambda item: (
        item.candidate.game_id, item.candidate.ply, fen_key(item.candidate.fen)))


def lichess_url(candidate):
    side = "white" if chess.Board(candidate.fen).turn else "black"
    return f"https://lichess.org/{candidate.game_id}/{side}#{candidate.ply}"


def to_epd(blunder):
    """Serialize one labelled position with enough provenance to reproduce it."""
    candidate = blunder.candidate
    board = chess.Board(candidate.fen)
    played = board.san(candidate.played)
    identifier = f"LBG.{candidate.game_id}.{candidate.ply}"
    if blunder.chance_loss is not None:
        advice = "CpAdvice"
        comparison = (f"win-chance loss {blunder.chance_loss:.6f}; "
                      f"best {blunder.best_eval}; played {blunder.played_eval}; "
                      f"cp loss {blunder.cp_loss}")
    else:
        advice = "MateAdvice"
        comparison = (f"mate transition; best {blunder.best_eval}; "
                      f"played {blunder.played_eval}")
    details = (f"{candidate.user} vs {candidate.opponent}; {candidate.result}; "
               f"tc {candidate.time_control}; played {played}; "
               f"Lichess {advice} Blunder; {comparison}")
    settings = (f"{blunder.oracle}; scan {blunder.scan_nodes} nodes; "
                f"confirm {blunder.confirm_nodes} nodes; "
                f"stability {blunder.stability_nodes} nodes; "
                f"advice {LICHESS_ADVICE_COMMIT}; "
                f"eval {LICHESS_EVAL_COMMIT}; "
                f"blunder delta {LICHESS_BLUNDER_THRESHOLD}; "
                f"bm margin {blunder.best_margin}; "
                f"boundary guard {blunder.boundary_guard}; multipv {blunder.multipv}; "
                f"threads 1; hash 256 MB; pgn sha256 {blunder.source_sha}")
    settings += (f"; source games {blunder.source_game_count}; "
                 f"game order {blunder.game_order}")
    return board.epd(
        bm=blunder.best_moves,
        id=identifier,
        c0=lichess_url(candidate),
        c1=details,
        c2=settings,
        hmvc=board.halfmove_clock,
        fmvn=board.fullmove_number,
    )


def _require_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"invalid checkpoint {label}")


def _checkpoint_evaluation(value):
    _require_keys(value, ("cp", "mate"), "evaluation")
    for score in value.values():
        if score is not None and type(score) is not int:
            raise ValueError("invalid checkpoint evaluation score")
    try:
        return Evaluation(cp=value["cp"], mate=value["mate"])
    except ValueError as error:
        raise ValueError(f"invalid checkpoint evaluation: {error}") from error


def blunder_to_checkpoint(blunder):
    candidate = blunder.candidate
    return {
        "candidate": {
            "fen": candidate.fen,
            "played": candidate.played.uci(),
            "game_id": candidate.game_id,
            "ply": candidate.ply,
            "user": candidate.user,
            "opponent": candidate.opponent,
            "result": candidate.result,
            "time_control": candidate.time_control,
        },
        "best_moves": [move.uci() for move in blunder.best_moves],
        "best_eval": dataclasses.asdict(blunder.best_eval),
        "played_eval": dataclasses.asdict(blunder.played_eval),
        "oracle": blunder.oracle,
        "scan_nodes": blunder.scan_nodes,
        "confirm_nodes": blunder.confirm_nodes,
        "stability_nodes": blunder.stability_nodes,
        "best_margin": blunder.best_margin,
        "boundary_guard": blunder.boundary_guard,
        "multipv": blunder.multipv,
        "source_sha": blunder.source_sha,
        "game_order": blunder.game_order,
        "source_game_count": blunder.source_game_count,
    }


def blunder_from_checkpoint(value, expected_game_id):
    fields = (
        "candidate", "best_moves", "best_eval", "played_eval", "oracle",
        "scan_nodes", "confirm_nodes", "stability_nodes", "best_margin",
        "boundary_guard", "multipv", "source_sha", "game_order",
        "source_game_count",
    )
    _require_keys(value, fields, "blunder")
    candidate_value = value["candidate"]
    candidate_fields = (
        "fen", "played", "game_id", "ply", "user", "opponent", "result",
        "time_control",
    )
    _require_keys(candidate_value, candidate_fields, "candidate")
    if candidate_value["game_id"] != expected_game_id:
        raise ValueError("checkpoint blunder belongs to the wrong game")
    string_fields = (
        "fen", "played", "game_id", "user", "opponent", "result",
        "time_control",
    )
    if any(not isinstance(candidate_value[field], str) for field in string_fields):
        raise ValueError("invalid checkpoint candidate strings")
    if type(candidate_value["ply"]) is not int or candidate_value["ply"] <= 0:
        raise ValueError("invalid checkpoint candidate ply")
    try:
        board = chess.Board(candidate_value["fen"])
        played = chess.Move.from_uci(candidate_value["played"])
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid checkpoint candidate: {error}") from error
    if played not in board.legal_moves:
        raise ValueError("checkpoint played move is not legal")
    moves_value = value["best_moves"]
    if (not isinstance(moves_value, list) or not moves_value
            or any(not isinstance(move, str) for move in moves_value)):
        raise ValueError("invalid checkpoint best moves")
    try:
        best_moves = tuple(chess.Move.from_uci(move) for move in moves_value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid checkpoint best move: {error}") from error
    if (len(set(best_moves)) != len(best_moves)
            or any(move not in board.legal_moves for move in best_moves)
            or played in best_moves):
        raise ValueError("invalid checkpoint best-move set")
    integer_fields = (
        "scan_nodes", "confirm_nodes", "stability_nodes", "best_margin",
        "boundary_guard", "multipv", "source_game_count",
    )
    if any(type(value[field]) is not int for field in integer_fields):
        raise ValueError("invalid checkpoint numeric setting")
    if (value["scan_nodes"] <= 0 or value["confirm_nodes"] <= 0
            or value["stability_nodes"] <= 0 or value["multipv"] < 2
            or value["boundary_guard"] < 0 or value["source_game_count"] <= 0):
        raise ValueError("invalid checkpoint numeric setting")
    for field in ("oracle", "source_sha", "game_order"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError("invalid checkpoint text setting")
    candidate = Candidate(
        fen=candidate_value["fen"],
        played=played,
        game_id=candidate_value["game_id"],
        ply=candidate_value["ply"],
        user=candidate_value["user"],
        opponent=candidate_value["opponent"],
        result=candidate_value["result"],
        time_control=candidate_value["time_control"],
    )
    return Blunder(
        candidate=candidate,
        best_moves=best_moves,
        best_eval=_checkpoint_evaluation(value["best_eval"]),
        played_eval=_checkpoint_evaluation(value["played_eval"]),
        oracle=value["oracle"],
        scan_nodes=value["scan_nodes"],
        confirm_nodes=value["confirm_nodes"],
        stability_nodes=value["stability_nodes"],
        best_margin=value["best_margin"],
        boundary_guard=value["boundary_guard"],
        multipv=value["multipv"],
        source_sha=value["source_sha"],
        game_order=value["game_order"],
        source_game_count=value["source_game_count"],
    )


def funnel_to_checkpoint(funnel):
    return dataclasses.asdict(funnel)


def funnel_from_checkpoint(value):
    fields = ("examined", "scan_blunders", "confirmed_blunders", "stable_labels")
    _require_keys(value, fields, "funnel")
    if any(type(value[field]) is not int or value[field] < 0 for field in fields):
        raise ValueError("invalid checkpoint funnel count")
    counts = [value[field] for field in fields]
    if counts != sorted(counts, reverse=True):
        raise ValueError("invalid checkpoint funnel ordering")
    return Funnel(**value)


def checkpoint_record(game_identifier, blunders, funnel):
    payload = {
        "game_id": game_identifier,
        "blunders": [blunder_to_checkpoint(blunder) for blunder in blunders],
        "funnel": funnel_to_checkpoint(funnel),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"payload": payload, "sha256": digest}


def checkpoint_record_values(value, expected_game_id):
    _require_keys(value, ("payload", "sha256"), "game record")
    payload = value["payload"]
    if not isinstance(value["sha256"], str):
        raise ValueError("invalid checkpoint game checksum")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if value["sha256"] != actual:
        raise ValueError("checkpoint game checksum mismatch")
    _require_keys(payload, ("game_id", "blunders", "funnel"), "game payload")
    if payload["game_id"] != expected_game_id:
        raise ValueError("checkpoint record belongs to the wrong game")
    if not isinstance(payload["blunders"], list):
        raise ValueError("invalid checkpoint blunder list")
    blunders = [
        blunder_from_checkpoint(blunder, expected_game_id)
        for blunder in payload["blunders"]
    ]
    funnel = funnel_from_checkpoint(payload["funnel"])
    if len(blunders) != funnel.stable_labels:
        raise ValueError("checkpoint stable-label count does not match records")
    return blunders, funnel


def checkpoint_identity(games, user, args, source_sha256, engine_sha256, oracle):
    ordered = prioritize_games(games, user, args.losses_first)
    game_ids = [game_id(game) for game in ordered]
    if any(identifier == "unknown" for identifier in game_ids):
        raise ValueError("checkpointing requires a Lichess game id for every game")
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("checkpointing requires unique Lichess game ids")
    return {
        "source_sha256": source_sha256,
        "source_games": len(games),
        "game_ids": game_ids,
        "game_order": "losses-first" if args.losses_first else "archive",
        "user": user.lower(),
        "engine_sha256": engine_sha256,
        "oracle": oracle,
        "scan_nodes": args.scan_nodes,
        "confirm_nodes": args.confirm_nodes,
        "stability_nodes": args.confirm_nodes // 2,
        "best_margin": args.best_margin,
        "boundary_guard": args.boundary_guard,
        "multipv": args.multipv,
        "threads": 1,
        "hash_mb": 256,
        "advice_commit": LICHESS_ADVICE_COMMIT,
        "eval_commit": LICHESS_EVAL_COMMIT,
        "blunder_delta": LICHESS_BLUNDER_THRESHOLD,
    }


def _validate_checkpoint(document, identity):
    _require_keys(document, ("schema", "identity", "games"), "document")
    if document["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if document["identity"] != identity:
        stored = document["identity"] if isinstance(document["identity"], dict) else {}
        changed = sorted(
            key for key in set(stored) | set(identity)
            if stored.get(key) != identity.get(key)
        )
        details = ", ".join(changed) if changed else "identity"
        raise ValueError(f"checkpoint input/settings mismatch: {details}")
    records = document["games"]
    if not isinstance(records, dict):
        raise ValueError("invalid checkpoint game records")
    expected = set(identity["game_ids"])
    if unknown := set(records) - expected:
        raise ValueError(f"checkpoint contains unknown games: {sorted(unknown)}")
    for identifier, record in records.items():
        blunders, _ = checkpoint_record_values(record, identifier)
        for blunder in blunders:
            settings = {
                "oracle": identity["oracle"],
                "scan_nodes": identity["scan_nodes"],
                "confirm_nodes": identity["confirm_nodes"],
                "stability_nodes": identity["stability_nodes"],
                "best_margin": identity["best_margin"],
                "boundary_guard": identity["boundary_guard"],
                "multipv": identity["multipv"],
                "source_sha": identity["source_sha256"][:16],
                "game_order": identity["game_order"],
                "source_game_count": identity["source_games"],
            }
            if any(getattr(blunder, field) != value
                   for field, value in settings.items()):
                raise ValueError("checkpoint blunder settings do not match identity")


def load_checkpoint(path, identity):
    if not path.exists():
        return {"schema": CHECKPOINT_SCHEMA, "identity": identity, "games": {}}
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint {path}: {error}") from error
    _validate_checkpoint(document, identity)
    return document


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(path, document):
    text = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    atomic_write_text(path, text)


def download_pgn(url):
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/x-chess-pgn", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read().decode()


def date_to_millis(date):
    """End of a UTC date, in the milliseconds expected by Lichess."""
    day = datetime.datetime.strptime(date, "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc)
    return int(day.timestamp() * 1000) + 86_399_999


def fetch_pgn(user, games, include_casual=False, until=None):
    query = {
        "moves": "true",
        "clocks": "true",
        "evals": "true",
        "opening": "true",
        "perfType": "bullet,blitz,rapid,classical,correspondence",
        "sort": "dateDesc",
    }
    if not include_casual:
        query["rated"] = "true"
    if games:
        query["max"] = games
    if until:
        query["until"] = date_to_millis(until)
    url = f"https://lichess.org/api/games/user/{user}?{urllib.parse.urlencode(query)}"
    return download_pgn(url)


def fetch_game(game_id):
    query = urllib.parse.urlencode({
        "evals": "true",
        "clocks": "true",
        "opening": "true",
        "literate": "true",
    })
    return download_pgn(f"https://lichess.org/game/export/{game_id}?{query}")


def load_pgn(user, games, cache, refresh=False, game_id=None,
             include_casual=False, until=None):
    """Read a frozen PGN archive, fetching it once when absent or refreshed."""
    if cache and cache.exists() and not refresh:
        return cache.read_text()
    text = (fetch_game(game_id) if game_id else
            fetch_pgn(user, games, include_casual, until))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return text


def clear_hash(engine):
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})


def analyse(engine, board, limit, **kwargs):
    """Search from clean engine state so labels do not depend on probe order."""
    clear_hash(engine)
    return engine.analyse(board, limit, game=object(), **kwargs)


def analyse_candidate(engine, candidate, args, oracle, source_sha, funnel=None):
    board = chess.Board(candidate.fen)
    scan_limit = chess.engine.Limit(nodes=args.scan_nodes)
    best = analyse(engine, board, scan_limit)
    best_move = best.get("pv", [None])[0]
    if best_move is None or best_move == candidate.played:
        return None
    best_eval = evaluation(best["score"], board.turn)
    played = analyse(engine, board, scan_limit, root_moves=[candidate.played])
    played_eval = evaluation(played["score"], board.turn)
    if lichess_judgement(best_eval, played_eval) != "Blunder":
        return None
    if funnel:
        funnel.scan_blunders += 1

    confirm_limit = chess.engine.Limit(nodes=args.confirm_nodes)
    confirmed_best = analyse(engine, board, confirm_limit)
    confirmed_best_move = confirmed_best.get("pv", [None])[0]
    if confirmed_best_move is None:
        return None
    confirmed_best_eval = evaluation(confirmed_best["score"], board.turn)
    confirmed_played = analyse(
        engine, board, confirm_limit, root_moves=[candidate.played])
    confirmed_played_eval = evaluation(confirmed_played["score"], board.turn)
    if lichess_judgement(confirmed_best_eval, confirmed_played_eval) != "Blunder":
        return None
    if funnel:
        funnel.confirmed_blunders += 1

    legal_count = board.legal_moves.count()
    multipv = min(legal_count, args.multipv)
    stability_nodes = args.confirm_nodes // 2
    stability_limit = chess.engine.Limit(nodes=stability_nodes)
    stable_infos = analyse(engine, board, stability_limit, multipv=multipv)
    stable_ranked = [
        (info["pv"][0], score_rank(info["score"], board.turn))
        for info in stable_infos if info.get("pv")
    ]
    stable_moves = acceptable_moves(
        stable_ranked, legal_count, args.best_margin)
    if not stable_moves:
        return None

    infos = analyse(engine, board, confirm_limit, multipv=multipv)
    ranked = [(info["pv"][0], score_rank(info["score"], board.turn))
              for info in infos if info.get("pv")]
    best_moves = acceptable_moves(
        ranked, legal_count, args.best_margin, args.boundary_guard)
    if (not best_moves
            or set(stable_moves) != set(best_moves)
            or confirmed_best_move not in best_moves
            or candidate.played in best_moves):
        return None
    blunder = Blunder(
        candidate=candidate,
        best_moves=best_moves,
        best_eval=confirmed_best_eval,
        played_eval=confirmed_played_eval,
        oracle=oracle,
        scan_nodes=args.scan_nodes,
        confirm_nodes=args.confirm_nodes,
        stability_nodes=stability_nodes,
        best_margin=args.best_margin,
        boundary_guard=args.boundary_guard,
        multipv=args.multipv,
        source_sha=source_sha,
        game_order=("losses-first"
                    if getattr(args, "losses_first", False) else "archive"),
        source_game_count=getattr(args, "source_game_count", 0),
    )
    if funnel:
        funnel.stable_labels += 1
    return blunder


def configure_engine(engine):
    required = {"Threads", "Hash", "Clear Hash", "MultiPV"}
    if missing := required - engine.options.keys():
        raise ValueError(f"oracle lacks required UCI options: {sorted(missing)}")
    options = {"Threads": 1, "Hash": 256}
    if "UCI_AnalyseMode" in engine.options:
        options["UCI_AnalyseMode"] = True
    if options:
        engine.configure(options)


def add_funnel(total, part):
    total.examined += part.examined
    total.scan_blunders += part.scan_blunders
    total.confirmed_blunders += part.confirmed_blunders
    total.stable_labels += part.stable_labels


def build_corpus(games, engine, user, args, oracle, source_sha256, engine_sha256):
    found, funnel = [], Funnel()
    args.source_game_count = len(games)
    ordered = prioritize_games(games, user, args.losses_first)
    checkpoint = None
    records = {}
    if args.checkpoint:
        identity = checkpoint_identity(
            games, user, args, source_sha256, engine_sha256, oracle)
        checkpoint = load_checkpoint(args.checkpoint, identity)
        records = checkpoint["games"]
        if records:
            print(f"resuming {len(records)}/{len(ordered)} completed games from "
                  f"{args.checkpoint}", file=sys.stderr)
    previous_outcome = None
    for game_number, game in enumerate(ordered, start=1):
        identifier = game_id(game)
        outcome = user_outcome(game, user)
        if args.losses_first and outcome != previous_outcome:
            if previous_outcome is not None:
                print(f"{previous_outcome} tranche complete after "
                      f"{game_number - 1} games", file=sys.stderr)
            print(f"starting {outcome} tranche", file=sys.stderr)
            previous_outcome = outcome
        if identifier in records:
            game_found, game_funnel = checkpoint_record_values(
                records[identifier], identifier)
            expected = sum(1 for _ in candidates(game, user))
            if game_funnel.examined != expected:
                raise ValueError(
                    f"checkpoint move count mismatch for game {identifier}")
        else:
            game_found, game_funnel = [], Funnel()
            for candidate in candidates(game, user):
                game_funnel.examined += 1
                if blunder := analyse_candidate(
                        engine, candidate, args, oracle, source_sha256[:16],
                        game_funnel):
                    game_found.append(blunder)
                    if blunder.chance_loss is not None:
                        metric = (f"{blunder.chance_loss:.3f} win-chance loss, "
                                  f"{blunder.cp_loss} cp")
                    else:
                        metric = (f"mate transition {blunder.best_eval} to "
                                  f"{blunder.played_eval}")
                    print(f"{blunder.candidate.game_id} ply "
                          f"{blunder.candidate.ply}: {metric}", file=sys.stderr)
            if checkpoint is not None:
                records[identifier] = checkpoint_record(
                    identifier, game_found, game_funnel)
                save_checkpoint(args.checkpoint, checkpoint)
        found.extend(game_found)
        add_funnel(funnel, game_funnel)
        if args.losses_first and game_number % 50 == 0:
            print(f"progress {game_number}/{len(ordered)} games: "
                  f"{funnel.examined} moves, {funnel.scan_blunders} scan, "
                  f"{funnel.confirmed_blunders} confirmed, "
                  f"{funnel.stable_labels} stable", file=sys.stderr)
    if args.losses_first and previous_outcome is not None:
        print(f"{previous_outcome} tranche complete after {len(ordered)} games",
              file=sys.stderr)
    return deduplicate(found), funnel


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user", help="Lichess username of our bot")
    parser.add_argument("--games", type=int, default=100,
                        help="accepted games to use; 0 means the full archive")
    parser.add_argument("--game-id",
                        help="fetch one immutable game instead of a user slice")
    parser.add_argument("--pgn-cache", type=pathlib.Path,
                        help="read this PGN, or fetch and create it when absent")
    parser.add_argument("--refresh", action="store_true",
                        help="replace --pgn-cache from Lichess")
    parser.add_argument("--include-casual", action="store_true",
                        help="include casual alongside rated games (license them before commit)")
    parser.add_argument("--until", metavar="YYYY-MM-DD",
                        help="fetch games no later than this UTC date")
    parser.add_argument("--losses-first", action="store_true",
                        help="scan losses before wins and draws within the frozen window")
    parser.add_argument("--output", type=pathlib.Path,
                        help="write EPD here instead of stdout")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        help="atomically journal completed games for safe resume")
    parser.add_argument("--best-margin", type=int, default=30,
                        help="moves this many cp from best are all accepted")
    parser.add_argument("--boundary-guard", type=int, default=10,
                        help="required cp gap beyond the best-move cutoff")
    parser.add_argument("--multipv", type=int, default=5,
                        help="maximum moves ranked by the confirmation search")
    parser.add_argument("--scan-nodes", type=int, default=100_000)
    parser.add_argument("--confirm-nodes", type=int, default=1_000_000)
    parser.add_argument("--engine", default=shutil.which("stockfish"),
                        help="path to the Stockfish UCI executable")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if not args.engine:
        sys.exit("no Stockfish on PATH (brew install stockfish)")
    engine_path = shutil.which(args.engine) or args.engine
    if args.games < 0:
        sys.exit("--games must not be negative")
    if args.scan_nodes <= 0 or args.confirm_nodes < 2 * args.scan_nodes:
        sys.exit("confirmation nodes must be at least twice the positive scan nodes")
    if args.multipv < 2:
        sys.exit("--multipv must be at least 2")
    if args.boundary_guard < 0:
        sys.exit("--boundary-guard must not be negative")
    if args.checkpoint:
        protected = [path for path in (args.pgn_cache, args.output) if path]
        if any(args.checkpoint.resolve() == path.resolve() for path in protected):
            sys.exit("--checkpoint must differ from --pgn-cache and --output")

    text = load_pgn(
        args.user,
        args.games,
        args.pgn_cache,
        args.refresh,
        args.game_id,
        args.include_casual,
        args.until,
    )
    source_sha256 = hashlib.sha256(text.encode()).hexdigest()
    try:
        games = source_games(
            text, args.games, args.include_casual, args.game_id)
    except ValueError as error:
        sys.exit(str(error))
    engine = chess.engine.SimpleEngine.popen_uci([engine_path])
    try:
        configure_engine(engine)
        name = engine.id.get("name", "unknown UCI engine")
        engine_sha256 = file_sha256(engine_path)
        oracle = f"{name} sha256 {engine_sha256[:16]}"
        try:
            corpus, funnel = build_corpus(
                games, engine, args.user, args, oracle, source_sha256,
                engine_sha256)
        except ValueError as error:
            sys.exit(str(error))
    finally:
        engine.quit()

    output = "\n".join(map(to_epd, corpus)) + ("\n" if corpus else "")
    if args.output:
        atomic_write_text(args.output, output)
    else:
        print(output, end="")
    print(f"{len(games)} games, {funnel.examined} bot moves, "
          f"{funnel.scan_blunders} scan blunders, "
          f"{funnel.confirmed_blunders} confirmed blunders, "
          f"{funnel.stable_labels} stable labels, "
          f"{len(corpus)} deduplicated blunders", file=sys.stderr)


if __name__ == "__main__":
    main()
