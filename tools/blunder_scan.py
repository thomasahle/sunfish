#!/usr/bin/env python3
"""Turn confirmed mistakes from a Lichess bot archive into an EPD suite.

Typical reproducible workflow::

    tools/blunder_scan.py sunfish-engine --pgn-cache sunfish-games.pgn \
        --output lichess_blunders.epd

The first run downloads one PGN archive and caches it. Later runs read only
that file unless ``--refresh`` is passed. A quick fixed-node Stockfish pass
finds candidates. Equal-budget single-PV probes then confirm the loss, while
MultiPV probes at two larger budgets must agree on every accepted near-best
move. Positions too close to the MultiPV boundary are rejected rather than
incompletely labelled.

The output is ordinary EPD accepted by ``tools/tester.py best``. Its ``bm``
operation contains the certified alternatives, while ``c0`` through ``c2``
record the source game, played move, score loss, and oracle settings. Network
access and Stockfish are generation-time dependencies only; the committed EPD
is a deterministic test input.
"""
import argparse
import dataclasses
import datetime
import hashlib
import io
import pathlib
import shutil
import sys
import urllib.parse
import urllib.request

import chess
import chess.engine
import chess.pgn

MATE_CP = 100_000
USER_AGENT = "sunfish-blunder-corpus/1 (+https://github.com/thomasahle/sunfish)"


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
class Blunder:
    candidate: Candidate
    best_moves: tuple[chess.Move, ...]
    best_cp: int
    played_cp: int
    oracle: str
    scan_nodes: int
    confirm_nodes: int
    stability_nodes: int
    threshold: int
    best_margin: int
    boundary_guard: int
    multipv: int
    source_sha: str

    @property
    def loss(self):
        return self.best_cp - self.played_cp


def parse_games(text):
    """Parse every game in a PGN stream, preserving file order."""
    games, stream = [], io.StringIO(text)
    while (game := chess.pgn.read_game(stream)) is not None:
        if game.errors:
            raise ValueError(f"malformed PGN game: {game.errors}")
        games.append(game)
    return games


def game_id(game):
    """Extract the stable Lichess game id from a Site header."""
    path = urllib.parse.urlparse(game.headers.get("Site", "")).path
    return path.strip("/").split("/", 1)[0] or "unknown"


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


def score_cp(score, color):
    """Centipawns from ``color``; mate values live far outside normal evals."""
    score = score.pov(color)
    return score.score(mate_score=MATE_CP)


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
    """Keep the strongest confirmed example for each EPD position."""
    selected = {}
    for blunder in blunders:
        key = fen_key(blunder.candidate.fen)
        previous = selected.get(key)
        rank = (blunder.loss, blunder.candidate.game_id, blunder.candidate.ply)
        if previous is None:
            selected[key] = blunder
            continue
        old_rank = (previous.loss, previous.candidate.game_id, previous.candidate.ply)
        if rank > old_rank:
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
    details = (f"{candidate.user} vs {candidate.opponent}; {candidate.result}; "
               f"tc {candidate.time_control}; played {played}; loss {blunder.loss} cp; "
               f"best {blunder.best_cp}; played {blunder.played_cp}")
    settings = (f"{blunder.oracle}; scan {blunder.scan_nodes} nodes; "
                f"confirm {blunder.confirm_nodes} nodes; "
                f"stability {blunder.stability_nodes} nodes; "
                f"threshold {blunder.threshold}; bm margin {blunder.best_margin}; "
                f"boundary guard {blunder.boundary_guard}; multipv {blunder.multipv}; "
                f"threads 1; hash 256 MB; pgn sha256 {blunder.source_sha}")
    return board.epd(
        bm=blunder.best_moves,
        id=identifier,
        c0=lichess_url(candidate),
        c1=details,
        c2=settings,
        hmvc=board.halfmove_clock,
        fmvn=board.fullmove_number,
    )


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


def analyse_candidate(engine, candidate, args, oracle, source_sha):
    board = chess.Board(candidate.fen)
    scan_limit = chess.engine.Limit(nodes=args.scan_nodes)
    best = analyse(engine, board, scan_limit)
    best_move = best.get("pv", [None])[0]
    if best_move is None or best_move == candidate.played:
        return None
    best_cp = score_cp(best["score"], board.turn)
    played = analyse(engine, board, scan_limit, root_moves=[candidate.played])
    played_cp = score_cp(played["score"], board.turn)
    if best_cp - played_cp < args.threshold or best_cp <= -args.threshold:
        return None

    confirm_limit = chess.engine.Limit(nodes=args.confirm_nodes)
    confirmed_best = analyse(engine, board, confirm_limit)
    confirmed_best_move = confirmed_best.get("pv", [None])[0]
    if confirmed_best_move is None:
        return None
    confirmed_best_cp = score_cp(confirmed_best["score"], board.turn)
    confirmed_played = analyse(
        engine, board, confirm_limit, root_moves=[candidate.played])
    confirmed_played_cp = score_cp(confirmed_played["score"], board.turn)
    if (confirmed_best_cp - confirmed_played_cp < args.threshold
            or confirmed_best_cp <= -args.threshold):
        return None

    legal_count = board.legal_moves.count()
    multipv = min(legal_count, args.multipv)
    stability_nodes = args.confirm_nodes // 2
    stability_limit = chess.engine.Limit(nodes=stability_nodes)
    stable_infos = analyse(engine, board, stability_limit, multipv=multipv)
    stable_ranked = [
        (info["pv"][0], score_cp(info["score"], board.turn))
        for info in stable_infos if info.get("pv")
    ]
    stable_moves = acceptable_moves(
        stable_ranked, legal_count, args.best_margin)
    if not stable_moves:
        return None

    infos = analyse(engine, board, confirm_limit, multipv=multipv)
    ranked = [(info["pv"][0], score_cp(info["score"], board.turn))
              for info in infos if info.get("pv")]
    best_moves = acceptable_moves(
        ranked, legal_count, args.best_margin, args.boundary_guard)
    if (not best_moves
            or set(stable_moves) != set(best_moves)
            or confirmed_best_move not in best_moves
            or candidate.played in best_moves):
        return None
    return Blunder(
        candidate=candidate,
        best_moves=best_moves,
        best_cp=confirmed_best_cp,
        played_cp=confirmed_played_cp,
        oracle=oracle,
        scan_nodes=args.scan_nodes,
        confirm_nodes=args.confirm_nodes,
        stability_nodes=stability_nodes,
        threshold=args.threshold,
        best_margin=args.best_margin,
        boundary_guard=args.boundary_guard,
        multipv=args.multipv,
        source_sha=source_sha,
    )


def configure_engine(engine):
    required = {"Threads", "Hash", "Clear Hash", "MultiPV"}
    if missing := required - engine.options.keys():
        raise ValueError(f"oracle lacks required UCI options: {sorted(missing)}")
    options = {"Threads": 1, "Hash": 256}
    if "UCI_AnalyseMode" in engine.options:
        options["UCI_AnalyseMode"] = True
    if options:
        engine.configure(options)


def build_corpus(games, engine, user, args, oracle, source_sha):
    found, examined = [], 0
    for game in games:
        for candidate in candidates(game, user):
            examined += 1
            if blunder := analyse_candidate(engine, candidate, args, oracle, source_sha):
                found.append(blunder)
                print(f"{blunder.candidate.game_id} ply {blunder.candidate.ply}: "
                      f"{blunder.loss} cp", file=sys.stderr)
    return deduplicate(found), examined


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
                        help="games to fetch; 0 means the full archive")
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
    parser.add_argument("--output", type=pathlib.Path,
                        help="write EPD here instead of stdout")
    parser.add_argument("--threshold", type=int, default=300,
                        help="minimum confirmed centipawn loss")
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
    if args.scan_nodes <= 0 or args.confirm_nodes < 2 * args.scan_nodes:
        sys.exit("confirmation nodes must be at least twice the positive scan nodes")
    if args.multipv < 2:
        sys.exit("--multipv must be at least 2")
    if args.boundary_guard < 0:
        sys.exit("--boundary-guard must not be negative")

    text = load_pgn(
        args.user,
        args.games,
        args.pgn_cache,
        args.refresh,
        args.game_id,
        args.include_casual,
        args.until,
    )
    source_sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    games = parse_games(text)
    engine = chess.engine.SimpleEngine.popen_uci([engine_path])
    try:
        configure_engine(engine)
        name = engine.id.get("name", "unknown UCI engine")
        oracle = f"{name} sha256 {file_sha256(engine_path)[:16]}"
        corpus, examined = build_corpus(
            games, engine, args.user, args, oracle, source_sha)
    finally:
        engine.quit()

    output = "\n".join(map(to_epd, corpus)) + ("\n" if corpus else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")
    print(f"{len(games)} games, {examined} bot moves, "
          f"{len(corpus)} deduplicated blunders", file=sys.stderr)


if __name__ == "__main__":
    main()
