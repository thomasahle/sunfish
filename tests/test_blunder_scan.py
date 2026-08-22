import dataclasses
import importlib.util
from pathlib import Path

import chess
import chess.engine


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("blunder_scan", ROOT / "tools/blunder_scan.py")
blunder_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(blunder_scan)


PGN = """[Event "Regression fixture"]
[Site "https://lichess.org/abc12345"]
[Date "2026.08.22"]
[Round "-"]
[White "Sunfish-Engine"]
[Black "Opponent"]
[Result "*"]
[TimeControl "180+1"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""


def make_blunder(loss=500, game_id="abc12345"):
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    candidate = blunder_scan.Candidate(
        fen=board.fen(),
        played=chess.Move.from_uci("g1f3"),
        game_id=game_id,
        ply=3,
        user="sunfish-engine",
        opponent="Opponent",
        result="0-1",
        time_control="180+1",
    )
    return blunder_scan.Blunder(
        candidate=candidate,
        best_moves=(chess.Move.from_uci("f1c4"),),
        best_cp=100,
        played_cp=100 - loss,
        oracle="Stockfish fixture",
        scan_nodes=100,
        confirm_nodes=1000,
        threshold=300,
        best_margin=30,
        multipv=5,
        source_sha="0123456789abcdef",
    )


def test_parse_games_and_extract_bot_positions():
    games = blunder_scan.parse_games(PGN)
    assert len(games) == 1
    positions = list(blunder_scan.candidates(games[0], "sunfish-engine"))
    assert [position.ply for position in positions] == [1, 3, 5]
    assert positions[1].played == chess.Move.from_uci("g1f3")
    assert positions[1].game_id == "abc12345"
    assert positions[1].opponent == "Opponent"


def test_score_cp_preserves_point_of_view_and_separates_mates():
    cp = chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE)
    mate = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    mated = chess.engine.PovScore(chess.engine.Mate(0), chess.WHITE)
    assert blunder_scan.score_cp(cp, chess.WHITE) == 42
    assert blunder_scan.score_cp(cp, chess.BLACK) == -42
    assert blunder_scan.score_cp(mate, chess.WHITE) == blunder_scan.MATE_CP - 3
    assert blunder_scan.score_cp(mate, chess.BLACK) == 3 - blunder_scan.MATE_CP
    assert blunder_scan.score_cp(mated, chess.WHITE) == -blunder_scan.MATE_CP


def test_acceptable_moves_rejects_a_truncated_near_best_set():
    moves = [chess.Move.from_uci(move) for move in ("e2e4", "d2d4", "g1f3")]
    ranked = [(moves[0], 100), (moves[1], 85)]
    assert blunder_scan.acceptable_moves(ranked, legal_count=3, margin=30) == ()
    assert blunder_scan.acceptable_moves(ranked, legal_count=3, margin=10) == (moves[0],)
    assert blunder_scan.acceptable_moves(ranked, legal_count=2, margin=30) == tuple(moves[:2])


def test_deduplicate_keeps_strongest_loss_and_has_stable_order():
    weaker = make_blunder(loss=400, game_id="later")
    stronger = make_blunder(loss=700, game_id="earlier")
    other = make_blunder(loss=500, game_id="middle")
    other_candidate = dataclasses.replace(other.candidate, fen=chess.Board().fen(), ply=1)
    other = dataclasses.replace(other, candidate=other_candidate)
    result = blunder_scan.deduplicate([weaker, other, stronger])
    assert [item.candidate.game_id for item in result] == ["earlier", "middle"]
    assert result[0].loss == 700


def test_epd_round_trip_contains_labels_and_provenance():
    epd = blunder_scan.to_epd(make_blunder())
    board, operations = chess.Board.from_epd(epd)
    assert board.fen().split()[:4] == make_blunder().candidate.fen.split()[:4]
    assert operations["bm"] == [chess.Move.from_uci("f1c4")]
    assert operations["id"] == "LBG.abc12345.3"
    assert operations["c0"] == "https://lichess.org/abc12345/white#3"
    assert "Opponent; 0-1; tc 180+1; played Nf3" in operations["c1"]
    assert "confirm 1000" in operations["c2"]
    assert "multipv 5; threads 1; hash 256 MB" in operations["c2"]
    assert "pgn sha256 0123456789abcdef" in operations["c2"]


def test_pgn_cache_is_read_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "games.pgn"
    cache.write_text(PGN)

    def unexpected_fetch(*_args):
        raise AssertionError("cache hit attempted a network request")

    monkeypatch.setattr(blunder_scan, "fetch_pgn", unexpected_fetch)
    assert blunder_scan.load_pgn("sunfish-engine", 10, cache) == PGN


def test_date_cutoff_is_the_end_of_the_requested_utc_day():
    assert blunder_scan.date_to_millis("1970-01-01") == 86_399_999
    assert blunder_scan.date_to_millis("2023-02-28") == 1_677_628_799_999
