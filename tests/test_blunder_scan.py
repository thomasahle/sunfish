import dataclasses
import importlib.util
from pathlib import Path
import types
import urllib.parse

import chess
import chess.engine
import pytest


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


def source_pgn(game_id, event="rated blitz game", variant="Standard"):
    return f"""[Event "{event}"]
[Site "https://lichess.org/{game_id}"]
[Date "2026.08.22"]
[Round "-"]
[White "Sunfish-Engine"]
[Black "Opponent"]
[Result "*"]
[Variant "{variant}"]
[TimeControl "180+1"]

1. e4 e5 *

"""


def make_blunder(loss=500, game_id="abc12345"):
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    board.halfmove_clock = 18
    board.fullmove_number = 37
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
        best_eval=blunder_scan.Evaluation(cp=100),
        played_eval=blunder_scan.Evaluation(cp=100 - loss),
        oracle="Stockfish fixture",
        scan_nodes=100,
        confirm_nodes=1000,
        stability_nodes=500,
        best_margin=30,
        boundary_guard=10,
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


def test_source_games_filters_before_applying_the_requested_cap():
    text = "".join([
        source_pgn("variant1", variant="Chess960"),
        source_pgn("casual1", event="casual blitz game"),
        source_pgn("accepted1"),
        source_pgn("ultra001", event="rated ultraBullet game"),
        source_pgn("accepted2", event="rated rapid game"),
        source_pgn("surplus1", event="rated classical game"),
    ])
    rated = blunder_scan.source_games(text, games=2)
    assert [blunder_scan.game_id(game) for game in rated] == ["accepted1", "accepted2"]
    with_casual = blunder_scan.source_games(text, games=2, include_casual=True)
    assert [blunder_scan.game_id(game) for game in with_casual] == [
        "casual1", "accepted1"]


def test_single_game_source_rejects_nonstandard_chess():
    text = source_pgn("variant1", variant="Chess960")
    with pytest.raises(ValueError, match="only supports Standard chess"):
        blunder_scan.source_games(text, games=100, requested_game_id="variant1")
    selected = blunder_scan.source_games(
        source_pgn("standard"), games=100, requested_game_id="standard")
    assert [blunder_scan.game_id(game) for game in selected] == ["standard"]


def test_evaluation_preserves_point_of_view_and_separates_mates():
    cp = chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE)
    mate = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    mated = chess.engine.PovScore(chess.engine.Mate(0), chess.WHITE)
    assert blunder_scan.evaluation(cp, chess.WHITE) == blunder_scan.Evaluation(cp=42)
    assert blunder_scan.evaluation(cp, chess.BLACK) == blunder_scan.Evaluation(cp=-42)
    assert blunder_scan.evaluation(mate, chess.WHITE) == blunder_scan.Evaluation(mate=3)
    assert blunder_scan.evaluation(mate, chess.BLACK) == blunder_scan.Evaluation(mate=-3)
    assert blunder_scan.evaluation(mated, chess.WHITE) == blunder_scan.Evaluation(mate=0)


def test_equal_cp_losses_have_position_dependent_lichess_judgments():
    evaluation = blunder_scan.Evaluation
    assert blunder_scan.winning_chances(0) == 0
    assert blunder_scan.winning_chances(200) == pytest.approx(
        -blunder_scan.winning_chances(-200))

    # The same 300 cp loss is decisive near equality and minor in a won ending.
    assert blunder_scan.lichess_judgement(
        evaluation(cp=200), evaluation(cp=-100)) == "Blunder"
    assert blunder_scan.lichess_judgement(
        evaluation(cp=1000), evaluation(cp=700)) is None

    # There is no absolute "already losing" exclusion in Lichess CpAdvice.
    assert blunder_scan.lichess_judgement(
        evaluation(cp=-300), evaluation(cp=-700)) == "Blunder"


@pytest.mark.parametrize(("before", "after", "judgement"), [
    (("cp", -1000), ("mate", -3), "Inaccuracy"),
    (("cp", -701), ("mate", -3), "Mistake"),
    (("cp", -700), ("mate", -3), "Blunder"),
    (("mate", 3), ("cp", 1000), "Inaccuracy"),
    (("mate", 3), ("cp", 701), "Mistake"),
    (("mate", 3), ("cp", 700), "Blunder"),
    (("mate", 3), ("mate", -4), "Blunder"),
    (("mate", 3), ("mate", 8), None),
])
def test_lichess_mate_advice_boundaries(before, after, judgement):
    def evaluation(spec):
        kind, value = spec
        return blunder_scan.Evaluation(**{kind: value})

    assert blunder_scan.lichess_judgement(
        evaluation(before), evaluation(after)) == judgement


def test_analyse_resets_the_hash_and_uci_game_for_every_probe():
    class FakeEngine:
        options = {"Clear Hash": object()}

        def __init__(self):
            self.configured = []
            self.games = []

        def configure(self, options):
            self.configured.append(options)

        def analyse(self, _board, _limit, **kwargs):
            self.games.append(kwargs["game"])
            return {}

    engine = FakeEngine()
    limit = chess.engine.Limit(nodes=100)
    blunder_scan.analyse(engine, chess.Board(), limit)
    blunder_scan.analyse(engine, chess.Board(), limit)
    assert engine.configured == [{"Clear Hash": None}, {"Clear Hash": None}]
    assert engine.games[0] is not engine.games[1]


def test_acceptable_moves_rejects_a_truncated_near_best_set():
    moves = [chess.Move.from_uci(move) for move in ("e2e4", "d2d4", "g1f3")]
    ranked = [(moves[0], 100), (moves[1], 85)]
    assert blunder_scan.acceptable_moves(ranked, legal_count=3, margin=30) == ()
    assert blunder_scan.acceptable_moves(ranked, legal_count=3, margin=10) == (moves[0],)
    assert blunder_scan.acceptable_moves(ranked, legal_count=2, margin=30) == tuple(moves[:2])


def test_acceptable_moves_requires_a_guard_beyond_the_cutoff():
    moves = [chess.Move.from_uci(move) for move in ("e2e4", "d2d4", "g1f3")]
    ambiguous = [(moves[0], 100), (moves[1], 61), (moves[2], 20)]
    separated = [(moves[0], 100), (moves[1], 60), (moves[2], 20)]
    assert blunder_scan.acceptable_moves(
        ambiguous, legal_count=4, margin=30, boundary_guard=10) == ()
    assert blunder_scan.acceptable_moves(
        separated, legal_count=4, margin=30, boundary_guard=10) == (moves[0],)


def test_deduplicate_keeps_strongest_loss_and_has_stable_order():
    weaker = make_blunder(loss=400, game_id="later")
    stronger = make_blunder(loss=700, game_id="earlier")
    other = make_blunder(loss=500, game_id="middle")
    other_candidate = dataclasses.replace(other.candidate, fen=chess.Board().fen(), ply=1)
    other = dataclasses.replace(other, candidate=other_candidate)
    result = blunder_scan.deduplicate([weaker, other, stronger])
    assert [item.candidate.game_id for item in result] == ["earlier", "middle"]
    assert result[0].cp_loss == 700


def test_deduplicate_preserves_distinct_fifty_move_states():
    first = make_blunder(game_id="first")
    fields = first.candidate.fen.split()
    fields[4] = str(int(fields[4]) + 1)
    second_candidate = dataclasses.replace(
        first.candidate, fen=" ".join(fields), game_id="second")
    second = dataclasses.replace(first, candidate=second_candidate)
    assert len(blunder_scan.deduplicate([first, second])) == 2


def test_epd_round_trip_contains_labels_and_provenance():
    epd = blunder_scan.to_epd(make_blunder())
    board, operations = chess.Board.from_epd(epd)
    assert board.fen() == make_blunder().candidate.fen
    assert operations["bm"] == [chess.Move.from_uci("f1c4")]
    assert operations["id"] == "LBG.abc12345.3"
    assert operations["c0"] == "https://lichess.org/abc12345/white#3"
    assert "Opponent; 0-1; tc 180+1; played Nf3" in operations["c1"]
    assert "Lichess Blunder; win-chance loss" in operations["c1"]
    assert "best +100 cp; played -400 cp; cp loss 500" in operations["c1"]
    assert "confirm 1000 nodes; stability 500 nodes" in operations["c2"]
    assert f"advice {blunder_scan.LICHESS_ADVICE_COMMIT}" in operations["c2"]
    assert f"eval {blunder_scan.LICHESS_EVAL_COMMIT}" in operations["c2"]
    assert "blunder delta 0.3" in operations["c2"]
    assert "boundary guard 10; multipv 5; threads 1; hash 256 MB" in operations["c2"]
    assert "pgn sha256 0123456789abcdef" in operations["c2"]
    assert operations["hmvc"] == 18
    assert operations["fmvn"] == 37


def test_committed_corpus_preserves_rule_state_and_legal_labels():
    expected_seed_clocks = {
        "LBG.eWjtwAtB.59": (0, 30),
        "LBG.eWjtwAtB.149": (18, 75),
    }
    expected_settings = (
        "Stockfish 16 sha256 1967ae9001b4d18b; scan 100000 nodes; "
        "confirm 1000000 nodes; stability 500000 nodes; threshold 300; "
        "bm margin 30; boundary guard 10; multipv 5; threads 1; "
        "hash 256 MB; pgn sha256 6db73d5491270df5"
    )
    seen, clocks, states = set(), {}, set()
    corpus = ROOT / "tests/files/lichess_blunders.epd"
    lines = corpus.read_text().splitlines()
    assert len(lines) == 40
    for line in lines:
        board, operations = chess.Board.from_epd(line)
        identifier = operations["id"]
        assert identifier.startswith("LBG.")
        assert identifier not in seen
        seen.add(identifier)
        clocks[identifier] = (board.halfmove_clock, board.fullmove_number)
        assert (operations["hmvc"], operations["fmvn"]) == clocks[identifier]
        state = " ".join(board.fen().split()[:5])
        assert state not in states
        states.add(state)
        assert operations["bm"]
        assert all(move in board.legal_moves for move in operations["bm"])
        game_id = identifier.split(".")[1]
        assert operations["c0"].startswith(f"https://lichess.org/{game_id}/")
        assert operations["c2"] == expected_settings
    assert expected_seed_clocks.items() <= clocks.items()
    assert len(seen) == len(states) == 40


def test_loss_confirmation_uses_equal_single_pv_budgets(monkeypatch):
    blunder = make_blunder()
    candidate = blunder.candidate
    board = chess.Board(candidate.fen)
    best = chess.Move.from_uci("f1c4")
    alternatives = [
        best,
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("b1c3"),
        chess.Move.from_uci("f2f4"),
        chess.Move.from_uci("d2d3"),
    ]

    def info(move, cp):
        return {
            "pv": [move],
            "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
        }

    replies = iter([
        info(best, 100),
        info(candidate.played, -400),
        info(best, 90),
        info(candidate.played, -410),
        [info(move, score) for move, score in zip(
            alternatives, (95, 40, 20, 10, 0))],
        [info(move, score) for move, score in zip(
            alternatives, (95, 40, 20, 10, 0))],
    ])
    calls = []

    def fake_analyse(_engine, _board, limit, **kwargs):
        calls.append((limit.nodes, kwargs))
        return next(replies)

    monkeypatch.setattr(blunder_scan, "analyse", fake_analyse)
    args = types.SimpleNamespace(
        scan_nodes=100,
        confirm_nodes=1000,
        multipv=5,
        best_margin=30,
        boundary_guard=10,
    )
    funnel = blunder_scan.Funnel(examined=1)
    result = blunder_scan.analyse_candidate(
        object(), candidate, args, "Stockfish fixture", "source", funnel)
    assert result.best_eval == blunder_scan.Evaluation(cp=90)
    assert result.played_eval == blunder_scan.Evaluation(cp=-410)
    assert result.cp_loss == 500
    assert funnel == blunder_scan.Funnel(
        examined=1, scan_blunders=1, confirmed_blunders=1, stable_labels=1)
    assert [nodes for nodes, _ in calls] == [100, 100, 1000, 1000, 500, 1000]
    assert calls[2][1] == {}
    assert calls[3][1] == {"root_moves": [candidate.played]}
    assert calls[4][1] == {"multipv": 5}
    assert calls[5][1] == {"multipv": 5}


def test_multipv_labels_must_be_stable_between_budgets(monkeypatch):
    blunder = make_blunder()
    candidate = blunder.candidate
    board = chess.Board(candidate.fen)
    best = chess.Move.from_uci("f1c4")
    second = chess.Move.from_uci("d2d4")
    others = [
        chess.Move.from_uci("b1c3"),
        chess.Move.from_uci("f2f4"),
        chess.Move.from_uci("d2d3"),
    ]

    def info(move, cp):
        return {
            "pv": [move],
            "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
        }

    replies = iter([
        info(best, 100),
        info(candidate.played, -400),
        info(best, 90),
        info(candidate.played, -410),
        [info(move, score) for move, score in zip(
            [best, second] + others, (95, 70, 40, 20, 0))],
        [info(move, score) for move, score in zip(
            [best, second] + others, (95, 50, 40, 20, 0))],
    ])
    monkeypatch.setattr(
        blunder_scan, "analyse", lambda *_args, **_kwargs: next(replies))
    args = types.SimpleNamespace(
        scan_nodes=100,
        confirm_nodes=1000,
        multipv=5,
        best_margin=30,
        boundary_guard=10,
    )
    assert blunder_scan.analyse_candidate(
        object(), candidate, args, "Stockfish fixture", "source") is None


def test_pgn_cache_is_read_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "games.pgn"
    cache.write_text(PGN)

    def unexpected_fetch(*_args):
        raise AssertionError("cache hit attempted a network request")

    monkeypatch.setattr(blunder_scan, "fetch_pgn", unexpected_fetch)
    assert blunder_scan.load_pgn("sunfish-engine", 10, cache) == PGN


def test_include_casual_omits_the_rated_only_filter(monkeypatch):
    urls = []
    monkeypatch.setattr(
        blunder_scan, "download_pgn", lambda url: urls.append(url) or PGN)
    blunder_scan.fetch_pgn("sunfish-engine", 10)
    blunder_scan.fetch_pgn("sunfish-engine", 10, include_casual=True)
    rated_query = urllib.parse.parse_qs(urllib.parse.urlparse(urls[0]).query)
    all_query = urllib.parse.parse_qs(urllib.parse.urlparse(urls[1]).query)
    assert rated_query["rated"] == ["true"]
    assert "rated" not in all_query


def test_date_cutoff_is_the_end_of_the_requested_utc_day():
    assert blunder_scan.date_to_millis("1970-01-01") == 86_399_999
    assert blunder_scan.date_to_millis("2023-02-28") == 1_677_628_799_999
