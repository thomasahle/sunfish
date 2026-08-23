import dataclasses
import importlib.util
import json
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


def source_pgn(game_id, event="rated blitz game", variant="Standard", result="*"):
    return f"""[Event "{event}"]
[Site "https://lichess.org/{game_id}"]
[Date "2026.08.22"]
[Round "-"]
[White "Sunfish-Engine"]
[Black "Opponent"]
[Result "{result}"]
[Variant "{variant}"]
[TimeControl "180+1"]

1. e4 e5 {result}

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
        source_game_count=1,
    )


def checkpoint_args(checkpoint):
    return types.SimpleNamespace(
        checkpoint=checkpoint,
        losses_first=False,
        scan_nodes=100,
        confirm_nodes=1000,
        multipv=5,
        best_margin=30,
        boundary_guard=10,
    )


def checkpoint_blunder(candidate, args, oracle, source_sha):
    fields = candidate.fen.split()
    fields[2] = {"1": "KQkq", "2": "KQk", "3": "KQ"}[candidate.game_id[-1]]
    candidate = dataclasses.replace(candidate, fen=" ".join(fields))
    board = chess.Board(candidate.fen)
    best_move = next(move for move in board.legal_moves if move != candidate.played)
    return blunder_scan.Blunder(
        candidate=candidate,
        best_moves=(best_move,),
        best_eval=blunder_scan.Evaluation(cp=100),
        played_eval=blunder_scan.Evaluation(cp=-400),
        oracle=oracle,
        scan_nodes=args.scan_nodes,
        confirm_nodes=args.confirm_nodes,
        stability_nodes=args.confirm_nodes // 2,
        best_margin=args.best_margin,
        boundary_guard=args.boundary_guard,
        multipv=args.multipv,
        source_sha=source_sha,
        game_order="losses-first" if args.losses_first else "archive",
        source_game_count=args.source_game_count,
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


def test_loss_first_order_is_stable_within_each_outcome():
    text = "".join([
        source_pgn("win00001", result="1-0"),
        source_pgn("draw0001", result="1/2-1/2"),
        source_pgn("loss0001", result="0-1"),
        source_pgn("win00002", result="1-0"),
        source_pgn("loss0002", result="0-1"),
    ])
    games = blunder_scan.parse_games(text)
    assert blunder_scan.prioritize_games(
        games, "sunfish-engine") == games
    ordered = blunder_scan.prioritize_games(
        games, "sunfish-engine", losses_first=True)
    assert [blunder_scan.game_id(game) for game in ordered] == [
        "loss0001", "loss0002", "win00001", "win00002", "draw0001"]


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


def test_deduplicate_keeps_first_source_and_has_stable_order():
    weaker = make_blunder(loss=400, game_id="later")
    stronger = make_blunder(loss=700, game_id="earlier")
    other = make_blunder(loss=500, game_id="middle")
    other_candidate = dataclasses.replace(other.candidate, fen=chess.Board().fen(), ply=1)
    other = dataclasses.replace(other, candidate=other_candidate)
    result = blunder_scan.deduplicate([weaker, other, stronger])
    assert [item.candidate.game_id for item in result] == ["later", "middle"]
    assert result[0].cp_loss == 400


def test_deduplicate_uses_the_four_epd_position_fields():
    first = make_blunder(game_id="first")
    fields = first.candidate.fen.split()
    fields[4] = str(int(fields[4]) + 1)
    second_candidate = dataclasses.replace(
        first.candidate, fen=" ".join(fields), game_id="second")
    second = dataclasses.replace(first, candidate=second_candidate)
    assert len(blunder_scan.deduplicate([first, second])) == 1


def test_epd_round_trip_matches_the_wac_format():
    blunder = dataclasses.replace(make_blunder(), best_moves=(
        chess.Move.from_uci("f1c4"), chess.Move.from_uci("d2d4")))
    epd = blunder_scan.to_epd(blunder)
    board, operations = chess.Board.from_epd(epd)
    assert " ".join(board.fen().split()[:4]) == " ".join(
        blunder.candidate.fen.split()[:4])
    assert operations["bm"] == [
        chess.Move.from_uci("f1c4"), chess.Move.from_uci("d2d4")]
    assert operations["id"] == "LBG.abc12345.3"
    assert set(operations) == {"bm", "id"}
    assert epd.endswith('bm Bc4 d4; id "LBG.abc12345.3";')


def test_committed_corpus_has_compact_legal_best_move_labels():
    seen, states, multiple = set(), set(), False
    corpus = ROOT / "tests/files/lichess_blunders.epd"
    lines = corpus.read_text().splitlines()
    assert len(lines) == 1736
    for line in lines:
        assert len(line.split(" bm ", 1)[0].split()) == 4
        board, operations = chess.Board.from_epd(line)
        identifier = operations["id"]
        assert identifier.startswith("LBG.")
        assert identifier not in seen
        seen.add(identifier)
        assert set(operations) == {"bm", "id"}
        state = " ".join(board.fen().split()[:4])
        assert state not in states
        states.add(state)
        assert operations["bm"]
        assert all(move in board.legal_moves for move in operations["bm"])
        multiple |= len(operations["bm"]) > 1
    assert len(seen) == len(states) == 1736
    assert multiple


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


def test_pgn_cache_refresh_is_atomic(tmp_path, monkeypatch):
    cache = tmp_path / "games.pgn"
    cache.write_text("old")
    writes = []
    monkeypatch.setattr(blunder_scan, "fetch_pgn", lambda *_args: PGN)
    monkeypatch.setattr(
        blunder_scan, "atomic_write_text",
        lambda path, text: writes.append((path, text)))
    assert blunder_scan.load_pgn(
        "sunfish-engine", 10, cache, refresh=True) == PGN
    assert writes == [(cache, PGN)]


def test_checkpoint_resume_is_byte_identical_to_uninterrupted(tmp_path, monkeypatch):
    games = blunder_scan.parse_games("".join([
        source_pgn("resume01"),
        source_pgn("resume02"),
        source_pgn("resume03"),
    ]))
    checkpoint = tmp_path / "scan.json"
    source_sha = "a" * 64
    engine_sha = "b" * 64
    oracle = f"Stockfish fixture sha256 {engine_sha[:16]}"
    args = checkpoint_args(checkpoint)
    interrupted_calls = []

    def interrupted(_engine, candidate, settings, name, digest, funnel):
        interrupted_calls.append(candidate.game_id)
        if candidate.game_id == "resume02":
            raise KeyboardInterrupt
        funnel.scan_blunders += 1
        funnel.confirmed_blunders += 1
        funnel.stable_labels += 1
        return checkpoint_blunder(candidate, settings, name, digest)

    monkeypatch.setattr(blunder_scan, "analyse_candidate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        blunder_scan.build_corpus(
            games, object(), "sunfish-engine", args, oracle, source_sha,
            engine_sha)
    saved = json.loads(checkpoint.read_text())
    assert interrupted_calls == ["resume01", "resume02"]
    assert list(saved["games"]) == ["resume01"]

    resumed_calls = []

    def completed(_engine, candidate, settings, name, digest, funnel):
        resumed_calls.append(candidate.game_id)
        funnel.scan_blunders += 1
        funnel.confirmed_blunders += 1
        funnel.stable_labels += 1
        return checkpoint_blunder(candidate, settings, name, digest)

    monkeypatch.setattr(blunder_scan, "analyse_candidate", completed)
    resumed, resumed_funnel = blunder_scan.build_corpus(
        games, object(), "sunfish-engine", args, oracle, source_sha, engine_sha)
    assert resumed_calls == ["resume02", "resume03"]

    uninterrupted_args = checkpoint_args(None)
    uninterrupted, uninterrupted_funnel = blunder_scan.build_corpus(
        games, object(), "sunfish-engine", uninterrupted_args, oracle,
        source_sha, engine_sha)
    assert len(resumed) == len(uninterrupted) == 3
    assert [blunder_scan.to_epd(item) for item in resumed] == [
        blunder_scan.to_epd(item) for item in uninterrupted
    ]
    assert resumed_funnel == uninterrupted_funnel == blunder_scan.Funnel(
        examined=3, scan_blunders=3, confirmed_blunders=3, stable_labels=3)
    assert set(json.loads(checkpoint.read_text())["games"]) == {
        "resume01", "resume02", "resume03"}


def test_checkpoint_rejects_corruption_and_identity_mismatch(tmp_path):
    games = blunder_scan.parse_games(source_pgn("resume01"))
    checkpoint = tmp_path / "scan.json"
    source_sha = "a" * 64
    engine_sha = "b" * 64
    oracle = f"Stockfish fixture sha256 {engine_sha[:16]}"
    args = checkpoint_args(checkpoint)
    identity = blunder_scan.checkpoint_identity(
        games, "sunfish-engine", args, source_sha, engine_sha, oracle)
    assert blunder_scan.GENERATOR_SHA256 == blunder_scan.file_sha256(
        ROOT / "tools/blunder_scan.py")
    assert identity["generator_sha256"] == blunder_scan.GENERATOR_SHA256
    assert identity["python"] == blunder_scan.PYTHON_RUNTIME
    assert identity["python_chess"] == chess.__version__
    document = {
        "schema": blunder_scan.CHECKPOINT_SCHEMA,
        "identity": identity,
        "games": {},
    }
    blunder_scan.save_checkpoint(checkpoint, document)

    changed_args = checkpoint_args(checkpoint)
    changed_args.scan_nodes += 1
    changed = blunder_scan.checkpoint_identity(
        games, "sunfish-engine", changed_args, source_sha, engine_sha, oracle)
    with pytest.raises(ValueError, match="settings mismatch: scan_nodes"):
        blunder_scan.load_checkpoint(checkpoint, changed)

    changed_input = blunder_scan.checkpoint_identity(
        games, "sunfish-engine", args, "c" * 64, engine_sha, oracle)
    with pytest.raises(ValueError, match="settings mismatch: source_sha256"):
        blunder_scan.load_checkpoint(checkpoint, changed_input)

    changed_generator = dict(identity, generator_sha256="c" * 64)
    with pytest.raises(ValueError, match="settings mismatch: generator_sha256"):
        blunder_scan.load_checkpoint(checkpoint, changed_generator)

    corrupt = {
        "schema": blunder_scan.CHECKPOINT_SCHEMA,
        "identity": identity,
        "games": {
            "resume01": blunder_scan.checkpoint_record(
                "resume01", [], blunder_scan.Funnel(examined=1)),
        },
    }
    corrupt["games"]["resume01"]["payload"]["funnel"]["examined"] = 2
    blunder_scan.save_checkpoint(checkpoint, corrupt)
    with pytest.raises(ValueError, match="checksum mismatch"):
        blunder_scan.load_checkpoint(checkpoint, identity)

    checkpoint.write_text("{not json")
    with pytest.raises(ValueError, match="cannot read checkpoint"):
        blunder_scan.load_checkpoint(checkpoint, identity)


def test_checkpoint_replacement_failure_preserves_previous_file(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "scan.json"
    blunder_scan.save_checkpoint(checkpoint, {"generation": 1})
    original = checkpoint.read_bytes()
    replaced = []

    def fail_replace(source, target):
        source, target = Path(source), Path(target)
        replaced.append((source, target))
        assert source.parent == target.parent == tmp_path
        assert source.exists()
        raise OSError("simulated preemption")

    monkeypatch.setattr(blunder_scan.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated preemption"):
        blunder_scan.save_checkpoint(checkpoint, {"generation": 2})
    assert checkpoint.read_bytes() == original
    assert replaced and replaced[0][1] == checkpoint
    assert list(tmp_path.iterdir()) == [checkpoint]


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
