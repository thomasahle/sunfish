"""tools/book: attribution of game results to book nodes, and reweighting.

The two tools are a measurement instrument, so the properties that matter are
the ones that keep them from inventing signal: no information in must mean no
change out, equal information must mean no change out, and a change out must be
bounded by the amount of information that produced it.  Those three are the
first three rebuild tests.  The attribution tests pin the semantics that were
chosen to match the deployed lichess-bot: the 15-ply horizon, credit to both
sides, re-probing after a book exit, and polyglot's own castling spelling.
"""

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import chess
import chess.polyglot
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "book"))

import attribute  # noqa: E402
import rebuild  # noqa: E402


# ---------------------------------------------------------------- helpers

def raw_move(board, move):
    """Encode a move the way polyglot does: king-takes-rook for castling."""
    if board.is_castling(move):
        rook = board.rook_square = chess.square(7 if board.is_kingside_castling(move) else 0,
                                                chess.square_rank(move.from_square))
        move = chess.Move(move.from_square, rook)
    promo = 0 if move.promotion is None else move.promotion - 1
    return move.to_square | (move.from_square << 6) | (promo << 12)


def make_book(path, lines, weight=1, drop=()):
    """Build a polyglot book containing every (position, move) on `lines` (UCI strings).

    `drop` removes (prefix, uci) pairs afterwards, which is how a line can be reachable
    deeper in the book without its own first move being a book move.
    """
    entries = {}
    for line in lines:
        board = chess.Board()
        for uci in line.split():
            move = chess.Move.from_uci(uci)
            entries.setdefault((chess.polyglot.zobrist_hash(board), raw_move(board, move)), weight)
            board.push(move)
    for prefix, uci in drop:
        board = chess.Board()
        for u in prefix.split(): board.push(chess.Move.from_uci(u))
        entries.pop((chess.polyglot.zobrist_hash(board), raw_move(board, chess.Move.from_uci(uci))))
    rows = [(k, m, w, 0) for (k, m), w in sorted(entries.items())]
    rebuild.write_raw(str(path), rows)
    return rows


def make_pgn(path, games, white="alice", black="bob"):
    """games: list of (uci_line, result).  Written as a minimal but real PGN."""
    out = []
    for line, result in games:
        board = chess.Board()
        sans = []
        for uci in line.split():
            move = chess.Move.from_uci(uci)
            sans.append(board.san(move))
            board.push(move)
        body = " ".join("%d. %s" % (i // 2 + 1, s) if i % 2 == 0 else s for i, s in enumerate(sans))
        out.append('[Event "t"]\n[White "%s"]\n[Black "%s"]\n[Result "%s"]\n\n%s %s\n'
                   % (white, black, result, body, result))
    path.write_text("\n".join(out))
    return path


def attribute_to(tmp_path, book, pgn, **kw):
    out = tmp_path / "stats.json"
    argv = [str(pgn), "--book", str(book), "--out", str(out)]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else sum(([flag, str(x)] for x in (v if isinstance(v, list) else [v])), [])
    assert attribute.main(argv) == 0
    return json.loads(out.read_text())


def node_of(stats, uci_line):
    board = chess.Board()
    for uci in uci_line.split():
        board.push(chess.Move.from_uci(uci))
    key = "%016x" % chess.polyglot.zobrist_hash(board)
    return next((n for n in stats["nodes"] if n["key"] == key), None)


def move_of(node, uci):
    return next((m for m in node["moves"] if m["uci"] == uci), None)


E4E5 = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"
SICILIAN = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4"


# ---------------------------------------------------------------- attribution

def test_credits_both_sides_with_mover_relative_results(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5])
    pgn = make_pgn(tmp_path / "g.pgn", [(E4E5, "1-0"), (E4E5, "0-1"), (E4E5, "1/2-1/2")])
    stats = attribute_to(tmp_path, book, pgn)

    assert stats["meta"]["games"] == stats["meta"]["games_scored"] == 3
    assert stats["meta"]["max_plies"] == 15                      # the deployed horizon
    white = move_of(node_of(stats, ""), "e2e4")                  # White's 1.e4: 1 win, 1 loss, 1 draw
    assert (white["games"], white["w"], white["d"], white["l"]) == (3, 1, 1, 1)
    assert white["score"] == 1.5
    black = move_of(node_of(stats, "e2e4"), "e7e5")              # Black's 1...e5: mirrored
    assert (black["games"], black["w"], black["d"], black["l"]) == (3, 1, 1, 1)
    assert node_of(stats, "")["stm"] == "w" and node_of(stats, "e2e4")["stm"] == "b"
    assert stats["meta"]["credited_plies"] == 3 * 6


def test_unfinished_games_are_not_scored(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5])
    pgn = make_pgn(tmp_path / "g.pgn", [(E4E5, "*"), (E4E5, "1-0")])
    stats = attribute_to(tmp_path, book, pgn)
    assert stats["meta"]["games"] == 2 and stats["meta"]["games_scored"] == 1
    assert move_of(node_of(stats, ""), "e2e4")["games"] == 1


def test_transpositions_land_on_one_node(tmp_path):
    a, b = "d2d4 g8f6 c2c4 e7e6 b1c3", "c2c4 g8f6 d2d4 e7e6 b1c3"
    book = tmp_path / "b.bin"
    make_book(book, [a, b])
    pgn = make_pgn(tmp_path / "g.pgn", [(a, "1-0"), (b, "1-0")])
    stats = attribute_to(tmp_path, book, pgn)
    shared = node_of(stats, "d2d4 g8f6 c2c4 e7e6")
    assert node_of(stats, "c2c4 g8f6 d2d4 e7e6")["key"] == shared["key"]   # same position, same key
    assert move_of(shared, "b1c3")["games"] == 2                           # both orders credit it once


def test_horizon_is_max_depth_full_moves(tmp_path):
    line = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 b1c3 d7d6"   # 10 plies
    book = tmp_path / "b.bin"
    make_book(book, [line])
    assert attribute_to(tmp_path, book, make_pgn(tmp_path / "g.pgn", [(line, "1-0")]),
                        max_depth=8)["meta"]["credited_plies"] == 10       # 15-ply horizon: all of it
    assert attribute_to(tmp_path, book, make_pgn(tmp_path / "g.pgn", [(line, "1-0")]),
                        max_depth=3)["meta"]["credited_plies"] == 5        # 3*2-1 = 5 plies
    assert attribute_to(tmp_path, book, make_pgn(tmp_path / "g.pgn", [(line, "1-0")]),
                        max_plies=4)["meta"]["credited_plies"] == 4        # explicit override wins


def test_book_exit_is_recorded_and_re_entry_is_configurable(tmp_path):
    book = tmp_path / "b.bin"
    # The Caro-Kann line is IN the book from move 2 on, but 1...c6 itself is not a book
    # move -- exactly the shape that separates "re-probe" from "latch".
    played = "e2e4 c7c6 g1f3 d7d6 d2d4"
    make_book(book, [E4E5, SICILIAN, played], drop=[("e2e4", "c7c6")])
    pgn = make_pgn(tmp_path / "g.pgn", [(played, "1-0")])

    loose = attribute_to(tmp_path, book, pgn)
    assert loose["exits"] == {"b1": 1}                                     # one black exit, at ply 1
    assert move_of(node_of(loose, "e2e4 c7c6 g1f3"), "d7d6")["games"] == 1  # re-probed, so credited

    strict = attribute_to(tmp_path, book, pgn, latch_exit=True)
    assert strict["exits"] == {"b1": 1}
    assert node_of(strict, "e2e4 c7c6 g1f3") is None                       # latched out, never credited
    assert move_of(node_of(strict, ""), "e2e4")["games"] == 1              # white is unaffected


def test_player_filter_credits_only_us(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5])
    pgn = make_pgn(tmp_path / "g.pgn", [(E4E5, "1-0")], white="sunfish-engine", black="human")
    stats = attribute_to(tmp_path, book, pgn, player="sunfish-engine")
    assert stats["meta"]["players"] == ["sunfish-engine"]
    assert move_of(node_of(stats, ""), "e2e4")["games"] == 1
    assert node_of(stats, "e2e4") is None                                  # the opponent's reply is not ours


def test_castling_keeps_polyglots_spelling(tmp_path):
    line = "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1"
    book = tmp_path / "b.bin"
    make_book(book, [line])
    stats = attribute_to(tmp_path, book, make_pgn(tmp_path / "g.pgn", [(line, "1-0")]))
    castle = node_of(stats, "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5")["moves"][0]
    assert (castle["uci"], castle["raw"], castle["san"], castle["games"]) == ("e1g1", "e1h1", "O-O", 1)
    # ...and the join key is the one rebuild.py uses, so the credit is not dropped.
    src = rebuild.read_raw(str(book))
    assert castle["raw"] in {rebuild.raw_move_uci(e[1]) for e in src}


def test_non_standard_start_positions_are_skipped(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5])
    pgn = tmp_path / "g.pgn"
    pgn.write_text('[Event "t"]\n[Result "1-0"]\n[FEN "8/8/8/8/8/8/6k1/6K1 w - - 0 1"]\n[SetUp "1"]\n\n1. Kf1 1-0\n')
    stats = attribute_to(tmp_path, book, pgn)
    assert stats["meta"]["games"] == 1 and stats["meta"]["games_scored"] == 0


# ---------------------------------------------------------------- reweighting

def uniform_stats(book, games=10, score_each=0.5):
    """Every book move played `games` times at the same rate -- the null input."""
    nodes = {}
    for key, mv, _w, _l in rebuild.read_raw(str(book)):
        nodes.setdefault(key, []).append(mv)
    return {"meta": {}, "nodes": [
        {"key": "%016x" % k, "moves": [{"uci": rebuild.raw_move_uci(m), "raw": rebuild.raw_move_uci(m),
                                        "games": games, "score": games * score_each} for m in mvs]}
        for k, mvs in sorted(nodes.items())]}


def run_rebuild(tmp_path, book, stats, **kw):
    out = tmp_path / "out.bin"
    sfile = tmp_path / "s.json"
    sfile.write_text(json.dumps(stats))
    argv = ["--book", str(book), "--stats", str(sfile), "--out", str(out)]
    for k, v in kw.items(): argv += ["--" + k.replace("_", "-"), str(v)]
    assert rebuild.main(argv) == 0
    return out


def test_empty_stats_reproduce_the_book_bit_for_bit(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5, SICILIAN], weight=37)
    out = run_rebuild(tmp_path, book, {"meta": {}, "nodes": []})
    assert out.read_bytes() == book.read_bytes()


def test_uniform_stats_reproduce_the_book_bit_for_bit(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5, SICILIAN, "d2d4 d7d5 c2c4 e7e6"], weight=1)
    for rate in (0.0, 0.5, 1.0):                      # a level shift is not a preference
        out = run_rebuild(tmp_path, book, uniform_stats(book, games=500, score_each=rate))
        assert out.read_bytes() == book.read_bytes(), "rate %s moved the book" % rate


def test_one_dominant_line_shifts_by_the_calibrated_factor(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, ["e2e4", "d2d4"])                 # two siblings at the root, nothing else
    n, alpha = 30, 60.0
    stats = {"meta": {}, "nodes": [{"key": "%016x" % chess.polyglot.zobrist_hash(chess.Board()), "moves": [
        {"uci": "d2d4", "raw": "d2d4", "games": n, "score": 0.0},
        {"uci": "e2e4", "raw": "e2e4", "games": n, "score": float(n)}]}]}
    out = run_rebuild(tmp_path, book, stats, alpha=alpha)
    w = {rebuild.raw_move_uci(m): weight for _k, m, weight, _l in rebuild.read_raw(str(out))}
    # alpha = 2*N_min makes N_min games of a 100%/0% split worth exactly one doubling.
    # rel 1e-2 is the ushort quantisation, not slack in the rule: 667/333 is 2.003.
    assert w["e2e4"] / w["d2d4"] == pytest.approx(1 + 2 * n / alpha, rel=1e-2) == pytest.approx(2.0, rel=1e-2)
    assert w["d2d4"] / (w["d2d4"] + w["e2e4"]) > 0.02  # and the loser keeps its exploration share


def test_the_exploration_floor_bounds_the_worst_case(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, ["e2e4", "d2d4", "g1f3"])
    huge = 100000
    stats = {"meta": {}, "nodes": [{"key": "%016x" % chess.polyglot.zobrist_hash(chess.Board()), "moves": [
        {"uci": u, "raw": u, "games": huge, "score": float(huge) if u == "e2e4" else 0.0}
        for u in ("d2d4", "e2e4", "g1f3")]}]}
    out = run_rebuild(tmp_path, book, stats, floor=0.05)
    w = {rebuild.raw_move_uci(m): weight for _k, m, weight, _l in rebuild.read_raw(str(out))}
    total = sum(w.values())
    assert min(w.values()) / total == pytest.approx(0.05, abs=1e-3)          # floor is exact
    assert max(w.values()) / min(w.values()) == pytest.approx(0.9 / 0.05, rel=1e-2)  # (1-(m-1)f)/f


def test_a_floor_that_cannot_be_met_degrades_to_uniform(tmp_path):
    assert rebuild.apply_floor([0.97] + [0.001] * 30, 0.05) == pytest.approx([1 / 31] * 31)


def test_rebuild_is_deterministic_and_only_weights_move(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5, SICILIAN], weight=3)
    stats = uniform_stats(book, games=8)
    stats["nodes"][0]["moves"][0]["score"] = 8.0                             # one real preference
    second = tmp_path / "second"
    second.mkdir()
    assert run_rebuild(tmp_path, book, stats).read_bytes() == run_rebuild(second, book, stats).read_bytes()
    src, out = rebuild.read_raw(str(book)), rebuild.read_raw(str(tmp_path / "out.bin"))
    assert [(k, m, ln) for k, m, _w, ln in src] == [(k, m, ln) for k, m, _w, ln in out]
    assert all(1 <= w <= 65535 for _k, _m, w, _l in out)


def test_min_games_gates_a_node_out_entirely(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, ["e2e4", "d2d4"])
    stats = {"meta": {}, "nodes": [{"key": "%016x" % chess.polyglot.zobrist_hash(chess.Board()), "moves": [
        {"uci": "d2d4", "raw": "d2d4", "games": 4, "score": 0.0},
        {"uci": "e2e4", "raw": "e2e4", "games": 4, "score": 4.0}]}]}
    assert run_rebuild(tmp_path, book, stats, min_games=20).read_bytes() == book.read_bytes()
    assert run_rebuild(tmp_path, book, stats, min_games=8).read_bytes() != book.read_bytes()


def test_attribute_then_rebuild_round_trips(tmp_path):
    book = tmp_path / "b.bin"
    make_book(book, [E4E5, SICILIAN])
    pgn = make_pgn(tmp_path / "g.pgn", [(E4E5, "1-0")] * 20 + [(SICILIAN, "0-1")] * 20)
    sfile = tmp_path / "stats.json"
    assert attribute.main([str(pgn), "--book", str(book), "--out", str(sfile)]) == 0
    out = tmp_path / "v1.bin"
    assert rebuild.main(["--book", str(book), "--stats", str(sfile), "--out", str(out),
                         "--report", str(tmp_path / "r.json")]) == 0
    src, new = rebuild.read_raw(str(book)), rebuild.read_raw(str(out))
    assert [(k, m) for k, m, _w, _l in src] == [(k, m) for k, m, _w, _l in new]
    w = {rebuild.raw_move_uci(m): weight for k, m, weight, _l in new
         if k == chess.polyglot.zobrist_hash(chess.Board())}
    assert w["e2e4"] > w["d2d4"] if "d2d4" in w else True
    with chess.polyglot.open_reader(str(out)) as r:                          # still a readable book
        assert {r_.move.uci() for r_ in r.find_all(chess.Board())} == {"e2e4"}
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["meta"]["nodes_changed"] == len(report["nodes"]) > 0
    assert report["meta"]["stats_nodes_not_in_book"] == 0


# ---------------------------------------------------------------- the real format

@pytest.mark.skipif(shutil.which("polyglot") is None, reason="polyglot adapter not installed")
def test_agrees_with_a_book_built_by_polyglot_itself(tmp_path):
    """The raw 16-byte reader must agree with a book PolyGlot produced, not just with ours."""
    pgn = make_pgn(tmp_path / "src.pgn", [(E4E5, "1-0"), (SICILIAN, "0-1"), (E4E5, "1/2-1/2")])
    book = tmp_path / "made.bin"
    subprocess.run(["polyglot", "make-book", "-pgn", str(pgn), "-bin", str(book), "-min-game", "1", "-uniform"],
                   check=True, capture_output=True, cwd=tmp_path)
    src = rebuild.read_raw(str(book))
    assert src and len(book.read_bytes()) == 16 * len(src)
    with chess.polyglot.open_reader(str(book)) as r:
        assert len(r) == len(src)
        assert {e.raw_move for e in r.find_all(chess.Board())} == {m for k, m, _w, _l in src
                                                                   if k == chess.polyglot.zobrist_hash(chess.Board())}
    stats = attribute_to(tmp_path, book, pgn)
    assert stats["meta"]["credited_plies"] > 0
    out = run_rebuild(tmp_path, book, {"meta": {}, "nodes": []})
    assert out.read_bytes() == book.read_bytes()                             # identity on a real book


def test_entries_must_be_sorted_by_key(tmp_path):
    with pytest.raises(SystemExit):
        rebuild.write_raw(str(tmp_path / "bad.bin"), [(9, 1, 1, 0), (2, 1, 1, 0)])


def test_a_truncated_book_is_rejected(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(struct.pack(">QHHI", 1, 2, 3, 4)[:-1])
    with pytest.raises(SystemExit):
        rebuild.read_raw(str(bad))
