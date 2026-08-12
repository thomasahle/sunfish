"""Regression tests for bugs found in the 2026-08 live-game and test-suite
audits. Each test names the lichess game or audit finding it guards.

All tests are deterministic: fixed-depth searches and "stop-scans" that
enumerate every point the iterative-deepening loop could have been
interrupted (a time-starved search stops at exactly one of these points,
so a move that never appears at any stop point can never be played, on
any hardware, under any load).

Not covered here: the 2026-08-05 engine hang trio (probable OOM on the
1GB bot VM plus a suspected ponder race) is an ops/integration concern —
memory sizing is a config matter and the protocol path is exercised by
tests/test_bot_integration.py.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_sunfish():
    """Import sunfish.py without triggering its UCI interface.

    Handles both layouts: with a main() entry point (post-#138) a plain
    exec is safe (the __main__ guard does not fire); before that, the
    interface ran at module level and we cut the source at the last
    minifier-hide marker (everything above it is pure definitions).
    """
    src = (ROOT / "sunfish.py").read_text()
    if "def main():" not in src:
        src = src[: src.rindex("# minifier-hide start")]
    module = type(sys)("sunfish_under_test")
    module.__file__ = str(ROOT / "sunfish.py")
    exec(compile(src, "sunfish.py", "exec"), module.__dict__)
    return module


sf = load_sunfish()

import sunfish_ui.uci as uci  # noqa: E402

uci.sunfish = sf  # the UCI module resolves the engine module via this global


def hist_from_fen(fen, moves=()):
    parts = fen.split()
    pos = uci.from_fen(*(parts + ["0", "1"] * ((6 - len(parts)) // 2))[:6])
    hist = [pos] if parts[1] == "w" else [pos.rotate(), pos]
    for m in moves:
        hist.append(hist[-1].move(uci.parse_move(m, len(hist) % 2 == 1)))
    return hist


def stop_scan(hist, max_depth):
    """Yield (stop_point, root_move) for every possible interruption point."""
    searcher = sf.Searcher()
    for n, (depth, gamma, score, move) in enumerate(searcher.search(hist)):
        if depth > max_depth:
            break
        yield n, searcher.tp_move.get(hist[-1])


def render(hist, move):
    return uci.render_move(move, white_pov=len(hist) % 2 == 1)


class TestStalemateBlindness:
    """lichess SSPx1Gr0 (2026-08-05): with Q+R vs bare K and mate-in-2 on
    the board, the deployed engine played Qc4?? stalemate. Root cause: a
    depth<=2 child scored stalemating as +MATE_UPPER, poisoning tp_move,
    and a node-starved bullet search (stop points 5-8, 87-1279 nodes)
    served it. Fixed in 8440d8f (PR #136)."""

    FEN = "8/8/7p/5p2/4q3/K7/4rk2/8 b - - 3 88"

    def test_qc4_never_chosen_at_any_stop_point(self):
        hist = hist_from_fen(self.FEN)
        seen = 0
        for n, move in stop_scan(hist, max_depth=5):
            if move is not None:
                assert render(hist, move) != "e4c4", f"stop point {n}"
                seen += 1
        assert seen > 0


class TestSpiteCheckPoisoning:
    """lichess n4FD0p5Q (2023-02-01): with Q vs 2R holding a fortress
    draw, the 2023 engine played Qf2+?? Kxf2. Mechanism (reproduced on
    the era engine): a routine mate-level MTD-bi bracket probe scored
    every sane retreat as a false mate, stored the spite check as the
    hash move, and an uninterruptible probe boundary served it. The
    current engine must never surface Qf2+ at any stop point."""

    FEN = "8/8/2R5/3R4/8/1k2K3/8/5q2 b - - 10 90"

    def test_qf2_never_chosen_at_any_stop_point(self):
        hist = hist_from_fen(self.FEN)
        seen = 0
        for n, move in stop_scan(hist, max_depth=6):
            if move is not None:
                assert render(hist, move) != "f1f2", f"stop point {n}"
                seen += 1
        assert seen > 0


class TestCappedNullMove:
    """The old mate-band null semantics needed a second boundary probe.

    This position exercises the rare case where that probe vetoed the pass.
    The monotone replacement declares the null option to be the smaller of
    the pass value and static evaluation plus one MTD score bucket, so the
    first child report is sufficient and the result is the static cap.
    """

    FEN = "8/6p1/6R1/k7/2K5/8/8/8 w - - 0 1"

    def test_static_cap_replaces_mate_boundary_probe(self):
        sf.pst["K"] = sf.K_MID
        old_engine, uci.sunfish = uci.sunfish, sf
        try:
            pos = hist_from_fen(self.FEN)[-1]
        finally:
            uci.sunfish = old_engine
        searcher = sf.Searcher()
        searcher.root, searcher.history = pos, set()
        calls, bound = [], searcher.bound

        def observed(pos, gamma, depth, root=False):
            calls.append((gamma, depth, root))
            return bound(pos, gamma, depth, root)

        searcher.bound = observed
        score = bound(pos, 0, 6)

        assert score == pos.score + sf.EVAL_ROUGHNESS == 409
        assert not any(gamma == 1 - sf.MATE_LOWER for gamma, _, _ in calls)


class TestNullSentinelMasking:
    """Audit finding A1: in pawn endings the null-move gate
    (abs(score) < 500) admits a "pass" that yields a normal material
    score, masking the -MATE_UPPER stalemate sentinel. Consequences on
    unfixed engines: contradictory root bounds (lower > upper) in a KPK
    probe, and a thrown KPK race (the winning side stalemates the bare
    king at fixed depth 8).

    These began life as strict xfails; the A1 fix PR claimed them as
    passing tests, per the golden-floor doctrine."""

    PROBE_FEN = "k7/P7/1K6/8/8/8/8/8 w - - 0 1"
    # A textbook WON KPK (king in front, opposition). The audit's original
    # race position (8/5k2/... rook pawn) turned out to be a THEORETICAL
    # DRAW - the defender reaches the corner - so "converts to mate" was
    # the wrong assertion there; the bug it exhibited (crossed bounds,
    # stalemate delivered) is covered by the probe test above and the
    # no-stalemate assertion below.
    RACE_FEN = "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"

    def test_kpk_probe_bounds_do_not_cross(self):
        hist = hist_from_fen(self.PROBE_FEN)
        searcher = sf.Searcher()
        lower, upper = None, None
        for depth, gamma, score, move in searcher.search(hist):
            if depth > 6:
                break
            if score >= gamma:
                lower = score
            else:
                upper = score
        assert lower is not None and upper is not None
        assert lower <= upper, f"contradictory root bounds [{lower}, {upper}]"

    def test_kpk_race_win_is_not_thrown(self):
        import chess

        board = chess.Board(self.RACE_FEN)
        hist = hist_from_fen(self.RACE_FEN)
        for ply in range(120):
            searcher = sf.Searcher()
            for depth, gamma, score, move in searcher.search(hist):
                if depth > 8:
                    break
            best = searcher.tp_move.get(hist[-1])
            assert best is not None, f"no move at ply {ply}"
            uci_move = chess.Move.from_uci(render(hist, best))
            assert uci_move in board.legal_moves, f"illegal move at ply {ply}"
            board.push(uci_move)
            hist.append(hist[-1].move(best))
            assert not board.is_stalemate(), (
                f"stalemate delivered at ply {ply + 1} in a won KPK"
            )
            if board.is_checkmate():
                return  # converted correctly
            assert board.halfmove_clock < 100, (
                f"50-move draw at ply {ply + 1} in a won KPK"
            )
        pytest.fail("no conversion within 120 plies")


class TestMateDistance:
    """github.com/thomasahle/sunfish/issues/11 ("Tempo", 2014): "a mate in 6
    is considered the same as a mate in 1".

    Every checkmate used to score the flat ``-MATE_LOWER``, so a mating line
    carried no information about HOW FAR the mate was: at the fold, every
    winning move tied, and the losing side had no reason to hold out. The
    terminal correction now deposits the depth still unspent when the mate
    was found, one ``EVAL_ROUGHNESS`` per ply, which negation carries home
    as ``MATE_LOWER + (depth - plies) * EVAL_ROUGHNESS``.

    The scale matters: MTD-bi stops bisecting at ``upper - lower <=
    EVAL_ROUGHNESS``, so at one point per ply the driver's final window
    could not separate two mating moves and the ordering never reached the
    root. A whole bracket per ply is what makes it visible.

    The position below is the complaint in miniature: three mating moves
    and eight moves that mate in three, all scoring exactly 47923 on
    master, and 47998 vs 47938 at depth 6 here."""

    FEN = "8/3Q4/8/8/8/3R4/5K1k/8 w - - 0 1"
    FAST = ("d3h3", "d7h3", "d7h7")  # mate in 1
    SLOW = ("d3a3", "d3b3", "d3c3", "d3d2")  # mate in 3

    def yield_of(self, hist, uci_move, depth):
        """The parent's view of one move at a fixed depth: -bound(child)."""
        child = hist[-1].move(uci.parse_move(uci_move, len(hist) % 2 == 1))
        searcher = sf.Searcher()
        return -searcher.bound(child, -sf.MATE_LOWER, depth - 1, root=True)

    @pytest.mark.parametrize("depth", [4, 5, 6])
    def test_faster_mate_scores_strictly_better(self, depth):
        hist = hist_from_fen(self.FEN)
        fast = [self.yield_of(hist, m, depth) for m in self.FAST]
        slow = [self.yield_of(hist, m, depth) for m in self.SLOW]
        assert min(fast) >= sf.MATE_LOWER, f"mate in 1 left the band: {fast}"
        assert min(slow) >= sf.MATE_LOWER, f"mate in 3 left the band: {slow}"
        assert min(fast) - max(slow) >= sf.EVAL_ROUGHNESS, (
            f"depth {depth}: mate in 1 scored {fast}, mate in 3 scored {slow} "
            f"- the gap must clear EVAL_ROUGHNESS ({sf.EVAL_ROUGHNESS}) or the "
            "driver's final bracket swallows it"
        )
        assert len(set(fast)) == 1 and len(set(slow)) == 1, (
            f"same distance, different score: {fast} / {slow}"
        )

    @pytest.mark.parametrize("depth", [2, 3, 4, 5, 6])
    def test_mate_in_one_score_carries_the_distance(self, depth):
        # Ra8# from a bare-rook mate: the score is the band floor plus the
        # depth the search still had in hand.
        hist = hist_from_fen("3k4/8/3K4/8/8/8/8/7R w - - 0 1")
        searcher = sf.Searcher()
        score = searcher.bound(hist[-1], sf.MATE_LOWER, depth, root=True)
        want = sf.MATE_LOWER + (depth - 1) * sf.EVAL_ROUGHNESS
        assert score == want, (
            f"depth {depth}: mate in 1 scored {score}, expected {want}"
        )
        assert render(hist, searcher.tp_move[hist[-1]]) == "h1h8"

    @pytest.mark.parametrize("depth", [4, 6])
    def test_engine_plays_the_mate_in_one(self, depth):
        hist = hist_from_fen(self.FEN)
        searcher = sf.Searcher()
        for d, gamma, score, move in searcher.search(hist):
            if d > depth:
                break
        played = render(hist, searcher.tp_move[hist[-1]])
        assert played in self.FAST, f"depth {depth}: dawdled with {played}"

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_checkmated_node_reports_its_distance(self, depth):
        """The mated side's half of the contract: being mated with `depth`
        still unspent is worth exactly -MATE_LOWER - depth*EVAL_ROUGHNESS, so
        of two lost replies the one that postpones the mate scores strictly
        higher - by a whole bracket per ply."""
        hist = hist_from_fen("3k3R/8/3K4/8/8/8/8/8 b - - 1 1")
        searcher = sf.Searcher()
        score = searcher.bound(hist[-1], sf.MATE_LOWER, depth, root=True)
        want = -sf.MATE_LOWER - depth * sf.EVAL_ROUGHNESS
        assert score == want, (
            f"depth {depth}: checkmated node scored {score}, expected {want}"
        )
        assert score > -sf.MATE_UPPER, (
            "a mate value collided with the illegal-move sentinel"
        )
