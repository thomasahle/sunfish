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

    Probed at depth 5: the pass is a score candidate only on `2 < depth < 6`.
    From depth 6 it is a fuel oracle that never contributes a score, so the
    cap has nothing to cap there (see TestFuelOracle).
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
        nullpos = pos.rotate(nullmove=True)

        def observed(pos, gamma, depth, root=False):
            calls.append((gamma, depth, root))
            if pos == nullpos and gamma == 1 and depth == 2:
                return -sf.MATE_LOWER
            return bound(pos, gamma, depth, root)

        searcher.bound = observed
        score = bound(pos, 0, 5)

        assert score == pos.score + sf.EVAL_ROUGHNESS == 409
        assert not any(gamma == 1 - sf.MATE_LOWER for gamma, _, _ in calls)


class TestFuelOracle:
    """From depth 6 the pass is a fuel oracle, not a score candidate.

    The probe runs at ONE fixed target, `pos.score + NULL_MARGIN`, which
    depends on `(pos, depth)` and not on the caller's `gamma`. That is what
    makes the "hot" bit position-determined -- and so table-cacheable and
    stable across the driver's windows -- and it is the premise the Lean
    proof leans on (`hot_bit_determined`, `hot_bit_stable`). If the window
    ever picks up a `gamma`, the bit stops being a function of the position
    and the trichotomy loses its footing, so it is pinned here.
    """

    FEN = "3k4/8/3K4/8/8/8/8/7R w - - 0 1"

    def probe_windows(self, gamma):
        pos = hist_from_fen(self.FEN)[-1]
        passed = pos.rotate(nullmove=True)
        searcher = sf.Searcher()
        searcher.root, searcher.history = pos, set()
        seen, bound = [], searcher.bound

        def observed(p, g, depth, root=False):
            if p == passed:
                seen.append((g, depth))
            return bound(p, g, depth, root)

        searcher.bound = observed
        bound(pos, gamma, 8)
        return seen

    def test_probe_window_is_gamma_free(self):
        pos = hist_from_fen(self.FEN)[-1]
        target = pos.score + sf.NULL_MARGIN
        windows = [self.probe_windows(g) for g in (0, 200, -200, sf.MATE_LOWER)]
        for gamma, seen in zip((0, 200, -200, sf.MATE_LOWER), windows):
            assert seen, f"gamma {gamma}: the fuel probe never ran"
            assert all(g == 1 - target and d == 1 for g, d in seen), (
                f"gamma {gamma}: probe windows {seen} - expected only "
                f"({1 - target}, 1), the gamma-free fixed target"
            )
        assert len({tuple(w) for w in windows}) == 1, (
            f"the probe window moved with gamma: {windows}"
        )

    def test_no_pass_score_candidate_above_the_horizon(self):
        # Below 6 the capped pass yields a score; from 6 it never does, so
        # the deep null can no longer fail high on its own.
        pos = hist_from_fen("8/6p1/6R1/k7/2K5/8/8/8 w - - 0 1")[-1]
        passed = pos.rotate(nullmove=True)
        for depth, want in ((3, True), (4, True), (5, True), (6, False), (7, False)):
            searcher = sf.Searcher()
            searcher.root, searcher.history = pos, set()
            seen, bound = [], searcher.bound

            def observed(p, g, d, root=False, _b=bound, _s=seen):
                if p == passed:
                    _s.append(g)
                return _b(p, g, d, root)

            searcher.bound = observed
            bound(pos, 0, depth)
            scored = any(g == 1 - 0 for g in seen)
            assert scored == want, (
                f"depth {depth}: pass probed at the caller's window "
                f"{'unexpectedly' if scored else 'never'} ({seen})"
            )


class TestIntrinsicLMR:
    """Deep moves below one fixed intrinsic threshold spend an extra ply.

    The edge cost depends only on position, nominal depth, and move value.
    A cached killer must therefore receive exactly the same depth as that
    move receives later in the intrinsic ordering.
    """

    FEN = "4k3/8/8/3p4/4P3/8/8/N3K3 w - - 0 1"

    def observed_depths(self, depth, pass_score, fen=FEN, root=False):
        pos = hist_from_fen(fen)[-1]
        passed = pos.rotate(nullmove=True)
        moves = list(pos.gen_moves())
        children = {pos.move(move): move for move in moves}
        killer = next(move for move in moves if pos.value(move) < sf.LMR)
        searcher = sf.Searcher()
        searcher.root, searcher.history = pos, set()
        searcher.tp_move[pos] = killer
        seen = []

        def observed(child, gamma, child_depth, root=False):
            if child == passed:
                return -pass_score
            if child in children:
                seen.append((children[child], child_depth))
            return 0

        searcher.bound = observed
        sf.Searcher.bound(searcher, pos, sf.MATE_UPPER, depth, root=root)
        assert seen[0][0] == killer
        return pos, moves, seen

    def test_edge_cost_is_intrinsic_and_killer_independent(self):
        cases = (
            (self.FEN, 5, 0),
            (self.FEN, 6, 0),
            (self.FEN, 6, sf.NULL_MARGIN),
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 6, sf.NULL_MARGIN),
            ("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1", 6, sf.NULL_MARGIN),
        )
        for fen, depth, offset in cases:
            pos = hist_from_fen(fen)[-1]
            pass_score = pos.score + offset
            pos, moves, seen = self.observed_depths(depth, pass_score, fen)
            guard = depth >= 6 and abs(pos.score) < 750 and any(c in pos.board for c in "RBNQ")
            hot = guard and pass_score >= pos.score + sf.NULL_MARGIN
            for move in moves:
                expected = depth - hot - 1 - (guard and pos.value(move) < sf.LMR)
                assert {d for m, d in seen if m == move} == {expected}

    def test_root_moves_are_not_intrinsically_reduced(self):
        pos = hist_from_fen(self.FEN)[-1]
        _, moves, seen = self.observed_depths(6, pos.score + sf.NULL_MARGIN, root=True)
        for move in moves:
            assert {d for m, d in seen if m == move} == {4}


class TestShallowNullMateFloor:
    """#205 coupled the shallow null candidate to the deep probe's reduction.

    Together with root LMR, that hid this forced mate completely. Shallow null
    keeps its three-ply recurrence, root moves get no intrinsic reduction, and
    the finite shallow move cap may delay but not erase the proof.
    """

    FEN = "2q1r3/4pR2/3rQ1pk/p1pnN2p/Pn5B/8/1P4PP/3R3K w - - 1 0"

    def test_depth_eight_search_reports_mate(self):
        pos = hist_from_fen(self.FEN)[-1]
        result = None
        for depth, _, score, move in sf.Searcher().search([pos]):
            if depth == 8:
                result = score, move
            elif depth > 8:
                break
        assert result is not None and result[0] >= sf.MATE_LOWER


class TestStaticMoveCap:
    """Ordinary depth-two and depth-three moves have a fixed static upper cap.

    A cap below the current window proves fail-low without a child search.
    Only a king capture bypasses it and retains the exact mate sentinel.
    """

    def test_fail_low_caps_skip_all_starting_children(self):
        pos = sf.Position(sf.initial, 0, (True, True), (True, True), 0, 0)
        searcher = sf.Searcher()
        searcher.root, searcher.history = pos, set()
        score = searcher.bound(pos, 200, 2, root=True)
        caps = [min(sf.MATE_LOWER - 1, pos.score + pos.value(m) + sf.QS_A)
            for m in pos.gen_moves()]

        assert score == max(caps) == 186
        assert searcher.nodes == 1

    def test_bk15_forcing_capture_recovers_above_cap_horizon(self):
        hist = hist_from_fen(
            "2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1")
        moves = [render(hist, move) for _, move in stop_scan(hist, 7) if move]

        assert moves[-1] == "g4g7"

    def test_en_passant_uses_the_ordinary_move_cap(self):
        pos = hist_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")[-1]
        ep = next(move for move in pos.gen_moves() if render([pos], move) == "e5d6")
        child = pos.move(ep)
        searcher = sf.Searcher()
        searcher.root, searcher.history = pos, set()
        calls, bound = [], searcher.bound

        def observed(pos, gamma, depth, root=False):
            calls.append(pos)
            return bound(pos, gamma, depth, root)

        searcher.bound = observed
        searcher.bound(pos, 1000, 2, root=True)

        assert child not in calls

    def test_king_capture_keeps_exact_sentinel(self):
        pos = hist_from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")[-1]
        searcher = sf.Searcher()
        searcher.root, searcher.history = pos, set()

        def recursive_call(*args, **kwargs):
            pytest.fail("a king capture should be resolved before searching its kingless child")

        searcher.bound = recursive_call
        assert sf.Searcher.bound(
            searcher, pos, sf.MATE_LOWER, 2, root=True) == sf.MATE_UPPER

    def test_null_capture_substitution_ignores_a_quiet_killer(self):
        pos = hist_from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")[-1]
        passed = pos.rotate(nullmove=True)
        quiet = next(move for move in pos.gen_moves() if pos.value(move) < sf.MATE_LOWER)
        searcher = sf.Searcher()
        searcher.root, searcher.history = None, set()
        searcher.tp_move[pos] = quiet

        def null_probe(child, gamma, depth, root=False):
            assert child == passed
            return 0

        searcher.bound = null_probe
        assert sf.Searcher.bound(searcher, pos, 0, 3) == sf.MATE_UPPER


class TestPositiveDepthEvasion:
    """Every positive-depth legal evasion must be searched before certifying mate."""

    CHILD = "8/8/8/8/8/8/1Q6/K1k5 b - - 0 1"

    def test_complete_move_fold_removes_false_mate(self):
        depth = 1
        sf.pst["K"] = sf.K_MID
        child = hist_from_fen(self.CHILD)[-1]
        legal = [m for m in child.gen_moves() if not child.move(m).king_capture()]
        assert [child.value(m) for m in legal] == [-159]
        assert max(child.value(m) for m in legal) < sf.QS - sf.QS_A

        searcher = sf.Searcher()
        searcher.root, searcher.history = child, set()
        assert searcher.bound(child, 1 - sf.MATE_LOWER, depth, root=True) == -1108
        assert searcher.tp_move[child] in legal


class TestNullSentinelMasking:
    """Audit finding A1: in pawn endings the null-move gate
    (abs(score) < 750) admits a "pass" that yields a normal material
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
    was found, one ``EVAL_ROUGHNESS`` per ply -- ``max(1 - MATE_UPPER,
    -MATE_LOWER - depth * EVAL_ROUGHNESS)`` -- which negation carries home
    as ``MATE_LOWER + (depth - plies) * EVAL_ROUGHNESS``.

    The scale matters: MTD-bi stops bisecting at ``upper - lower <=
    EVAL_ROUGHNESS``, so at one point per ply the driver's final window
    could not separate two mating moves and the ordering never reached the
    root. A whole bracket per ply is what makes it visible.

    The position below is the complaint in miniature: three mating moves
    and eight moves that mate in three, all scoring exactly 47923 on
    master, and 47998 vs 47938 at depth 6 here. The finite move cap can keep
    a proof below the mate band at depths two and three; these guarantees
    begin once the proof has moved above that frontier."""

    FEN = "8/3Q4/8/8/8/3R4/5K1k/8 w - - 0 1"
    FAST = ("d3h3", "d7h3", "d7h7")  # mate in 1
    SLOW = ("d3a3", "d3b3", "d3c3", "d3d2")  # mate in 3

    def yield_of(self, hist, uci_move, depth):
        """The parent's view of one move at a fixed depth: -bound(child)."""
        child = hist[-1].move(uci.parse_move(uci_move, len(hist) % 2 == 1))
        searcher = sf.Searcher()
        return -searcher.bound(child, -sf.MATE_LOWER, depth - 1, root=True)

    @pytest.mark.parametrize("depth", [6, 7, 8])
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

    @pytest.mark.parametrize("depth", [4, 5, 6, 7, 8, 9, 10])
    def test_mate_in_one_score_carries_the_distance(self, depth):
        # Ra8# from a bare-rook mate: the score is the band floor plus the
        # depth the search still had in hand.
        #
        # From depth 6 the deep-null fuel oracle is live, and in this
        # position it is hot (White is a rook up), so every real edge costs
        # TWO plies instead of one and the mate arrives with one ply less in
        # hand. That is the whole trade -- a bounded, uniform cost per edge
        # instead of a null cutoff -- and pinning it here is what stops the
        # cost from silently growing: `fuel_edge_cost` proves it is in
        # {1, 2}, and this is the {2} branch, measured.
        hist = hist_from_fen("3k4/8/3K4/8/8/8/8/7R w - - 0 1")
        searcher = sf.Searcher()
        score = searcher.bound(hist[-1], sf.MATE_LOWER, depth, root=True)
        fuel = 1 if depth >= 6 else 0
        want = sf.MATE_LOWER + (depth - 1 - fuel) * sf.EVAL_ROUGHNESS
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
