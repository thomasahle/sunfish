"""Executable correspondence checks for intrinsic late-move reduction."""

import sunfish


def move_named(pos, name):
    return next(m for m in pos.gen_moves()
                if sunfish.render(m.i) + sunfish.render(m.j) == name)


def test_eligible_killer_keeps_its_intrinsic_child_depth():
    """Killer status may reorder a move, but must not choose its depth."""
    pos = sunfish.Position(
        sunfish.initial, 0, (True, True), (True, True), 0, 0)
    pos = pos.move(move_named(pos, "c2c3")).rotate()
    killer = move_named(pos, "d1a4")
    child = pos.move(killer)

    assert pos.board[killer.i] == "Q"
    assert pos.board[killer.j] == "."
    assert pos.value(killer) < 0

    searcher = sunfish.Searcher()
    searcher.history = set()
    searcher.tp_move[pos] = killer
    calls = []
    bound = searcher.bound

    def observed(p, gamma, depth, root=False):
        if p == child:
            calls.append((gamma, depth, root))
        return bound(p, gamma, depth, root)

    searcher.bound = observed
    searcher.bound(pos, sunfish.MATE_UPPER, 5)

    assert calls
    assert {depth for _, depth, _ in calls} == {3}

