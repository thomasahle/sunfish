"""Executable correspondence checks for pawn-protected move debt."""

import sunfish


def move_named(pos, name):
    return next(m for m in pos.gen_moves()
                if sunfish.render(m.i) + sunfish.render(m.j) == name)


def test_protected_killer_keeps_its_fixed_child_depth():
    """Killer status may reorder an eligible move, but cannot choose its depth."""
    pos = sunfish.Position(
        sunfish.initial, 0, (True, True), (True, True), 0, 0)
    for name in ("a2a4", "b2b4", "c2c3", "h2h3", "d2d4"):
        pos = pos.move(move_named(pos, name)).rotate()

    killer = move_named(pos, "c1h6")
    child = pos.move(killer)
    assert pos.board[killer.i] == "B"
    assert pos.board[killer.j] == "."
    assert (pos.board[killer.j + sunfish.N + sunfish.W] == "p"
            or pos.board[killer.j + sunfish.N + sunfish.E] == "p")

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
    searcher.bound(pos, sunfish.MATE_UPPER, 4)

    assert calls
    assert {depth for _, depth, _ in calls} == {2}
