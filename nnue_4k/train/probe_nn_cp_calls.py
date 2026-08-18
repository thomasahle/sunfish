"""How many times does nn_cp run per NODE?  The two live readings disagree by
20x (main ledger: ~21.5/node; factor lane: ~1.01/node) and every speed price in
the portfolio scales with the answer.  Settle it by counting."""
import sys, time
sys.path.insert(0, "/Users/ahle/repos/sunfish-packed/nnue_4k")
sys.path.insert(0, "/Users/ahle/repos/sunfish-packed")
import replnet_proto as R

calls = [0]
_orig = R.nn_cp
def counted(acc, pf):
    calls[0] += 1
    return _orig(acc, pf)
R.nn_cp = counted

for name, fen_moves in (("startpos", []),
                        ("kiwipete-ish", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])):
    pos = R.from_board(R.initial)
    for mv in fen_moves:
        i = R.parse(mv[:2]); j = R.parse(mv[2:4])
        pos = pos.move(R.Move(i, j, "") if hasattr(R, "Move") else (i, j, ""))
    s = R.Searcher()
    calls[0] = 0
    t0 = time.perf_counter()
    nodes = 0
    for depth, gamma, score, move in s.search([pos]):
        nodes = s.nodes if hasattr(s, "nodes") else nodes
        if depth >= 5:
            break
    el = time.perf_counter() - t0
    n = getattr(s, "nodes", None)
    print("%-14s depth=%d  nn_cp calls=%d  searcher.nodes=%s  ratio=%s  %.2fs"
          % (name, depth, calls[0], n,
             ("%.2f" % (calls[0] / n)) if n else "n/a", el))
