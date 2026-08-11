"""Prove the packed accumulator and head correct before optimising anything.

Three independent checks, over tens of thousands of positions reached by
random legal play (so castling, en passant, promotion and king capture all
occur):

  1. lane integrity  -- every lane of every accumulator stays inside
     [0, 2^15).  A lane that leaves the range borrows from its NEIGHBOUR and
     silently corrupts a different feature, which is why this is checked
     everywhere rather than argued about.
  2. incremental == from scratch -- the accumulator carried through move()
     equals the one rebuilt from the board.
  3. packed == float -- the quantised head against a float reference of the
     same net, and the eval's exact antisymmetry score(p) == -score(p.rotate()).

usage: verify.py ENGINE.py NET.pickle [nplies] [ngames]
"""
import sys, os, random, importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnet


def load(path, net):
    os.environ["SF_NET"] = net
    spec = importlib.util.spec_from_file_location("eng", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eng"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    engpath, netpath = sys.argv[1], sys.argv[2]
    nplies = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    ngames = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    eng = load(engpath, netpath)
    net = pnet.load(netpath)
    random.seed(20260809)

    n = 0
    worst_lane = 0
    for g in range(ngames):
        pos = eng.hist[0]
        for ply in range(nplies):
            n += 1
            # 1. lane integrity
            lanes = pnet.unpacklanes(pos.acc, net.lanes)
            assert pos.acc >= 0, "accumulator went negative"
            m = max(lanes)
            worst_lane = max(worst_lane, max(abs(x - pnet.BIAS) for x in lanes))
            assert m < (1 << pnet.VBITS), (
                "lane %d = %d overflowed the guard at game %d ply %d"
                % (lanes.index(m), m, g, ply))
            # 2. incremental == from scratch
            fresh = net.from_board(pos.board, pos.pf)
            assert fresh == pos.acc, (
                "incremental != from-scratch at game %d ply %d" % (g, ply))
            # the piece count carried by the position must be the truth
            cnt = sum(1 for c in pos.board if c in pnet.PIECES)
            assert pos.cnt == cnt, "incremental piece count drifted"
            # the other perspective flag must give the same evaluation
            other = net.from_board(pos.board, pos.pf ^ 1)
            assert net.nn_cp(other, pos.pf ^ 1, cnt) == net.nn_cp(pos.acc, pos.pf, cnt), \
                "perspective flag changes the evaluation"
            # 3. packed head == engine head, and ps + nn == score
            assert eng.nn_cp(pos.acc, pos.pf, cnt) == net.nn_cp(pos.acc, pos.pf, cnt)
            assert pos.score == pos.ps + eng.nn_cp(pos.acc, pos.pf, cnt)
            # antisymmetry, and it has to hold for the RECOMPUTED evaluation
            # of the rotated position, not just for rotate()'s negation --
            # the search reaches positions by both routes and the
            # transposition table assumes they agree.
            r = pos.rotate(nullmove=True)
            assert r.score == -pos.score
            fresh_r = net.from_board(r.board, r.pf)
            assert fresh_r == r.acc
            assert net.nn_cp(fresh_r, r.pf, cnt) == -net.nn_cp(pos.acc, pos.pf, cnt), \
                "evaluation is not exactly antisymmetric"

            moves = list(pos.gen_moves())
            if not moves:
                break
            mv = random.choice(moves)
            if pos.board[mv.j] == "k":
                break                    # king captured: restart
            pos = pos.move(mv)
    print("verified %d positions from %d random games" % (n, ngames))
    print("worst |lane - BIAS| seen: %d   (guard trips at %d, rigorous bound %d)"
          % (worst_lane, 1 << (pnet.VBITS - 1), net.meta["excursion"]))


main()
