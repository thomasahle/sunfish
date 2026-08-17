"""Certify the factored pricing variant: the engine's rows ARE U @ V.

The byte and speed numbers in the ledger are for artifacts that RUN; this is
the separate claim that they also COMPUTE THE RIGHT THING, which is what makes
the pricing variant a legitimate stand-in for a trained one.  It is the
`verify_export` triangle minus the torch leg (there is no trained net yet):

  (a) payload decode == the emitter's own factors  -- the stream round-trips
  (b) engine ROWS == an independent numpy-free reconstruction of U @ V
  (c) engine nn_cp == an independent integer reference read-out, on real
      boards, in both perspectives, and antisymmetric under rotation

Failing any of these means the pricing variant prices something other than
the design, so this runs before the ledger quotes a byte number.

    python3 factor_check.py [--r 4] [--N 32] [--lane-bits 16]
"""
import argparse
import importlib.util
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_factor_proto as M                                    # noqa: E402


def load(r, N, lane_bits, zeros, seed):
    with open(os.path.join(HERE, "..", "replnet_proto.py")) as f:
        src = f.read()
    out, ndig, _ = M.build(src, r, N, lane_bits, zeros, seed)
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(out)
    spec = importlib.util.spec_from_file_location("fv", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.unlink(path)
    return m, ndig


def ref_weights(q, r, N):
    """Independent reconstruction: W[piece][square][lane] = sum_j U*V."""
    voff, foff = 1 + 2 * N, 1 + 2 * N + r * N
    V = [[q[voff + j * N + k] - 44 for k in range(N)] for j in range(r)]
    nchunk = (r + 3) // 4
    W = []
    for i in range(12):
        rows = []
        for f in range(64):
            u = []
            for c in range(nchunk):
                d = q[foff + (i * 64 + f) * nchunk + c]
                u += [d // 3 ** j % 3 - 1 for j in range(min(4, r - c * 4))]
            rows.append([sum(u[j] * V[j][k] for j in range(r)) for k in range(N)])
        W.append(rows)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--lane-bits", type=int, default=16)
    ap.add_argument("--zeros", type=float, default=0.43)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--boards", type=int, default=200)
    a = ap.parse_args()
    m, ndig = load(a.r, a.N, a.lane_bits, a.zeros, a.seed)
    r, N, LB = a.r, a.N, a.lane_bits
    q = m._q
    assert len(q) == ndig, ("payload length", len(q), ndig)

    # (a) the emitter's factors survive the codec
    want = M.emit_payload(r, N, LB - 1, a.zeros, a.seed)
    assert q == want, "payload did not round-trip through the base-90 literal"

    # (b) engine rows == independent reconstruction
    W = ref_weights(q, r, N)
    sqs = [21 + f // 8 * 10 + f % 8 for f in range(64)]
    for i, p in enumerate(m._PIECES):
        for f in range(64):
            got = m._half[p][sqs[f]]
            wnt = sum(W[i][f][k] << LB * k for k in range(N))
            assert got == wnt, ("half row", p, f, got, wnt)
    for p in m._PIECES:
        for s in range(120):
            assert m.ROWS[1][p][s] == m.ROWS[0][p.swapcase()][119 - s], (p, s)
    nonzero = sum(1 for p in m._PIECES for s in sqs if m._half[p][s])
    assert nonzero > 12 * 64 * 0.3, ("rows are mostly dead", nonzero)

    # (c) engine nn_cp == an independent integer reference, on real boards
    # `pf` is not a sign: from_board picks ROWS[pf], which swaps the two
    # perspective blocks, and nn_cp negates -- the two flips cancel, so the
    # eval of one board string is pf-INVARIANT (verify_export asserts the same
    # thing as "perspective flag changed the eval").  The reference therefore
    # takes no pf, and pf-invariance is checked separately below.
    def ref_cp(board):
        acc = [0] * N
        for s, c in enumerate(board):
            if c in m._PIECES:
                i = m._PIECES.index(c)
                f = (s - 21) // 10 * 8 + (s - 21) % 10
                for k in range(N):
                    acc[k] += W[i][f][k]
        for k in range(N):                       # bias, offset-binary
            acc[k] += q[1 + N + k] - 44
        opp = [0] * N
        for s, c in enumerate(board):            # the them-block, mirrored
            if c in m._PIECES:
                i = m._PIECES.index(c.swapcase())
                s2 = 119 - s
                f = (s2 - 21) // 10 * 8 + (s2 - 21) % 10
                for k in range(N):
                    opp[k] += W[i][f][k]
        for k in range(N):
            opp[k] += q[1 + N + k] - 44
        cap = [q[1 + k] * (1 << max(0, LB - 8)) for k in range(N)]
        tot = (sum(min(max(acc[k], 0), cap[k]) for k in range(N))
               - sum(min(max(opp[k], 0), cap[k]) for k in range(N)))
        return max(-m.CLAMP, min(m.CLAMP, int(tot / (1 << m.SHIFT))))

    rng = random.Random(a.seed)
    board = list(m.initial)
    fired = 0
    for t in range(a.boards):
        wnt = ref_cp("".join(board))
        reads = []
        for pf in (0, 1):
            pos = m.from_board("".join(board), pf=pf)
            got = m.nn_cp(pos.acc, pos.pf)
            assert got == wnt, ("nn_cp != int-ref", t, pf, got, wnt)
            reads.append(got)
            fired += got != 0
        assert reads[0] == reads[1], ("perspective flag changed the eval", t)
        # antisymmetry: the rotated board from the other side negates
        rot = "".join(c.swapcase() for c in reversed("".join(board)))
        p0, p1 = m.from_board("".join(board), pf=0), m.from_board(rot, pf=0)
        assert m.nn_cp(p0.acc, 0) == -m.nn_cp(p1.acc, 0), ("antisymmetry", t)
        # perturb: move one man to a random empty square
        occ = [s for s in sqs if board[s] in m._PIECES]
        emp = [s for s in sqs if board[s] == "."]
        if occ and emp:
            s, d = rng.choice(occ), rng.choice(emp)
            board[d], board[s] = board[s], "."

    print("factored r=%d N=%d lane_bits=%d: PASS" % (r, N, LB))
    print("  payload %d digits, round-trips through the literal" % ndig)
    print("  %d/%d piece-square rows == an independent U@V reconstruction"
          % (12 * 64, 12 * 64))
    print("  %d boards x 2 views: nn_cp == int-ref, antisymmetry holds, "
          "net fired on %d of %d reads" % (a.boards, fired, 2 * a.boards))


if __name__ == "__main__":
    main()
