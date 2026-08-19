"""Generate the CARRIER-PORTED engine, and gate it bit-exact against the
shipped one.

The measurement behind this file (ledger, 2026-08-19, "THE EVALUATION TAX,
DECOMPOSED"): the packed net's per-node cost is dominated NOT by the
accumulator update (already incremental) nor by the readout, but by the
carrier -- `sunfish_nnue.py` rebuilds the mirrored board with
`board[::-1].swapcase()` on every node (~917 ns measured, more than the whole
N=32 readout), where `pst_entry.py` carries it in the tuple as `r` and pays two
slice-concats instead.  Grafting the accumulator onto the ENTRY's carrier takes
nps retention from 60.0% to 83.8% of the entry with a BIT-IDENTICAL evaluation.

This file exists so that number is reproducible and so the port can never drift
from either parent: it is generated FROM `pst_entry.py`, and gated AGAINST
`sunfish_nnue.py`.  It builds nothing that ships; a real 4k artifact re-prices
from the bytes-literal decoder.

usage:
    carrier_port.py OUT.py                # generate
    carrier_port.py OUT.py --gate NET     # generate, then prove bit-exact
"""
import os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(HERE, os.pardir, "pst_entry.py")

# The net loader and read-out, in the pure-int form a 4k artifact would carry:
# no json/base64 extension machinery, no bucket tables, no float tail.
LOADER = '''
###############################################################################
# Packed big-integer NNUE residual, riding the ENTRY's carrier
###############################################################################
import json as _json
from base64 import b64decode as _b64
NET_PATH = os.environ["SF_NET"]
_h, _rr = open(NET_PATH).read().split("\\n", 1)
_d = _json.loads(_h)
_it = (int.from_bytes(_b64(t), "little", signed=True) for t in _rr.split())
_PIECES = "PNBRQKpnbrqk"
ACC_BASE, MGP = next(_it), next(_it)
_r0 = {p: [next(_it) for _ in range(120)] for p in _PIECES}
ROWS = (_r0, {p: [_r0[p.swapcase()][119 - s] for s in range(120)]
              for p in _PIECES})
_N = _d["N"]
NLANE, LBITS, VBITS = 2 * _N, 16, 15
ONES = (1 << VBITS) - 1
HALF = _N * LBITS
SHIFT, CLAMP = _d["shift"], _d["clampcp"]
MH = sum(1 << (VBITS + LBITS * i) for i in range(NLANE))
MLO = MH >> 1
MVAL = MH - (MH >> VBITS)
MGH = MGP | MH
MASKLO = (1 << HALF) - 1
M16 = (1 << LBITS) - 1
del _d


def nn_cp(acc, pf, bd=""):
    """Byte-for-byte the shipped read-out for the pure-int family; `bd` is
    accepted and ignored so packed/verify.py runs unmodified."""
    m = ((acc & MLO) >> 14) * ONES
    y = ((acc & m) | MLO) - MLO
    m = (((MGH - y) & MH) >> VBITS) * ONES
    y = (y & m) | (MGP & (m ^ MVAL))
    v = (y & MASKLO) % M16 - ((y >> HALF) & MASKLO) % M16
    if pf: v = -v
    v = (v >> SHIFT) if v >= 0 else -((-v) >> SHIFT)
    return -CLAMP if v < -CLAMP else (CLAMP if v > CLAMP else v)

'''

# (old, new) applied in order, each exactly once.  An assertion failure here
# means `pst_entry.py` moved and the port must be re-derived, which is the
# whole point of generating rather than forking.
EDITS = [
    ("###############################################################################\n"
     "# Global constants",
     LOADER + "###############################################################################\n"
     "# Global constants"),
    ('class Position(namedtuple("P", "board score wc bc ep kp r")):',
     'class Position(namedtuple("P", "board score wc bc ep kp r acc pf ps")):'),
    # rotate(): the accumulator is UNCHANGED; only which lane block is ours flips
    ("""        return Position(
            self.r, -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not n else 0,
            119 - self.kp if self.kp and not n else 0,
            self.board,
        )""",
     """        return Position(
            self.r, -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not n else 0,
            119 - self.kp if self.kp and not n else 0,
            self.board, self.acc, self.pf ^ 1, -self.ps,
        )"""),
    # move(): every board mutation gets its packed row delta, exactly as the
    # shipped carrier does.  `ps` stays the exact incremental pst so that
    # value(move) remains an exact delta of it.
    ("""        score = self.score + self.value(move)
        # Actual move, applied to BOTH orientations so neither is ever rebuilt
        board, r = put(board, j, p), put(r, 119 - j, p.swapcase())""",
     """        ps = self.ps + self.value(move)
        row = ROWS[self.pf]
        acc = self.acc + row[p][j] - row[p][i]
        if q != ".":
            acc -= row[q][j]
        # Actual move, applied to BOTH orientations so neither is ever rebuilt
        board, r = put(board, j, p), put(r, 119 - j, p.swapcase())"""),
    ('''                board, r = put(board, rk, "."), put(r, 119 - rk, ".")
                board, r = put(board, kp, "R"), put(r, 119 - kp, "r")''',
     '''                board, r = put(board, rk, "."), put(r, 119 - rk, ".")
                board, r = put(board, kp, "R"), put(r, 119 - kp, "r")
                acc += row["R"][kp] - row["R"][rk]'''),
    ("                board, r = put(board, j, prom), put(r, 119 - j, prom.swapcase())",
     '                board, r = put(board, j, prom), put(r, 119 - j, prom.swapcase())\n'
     '                acc += row[prom][j] - row["P"][j]'),
    ('                board, r = put(board, j + S, "."), put(r, 119 - j - S, ".")',
     '                board, r = put(board, j + S, "."), put(r, 119 - j - S, ".")\n'
     '                acc -= row["p"][j + S]'),
    # the read-out is mover-signed by the NEW perspective flag, hence `+`
    ("""        return Position(r, -score, bc, wc,
                        119 - ep if ep else 0, 119 - kp if kp else 0, board)""",
     """        pf = self.pf ^ 1
        return Position(r, -ps + nn_cp(acc, pf), bc, wc,
                        119 - ep if ep else 0, 119 - kp if kp else 0, board,
                        acc, pf, -ps)"""),
    ('''def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0):
    """Build a position from scratch; `board` is in the mover's orientation."""
    score = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
                for i, p in enumerate(board) if p.isalpha())
    return Position(board, score, wc, bc, ep, kp, board[::-1].swapcase())''',
     '''def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position from scratch; `board` is in the mover's orientation."""
    ps = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
             for i, p in enumerate(board) if p.isalpha())
    row = ROWS[pf]
    acc = ACC_BASE
    for s, c in enumerate(board):
        if c in _PIECES:
            acc += row[c][s]
    return Position(board, ps + nn_cp(acc, pf), wc, bc, ep, kp,
                    board[::-1].swapcase(), acc, pf, ps)'''),
]


def generate(out):
    src = open(ENTRY).read()
    for old, new in EDITS:
        assert src.count(old) == 1, "pst_entry.py moved; re-derive: %r" % old[:60]
        src = src.replace(old, new, 1)
    open(out, "w").write(src)
    return out


def gate(out, net, ngames=60, nplies=60):
    """The evaluation must be IDENTICAL to the shipped carrier at every node --
    otherwise the port is a new engine and its fixed-node Elo is unknown."""
    import importlib.util
    os.environ["SF_NET"] = net
    sys.path.insert(0, os.path.join(HERE, os.pardir))
    import sunfish_nnue as F
    spec = importlib.util.spec_from_file_location("ported", out)
    E = importlib.util.module_from_spec(spec)
    sys.modules["ported"] = E
    spec.loader.exec_module(E)

    random.seed(4096)
    checked = 0
    for _ in range(ngames):
        f, e = F.hist[0], E.hist[0]
        for _ in range(nplies):
            assert f.board == e.board and f.ps == e.ps and f.acc == e.acc
            assert f.score == e.score, (f.score, e.score)
            assert f.pf == e.pf
            # incremental == from scratch, in the ported carrier's own frame
            reb, row = E.ACC_BASE, E.ROWS[e.pf]
            for s, c in enumerate(e.board):
                if c in E._PIECES: reb += row[c][s]
            assert reb == e.acc
            assert F.nn_cp(f.acc, f.pf, f.board) == E.nn_cp(e.acc, e.pf)
            # rotate() must agree too: it is where the port changes most
            assert f.rotate(True).score == e.rotate(True).score
            assert f.rotate(True).acc == e.rotate(True).acc
            checked += 1
            ms = sorted(tuple(m) for m in f.gen_moves())
            if not ms: break
            mv = ms[random.randrange(len(ms))]
            assert ms == sorted(tuple(m) for m in e.gen_moves())
            f, e = f.move(F.Move(*mv)), e.move(mv)
    return checked


if __name__ == "__main__":
    out = generate(sys.argv[1])
    print("generated", out)
    if "--gate" in sys.argv:
        net = sys.argv[sys.argv.index("--gate") + 1]
        n = gate(out, net)
        print("GATE PASS: %d nodes -- score, ps, acc, read-out, from-scratch "
              "rebuild and rotate() all identical to sunfish_nnue.py" % n)
