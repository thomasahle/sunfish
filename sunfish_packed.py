#!/bin/sh
""":"
# Polyglot header: run with pypy3 when available, else python3 (issue #102).
# No -u needed: all UCI output is flushed explicitly (tools/uci.py).
for cmd in pypy3 python3; do
   command -v "$cmd" > /dev/null && exec "$cmd" "$0" "$@"
done
echo "Error: sunfish requires pypy3 or python3" >&2
exit 1
":"""

import os
import time
from itertools import count
from collections import namedtuple

__version__ = "2026-packed"
version = "sunfish " + __version__

###############################################################################
# Packed big-integer NNUE residual
###############################################################################
# The evaluation is
#     score = pst(pos)  +  clip(nn(pos), -CLAMP, CLAMP)
# where pst() is classic sunfish's exact incremental piece-square score (so
# `value(move)` stays exact for move ordering, the QS gate and futility) and
# nn() is a 768 -> N -> 1 net whose whole accumulator and whole head live in
# ONE Python int.  See packed/pnet.py for the lane layout and why the head
# needs no per-lane multiply.

import pickle as _pickle

NET_PATH = os.environ.get("SF_NET", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "packed", "net.pickle"))
with open(NET_PATH, "rb") as _f:
    _d = _pickle.load(_f)

NLANE = 2 * _d["N"] + 2 * _d.get("nb", 0)
LBITS = 16
VBITS = 15
BIAS = 1 << 14
ONES = (1 << VBITS) - 1
HALF = _d["N"] * LBITS            # bit offset of the second lane block
SHIFT = _d["shift"]
CLAMP = _d["clampcp"]
ACC_BASE = _d["base"]


def _rep(v, n):
    r = 0
    for _ in range(n):
        r = (r << LBITS) | v
    return r


MH = _rep(1 << VBITS, NLANE)      # guard bits
MVAL = _rep(ONES, NLANE)          # value bits
MLO = _rep(BIAS, NLANE)           # offset-binary zero, and the bit-14 probe
MGP = _d["gp"]                    # per-lane activation ceilings G_k
MGH = MGP | MH
MASKLO = (1 << HALF) - 1
M16 = (1 << LBITS) - 1
# One packed constant M_k*t_i per extra activation segment.  Empty for plain
# clipped ReLU; three breakpoints approximate squared clipped ReLU.
MTS = tuple(_d.get("ts", ()))

# ---- extensions: bilinear lanes, narrow odd tail, phase output scale.
# Their read-out runs in floats; exact antisymmetry survives because every
# float input is exactly negated or exactly invariant under perspective
# swap, IEEE arithmetic is sign-symmetric, and the final truncation rounds
# toward zero (packed/pnet.py ext_cp is the verified reference).
NB = _d.get("nb", 0)              # bilinear lanes per perspective
BM = _d.get("m", 4)               # bilinear groups
PHASE_S = tuple(_d.get("phase_s") or ()) or None
EXT = bool(NB or PHASE_S)
if NB:
    NBG = NB // BM                # lanes per group (contiguous runs)
    BGMASK = (1 << (NBG * LBITS)) - 1
    BOFFX = tuple((2 * _d["N"] + s * NBG) * LBITS for s in range(BM))
    BOFFY = tuple((2 * _d["N"] + NB + s * NBG) * LBITS for s in range(BM))
    CB2 = float(1 << (2 * _d["bshift"]))
    BU = tuple(_d["u"])
    BTAIL = _d.get("tail")
    if BTAIL:
        T1W = tuple(tuple(r) for r in BTAIL["t1w"])
        T1B = tuple(BTAIL["t1b"])
        T2W = tuple(BTAIL["t2w"][0])
        T2B = BTAIL["t2b"][0]

_PIECES = "PNBRQKpnbrqk"
# Own-king buckets per perspective.  B == 1 is the plain net; B > 1 nets
# condition the first-layer rows on each side's own king bucket, and a king
# move that crosses a bucket boundary rebuilds the accumulator from scratch
# (rare, ~32 adds).  ROWS[pf][kb][piece][square] is the packed contribution
# of one man on one square, in the frame of the side to move, for the
# ABSOLUTE king-bucket pair kb = B*bucket(white) + bucket(black); pf==1
# shares the very same int objects.  For B == 1 there is a single table and
# kb is always 0, so the hot path is unchanged.
B = _d.get("B", 1)


assert B in (1, 4, 8), "unknown king-bucket scheme B=%r" % B


def kbucket(s):
    """Bucket of a perspective's OWN king on its OWN-frame square.  The
    scheme is selected by B and must match packed/pnet.py (verify.py checks
    the composition): B == 4 is back-two-ranks vs advanced times queenside
    vs kingside; B == 8 refines the file split to pairs (ab/cd/ef/gh)."""
    r, f = divmod(s, 10)
    if B == 8:
        return (r <= 7) * 4 + (f - 1) // 2
    return (r <= 7) * 2 + (f >= 5)


if B == 1:
    _rows0 = _d["rows"]
    _rows1 = {p: [_rows0[p.swapcase()][119 - s] for s in range(120)]
              for p in _PIECES}
    ROWS = ([_rows0], [_rows1])
else:
    _r0, _r1 = [], []
    for _bw in range(B):
        for _bb in range(B):
            _c0 = {p: [_d["rowsW"][_bw][p][s] + _d["rowsB"][_bb][p][s]
                       for s in range(120)] for p in _PIECES}
            _c1 = {p: [_c0[p.swapcase()][119 - s] for s in range(120)]
                   for p in _PIECES}
            _r0.append(_c0)
            _r1.append(_c1)
    ROWS = (_r0, _r1)
del _d


from math import tanh as _tanh


def _mlp(z):
    # term order matches pnet._mlp exactly: float addition is not
    # associative and engine == pnet must hold to the last bit
    acc = 0.0
    for b1, row, w2 in zip(T1B, T1W, T2W):
        for wk, zk in zip(row, z):
            b1 += wk * zk
        acc += w2 * _tanh(b1)
    return acc + T2B


def nn_cp(acc, pf, cnt=0):
    """Clipped centipawn output of the packed net, mover's point of view:
    SWAR clamp, two modular horizontal sums, and -- for extended nets only --
    bilinear group convolutions, the odd tail and the phase scale, in floats
    (pnet is the verified reference; antisymmetry is exact on both paths:
    integers by symmetric shift-rounding, floats by IEEE sign-symmetry and
    truncation toward zero)."""
    m = ((acc & MLO) >> 14) * ONES              # lane >= 0 ?
    y = ((acc & m) | MLO) - MLO                 # relu
    for T in MTS:                               # convex piecewise-linear:
        x = acc - T                             #   y = sum_i relu(a - t_i)
        m = ((x & MLO) >> 14) * ONES
        y += ((x & m) | MLO) - MLO
    m = (((MGH - y) & MH) >> VBITS) * ONES      # lane <= G_k ?
    y = (y & m) | (MGP & (m ^ MVAL))            # ...capped at G_k
    # 2^16 == 1 (mod 2^16-1), so each block's residue IS its lane sum
    v = (y & MASKLO) % M16 - ((y >> HALF) & MASKLO) % M16
    if pf:
        v = -v
    if not EXT:
        # round TOWARDS ZERO: floor does not commute with negation, and the
        # search reaches a position both by rotate() (negates) and move()
        # (recomputes) -- symmetric rounding keeps them in exact agreement
        v = (v >> SHIFT) if v >= 0 else -((-v) >> SHIFT)
        return -CLAMP if v < -CLAMP else (CLAMP if v > CLAMP else v)
    d = float(v) / (1 << SHIFT)
    if NB:
        A = [((y >> o) & BGMASK) % M16 for o in BOFFX]
        Bv = [((y >> o) & BGMASK) % M16 for o in BOFFY]
        if pf:
            A, Bv = Bv, A
        h = [0] * BM
        f = [0] * BM
        for s in range(BM):
            a, b = A[s], Bv[s]
            for t in range(BM):
                g = (s + t) % BM
                h[g] += A[t] * a - Bv[t] * b
                f[g] += a * Bv[t]
        for g in range(BM):
            h[g] /= CB2
            d += BU[g] * h[g]
        if BTAIL:
            z = [d / 300.0] + [v / 100.0 for v in h] + [v / 100.0 / CB2 for v in f]
            zn = [-x for x in z[:1 + BM]] + z[1 + BM:]
            d += 150.0 * (_mlp(z) - _mlp(zn))
    if PHASE_S:
        b = (cnt - 1) * len(PHASE_S) // 32
        d *= PHASE_S[min(max(b, 0), len(PHASE_S) - 1)]
    d = -CLAMP if d < -CLAMP else (CLAMP if d > CLAMP else d)
    return int(d)                       # trunc toward zero: symmetric

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################

# With xz compression this whole section takes 652 bytes.
# That's pretty good given we have 64*6 = 384 values.
# Though probably we could do better...
# For one thing, they could easily all fit into int8.
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

# Mop-up: once one side is down to a bare king, the midgame king table
# gives the search no progress signal - every shuffle scores alike and
# won KRK/KQK endings drift to the 50-move horizon. Swap in a formulaic
# centralization gradient (value falls with distance from the center):
# because both kings share the table (zero-sum via rotation), the same
# swap simultaneously rewards driving the bare king to the edge and
# marching our own king up - the two halves of classical mop-up.
K_MID = pst["K"]
K_END = (0,) * 20 + sum(
    ((0,) + tuple(
        piece["K"] + 70 - 10 * (abs(2 * rank - 7) + abs(2 * file - 7))
        for file in range(8)) + (0,)
     for rank in range(8)), ()) + (0,) * 20

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
# The margin must cover the largest army a kingless side can face: nine
# queens (8 promotions + original) plus 2R+2B+2N with piece-square
# bonuses sums to 11749, which the old 10-queen margin (9290) missed by
# 2459 - a kingless position could evade the king-gone check below.
# 13 queens covers it with slack (formal/Sunfish/EvalBounds.lean proves
# both the leak and the repair).
MATE_LOWER = piece["K"] - 13 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]
# Every static evaluation must stay within [-MATE_UPPER, MATE_UPPER]: the
# transposition table's fresh entries assume it (formal/Sunfish/Tricks.lean,
# `Bounded`). The tables above guarantee it; keep it true if you change them.

# Constants for tuning search
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15
# Max entries kept in each transposition table, roughly 1GB per million.
# Python dicts keep insertion order, so we cheaply evict the oldest entry
# when full (see issue #95).
TABLE_SIZE = 10**6

# minifier-hide start
opt_ranges = dict(
    QS = (0, 300),
    QS_A = (0, 300),
    EVAL_ROUGHNESS = (0, 50),
    TABLE_SIZE = (10**4, 10**8),
)
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score ps wc bc ep kp acc pf kb cnt")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation: ps + the clipped net residual
    ps -- the piece-square part of the score alone, kept exactly incremental
          so that value(move) below stays an exact delta of it
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    acc -- the packed NNUE accumulator (one big int, 2N + 2*nb lanes)
    pf -- perspective flag: which of the two lane blocks is the mover's
    kb -- combined king-bucket index B*bucket(white) + bucket(black), in
          ABSOLUTE colours (0 for plain B == 1 nets)
    cnt -- number of men on the board, kept incrementally (captures and en
           passant decrement it); feeds the phase output scale

    score/ps/acc/pf/kb/cnt are all functions of the other fields, so
    identity -- what the transposition table, the killer table and the
    repetition set key on -- deliberately ignores them.  Keeping the
    accumulator out of __hash__ also keeps hashing off the big int, which
    would otherwise cost more than the evaluation it feeds.
    """

    def __hash__(self):
        return hash((self.board, self.wc, self.bc, self.ep, self.kp))

    def __eq__(self, o):
        return (self.board == o.board and self.ep == o.ep and self.kp == o.kp
                and self.wc == o.wc and self.bc == o.bc)

    def __ne__(self, o):
        return not self.__eq__(o)

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        # NB: `in <literal-str>` is ~30% faster than the equivalent .isupper() /
        # .isspace() / .islower() method calls in CPython; this matters because
        # these checks run millions of times per search.
        for i, p in enumerate(self.board):
            if p not in "PNBRQK":
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q in " \nPNBRQK":
                        break
                    # Pawn move, double move and capture
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                            #and j != self.ep and abs(j - self.kp) >= 2
                        ):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q in "pnbrqk":
                        break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove.
        The accumulator is unchanged; only which block is "ours" flips, and
        that flips the sign of the net residual exactly as it flips ps."""
        return Position(
            self.board[::-1].swapcase(), -self.score, -self.ps, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
            self.acc, self.pf ^ 1, self.kb, self.cnt,
        )

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        cnt = self.cnt - (q != ".")
        ps = self.ps + self.value(move)
        # Every board mutation below is mirrored by a packed row delta. The
        # rows are exact per-lane integers, so the order does not matter:
        # only the final accumulator has to sit inside the lane range.
        kb = self.kb
        row = ROWS[self.pf][kb]
        acc = self.acc + row[p][j] - row[p][i]
        if q != ".":
            acc -= row[q][j]
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                r = A1 if j < i else H1
                board = put(board, r, ".")
                board = put(board, kp, "R")
                acc += row["R"][kp] - row["R"][r]
            if B > 1:
                # Mover's own bucket may change; the frame is the mover's
                # own, so kbucket(j) IS the own-frame bucket.  The mover is
                # absolute white iff pf == 0.
                nb = kbucket(j)
                ob = kb // B if self.pf == 0 else kb % B
                if nb != ob:
                    kb = nb * B + kb % B if self.pf == 0 else kb - ob + nb
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
                acc += row[prom][j] - row["P"][j]
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
                acc -= row["p"][j + S]
                cnt -= 1
        # A king move across a bucket boundary invalidates every one of our
        # own-perspective lanes at once: rebuild from the (still mover-
        # oriented) final board with the new bucket's table.  Rare, ~32 adds.
        if kb != self.kb:
            row = ROWS[self.pf][kb]
            acc = ACC_BASE
            for s, c in enumerate(board):
                if c in _PIECES:
                    acc += row[c][s]
        # We rotate the returned position, so it's ready for the next player
        pf = self.pf ^ 1
        return Position(board[::-1].swapcase(), -ps + nn_cp(acc, pf, cnt), -ps,
                        bc, wc, 119 - ep if ep else 0, 119 - kp if kp else 0,
                        acc, pf, kb, cnt)

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        # Castling
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        # Special pawn stuff
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
        return score


###############################################################################
# Search logic
###############################################################################

# Raised inside the search when the wall-clock deadline passes
class Stop(Exception): pass


# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score, self.tp_move, self.history = {}, {}, set()
        self.nodes, self.deadline = 0, None

    def bound(self, pos, gamma, depth, root=False):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) """
        self.nodes += 1
        # Enforce the time budget inside the search: iteration boundaries can
        # be seconds apart on slow hardware, this is checked every ~2k nodes.
        if self.deadline is not None and self.nodes % 2048 == 0 \
                and time.time() > self.deadline:
            raise Stop

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        # This reads `ps`, the piece-square part, NOT the evaluated score: the
        # sentinel has to mean "the king is literally gone", and a net residual
        # of up to CLAMP could otherwise lift a kingless position back over the
        # threshold and hide the capture. On ps the test is bit-for-bit
        # classic's.
        if pos.ps <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # Driver probes (the search root, and IID below) are UNSTORED: they
        # skip the table in both directions, the repetition-0 and the null
        # option, and store nothing - so every entry in the table describes
        # ONE value function, determined by (pos, depth) alone, and the key
        # needs no flag. (The root's own entry was provably dead weight: the
        # driver picks each gamma strictly inside its bracket, which is the
        # same two numbers the entry held.)
        entry = Entry(-MATE_UPPER, MATE_UPPER)
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma: return entry.lower
            if entry.upper < gamma: return entry.upper

        # Let's not repeat positions. We don't chat
        # - at the root (a driver probe) since it is in history, but not a draw.
        # - at depth=0, since it would be expensive and break "futility pruning".
        if not root and depth > 0 and pos in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        # If depth == 0 we only try moves with high intrinsic score (captures and
        # promotions). Otherwise we do all moves. This is called quiescent search.
        val_lower = QS - depth * QS_A

        def moves():
            # First try not moving at all, i.e. the null move. Two caveats,
            # both now measured and consciously accepted (formal/README.md):
            # - Zugzwang: passing may be an un-chess-like free tempo; the
            #   score guard below limits exposure. Re-adding the classic
            #   majors-only guard measured -5 +/- 30 ELO: a wash, declined.
            # - King-capturable nodes: a null cutoff can mask the MATE_UPPER
            #   sentinel that mate/stalemate detection consumes. The killer
            #   path is provably safe (formal/Sunfish/Killer.lean); the null
            #   path is the one remaining exception. Closing it measured
            #   -28 ELO (move-scan check) and -12 with no benchmark gain
            #   (killer-only check): declined, exception tolerated.
            # Null move in QS (a search-verified stand-pat) measured
            # -13 +/- 34 ELO: declined.
            # The zugzwang guard also reads ps, so the gate opens on exactly
            # the positions it opens on in classic rather than on a threshold
            # jittered by the net residual.
            if depth > 2 and not root and abs(pos.ps) < 500 and any(
                    c in pos.board for c in "RBNQ"):
                # A pass claiming a mate-band value is redundant (if passing
                # wins the king, capturing it is a real move too) and can be a
                # false mate that poisons tp_move - suppress it. (A1;
                # Stalemate.lean: a1_unfixed_not_sound / a1_fix_repairs.)
                score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)
                yield None, score if score < MATE_LOWER else -MATE_UPPER

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # Look for the strongest move from last time, the hash-move.
            killer = self.tp_move.get(pos)

            # If there isn't one, try to find one with a more shallow search.
            # This is known as Internal Iterative Deepening (IID). The probe
            # runs as a driver probe (root=True): no null cutoff that would
            # end it without storing a move, no repetition truncation, and
            # no table entry under deviant semantics.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, root=True)
                killer = self.tp_move.get(pos)

            # Only play the move if it would be included at the current val-limit,
            # since otherwise we'd get search instability.
            # We will search it again in the main loop below, but the tp will fix
            # things for us.
            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Then all the other moves
            for val, move in sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True):
                # Quiescent search
                if val < val_lower:
                    break

                # If the new score is less than gamma, the opponent will for sure just
                # stand pat, since ""pos.score + val < gamma === -(pos.score + val) >= 1-gamma""
                # This is known as futility pruning.
                if depth <= 1 and pos.score + val < gamma:
                    # Need special case for MATE, since it would normally be caught
                    # before standing pat.
                    yield move, pos.score + val if val < MATE_LOWER else MATE_UPPER
                    # We can also break, since we have ordered the moves by value,
                    # so it can't get any better than this.
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        # Run through the moves, shortcutting when possible
        # best_real sees only real-move yields: the mate/stalemate sentinel
        # below must not be masked by the null option (a pass yielding a
        # normal material score hid genuine stalemates in pawn endings - A1).
        best = best_real = -MATE_UPPER
        for move, score in moves():
            best, best_real = max(best, score), max(best_real, score) if move else best_real
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos] = move
                    # Never evict the current search root: its killer is the
                    # answer the go loop plays, and once the table churns
                    # more than TABLE_SIZE stores in one deep probe, FIFO
                    # would age it out MID-SEARCH -- a later deep fail-low
                    # probe then stores whatever capture sorts first and a
                    # timeout plays it (classic's Qxc6 giveaway class).
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actually a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.

        # We fix this problem another way: the sorted move loop above always
        # yields a king capture first when one exists (it has the highest
        # value), so barring the exceptions below, bound returns MATE_UPPER
        # whenever the king is capturable. If we see best == -MATE_UPPER here,
        # every reply lost the king (or there were no replies): we are either
        # mated or stalemated, and it suffices to check whether we're in check.

        # Exceptions and non-exceptions (formal/Sunfish/Stalemate.lean and
        # Killer.lean make these precise):
        # - The killer path is provably safe: a killer stored at a king-
        #   capturable position is always itself a king capture (Killer.lean,
        #   boundKill_spec), so a killer cutoff still returns the sentinel.
        #   This relies on tp_move being keyed by exact position and on king
        #   captures sorting first; change either and the proof breaks.
        # - The null-move yield is the one remaining path that can end the
        #   loop below MATE_UPPER at a king-capturable node (it yields
        #   move=None, so nothing corrects it). Only the abs(pos.score) < 500
        #   zugzwang guard limits this; deeper search corrects the rare
        #   cases that slip through.
        # - At low depths we may have pruned all legal moves, so sunfish may
        #   report "mate" and retract it after more search. That's fair.

        # If the quiescent val-limit never skipped a move (no legal move falls below val_lower), then
        # exhausting the generator means every legal move was searched and lost the
        # king, so the mate/stalemate test is sound at ANY depth:
        # - best == -MATE_UPPER implies the generator was exhausted, since the
        #   consumption break needs best >= gamma and gamma > -MATE_UPPER here
        #   (a call with gamma <= -MATE_UPPER is answered by the entry cutoff).
        # - the futility break always yields a value > -MATE_UPPER first, so it
        #   cannot leave best == -MATE_UPPER with moves unseen.
        # - at depth == 0 stand-pat yields pos.score > -MATE_LOWER, so the
        #   correction still never fires in plain QS nodes.
        # Without this, a depth <= 2 node above a stalemate returns +MATE_UPPER for
        # the stalemating move, poisoning tp_move at the root (Qc4?? on lichess).
        if best < gamma and best_real == -MATE_UPPER and all(
                pos.value(m) >= val_lower for m in pos.gen_moves()):
            flipped = pos.rotate(nullmove=True)
            # Hopefully this is already in the TT because of null-move
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            # Mated scores as -MATE_LOWER, not -MATE_UPPER: the latter stays a
            # reserved sentinel meaning "the king is literally capturable",
            # which the best == -MATE_UPPER test above depends on. A parent
            # whose child is merely mated must not look king-capturable itself.
            best = -MATE_LOWER if in_check else 0

        # Table part 2. Every search decision is gamma-independent, so all
        # bounds target one value function determined by the key and stored
        # entries can never contradict each other (lower > upper). A change
        # that lets gamma (or any key-external state) select between
        # incomparable evaluations of a move breaks this - that is a bug,
        # not a configuration; see formal/README.md.
        if not root:
            self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        if len(self.tp_score) > TABLE_SIZE:
            del self.tp_score[next(iter(self.tp_score))]

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes, self.history = 0, set(history)
        self.tp_score.clear()
        # Table choice is fixed for the whole search (and tp_score is
        # cleared above), so every bound targets one value function.
        pos = self.root = history[-1]
        bare = sum(c.isupper() for c in pos.board) == 1 or sum(c.islower() for c in pos.board) == 1
        pst["K"] = K_END if bare else K_MID

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            # This probe range also keeps the stalemate correction in bound()
            # sound: it is only valid for windows inside (-MATE_UPPER,
            # MATE_UPPER] (formal/Sunfish/Stalemate.lean proves both
            # directions). Widen this range and stalemates break silently.
            lower, upper = -MATE_LOWER, MATE_LOWER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma: lower = score
                if score < gamma: upper = score
                yield depth, gamma, score, self.tp_move.get(pos)
                gamma = (lower + upper + 1) // 2


###############################################################################
# UCI User interface
###############################################################################

# parse/render/hist live at module level: tools/uci.py (and the tests)
# reach them as engine-module attributes, and main() uses hist before its
# own body would define it.
def parse(c): return A1 + ord(c[0]) - ord("a") - 10 * (int(c[1]) - 1)
def render(i): return chr((i - A1) % 10 + ord("a")) + str(1 - (i - A1) // 10)

def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position (and its accumulator) from scratch. `board` is
    already in the side-to-move's orientation."""
    ps = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
             for i, p in enumerate(board) if p.isalpha())
    kb = 0
    if B > 1:
        own, opp = kbucket(board.index("K")), kbucket(119 - board.index("k"))
        kb = own * B + opp if pf == 0 else opp * B + own
    acc = ACC_BASE
    row = ROWS[pf][kb]
    cnt = 0
    for i, p in enumerate(board):
        if p in _PIECES:
            acc += row[p][i]
            cnt += 1
    return Position(board, ps + nn_cp(acc, pf, cnt), ps, wc, bc, ep, kp,
                    acc, pf, kb, cnt)


hist = [from_board(initial)]


def main():
    # minifier-hide start
    # Development checkout: use the full-featured UCI interface in
    # tools/ (pondering, Hash option, spec-complete go parsing). An
    # installed or packed sunfish has no tools/ and falls through to
    # the built-in loop below, which is all a GUI needs.
    try:
        import sys, tools.uci
        return tools.uci.run(sys.modules[__name__], hist[-1])
    except ImportError:
        pass
    # minifier-hide end

    searcher = Searcher()
    while True:
        args = input().split()
        if args[0] == "uci":
            print("id name", version)
            print("uciok")

        elif args[0] == "isready":
            print("readyok")

        elif args[0] == "quit":
            break

        elif args[:2] == ["position", "startpos"]:
            del hist[1:]
            for ply, move in enumerate(args[3:]):
                i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
                if ply % 2 == 1:
                    i, j = 119 - i, 119 - j
                hist.append(hist[-1].move(Move(i, j, prom)))

        elif args[0] == "go":
            # The times may come in any order and combination, e.g. "go wtime 100 btime 100"
            times = dict(zip(args[1::2], map(int, args[2::2])))
            side = "wb"[len(hist) % 2 == 0]
            wtime, winc = times.get(side + "time", 60000), times.get(side + "inc", 0)
            think = min(wtime / 40 + winc, wtime / 2 - 1000)
            think = times.get("movetime", think) / 1000

            start = time.time()
            # Hard in-search deadline: iteration boundaries can be seconds
            # apart at deep depths, so the soft 0.8*think break alone can
            # overrun arbitrarily and forfeit on time.  max(.05) keeps a
            # degenerate clock from stopping before any move exists.
            searcher.deadline = start + max(think, .05)
            # best is COMMITTED only when a depth completes (its bracket
            # converged): a mid-depth fail-high can come from a deep
            # fail-low dive probe at an absurd gamma and is only a
            # candidate (classic's Qxc6 giveaway class).
            best, cand, d0 = None, None, 1
            try:
                for depth, gamma, score, move in searcher.search(hist):
                    if depth > d0:
                        best, d0 = cand or best, depth
                    if score >= gamma:
                        i, j = move.i, move.j
                        if len(hist) % 2 == 0:
                            i, j = 119 - i, 119 - j
                        cand = render(i) + render(j) + move.prom.lower()
                        print("info depth", depth, "score cp", score, "pv", cand)
                    if (best or cand) and time.time() - start > think * 0.8:
                        break
            except Stop:
                pass

            print("bestmove", best or cand or '(none)')


if __name__ == "__main__":
    main()
