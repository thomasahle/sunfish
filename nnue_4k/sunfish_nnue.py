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

import json as _json
from base64 import b64decode as _b64

# The .sfnn net format: one JSON header line (no code execution -- pickles
# never ship), then base64 tokens, one per big int, in canonical order:
# base, gp, ts, then the piece rows ("PNBRQKpnbrqk" x 120 squares; kb nets
# store rowsW buckets then rowsB buckets).
NET_PATH = os.environ.get("SF_NET", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "net128kb8.sfnn"))
_h, _r = open(NET_PATH).read().split("\n", 1)
_d = _json.loads(_h)
_it = (int.from_bytes(_b64(t), "little", signed=True) for t in _r.split())
_PIECES = "PNBRQKpnbrqk"
_row = lambda: {p: [next(_it) for _ in range(120)] for p in _PIECES}
ACC_BASE, MGP = next(_it), next(_it)
MTS = tuple(next(_it) for _ in range(_d.pop("nts", 0)))
B = _d.get("B", 1)
assert B in (1, 4, 8, 16), "unknown king-bucket scheme B=%r" % B
# All B*B absolute bucket pairs combined once (a single entry for B == 1);
# the pf==1 view relabels the very same int objects.
if B == 1:
    _r0 = [_row()]
else:
    _w = [_row() for _ in range(B)]
    _b = [_row() for _ in range(B)]
    _r0 = [{p: [_w[bw][p][s] + _b[bb][p][s] for s in range(120)]
            for p in _PIECES} for bw in range(B) for bb in range(B)]
ROWS = (_r0, [{p: [c[p.swapcase()][119 - s] for s in range(120)]
               for p in _PIECES} for c in _r0])

NLANE = 2 * _d["N"] + 2 * _d.get("nb", 0)
LBITS = 16
VBITS = 15
ONES = (1 << VBITS) - 1
HALF = _d["N"] * LBITS            # bit offset of the second lane block
SHIFT = _d["shift"]
CLAMP = _d["clampcp"]
BASE = _d.get("base_kind", "pst")   # score base: pst | mat (mat = dev only)


MH = sum(1 << (VBITS + LBITS * i) for i in range(NLANE))   # guard bits
MLO = MH >> 1                     # offset-binary zero, the bit-14 probe
MVAL = MH - (MH >> VBITS)         # value bits (each lane 2^15 - 1)
MGH = MGP | MH                    # per-lane activation ceilings, guard set
MASKLO = (1 << HALF) - 1
M16 = (1 << LBITS) - 1

# ---- extensions: bilinear lanes, narrow odd tail, phase output scale.
# Their read-out runs in floats; exact antisymmetry survives because every
# float input is exactly negated or exactly invariant under perspective
# swap, IEEE arithmetic is sign-symmetric, and the final truncation rounds
# toward zero (packed/pnet.py ext_cp is the verified reference).
NB = _d.get("nb", 0)              # bilinear lanes per perspective
BM = _d.get("m", 4)               # bilinear groups
PHASE_S = tuple(_d.get("phase_s") or ()) or None
RFF = _d.get("rff", 0)            # phase-sketch (angle) lanes per perspective
# rff angle fields are 32-BIT, above the 16-bit lane grid: present-piece
# quanta sums stay < 2^21 (no overflow), removals subtract exactly what
# was added (never negative) -- plain adds, no wrap machinery, and the
# mod-2^15 circle is the read-out mask.
EXT = bool(NB or PHASE_S or RFF)
# minifier-hide start
# The extension machinery is DEV-BUILD ONLY: the 4k artifact ships the
# pure-int family (its net is pure-int too) and refuses ext nets loudly
# below.  One source file, two shapes -- the same pattern as tools/.
_FULL = 1
if NB:
    NBG = NB // BM                # lanes per group (contiguous runs)
    BGMASK = (1 << (NBG * LBITS)) - 1
    BOFFX = tuple((2 * _d["N"] + s * NBG) * LBITS for s in range(BM))
    BOFFY = tuple(o + NB * LBITS for o in BOFFX)
    CB2 = float(1 << (2 * _d["bshift"]))
    BU = _d["u"]
    BTAIL = _d.get("tail")
    if BTAIL:
        T1W, T1B = BTAIL["t1w"], BTAIL["t1b"]
        T2W, T2B = BTAIL["t2w"][0], BTAIL["t2b"][0]
if RFF:
    _FB = (2 * _d["N"] + 2 * NB) * LBITS
    ROFFX = tuple(_FB + 32 * k for k in range(RFF))
    ROFFY = tuple(_FB + 32 * (RFF + k) for k in range(RFF))
    RW = tuple(_d["rw"])
# minifier-hide end
if (EXT or MTS) and "_FULL" not in dir():
    raise SystemExit("extended/segmented net: use the repo engine")
# The base tables ship IN the net file: they are eval data exactly like
# the packed rows (classic pst incl. piece values, padded 120-wide, plus
# the bare-king mop-up table; material-base nets simply carry flat
# tables -- no engine branch needed).
pst = {p: tuple(v) for p, v in _d["pst"].items()}
K_MID, K_END = pst["K"], tuple(_d["kend"])

# Own-king buckets per perspective.  B == 1 is the plain net; B > 1 nets
# condition the first-layer rows on each side's own king bucket, and a king
# move that crosses a bucket boundary rebuilds the accumulator from scratch
# (rare, ~32 adds).  ROWS[pf][kb][piece][square] is combined at load above.
def kbucket(s):
    """Bucket of a perspective's OWN king on its OWN-frame square.  The
    scheme is selected by B and must match packed/pnet.py (verify.py checks
    the composition): B == 4 is back-two-ranks vs advanced times queenside
    vs kingside; B == 8 refines the file split to pairs (ab/cd/ef/gh)."""
    r, f = divmod(s, 10)
    if B == 16:
        return (9 - r) // 2 * 4 + (f - 1) // 2
    return (r <= 7) * (B >> 1) + ((f - 1) // 2 if B == 8 else (f >= 5))


del _d


def nn_cp(acc, pf, bd=""):
    """Clipped centipawn output of the packed net, mover's point of view:
    SWAR clamp, two modular horizontal sums, and -- for extended nets only,
    dev builds only -- the float tail via _ext (pnet is the verified
    reference; antisymmetry is exact on both paths: integers by symmetric
    shift-rounding, floats by IEEE sign-symmetry and truncation)."""
    m = ((acc & MLO) >> 14) * ONES              # lane >= 0 ?
    y = ((acc & m) | MLO) - MLO                 # relu
    # minifier-hide start
    for T in MTS:                               # convex piecewise-linear:
        x = acc - T                             #   y = sum_i relu(a - t_i)
        m = ((x & MLO) >> 14) * ONES
        y += ((x & m) | MLO) - MLO
    # minifier-hide end
    m = (((MGH - y) & MH) >> VBITS) * ONES      # lane <= G_k ?
    y = (y & m) | (MGP & (m ^ MVAL))            # ...capped at G_k
    # 2^16 == 1 (mod 2^16-1), so each block's residue IS its lane sum
    v = (y & MASKLO) % M16 - ((y >> HALF) & MASKLO) % M16
    if pf:
        v = -v
    # minifier-hide start
    if EXT:
        return _ext(y, v, pf, bd, acc)
    # minifier-hide end
    # round TOWARDS ZERO: floor does not commute with negation, and the
    # search reaches a position both by rotate() (negates) and move()
    # (recomputes) -- symmetric rounding keeps them in exact agreement
    v = (v >> SHIFT) if v >= 0 else -((-v) >> SHIFT)
    return -CLAMP if v < -CLAMP else (CLAMP if v > CLAMP else v)


# minifier-hide start
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


def _rff_term(acc, pf):
    from math import cos
    TAU = 6.283185307179586 / 32768.0
    offA, offB = (ROFFX, ROFFY) if pf == 0 else (ROFFY, ROFFX)
    return sum(w * (cos(TAU * ((acc >> oa) & 32767))
                    - cos(TAU * ((acc >> ob) & 32767)))
               for w, oa, ob in zip(RW, offA, offB))


def _ext(y, v, pf, bd, acc):
    """Extended evaluation tail (dev builds; the 4k artifact refuses ext
    nets at load).  v is already mover-signed; the phase piece count is
    derived from the board -- dev-only cost, no state to drift."""
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
            z = [d / 300.0] + [x / 100.0 for x in h] + [x / 100.0 / CB2 for x in f]
            zn = [-x for x in z[:1 + BM]] + z[1 + BM:]
            d += 150.0 * (_mlp(z) - _mlp(zn))
    if RFF:
        d += _rff_term(acc, pf)
    if PHASE_S:
        # 64 squares minus the empties: one C-level count instead of a
        # 120-step genexpr, and only for nets that actually have phase
        # buckets (it used to run unconditionally -- a third of _ext's
        # cost, spent on nothing, for every phase-less ext net).
        cnt = 64 - bd.count(".")
        b = (cnt - 1) * len(PHASE_S) // 32
        d *= PHASE_S[min(max(b, 0), len(PHASE_S) - 1)]
    d = -CLAMP if d < -CLAMP else (CLAMP if d > CLAMP else d)
    return int(d)                       # trunc toward zero: symmetric
# minifier-hide end

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################

# With xz compression this whole section takes 652 bytes.
# That's pretty good given we have 64*6 = 384 values.
# Though probably we could do better...
# For one thing, they could easily all fit into int8.
# MATE values derive from the classic piece values (K=60000, Q=929);
# the tables themselves ride in the net file (see the loader above).
MATE_LOWER = 60000 - 13 * 929
MATE_UPPER = 60000 + 10 * 929

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
# Every static evaluation must stay within [-MATE_UPPER, MATE_UPPER]: the
# transposition table's fresh entries assume it (formal/Sunfish/Tricks.lean,
# `Bounded`). The tables above guarantee it; keep it true if you change them.

# Constants for tuning search
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15
# Probes per depth before the MTD driver gives up and commits what it has.
# The stable engine provably needs <= 15; this engine may not, so the bound
# is enforced rather than assumed.
PROBE_CAP = 40
# Late move reduction: reduce quiet moves whose static value is below this,
# once past the first few in the sorted list. 0 disables (classic parity).
LMR = 60
# Minimum sunfish_ui driver version this engine will run against; see the
# check in main(). Raise it in the same commit that bumps DRIVER_VERSION.
REQUIRED_DRIVER = 2
# Reverse futility pruning margin per ply. 0 disables.
# HELD AT 0: implemented and gated, but it INTERACTS with LMR. On the
# mate-in-1 suite, baseline finds 5/8, LMR alone 5/8, RFP alone 5/8 --
# and LMR+RFP together only 3/8, with the guards reporting brackets like
# "lower 47923 upper -143" (a mate-band claim against a negative upper).
# Each is harmless alone and the pair is not, which is exactly why these
# land one at a time with their own screen. RFP gets enabled and screened
# on top of LMR once LMR's own screen reports, with the mate suite as an
# acceptance gate rather than an afterthought.
RFP = 0
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


class Position(namedtuple("Position", "board score ps wc bc ep kp acc pf kb")):
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

    score/ps/acc/pf/kb are all functions of the other fields, so
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
            self.acc, self.pf ^ 1, self.kb,
        )

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
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
        wc = (wc[0] and i != A1, wc[1] and i != H1)
        bc = (bc[0] and j != H8, bc[1] and j != A8)
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
        return Position(board[::-1].swapcase(), -ps + nn_cp(acc, pf, board), -ps,
                        bc, wc, 119 - ep if ep else 0, 119 - kp if kp else 0,
                        acc, pf, kb)

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q in "pnbrqk":
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

    def king_capture(self):
        """The move that takes the opponent king, if any - i.e. the proof
        that this position was reached by an illegal move. Same test as
        gen_moves/value: the target is the king, or within one of the
        king-passant square (kp == 0 is safe: targets are >= A8 > 1).
        Serves double duty: found from a position it is the sentinel
        witness the search substitutes for a virtual cutoff; found from
        the null-rotation it says the side to move is in check."""
        return next((m for m in self.gen_moves()
                     if self.board[m.j] == "k" or abs(m.j - self.kp) < 2), None)


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
        self.nodes, self.deadline = 0, 1 << 63
        # minifier-hide start
        self.node_cap = 1 << 62          # testing only; see bound()
        # minifier-hide end

    def bound(self, pos, gamma, depth, root=False):
        """ Let s* be the score of the sub-tree from pos at this depth, as
            a function of (pos, depth) alone. This includes null moves and
            QS pruning, and global parameters like self.history that don't
            change during search. (Things that change, like tp_move or gamma,
            are not allowed to change the sub-tree and value of s*.)

            It is assumed 1 - MATE_UPPER < gamma <= MATE_UPPER.

            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound)

            Note, bound() is not guaranteed to be deterministic: stored values
            in self.tp_score may be used to return a bound that is not the best
            possible, but it is guaranteed to be valid according to the rules above.

            On top of the bound, three exact promises:
            - our own king already captured: r = -MATE_UPPER.
            - if depth >= 1:
                - if the opponent king capturable: r = MATE_UPPER
                  (note this is stronger than just gamma <= r <= s*.)
                - if mate/stalemate returns the exact -MATE_LOWER / 0.
            - if gamma <= r, tp_move[pos] will hold a legal move achieving r.
            """

        self.nodes += 1
        # minifier-hide start
        # Node budget enforced INSIDE the search, at the same granularity as
        # the deadline. Checking a node cap only between completed depths
        # rewards whichever engine prunes LESS: its last iteration is bigger,
        # so it sails further past the cap. Measured at a 20000 cap, classic
        # (no LMR) reached 34742 nodes -- 1.74x -- against 26336 for the same
        # engine with LMR, a ~30% free advantage worth ~38 Elo. Testing-only:
        # the 4k rules mandate no node command, so the artifact carries none
        # of this.
        if self.nodes % 2048 == 0 and self.nodes > self.node_cap: raise Stop
        # minifier-hide end
        # Enforce the time budget inside the search: iteration boundaries can
        # be seconds apart on slow hardware, this is checked every ~2k nodes.
        if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        # This reads `ps`, the piece-square part, NOT the evaluated score:
        # the sentinel means "the king is literally gone", and a net residual
        # of up to CLAMP could otherwise lift a kingless position back over
        # the threshold and hide the capture. On ps the test is bit-for-bit
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
        # same two numbers the entry held.) At the root 'entry' stays
        # unbound - its only other reader is the store below, also skipped.
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma: return entry.lower
            if entry.upper < gamma: return entry.upper

            # Let's not repeat positions. We don't chat
            # - at the root (a driver probe) since it is in history, but not a draw.
            # - at depth=0, since it would be expensive and break "futility pruning".
            if depth > 0 and pos in self.history: return 0

            # REVERSE FUTILITY PRUNING (packed only). If the static score is
            # already a margin above gamma, assume no reply drags it below and
            # fail high without searching. This is a GAMMA-DEPENDENT cutoff --
            # gamma selects the value being bounded, which the stable contract
            # forbids outright -- so it lives here and never in classic.
            #
            # It also returns a fail-high with NO move stored, so this node
            # breaks the "gamma <= r implies tp_move holds a legal move"
            # promise. That is survivable because the promise's consumers
            # degrade safely: the parent stores the move that reached here (so
            # ITS promise holds), and the null-move verifier finding no
            # tp_move falls through to its boundary probe, i.e. it declines to
            # cut rather than cutting unsoundly.
            #
            # Guards, borrowed from the null move because they encode the same
            # assumption -- that passing is not better than moving: non-pawn
            # material on the board (zugzwang), and a score well inside the
            # mate band so the exact mate/stalemate promises are not the thing
            # being pruned away.
            if RFP and 0 < depth < 5 and pos.score - RFP * depth >= gamma \
                    and abs(pos.ps) < 500 and any(c in pos.board for c in "RBNQ"):
                return pos.score - RFP * depth

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # Look for the strongest move from earlier searches of this position.
            # See https://chessprogramming.org/Killer_Move for details.
            # We read this "killer move" before null-move in case it would get
            # evicted from the table or replaced with something else worse.
            killer = self.tp_move.get(pos)

            # First try not moving at all, i.e. the null move.
            # See https://chessprogramming.org/Null_Move for details.
            # The idea is that "doing nothing" is a lower bound on the score
            # of the position, but we have to be careful with zugzwang, where
            # passing is better than any move - the piece test guards that
            # (K+P endings). The score cap has a different role: in decided
            # positions every pass fails high, sound but lazy bounds that
            # crowd out the precision needed to convert (dropping the cap was
            # Elo-neutral over 900 games yet cost a mate-in-3 at the CI
            # fixed-depth floor). Both halves stay. No null at root, so we
            # can always return a move.
            if not root and depth > 2 and abs(pos.ps) < 500 and any(c in pos.board for c in "RBNQ"):
                score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)
                # A fail high is a virtual claim, and needs verification
                # before it may cut: if the king is capturable the capture is
                # substituted (the node must report the exact MATE_UPPER)
                proof = score >= gamma and (self.tp_move.get(pos) or pos.king_capture())
                if proof and pos.value(proof) >= MATE_LOWER:
                    yield proof, MATE_UPPER
                # a remaining mate-band claim is vacuous (if passing wins the
                # king, capturing it is a real move too). Otherwise one probe
                # at the band boundary is decisive both ways
                # (boundary_window_decisive): fail-low means the pass really
                # wins in the band - vetoed by omission - and fail-high
                # certifies the value sub-band, letting the cutoff stand
                # with no chess assumption (the premise it replaced is false
                # in real chess: 8/6p1/6R1/k7/2K5/8/8/8 w).
                elif score < gamma or self.bound(pos.rotate(nullmove=True),
                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:
                    yield None, score

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else. (Note depth at root is always > 0.)
            if depth == 0:
                yield None, pos.score

            # Back to killer moves: This heuristic is so good, that if there
            # is no registered move, it's worth it to run a shallow search to find one.
            # See https://chessprogramming.org/Internal_Iterative_Deepening for detais.
            # This is known as Internal Iterative Deepening (IID). The probe
            # runs as a driver probe (root=True): no null cutoff that would
            # end it without storing a move, no repetition truncation, and
            # no table entry under deviant semantics.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, root=True)
                killer = self.tp_move.get(pos)

            # We only generate moves with an intrinsic score above some treshold
            # that decreases with depth. This is a generalization of Quiescent Search,
            # See https://chessprogramming.org/Quiescence_Search for details.
            val_lower = QS - depth * QS_A

            # Now finally play the killer move. But note that we have to respect
            # the QS lower bound, otherwise we would get search instability.
            # We will search it again in the main loop below, but the tp will
            # make this mostly free.
            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Then all the other moves
            # Quiescent search: only moves above the val-limit are admitted -
            # filtering BEFORE the sort skips sorting the sub-threshold tail
            # (most of the list at QS nodes), and is literally the model's
            # movesAbove form (formal/Sunfish/Stalemate.lean).
            # NOTE the iteration order is a soundness contract, not a
            # heuristic: the futility break below discards the rest of the
            # list, which is only valid when iteration descends in static
            # value. A history-credit order key tried here scrambled that
            # order and paid -449 Elo (ledger 5f5f34d); made sound, the
            # history table measured a 1.01 node ratio -- worthless.
            for cnt, (val, move) in enumerate(sorted(((v, m) for m in pos.gen_moves()
                                     if (v:=pos.value(m)) >= val_lower), reverse=True)):
                # If the new score is less than gamma, the opponent will for sure just
                # stand pat, since ""pos.score + val < gamma === -(pos.score + val) >= 1-gamma""
                # This is known as futility pruning.
                if depth <= 1 and pos.score + val < gamma:
                    # Need special case for MATE, since it would normally be caught
                    # before standing pat. A sub-mate futility yield estimates
                    # the child's stand-pat without searching it, so it is
                    # value evidence only, never legality evidence: it goes
                    # out as a virtual (None) yield - it can never cut (its
                    # score is below gamma by construction), and it must not
                    # set 'live' and mask the terminality correction below.
                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)
                    # We can also break, since we have ordered the moves by value,
                    # so it can't get any better than this.
                    break

                # LATE MOVE REDUCTION. The move list is sorted by static
                # value, so a quiet move arriving late is one the ordering
                # already judged unpromising: search it a ply shallower and
                # only pay full depth if it surprises us. This is the first
                # deliberate break of one-value-per-key -- the reduction
                # depends on cnt, which depends on ordering, which depends on
                # mutable table state, so the same position can now be
                # searched to different depths on different visits. The MTD
                # guards in search() exist for exactly this.
                # A null-window driver makes the verification re-search cheap:
                # the reduced search only needs to be trusted when it FAILS
                # LOW (score < gamma), and a fail-high is re-run at full depth.
                red = LMR and depth > 2 and cnt > 2 and val < LMR
                score = -self.bound(pos.move(move), 1 - gamma, depth - 2 if red else depth - 1)
                if red and score >= gamma:
                    score = -self.bound(pos.move(move), 1 - gamma, depth - 1)
                yield move, score

        # Run through the moves, shortcutting when score >= gamma.
        # live is True if we saw a legal (not null, score > -MATE_UPPER) move
        best, live = -MATE_UPPER, False
        for move, score in moves():
            best = max(best, score)
            live |= move is not None and score > -MATE_UPPER
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None and depth:
                    self.tp_move[pos] = move
                    # Never evict the current search root: its killer is the
                    # answer go_loop plays, and once the table churns more
                    # than TABLE_SIZE stores in one deep probe, FIFO would
                    # age it out MID-SEARCH - a later deep fail-low probe
                    # then stores whatever capture sorts first and a timeout
                    # plays it (three -5ish queen/piece giveaways in 145
                    # production games).
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break

        # If we didn't see any legal moves, it might just be that we failed
        # high on a null move and stopped searching, but it could also be that
        # we genuinely re in checkmate or stalemate. There's no way to know but
        # to check.
        if depth and not live and all(
                pos.move(m).king_capture() for m in pos.gen_moves()):
            # We can't move, but is it a checkmate or stalemate?
            best = -MATE_LOWER if pos.rotate(nullmove=True).king_capture() else 0

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

        # Bare-king endings: swap in the centralization gradient (packed's
        # own measured condition; classic keys on queens-off instead).
        # Both directions every search: table state must never outlive the
        # condition.
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
            lower, upper, probes = 1 - MATE_UPPER, MATE_UPPER, 0
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                # INSTABILITY GUARDS. This engine deliberately breaks
                # one-value-per-key (reductions, history, gamma-dependent
                # cutoffs), and MTD-bi has no real window in which to absorb
                # a contradiction: it only ever probes null windows and
                # bisects on the answers, assuming they bracket ONE function.
                # Two probes can now disagree (">= 100" at gamma=100, then
                # "< 50" at gamma=50), so:
                #  (a) tighten MONOTONICALLY -- max/min rather than plain
                #      assignment -- so a contradictory probe can never widen
                #      the bracket and spin the loop forever;
                #  (b) stop if the bracket CROSSES, committing the last
                #      self-consistent value rather than bisecting nonsense;
                #  (c) cap the probes per depth. We used to PROVE <= 15; that
                #      proof does not survive instability, so the invariant
                #      becomes a runtime check that trips loudly (dev builds)
                #      instead of silently ceasing to hold.
                if score >= gamma: lower = max(lower, score)
                else: upper = min(upper, score)
                probes += 1
                yield depth, gamma, score, self.tp_move.get(pos)
                if lower > upper or probes > PROBE_CAP:
                    # minifier-hide start
                    if probes > PROBE_CAP:
                        print("info string MTD-GUARD probe cap hit: depth", depth,
                              "lower", lower, "upper", upper, flush=True)
                    if lower > upper:
                        print("info string MTD-GUARD bracket crossed: depth", depth,
                              "lower", lower, "upper", upper, flush=True)
                    # minifier-hide end
                    break
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
    for i, p in enumerate(board):
        if p in _PIECES:
            acc += row[p][i]
    return Position(board, ps + nn_cp(acc, pf, board), ps, wc, bc, ep, kp,
                    acc, pf, kb)


hist = [from_board(initial)]


def main():
    # minifier-hide start
    # Development checkout: use the full-featured UCI interface in
    # sunfish_ui/ (pondering, Hash option, spec-complete go parsing,
    # FEN, node limits). An installed or packed sunfish has no
    # sunfish_ui/ and falls through to the built-in loop below, which is
    # all the 4k rules require.
    try:
        import sys
        import inspect
        # NOTE this puts THIS FILE'S GRANDPARENT at the front of sys.path,
        # ahead of PYTHONPATH and the repo, so any stray sunfish_ui/ sitting
        # there wins the import. A stale copy did exactly that once and
        # silently turned 425 fixed-node games into movetime games -- every
        # one a time forfeit -- because it predated `go nodes`. Hence the two
        # rules below: say which driver resolved, and refuse to run one that
        # is missing a capability rather than degrading into the builtin loop.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import sunfish_ui.uci as _drv
    except ImportError:
        _drv = None
    if _drv is not None:
        _p = inspect.signature(_drv.go_loop).parameters
        _nodes, _fen = "max_nodes" in _p, hasattr(_drv, "from_fen")
        # A capability check catches a MISSING feature; a stale copy that
        # merely predates a FIX passes every capability test while behaving
        # differently. The version stamp is the only thing that catches that,
        # and it is the cheapest insurance against tonight's worst failure
        # class -- three separate incidents from one shadowed driver.
        _ver = getattr(_drv, "DRIVER_VERSION", 0)
        print("info string driver", _drv.__file__, "v%d" % _ver,
              "nodes" if _nodes else "NO-NODES", "fen" if _fen else "NO-FEN",
              flush=True)
        if _ver < REQUIRED_DRIVER:
            raise SystemExit(
                "sunfish_ui driver at %s is version %d, need >= %d. This is a "
                "STALE copy shadowing the repo one: sys.path puts this file's "
                "grandparent first, so a scratch copy wins the import and "
                "silently behaves like an older engine (it voided 425 games "
                "once). Delete it or refresh it from the repo."
                % (_drv.__file__, _ver, REQUIRED_DRIVER))
        if not (_nodes and _fen):
            raise SystemExit(
                "sunfish_ui driver at %s lacks required capabilities "
                "(max_nodes=%s from_fen=%s). Refusing to fall back to the "
                "builtin loop: that fallback is what made the last failure "
                "silent (movetime-only, then startpos-only against an EPD "
                "book). Remove the stale copy or fix the driver."
                % (_drv.__file__, _nodes, _fen))
        return _drv.run(sys.modules[__name__], hist[-1])
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
            # increment-aware budget; see sunfish_ui/uci.py for the audit
            # numbers and the safety argument
            think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)
            # minifier-hide start
            # SUDDEN DEATH needs a flatter divisor. With winc == 0, /12 spends
            # 7% of the whole budget on one early move (12.8s of 180s on ply 9
            # in lichess.org/EAThUL0P) and the game is lost on time at move 73
            # without a single move overrunning: below 2s the wtime/2 - 1000
            # cap goes negative, the budget collapses to the 0.05s floor, and
            # ~200ms/move of unavoidable lag drains the rest.
            # /40 is what classic uses and classic does not flag, so this is a
            # constant with production evidence rather than a fit to one game.
            # Movecount-aware divisors were simulated and are WORSE: a
            # shrinking "moves remaining" divisor spends MORE per move as the
            # game lengthens, which is backwards for sudden death.
            # TCEC is 1800+3, so winc is always non-zero there and this line
            # is dead code in the artifact -- which is why it is hidden, and
            # why the artifact stays byte-for-byte unchanged. The increment
            # case is identical to the line above by construction.
            think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000)
            # minifier-hide end
            # A GUI-supplied movetime is a HARD limit that the GUI itself
            # enforces, so spending all of it forfeits: the node counter is
            # only checked every 2048 nodes, so the search returns at
            # movetime + epsilon and the GUI has already called the flag.
            # Keep 5% back (min 30ms) as polling slack. Measured the hard
            # way: 425 local fixed-node games, every single one a forfeit.
            think = times.get("movetime", think) / 1000
            if "movetime" in times: think -= max(think * .05, .03)

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
            # minifier-hide start
            # "go nodes N": equal-effort matches. Testing-only -- the 4k
            # rules mandate no such command, so the artifact does not carry
            # it. Without this a fixed-node match silently becomes a
            # movetime match and every game ends in a forfeit.
            max_nodes = times.get("nodes", 0)
            searcher.node_cap = max_nodes or 1 << 62
            # minifier-hide end
            try:
                for depth, gamma, score, move in searcher.search(hist):
                    if depth > d0:
                        best, d0 = cand or best, depth
                    # minifier-hide start
                    if max_nodes and searcher.nodes >= max_nodes and (best or cand):
                        break
                    # minifier-hide end
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
