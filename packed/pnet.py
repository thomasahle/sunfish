"""Packed big-integer NNUE: representation, loader and reference semantics.

The whole accumulator and the whole evaluation head live in ONE Python int.
That is the point: on PyPy an int of a few thousand bits is a single object
whose `+`, `-`, `&`, `|`, `>>` are C-level loops over 64-bit limbs, so the
per-node cost of a WIDE net is a handful of big-int ops instead of a
Python-level loop over lanes.

Architecture (chosen so that no per-lane multiply is ever needed)

    score_cp(pos) = pst(pos)                       # exact, incremental, classic
                  + clip(nn(pos), -CLAMP, +CLAMP)  # packed residual

    nn(pos) = (1/C) * sum_k  v_k * ( crelu(a_k(us)) - crelu(a_k(them)) )

`a_k(side)` is hidden unit k of a 768 -> N accumulator read from `side`'s
perspective; `v_k` is unit k's output weight.  The net is *symmetric*: the
same v_k serves both perspectives with opposite sign, which makes the eval
exactly antisymmetric (score(pos) == -score(pos.rotate())) and lets both
perspectives share one lane block layout.

Lane layout.  Let W and B be the white- and black-perspective accumulators
and let units be ordered so units 0..P-1 have v_k > 0 and P..N-1 have
v_k < 0.  The 2N lanes of the packed int are

    block0 = [ W[0..P) ; B[P..N) ]      lanes 0     .. N-1
    block1 = [ B[0..P) ; W[P..N) ]      lanes N     .. 2N-1

so that  sum(block0) - sum(block1) == C * nn  read from WHITE's point of
view, and swapping the two blocks is exactly "swap the perspectives".  We
never perform that swap: the position carries a perspective flag `pf` and
the two row tables (`rows[0]`, `rows[1]`) differ only by that relabelling,
so they share the very same Python int objects.

Lane arithmetic.  Lanes are L=16 bits, offset-binary with BIAS = 2^14, and
bit 15 of every lane is a borrow guard that must stay clear:

    lane_k = BIAS + G_k * a_k        with  G_k = round(C * |v_k|)

`G_k` folds the output weight INTO the input tables, so the head needs no
multiply at all: clipped-ReLU on unit k becomes a clamp of lane k to
[0, G_k], which the SWAR clamp does with per-lane bounds at zero extra
cost, and the head output is then a plain horizontal sum of lanes.

Intermediate values may leave the lane range with impunity -- Python ints
are exact, so `acc + rowNew - rowOld` is the correct sum of per-lane values
whatever order it is evaluated in.  Only the FINAL accumulator has to have
every lane inside [0, 2^15); `excursion_bound()` proves that over all legal
piece placements.

Horizontal sum.  2^16 == 1 (mod 2^16 - 1), so for a packed value with
non-negative 16-bit lanes,  x % (2**16 - 1)  IS the sum of the lanes, as
long as that sum stays below 2^16-1.  One single-limb division replaces a
Python-level loop over bytes and is ~1.7x faster than the fastest
`to_bytes()`-based form measured.  The clipped ReLU is what makes the
precondition checkable offline: lane k never exceeds G_k, so
`sum(G) <= 65534` -- asserted in `build()` -- is a static proof that the
reduction is exact.
"""

import pickle

L = 16                      # bits per lane
VBITS = L - 1               # value bits (bit 15 is the borrow guard)
BIAS = 1 << (VBITS - 1)     # 2^14, offset-binary zero
ONES = (1 << VBITS) - 1
PIECES = "PNBRQKpnbrqk"
SWAP = {c: c.swapcase() for c in PIECES}
# The 64 playable squares of sunfish's 120-char board.
SQUARES = tuple(r * 10 + f for r in range(2, 10) for f in range(1, 9))


def rep(val, n):
    """`val` replicated into n lanes."""
    r = 0
    for _ in range(n):
        r = (r << L) | val
    return r


def packlanes(vals):
    """Pack a list of per-lane integers (may be negative) into one int."""
    r = 0
    for i, v in enumerate(vals):
        r += v << (L * i)
    return r


def unpacklanes(x, n):
    m = (1 << L) - 1
    return [(x >> (L * i)) & m for i in range(n)]


class PackedNet:
    """Holds the packed tables and evaluates the head.

    The engine inlines the hot parts of `head()`; this class is the
    reference the engine is verified against, and the loader both use.
    """

    def __init__(self, d):
        self.N = N = d["N"]                 # hidden units per perspective
        self.P = d["P"]                     # units with positive output weight
        self.shift = d["shift"]             # nn_cp = (block0 - block1) >> shift
        self.clampcp = d["clampcp"]
        # The clip ceiling is not optional: it is the precondition that makes
        # the modular horizontal sum exact.
        self.crelu = True
        self.lanes = lanes = 2 * N
        self.nbytes = lanes * (L // 8)
        self.half = N * L                   # bit offset of block1
        self.base = d["base"]               # packed BIAS + G_k*bias_k
        self.gp = d["gp"]                   # packed G_k (clamp ceiling)
        self.G = d["G"]                     # per-lane gains, block order
        # rows[pf][piece][square120] -- pf==1 entries are the SAME int objects
        rows0 = d["rows"]
        rows1 = {p: [rows0[SWAP[p]][119 - s] for s in range(120)] for p in PIECES}
        self.rows = (rows0, rows1)
        # SWAR constants
        self.H = rep(1 << VBITS, lanes)             # guard bits
        self.VAL = rep(ONES, lanes)
        self.LO = rep(BIAS, lanes)
        self.GH = self.gp | self.H          # per-lane ceiling with guard set
        self.MASKLO = (1 << self.half) - 1
        self.M16 = (1 << L) - 1
        self.meta = {k: v for k, v in d.items() if k not in ("rows", "base", "gp", "G")}

    # ------------------------------------------------------------------ head
    def clamp(self, acc):
        """Per-lane clip to [0, G_k], returned as unsigned lanes."""
        LO, VAL = self.LO, self.VAL
        # relu: lane >= BIAS <=> bit 14 set (lanes are < 2^15).  ORing MLO
        # back in is free -- a surviving lane already has bit 14 set.
        m = ((acc & LO) >> (VBITS - 1)) * ONES
        y = ((acc & m) | LO) - LO
        if self.crelu:
            m = (((self.GH - y) & self.H) >> VBITS) * ONES     # y <= G_k ?
            y = (y & m) | (self.gp & (m ^ VAL))
        return y

    def hsum2(self, y):
        """(sum of block0 lanes, sum of block1 lanes) by modular reduction."""
        return (y & self.MASKLO) % self.M16, (y >> self.half) % self.M16

    def raw(self, acc, pf):
        """C * nn, from the perspective selected by pf (0 = white to move)."""
        x, y = self.hsum2(self.clamp(acc))
        return y - x if pf else x - y

    def nn_cp(self, acc, pf):
        v = self.raw(acc, pf) >> self.shift
        c = self.clampcp
        return -c if v < -c else (c if v > c else v)

    # ------------------------------------------------------- accumulator
    def from_board(self, board, pf):
        """Build the accumulator from scratch. `board` is mover-oriented."""
        acc = self.base
        rows = self.rows[pf]
        for i, p in enumerate(board):
            if p in PIECES:
                acc += rows[p][i]
        return acc

    # --------------------------------------------------------- diagnostics
    def lane_range(self, acc):
        return unpacklanes(acc, self.lanes)

    def check_lanes(self, acc):
        """True iff every lane is inside [0, 2^15) -- i.e. the packed
        representation is intact (no lane borrowed from its neighbour)."""
        if acc < 0:
            return False
        return all(v < (1 << VBITS) for v in unpacklanes(acc, self.lanes))


def load(path):
    with open(path, "rb") as f:
        return PackedNet(pickle.load(f))


def save(path, d):
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=4)


# --------------------------------------------------------------------------
# Building a packed net from float weights
# --------------------------------------------------------------------------
# Legal maxima per piece type per side (8 pawns; every other non-king piece
# can in principle reach 1 original + 8 promotions).  Used for a RIGOROUS
# bound on the accumulator excursion.
MAXCOUNT = {"P": 8, "N": 10, "B": 10, "R": 10, "Q": 9, "K": 1}


def _side_extreme(Wq_k, chars, sign):
    """Largest achievable sum of `sign`-signed weights for ONE side.

    A side is exactly one king plus at most 15 other pieces, with at most
    MAXCOUNT copies of each type (8 pawns; 1 original + 8 promotions for the
    others).  Taking the 15 best values from the capped pool is optimal
    because every piece costs the same one slot.
    """
    best = max(sign * v for v in Wq_k[chars[5]])          # the king
    pool = []
    for c in chars[:5]:
        vals = sorted((sign * v for v in Wq_k[c]), reverse=True)
        pool += vals[:MAXCOUNT[c.upper()]]
    pool.sort(reverse=True)
    return best + sum(v for v in pool[:15] if v > 0)


def excursion_bound(Wq, biasq, N):
    """Rigorous worst-case |lane - BIAS| over all legal piece placements."""
    white, black = PIECES[:6], PIECES[6:]
    out = []
    for k in range(N):
        hi = biasq[k] + _side_extreme(Wq[k], white, 1) + _side_extreme(Wq[k], black, 1)
        lo = biasq[k] - _side_extreme(Wq[k], white, -1) - _side_extreme(Wq[k], black, -1)
        out.append(max(abs(hi), abs(lo)))
    return out


def pick_shift(W, bias, v, limit=16000, sumlimit=65534, squares=SQUARES):
    """Largest gain C = 2**shift that satisfies BOTH packing constraints.

    guard  C * max_j(|v_j| * excursion_j) <= limit    (no lane leaves [0,2^15))
    hsum   C * sum_j |v_j|                <= sumlimit (the mod reduction is exact)

    Returns (shift, worst_guard_product, sum_abs_v).
    """
    N = len(v)
    sq = set(squares)
    Wf = [{p: [W[k][p][s] if s in sq else 0.0 for s in range(120)]
           for p in PIECES} for k in range(N)]
    exc = excursion_bound(Wf, list(bias), N)
    worst = max(abs(v[k]) * exc[k] for k in range(N)) or 1e-9
    sabs = sum(abs(x) for x in v) or 1e-9
    shift = 0
    while (1 << (shift + 1)) * worst <= limit and (1 << (shift + 1)) * sabs <= sumlimit:
        shift += 1
    return shift, worst, sabs


def build(W, bias, v, shift, clampcp=600, squares=SQUARES):
    """Quantise a float net into the packed representation.

    W[k][piece][sq120]  float input weights (activation units, clip at 1.0)
    bias[k]             float bias
    v[k]                float output weight, in centipawns per activation unit
    shift               nn_cp = raw >> shift, so the gain is C = 2**shift

    Returns the dict accepted by PackedNet.
    """
    N = len(v)
    squares = set(squares)
    C = 1 << shift
    order = sorted(range(N), key=lambda k: (v[k] <= 0, -abs(v[k])))
    P = sum(1 for x in v if x > 0)
    G = [max(1, round(C * abs(v[k]))) for k in order]

    # quantised weights, indexed by the NEW unit order
    Wq = [{p: [0] * 120 for p in PIECES} for _ in range(N)]
    biasq = [0] * N
    for j, k in enumerate(order):
        g = G[j]
        biasq[j] = round(g * bias[k])
        for p in PIECES:
            col = W[k][p]
            wq = Wq[j][p]
            for s in squares:
                wq[s] = round(g * col[s])

    # lane index of unit j for each perspective and block
    #   block0 = [ W-pos (P) ; B-neg (N-P) ]
    #   block1 = [ B-pos (P) ; W-neg (N-P) ]
    laneW = [0] * N   # lane holding the white-perspective copy of unit j
    laneB = [0] * N
    for j in range(N):
        if j < P:
            laneW[j] = j
            laneB[j] = N + j
        else:
            laneW[j] = N + j
            laneB[j] = j

    lanesG = [0] * (2 * N)
    lanebias = [0] * (2 * N)
    for j in range(N):
        lanesG[laneW[j]] = lanesG[laneB[j]] = G[j]
        lanebias[laneW[j]] = lanebias[laneB[j]] = biasq[j]

    if sum(G) > 65534:
        raise ValueError(
            "sum(G) = %d > 65534: a block's lane sum could reach the modulus "
            "and the horizontal sum would wrap. Lower `shift`." % sum(G))
    base = packlanes([BIAS + b for b in lanebias])
    gp = packlanes(lanesG)

    rows = {}
    for p in PIECES:
        pr = SWAP[p]
        col = []
        for s in range(120):
            lv = [0] * (2 * N)
            if s in squares:
                for j in range(N):
                    lv[laneW[j]] = Wq[j][p][s]
                    lv[laneB[j]] = Wq[j][pr][119 - s]
            col.append(packlanes(lv))
        rows[p] = col

    exc = excursion_bound(Wq, biasq, N)
    return {
        "kind": "packed-nnue-v1", "N": N, "P": P, "shift": shift,
        "clampcp": clampcp, "base": base, "gp": gp,
        "G": lanesG, "rows": rows,
        "excursion": max(exc), "excursion_per_unit": exc,
        "sum_G": sum(G),
    }


def float_nn(W, bias, v, board, pf, crelu=True):
    """Reference evaluation of the float net, mover's point of view.

    This is what the packed head approximates; the only differences are the
    per-lane roundings done in `build`.
    """
    N = len(v)
    aw = list(bias)
    ab = list(bias)
    for s, p in enumerate(board):
        if p in PIECES:
            q, t = (p, s) if pf == 0 else (SWAP[p], 119 - s)
            for k in range(N):
                aw[k] += W[k][q][t]
                ab[k] += W[k][SWAP[q]][119 - t]
    f = (lambda x: 0.0 if x < 0 else (1.0 if x > 1 else x)) if crelu \
        else (lambda x: max(0.0, x))
    out = sum(v[k] * (f(aw[k]) - f(ab[k])) for k in range(N))
    return out if pf == 0 else -out
