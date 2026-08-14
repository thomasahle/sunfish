"""Torch layers mirroring the ENGINE's packed big-int semantics, exactly.

The engine's one trick, stated as algebra: a big int with lanes at field
width F is the polynomial  X = sum_i a_i R^i,  R = 2^F.  Then

  X + Y          adds lanes            (exact iff no field overflows)
  X * Y          is the LINEAR CONVOLUTION of the lane sequences:
                 X*Y = sum_k c_k R^k,  c_k = sum_{i+j=k} a_i b_j
                 (exact READ-OUT iff every c_k < R -- the field budget)
  X*Y mod R^m-1  is the CIRCULAR convolution mod m (R^m == 1), the fix for
                 the rank-1 trap: x mod (R-1) alone gives (sum a)(sum b)
                 and every genuine second-order term cancels (ledger
                 2026-08-09, "the obvious read-out is rank-1")
  x mod (R-1)    is the horizontal lane sum (exact iff sum < R-1)
  SWAR clamp     per-lane min(relu(lane), G_k) via mask arithmetic
  v >> s         trunc-toward-zero renormalisation between layers

These modules compute the SAME functions on float64 tensors.  float64 is
the representation, integers are the semantics: every value this family
certifies is an integer of magnitude < 2^53, and IEEE float64 arithmetic
on such integers (add, subtract, multiply, trunc) is EXACT integer
arithmetic.  test_packed_layers.py holds every layer to that standard
against actual Python big-int evaluation -- bit-exact, no tolerances.
(Integer dtypes would carry no autograd; float32 would be a silent
approximation.  float64-with-a-certified-bound is the honest middle.)

Gradients.  A convolution is bilinear and the lane sum is linear: their
autograd gradients are the TRUE gradients of the polynomial the forward
computes -- no approximation anywhere.  Exactly two discrete points exist,
each with an explicit, documented straight-through rule:

  ShiftRenorm   forward trunc(v / 2^s); backward g / 2^s.  The forward is
                piecewise constant; the STE passes the gradient of the
                REAL division the shift approximates.  This is the depth
                enabler: without renormalisation the field budget grows
                multiplicatively per conv layer and collapses (the
                recorded wall that killed deep direct products).
  SwarClamp     forward min(relu(x), G); backward is the exact
                subgradient (pass inside [0, G], zero outside) BY DEFAULT.
                ste=True passes gradient everywhere instead; the house
                position stays "clamp is exact + satpen carries the
                pressure" (the kbbil lesson), so ste defaults off.

Weight quantization (round-to-grid, ternary) reuses constraints.py's STE.

The three recorded walls this family exists to respect (MEASUREMENTS.md,
2026-08-09..11):
  carry coupling      transient inter-lane borrows corrupt neighbours (the
                      wrap-AND rff variant was abandoned for exactly this);
                      here: certified NO-BORROW bounds per layer, enforced
                      before training starts (field_budget.py).
  field-budget collapse  products square the lane magnitudes, so deep
                      products overflow any fixed field; here: per-layer
                      field widths chosen by the analyzer + ShiftRenorm
                      between layers.
  quant-error compounding  rounding a float fit after training optimises
                      the wrong function at every layer; here: the
                      quantized grid sits INSIDE forward (STE), layer by
                      layer, and export-time verification is bit-exact.
"""
import torch

# ------------------------------------------------------------ int bridges
# Python-int side of the mirror: what the ENGINE does.  Tests evaluate
# these on packed big ints and demand equality with the torch modules.


def pack_lanes(vals, F):
    """Non-negative per-lane ints -> one big int at field width F."""
    r = 0
    for i, v in enumerate(vals):
        v = int(v)
        assert 0 <= v < (1 << F), "lane %d = %d does not fit %d bits" % (i, v, F)
        r |= v << (F * i)
    return r


def unpack_lanes(x, n, F):
    m = (1 << F) - 1
    return [(x >> (F * i)) & m for i in range(n)]


def bigint_linear_conv(a, b, F):
    """The engine op: one big-int multiply, fields read at width F."""
    x, y = pack_lanes(a, F), pack_lanes(b, F)
    return unpack_lanes(x * y, len(a) + len(b) - 1, F)


def bigint_circular_conv(a, b, m, F):
    """The engine op: multiply then fold mod 2^(F*m)-1 (R^m == 1)."""
    x, y = pack_lanes(a, F), pack_lanes(b, F)
    return unpack_lanes((x * y) % ((1 << (F * m)) - 1), m, F)


def bigint_hsum(vals, F):
    """The engine op: x mod (2^F - 1) is the lane sum."""
    return pack_lanes(vals, F) % ((1 << F) - 1)


def bigint_swar_clamp(lanes_offset, G, F):
    """The engine's SWAR clipped ReLU on offset-binary lanes, verbatim
    pnet.PackedNet.clamp / the entry's nn_cp mask arithmetic, at field
    width F (engine uses F=16, VBITS=15, BIAS=2^14)."""
    n = len(G)
    VB = F - 1
    BIAS = 1 << (VB - 1)
    ONES = (1 << VB) - 1

    def rep(val):
        r = 0
        for _ in range(n):
            r = (r << F) | val
        return r
    acc = pack_lanes(lanes_offset, F)
    H, LO = rep(1 << VB), rep(BIAS)
    gp = pack_lanes(G, F)
    GH = gp | H
    VAL = rep(ONES)
    m = ((acc & LO) >> (VB - 1)) * ONES
    y = ((acc & m) | LO) - LO
    m = (((GH - y) & H) >> VB) * ONES
    y = (y & m) | (gp & (m ^ VAL))
    return unpack_lanes(y, n, F)


# ------------------------------------------------------------ torch layers
class LanePack(torch.nn.Module):
    """Tensor lanes -> python big ints (one per batch row).  A BRIDGE for
    tests and engine-side builders, not a training-path op (python ints
    carry no grad); training stays in the float64 lane representation,
    which the certificate proves equivalent."""

    def __init__(self, F):
        super().__init__()
        self.F = F

    def forward(self, x):
        return [pack_lanes(row.tolist(), self.F) for row in x.detach().to(torch.int64)]


class LaneUnpack(torch.nn.Module):
    def __init__(self, F, n):
        super().__init__()
        self.F, self.n = F, n

    def forward(self, ints):
        return torch.tensor([unpack_lanes(x, self.n, self.F) for x in ints],
                            dtype=torch.float64)


class LaneConv(torch.nn.Module):
    """The big-int multiply as a layer: linear or circular convolution of
    two lane tensors (..., n).  Forward = exactly the integer convolution
    (float64-exact under the certificate); backward = the true bilinear
    gradients, delivered by autograd (conv is a polynomial -- there is
    nothing to approximate)."""

    def __init__(self, na, nb, mode="circular", m=4):
        super().__init__()
        self.mode, self.m = mode, m
        nout = m if mode == "circular" else na + nb - 1
        idx = torch.empty(na, nb, dtype=torch.long)
        for i in range(na):
            for j in range(nb):
                idx[i, j] = (i + j) % m if mode == "circular" else i + j
        self.register_buffer("idx", idx.reshape(-1))
        self.nout = nout

    def forward(self, a, b):
        prod = (a.unsqueeze(-1) * b.unsqueeze(-2)).reshape(*a.shape[:-1], -1)
        out = prod.new_zeros(*a.shape[:-1], self.nout)
        return out.index_add_(-1, self.idx, prod)


class _TruncShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, s):
        ctx.s = s
        return torch.div(v, float(1 << s), rounding_mode="trunc")

    @staticmethod
    def backward(ctx, g):
        return g / float(1 << ctx.s), None


class ShiftRenorm(torch.nn.Module):
    """v >> s, trunc toward zero (the engine's signed shift, which commutes
    with negation -- antisymmetry survives).  STE backward g/2^s; see the
    module docstring for why this is one of the two licensed STE points."""

    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, v):
        return _TruncShift.apply(v, self.s)


class SwarClamp(torch.nn.Module):
    """Per-lane min(relu(x), G): the engine's SWAR clipped ReLU on the lane
    VALUES (the engine's offset-binary BIAS bookkeeping is representation,
    not semantics; bigint_swar_clamp + the tests carry the equivalence).
    Backward: exact subgradient by default; ste=True passes through."""

    def __init__(self, ste=False):
        super().__init__()
        self.ste = ste

    def forward(self, x, G):
        y = x.clamp(min=torch.zeros_like(G), max=G)
        if self.ste:
            y = x + (y - x).detach()
        return y


class HSum(torch.nn.Module):
    """Horizontal lane sum -- the engine's single-limb `x % (2^F - 1)`.
    Torch side is a plain sum (identical value); the certificate carries
    the modulus precondition sum(lanes) <= 2^F - 2."""

    def forward(self, x):
        return x.sum(-1)


class RoundGrid(torch.autograd.Function):
    """Weight-to-integer-grid STE (distill_train.py's Round, kept here so
    multi-layer weights quantize the same way): snap forward, pass grad."""

    @staticmethod
    def forward(ctx, v, step):
        return torch.round(v / step) * step

    @staticmethod
    def backward(ctx, g):
        return g, None
