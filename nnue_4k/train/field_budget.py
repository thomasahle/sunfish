"""Field-budget certification: the analyzer that makes MULTI-LAYER packed
nets possible at all.

The three recorded walls (MEASUREMENTS.md 2026-08-09..11) and what this
module does about each:

  carry coupling         a field that outgrows its width borrows from its
                         neighbour and silently corrupts a DIFFERENT
                         feature (the abandoned wrap-AND rff variant; the
                         borrow-guard bit in the head).  Here: interval
                         bounds are propagated through every layer and the
                         no-borrow condition is checked per field, with
                         margins reported.
  field-budget collapse  a product SQUARES lane magnitudes, so a deep
                         chain of products overflows any fixed width.
                         Here: per-layer field widths are chosen by the
                         analyzer, and ShiftRenorm between layers is the
                         depth enabler -- the certificate computes the max
                         feasible depth for a width/shift policy instead
                         of hoping.
  quant-error compounding  handled in the layers (quantized grids inside
                         forward, STE), certified here only through the
                         exactness bound: every intermediate must stay
                         below 2^53 so the float64 training mirror IS
                         integer arithmetic, layer for layer.

A config whose certificate fails DOES NOT TRAIN: train.py calls
certify_or_raise before the first batch.  Training-time enforcement of the
assumed weight ranges is projection (the clamps the trainer already runs:
wclip, gain caps, bias bands) -- the certificate states the ranges it
assumed so the projection is checkable; a satpen-style penalty on observed
lane magnitudes is available for soft enforcement of data-dependent
bounds (constraints.saturation_penalty on any layer's pre-clamp output).
"""
import dataclasses
import json
from dataclasses import dataclass


@dataclass
class LaneBounds:
    """Per-lane closed integer intervals [lo_k, hi_k]."""
    lo: list
    hi: list

    def __len__(self):
        return len(self.lo)

    @classmethod
    def uniform(cls, n, lo, hi):
        return cls([lo] * n, [hi] * n)

    def absmax(self):
        return max(max(abs(l) for l in self.lo), max(abs(h) for h in self.hi))


# ---------------------------------------------------- bound transformers
def embed_bounds(n, wmax, bias_lo, bias_hi, max_features):
    """Accumulator bounds for a first layer: |row weight| <= wmax per lane,
    at most max_features active (32 pieces on a board).  pnet's
    excursion_bound is the sharper per-piece-placement version once
    concrete weights exist; this is the pre-training structural bound."""
    return LaneBounds([bias_lo - max_features * wmax] * n,
                      [bias_hi + max_features * wmax] * n)


def clamp_bounds(G):
    return LaneBounds([0] * len(G), list(G))


def conv_bounds(a, b, mode="circular", m=4):
    """Exact interval arithmetic through the lane convolution."""
    na, nb = len(a), len(b)
    nout = m if mode == "circular" else na + nb - 1
    lo = [0] * nout
    hi = [0] * nout
    for i in range(na):
        for j in range(nb):
            k = (i + j) % m if mode == "circular" else i + j
            prods = (a.lo[i] * b.lo[j], a.lo[i] * b.hi[j],
                     a.hi[i] * b.lo[j], a.hi[i] * b.hi[j])
            lo[k] += min(prods)
            hi[k] += max(prods)
    return LaneBounds(lo, hi)


def shift_bounds(b, s):
    """Trunc-toward-zero shift: magnitudes divide, signs kept."""
    t = lambda v: -((-v) >> s) if v < 0 else v >> s
    return LaneBounds([t(v) for v in b.lo], [t(v) for v in b.hi])


def weighted_sum_bounds(b, wmax):
    """Read-out sum_k u_k * x_k with |u_k| <= wmax: scalar interval."""
    hi = sum(wmax * max(abs(l), abs(h)) for l, h in zip(b.lo, b.hi))
    return LaneBounds([-hi], [hi])


# ---------------------------------------------------- certificate
EXACT53 = 1 << 53


@dataclass
class Check:
    name: str
    ok: bool
    value: int
    limit: int

    @property
    def margin(self):
        return self.limit - self.value


class Certificate:
    def __init__(self, name):
        self.name = name
        self.layers = []
        self.notes = []

    def layer(self, label, bounds, checks):
        self.layers.append({"label": label,
                            "bounds_absmax": bounds.absmax(),
                            "checks": [dataclasses.asdict(c) for c in checks]})

    @property
    def ok(self):
        return all(c["ok"] for l in self.layers for c in l["checks"])

    def report(self):
        lines = ["field-budget certificate: %s -- %s"
                 % (self.name, "CERTIFIED" if self.ok else "REFUSED")]
        for l in self.layers:
            lines.append("  %s (|lane| <= %d)" % (l["label"], l["bounds_absmax"]))
            for c in l["checks"]:
                lines.append("    %-24s %s  value %d  limit %d  margin %d"
                             % (c["name"], "ok " if c["ok"] else "FAIL",
                                c["value"], c["limit"], c["limit"] - c["value"]))
        for n in self.notes:
            lines.append("  note: " + n)
        return "\n".join(lines)

    def to_json(self):
        return json.dumps({"name": self.name, "ok": self.ok,
                           "layers": self.layers, "notes": self.notes}, indent=1)


def check_field_nonneg(b, F):
    """Non-negative product/activation fields: hi must fit the width."""
    return Check("no-carry (field < 2^%d)" % F, 0 <= min(b.lo) and b.absmax() < (1 << F),
                 b.absmax(), (1 << F) - 1)


def check_field_offset(b, F):
    """Offset-binary signed lanes with a borrow guard (engine head scheme:
    F=16, value bits F-1, BIAS 2^(F-2)): |value| must stay below 2^(F-2)."""
    return Check("no-borrow (|lane| < 2^%d)" % (F - 2), b.absmax() < (1 << (F - 2)),
                 b.absmax(), (1 << (F - 2)) - 1)


def check_hsum(b, F):
    """The modular horizontal sum precondition: sum of lane maxima."""
    s = sum(b.hi)
    return Check("hsum (sum <= 2^%d-2)" % F, 0 <= min(b.lo) and s <= (1 << F) - 2,
                 s, (1 << F) - 2)


def check_exact53(b):
    """float64 IS integer arithmetic below 2^53 -- the torch-mirror bound."""
    return Check("float64-exact (< 2^53)", b.absmax() < EXACT53, b.absmax(), EXACT53 - 1)


# ---------------------------------------------------- concrete certifiers
GMAX = 89          # replnet gain digit ceiling (payload codec)
BIAS_ABS = 44      # replnet bias digit band


def certify_grid_head(cert, cmax=1, gmax=GMAX, n=4):
    """Layer 1 with weights on the integer grid {-cmax..+cmax} * g_k.

    cmax=1 IS the shipped ternary scheme (engine-proven; certified here so
    the multi-layer chain starts from stated numbers): rows g*t, caps 32g,
    32 pieces max.  cmax>1 is the free-int lane the trained-structure
    modules ask about -- same engine code (a row is `sum g_k*c_k << 16k`
    for ANY int c_k), only the accumulator bound moves.  The activation
    cap stays 32g: it is the crelu's range, not a correctness condition."""
    acc = embed_bounds(n, gmax * cmax, -BIAS_ABS, BIAS_ABS + 1, 32)
    cert.layer("L1 accumulator (offset lanes, F=16, |w| <= %d*g)" % cmax, acc,
               [check_field_offset(acc, 16), check_exact53(acc)])
    y = clamp_bounds([32 * gmax] * n)
    cert.layer("L1 crelu output [0, 32g]", y,
               [check_hsum(y, 16), check_exact53(y)])
    return y


def certify_replnet_head(cert, gmax=GMAX):
    """The shipped ternary head: certify_grid_head at cmax=1."""
    return certify_grid_head(cert, 1, gmax)


def max_certified_grid(gmax=GMAX, n=4, limit=64):
    """Largest integer weight multiplier c with a certified L1 head.

    THE structural number for every trained-structure module: a weight of
    c lane-gain units puts 32*c*gmax + BIAS_ABS into an offset-binary lane
    that must stay under 2^14.  At gmax=89 this is 5 -- so a free-int
    codebook entry, or an unclipped low-rank composite, may reach 5 and no
    further.  Everything above is REFUSED, not rounded."""
    c = 1
    while c < limit:
        probe = Certificate("grid probe")
        certify_grid_head(probe, c + 1, gmax, n)
        if not probe.ok:
            return c
        c += 1
    return c


def certify_ml2(F2=32, m=4, umax=127, shift2=10, gmax=GMAX):
    """The first multi-layer config: layer 2 = circular self-convolution of
    the clamped head lanes (h = conv(A,A) - conv(B,B), odd by construction)
    via ONE extra big-int multiply, fields re-spaced to F2 bits, fold mod
    2^(F2*m)-1, integer read-out u then >> shift2.

    Sized by THIS certificate, not by hope: F2=16 fails no-carry (the
    recorded field-budget wall -- products square the magnitudes), F2=32
    holds with ~2 decades of margin; the group sums exceed 2^32-2 so the
    read-out is per-field mask+shift, never hsum."""
    cert = Certificate("ml2 F2=%d m=%d umax=%d shift2=%d" % (F2, m, umax, shift2))
    y = certify_replnet_head(cert, gmax)
    h = conv_bounds(y, y, "circular", m)         # one perspective's conv(A,A)
    cert.layer("L2 conv fields (product lanes, F=%d)" % F2, h,
               [check_field_nonneg(h, F2), check_exact53(h)])
    hs = check_hsum(h, F2)
    cert.notes.append("L2 group read-out: modular hsum precondition %s "
                      "(sum %d vs %d)%s"
                      % ("HOLDS -- one mod per block is legal" if hs.ok else "FAILS",
                         hs.value, hs.limit,
                         "" if hs.ok else "; read out per field by mask+shift"))
    # antisymmetric difference then integer read-out and renorm
    hd = LaneBounds([l - h2 for l, h2 in zip(h.lo, h.hi)],
                    [h2 - l for l, h2 in zip(h.lo, h.hi)])
    out = weighted_sum_bounds(hd, umax)
    cert.layer("L2 read-out sum u*h (|u| <= %d)" % umax, out, [check_exact53(out)])
    o2 = shift_bounds(out, shift2)
    cert.layer("L2 out >> %d (cp residual before clip)" % shift2, o2,
               [check_exact53(o2)])
    return cert


# ------------------------------------------- trained-structure certifiers
# Both modules below are PARAMETRIZATIONS of layer 1's weight table: the
# engine code is unchanged, the DECODER changes and the accumulator bound
# moves with the grid the parametrization can reach.  So each certifier is
# certify_grid_head at that structure's worst-case reach, plus the codec
# preconditions the shared base-90 body imposes (they are refusals too: a
# config the container cannot express must not train).

LOG2_90 = 6.4918530963296745       # log2(90): bits per base-90 payload char
LOG2_3 = 1.5849625007211562


def _chars(bits):
    """Payload bits -> base-90 chars = SOURCE BYTES before the codec.  Not
    a byte count of the artifact: lzma sees these chars in context and the
    bake-off measures what comes out.  PRICE-FIRST, always."""
    import math
    return int(math.ceil(bits / LOG2_90))


def cb_body_bits(K, block, N, nfeat=768):
    """Stored state of the trained-codebook body, EXACTLY as the shared
    codebook decoder pops it: 2 base-90 digits for the book size, K*block
    base-90 digits of base-3^N symbols, then nfeat/block base-K indices."""
    nblk = nfeat // block
    import math
    return (2 * LOG2_90 + K * block * LOG2_90 + nblk * math.log2(K),
            {"book_bits": K * block * LOG2_90, "index_bits": nblk * math.log2(K),
             "blocks": nblk})


def lr_body_bits(rank, nnz, N, nfeat=768, gap_radix=64):
    """Stored state of the low-rank body: rank digit, U (nfeat*rank trits),
    V (rank*N trits), then the residual's (gap, sign) pairs."""
    import math
    u = nfeat * rank * LOG2_3
    v = rank * N * LOG2_3
    r = nnz * (math.log2(gap_radix) + 1)
    return (5 * LOG2_90 + u + v + r,
            {"U_bits": u, "V_bits": v, "residual_bits": r, "nnz": nnz})


def certify_trained_cb(K, block, N=4, cmax=1, gmax=GMAX, nfeat=768):
    """Trained codebook (product quantization) over blocks of `block`
    consecutive features.  cmax is the grid the CODEBOOK ENTRIES live on:
    1 = ternary (the shipped grid; entries are directly the payload's
    trits), >1 = free-int, which certifies only while 32*cmax*gmax stays
    inside the offset lane (max_certified_grid)."""
    cert = Certificate("trained_cb K=%d block=%d N=%d cmax=%d" % (K, block, N, cmax))
    certify_grid_head(cert, cmax, gmax, N)
    cert.layer("codec preconditions (shared base-90 body)",
               LaneBounds([0], [K]),
               [Check("block divides %d" % nfeat, nfeat % block == 0,
                      nfeat % block, 0),
                Check("K <= 8100 (2 base-90 digits)", 1 <= K <= 8100, K, 8100),
                Check("K <= blocks (a book bigger than its data stores nothing)",
                      K <= nfeat // max(block, 1), K, nfeat // max(block, 1)),
                Check("book fits the entry (K*block digits)",
                      K * block <= 4096, K * block, 4096)])
    bits, parts = cb_body_bits(K, block, N, nfeat)
    cert.notes.append(
        "stored state: book %d entries x %d features = %.0f bits (%d base-90 "
        "chars), indices %d x log2(%d) = %.0f bits (%d chars); body total "
        "%.0f bits = %d chars PRE-CODEC (PRICE-FIRST: the artifact number is "
        "pack.sh's, via compress/bakeoff.py arm 'trained_cb')"
        % (K, block, parts["book_bits"], _chars(parts["book_bits"]),
           parts["blocks"], K, parts["index_bits"], _chars(parts["index_bits"]),
           bits, _chars(bits)))
    if cmax > 1:
        cert.notes.append(
            "free-int entries cost log2(%d)=%.2f bits/element vs log2(3)=1.58 "
            "ternary -- %.2fx the book, and they leave the shipped codec's "
            "representable set (no b81 denominator for the same net)"
            % (2 * cmax + 1, __import__("math").log2(2 * cmax + 1),
               __import__("math").log2(2 * cmax + 1) / LOG2_3))
    return cert


def certify_lowrank(rank, wmax=1, uvmax=1, rmax=1, N=4, gmax=GMAX, nfeat=768):
    """Low-rank + residual: W = clip(U@V + R, -wmax, +wmax), U (nfeat x
    rank) and V (rank x N) on the +-uvmax integer grid, R on the +-rmax
    grid (ternary at rmax=1).

    Two numbers matter and they are different:
      reach  the UNCLIPPED worst case, rank*uvmax^2 + rmax -- what the
             composite could produce if nothing clipped it;
      wmax   what the DECODER actually emits (it runs the clip), which is
             what the engine's lanes see and therefore what is certified.
    A wmax the head cannot hold is REFUSED.  A reach above wmax is legal
    (the clip is executed, one expression, encoder and decoder agree) and
    reported, so the free-int variant's ceiling is pre-stated."""
    cert = Certificate("lowrank rank=%d wmax=%d uvmax=%d rmax=%d"
                       % (rank, wmax, uvmax, rmax))
    certify_grid_head(cert, wmax, gmax, N)
    reach = rank * uvmax * uvmax + rmax
    cmax = max_certified_grid(gmax, N)
    cert.layer("composite grid W = clip(U@V + R)", LaneBounds([-wmax], [wmax]),
               [Check("wmax within the certified grid", 1 <= wmax <= cmax, wmax, cmax),
                Check("rank >= 1", rank >= 1, 1, rank)])
    cert.notes.append(
        "unclipped reach rank*uvmax^2 + rmax = %d; the certified grid ceiling "
        "is %d, so an UNCLIPPED composite certifies only to rank <= %d at "
        "uvmax=%d, rmax=%d.  At wmax=%d the decoder clips and the head is the "
        "shipped one." % (reach, cmax, max(0, (cmax - rmax) // (uvmax * uvmax)),
                          uvmax, rmax, wmax))
    return cert


def max_feasible_depth(F, m, start_hi, shift_policy, limit_layers=16):
    """How deep can conv+renorm chains go at field width F?  Iterate
    conv(x, x) -> >> s until no-carry fails.  With shift s chosen so
    bounds return to ~start (s ~= log2(m * start_hi)), depth is unbounded
    in principle -- the certificate quantifies the honest per-layer
    precision left after the shift instead."""
    b = LaneBounds.uniform(m, 0, start_hi)
    depth = 0
    detail = []
    for _ in range(limit_layers):
        h = conv_bounds(b, b, "circular", m)
        if h.absmax() >= (1 << F):
            break
        s = shift_policy(h)
        b = shift_bounds(h, s)
        depth += 1
        detail.append({"depth": depth, "conv_hi": h.absmax(), "shift": s,
                       "post_hi": b.absmax(),
                       "bits_kept": b.absmax().bit_length()})
    return depth, detail


def certify_or_raise(model_cfg):
    """train.py's pre-training hook: an uncertifiable config never trains."""
    arch = getattr(model_cfg, "arch", "residual")
    if arch == "ml2":
        cert = certify_ml2()
    elif arch == "cb":
        cert = certify_trained_cb(model_cfg.cb_k, model_cfg.cb_block,
                                  model_cfg.N, model_cfg.cb_cmax)
    elif arch == "lowrank":
        cert = certify_lowrank(model_cfg.lr_rank, model_cfg.lr_wmax, N=model_cfg.N)
    else:
        return None
    print(cert.report(), flush=True)
    if not cert.ok:
        raise SystemExit("field-budget certificate REFUSED -- fix the "
                         "architecture, not the check")
    return cert
