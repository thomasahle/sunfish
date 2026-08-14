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


def certify_replnet_head(cert, gmax=GMAX):
    """Layer 1, the shipped scheme (engine-proven; certified here so the
    multi-layer chain starts from stated numbers): ternary rows g*t, caps
    32g, 32 pieces max."""
    acc = embed_bounds(4, gmax, -BIAS_ABS, BIAS_ABS + 1, 32)
    cert.layer("L1 accumulator (offset lanes, F=16)", acc,
               [check_field_offset(acc, 16), check_exact53(acc)])
    y = clamp_bounds([32 * gmax] * 4)
    cert.layer("L1 crelu output [0, 32g]", y,
               [check_hsum(y, 16), check_exact53(y)])
    return y


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
    if getattr(model_cfg, "arch", "residual") != "ml2":
        return None
    cert = certify_ml2()
    print(cert.report(), flush=True)
    if not cert.ok:
        raise SystemExit("field-budget certificate REFUSED -- fix the "
                         "architecture, not the check")
    return cert
