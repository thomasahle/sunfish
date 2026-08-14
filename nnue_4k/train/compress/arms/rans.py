"""(c) Entropy coding with a static ternary prior: rANS, big-int state.

Python big ints make rANS degenerate-simple: no renormalization, no bit
IO -- the coder state IS the payload integer.  Decode pops a symbol with
one masked divmod:

    r = x & (M-1);  symbol by cumulative range;  x = f*(x >> 12) + r - C

Encoding runs the exact inverse in reverse symbol order, so the decoder
streams FORWARD (contexts computable from already-decoded trits).

Two priors, fitted on the actual net (parameters stored in the payload
and counted -- 4 base-90 digits per context):

  rc_o0   context = lane (4 contexts): pure order-0 skew.
  rc_run  context = (lane, zero-run bucket 0/1/2-4/5+) (16 contexts):
          the static ternary-RUN prior -- the spatial structure lzma
          normally finds, moved into the model.  Output is close to
          incompressible, which is exactly the layout-B case.
"""
from . import register
from .. import entrysrc

M = 4096
SHIFTB = 12


def _fit(counts):
    """counts (c_neg, c_zero, c_pos) -> 12-bit freqs summing to M, every
    present symbol >= 1."""
    total = sum(counts) or 1
    f = [max(1, round(M * c / total)) if c else 0 for c in counts]
    if sum(f) == 0:
        f[1] = M
    while sum(f) > M:
        i = max(range(3), key=lambda j: f[j])
        f[i] -= 1
        assert f[i] >= 1
    while sum(f) < M:
        i = max(range(3), key=lambda j: f[j])
        f[i] += 1
    return f


def _encode(stream, ctxs, tables):
    """stream: trits in {-1,0,+1}; ctxs: context id per position; tables:
    per-context (f_neg, f_zero, f_pos).  Returns final state (x0 = 1)."""
    x = 1
    for t, c in zip(reversed(stream), reversed(ctxs)):
        fn, fz, fp = tables[c]
        st, fr = ((0, fn) if t < 0 else (fn, fz) if t == 0 else (fn + fz, fp))
        assert fr > 0, "symbol with zero frequency in its context"
        x = (x // fr) * M + st + x % fr
    return x


def _params_body(tables, x):
    """params LSB-first (fn, fz per context, 2 base-90 digits each), then
    the rANS state as the high part."""
    body = x
    pairs = []
    for fn, fz, _ in tables:
        pairs += [(90, fn % 90), (90, fn // 90), (90, fz % 90), (90, fz // 90)]
    for r, d in reversed(pairs):
        body = body * r + d
    return body


SRC_PARAMS = """\
_P = []
for _j in range(%d):
    _w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
    _a = _d + 90 * _e
    _w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
    _P.append((_a, _d + 90 * _e))
"""

SRC_RANS_LOOP = """\
_T = [0] * 3072
for _k in range(NN):
    _z = 0
    for _f in range(768):
        _fn, _fz = _P[%s]
        _r = _w & 4095
        if _r < _fn:
            _t = -1; _w = _fn * (_w >> 12) + _r
        elif _r < _fn + _fz:
            _t = 0; _w = _fz * (_w >> 12) + _r - _fn
        else:
            _t = 1; _w = (4096 - _fn - _fz) * (_w >> 12) + _r - _fn - _fz
        _T[_f * NN + _k] = _t
        _z = 0 if _t else _z + 1
"""


def _bucket(z):
    return min(z, 2) + (z > 4)


class _Rans:
    native_a = False

    def __init__(self, name, nctx, ctxexpr, ctxfn, note):
        self.name, self.nctx, self.ctxexpr, self.ctxfn, self.note = \
            name, nctx, ctxexpr, ctxfn, note

    def encode(self, q):
        N = len(q.g)
        stream, ctxs = [], []
        for k in range(N):
            z = 0
            for f in range(768):
                t = q.trits[f][k]
                stream.append(t)
                ctxs.append(self.ctxfn(k, z))
                z = 0 if t else z + 1
        counts = [[0, 0, 0] for _ in range(self.nctx)]
        for t, c in zip(stream, ctxs):
            counts[c][t + 1] += 1
        tables = [_fit(c) for c in counts]
        x = _encode(stream, ctxs, tables)
        body = _params_body(tables, x)
        src = (SRC_PARAMS % self.nctx) + (SRC_RANS_LOOP % self.ctxexpr) \
            + entrysrc.SRC_HALF_FROM_T
        bits = x.bit_length()
        return body, src, "%s; state %d bits (%.0f B)" % (self.note, bits, bits / 8)


register(_Rans("rc_o0", 4, "_k", lambda k, z: k,
               "order-0 per-lane prior"))
register(_Rans("rc_run", 16, "_k * 4 + min(_z, 2) + (_z > 4)",
               lambda k, z: k * 4 + _bucket(z),
               "per-lane zero-run prior"))
