"""(a) The baseline: char-aligned base-3^4, one feature per base-90 digit.

Layout A is the RECORDED path (payload string spliced into the stock
entry, decoded by the entry's own loop) and must reproduce 3831/3834 B
on v1/repro_arm1 exactly -- that is the harness's instrument check.
Layout B stores the same integer as raw big-endian bytes; the decoder
body is the entry's own inline block, so the source delta is only the
prologue swap.
"""
from . import register
from .. import qnet

# The stock entry's own decode block, VERBATIM PER SEAM (one pass: pop
# digit, split trits, fold gains).  The baseline's patched cells (layout
# B, elided A) must carry the entry's own decoder shape, not a
# translation of it, so its decoder-cost column prices only the prologue
# swap -- hence body_src_for instead of the one-dialect body every other
# arm returns.
BODY_INLINE = {"v1": """\
_half = {}
for _p in _PIECES:
    _h = [0] * 120
    for _f in range(64):
        _w, _d = divmod(_w, 90)
        _r = 0
        for _k in range(NN):
            _d, _t = divmod(_d, 3); _r += _g[_k] * (_t - 1) << LBITS * _k
        _h[21 + _f // 8 * 10 + _f % 8] = _r
    _half[_p] = _h
""", "v2": """\
_half = {}
for _p in _PIECES:
    _half[_p] = _h = [0] * 120
    for _f in range(64):
        _w, _d = divmod(_w, 90)
        _h[21 + _f // 8 * 10 + _f % 8] = sum(
            _g[_k] * (_d // 3 ** _k % 3 - 1) << 16 * _k for _k in range(4))
"""}


@register
class Base81:
    name = "b81"
    native_a = True

    def body_src_for(self, seam):
        return BODY_INLINE[seam]

    def encode(self, q):
        syms = qnet.symbols81(q)
        body = 0
        for s in reversed(syms):
            body = body * 90 + s
        return body, BODY_INLINE["v1"], "1 char/feature, base-3^4 in base-90"
