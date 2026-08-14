"""Zero-run RLE that PRESERVES char alignment -- feed lzma less, same
alphabet.

Motivated by the first zoo round: entropy-coded arms lose to lzma's own
modeling of the char stream, so instead of replacing lzma, shorten what
it sees.  Tokens stay one base-90 digit: 0..80 = a feature's symbol,
81 + j = a run of (j + 2) all-zero features (runs 2..10; longer runs
emit multiple tokens).  The stream stays byte-aligned for the match
finder, zeros cost ~1/9th, and the decoder is a 3-line loop on top of
the stock one.
"""
from . import register
from .. import qnet

ZSYM = 40  # (0+1)*3^0+(0+1)*3^1+(0+1)*3^2+(0+1)*3^3 = 40: the all-zero char
MAXRUN = 10

# Inline in the stock one-pass loop: a pending-zero-run counter instead
# of a materialized symbol list (3 net source lines over stock; token
# 81+j covers this char plus j+1 more all-zero chars).
BODY = """\
_half = {}
_z = 0
for _p in _PIECES:
    _h = [0] * 120
    for _f in range(64):
        if _z:
            _z -= 1; _d = 40
        else:
            _w, _d = divmod(_w, 90)
            if _d > 80: _z = _d - 80; _d = 40
        _r = 0
        for _k in range(NN):
            _d, _t = divmod(_d, 3); _r += _g[_k] * (_t - 1) << LBITS * _k
        _h[21 + _f // 8 * 10 + _f % 8] = _r
    _half[_p] = _h
"""


@register
class RleZeros:
    name = "b81_rle"
    native_a = False

    def encode(self, q):
        syms = qnet.symbols81(q)
        toks, i = [], 0
        while i < len(syms):
            if syms[i] == ZSYM:
                j = i
                while j < len(syms) and syms[j] == ZSYM and j - i < MAXRUN:
                    j += 1
                if j - i >= 2:
                    toks.append(81 + (j - i) - 2)
                    i = j
                    continue
            toks.append(syms[i])
            i += 1
        # decode check is the harness's job; still, never emit a bad token
        assert all(0 <= t <= 89 for t in toks)
        body = 0
        for t in reversed(toks):
            body = body * 90 + t
        return body, BODY, "%d tokens from 768 chars" % len(toks)
