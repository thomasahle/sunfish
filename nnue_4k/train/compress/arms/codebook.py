"""(d) Codebook over weight groups + packed indices.

Lossless VQ: the codebook is the set of DISTINCT blocks of W consecutive
features' symbols (frequency-ordered, so common blocks get small
indices), indices packed in base-|D| mixed radix.  This wins exactly
when blocks repeat -- k-means with residuals is the lossy cousin; the
exact-dedupe form is the honest bit-exact version, and lowrank.py covers
the predictor+residual idea separately.
"""
from collections import Counter

from . import register, mixed_pack
from .. import qnet, entrysrc

BODY = """\
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_nd = _d + 90 * _e
_D = []
for _b in range(_nd):
    _e2 = []
    for _j in range(%d):
        _w, _d = divmod(_w, 90); _e2.append(_d)
    _D.append(_e2)
_S = []
for _b in range(%d):
    _w, _ix = divmod(_w, _nd)
    _S += _D[_ix]
"""


class _Codebook:
    native_a = False

    def __init__(self, w):
        self.w = w
        self.name = "cb%d" % w

    def encode(self, q):
        syms = qnet.symbols81(q)
        nblk = 768 // self.w
        blocks = [tuple(syms[b * self.w:(b + 1) * self.w]) for b in range(nblk)]
        freq = Counter(blocks)
        book = [b for b, _ in freq.most_common()]
        index = {b: i for i, b in enumerate(book)}
        nd = len(book)
        pairs = [(90, nd % 90), (90, nd // 90)]
        for b in book:
            pairs += [(90, s) for s in b]
        pairs += [(nd, index[b]) for b in blocks]
        body = mixed_pack(pairs)
        return body, BODY % (self.w, nblk) + entrysrc.SRC_HALF_FROM_S, \
            "%d distinct %d-feature blocks of %d" % (nd, self.w, nblk)


register(_Codebook(4))
register(_Codebook(8))
