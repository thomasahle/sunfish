"""(g) Significance/plane ordering: lane-major char packing.

The stock char interleaves the 4 lanes of one FEATURE.  This arm packs 4
consecutive features of one LANE per char instead: sparsity differs per
lane (a mostly-zero lane becomes long runs of the all-zero char), which
is exactly the structure lzma's match finder rewards.  Same digit count,
same alphabet, decoder pays only a different index expression.
"""
from . import register
from .. import entrysrc

BODY = """\
_T = [0] * 3072
for _k in range(NN):
    for _j in range(192):
        _w, _d = divmod(_w, 90)
        for _m in range(4):
            _d, _t = divmod(_d, 3)
            _T[(_j * 4 + _m) * NN + _k] = _t - 1
""" + entrysrc.SRC_HALF_FROM_T


@register
class LaneSplit:
    name = "b81_lanesplit"
    native_a = False

    def encode(self, q):
        pairs = []
        for k in range(len(q.g)):
            for j in range(192):
                sym = sum((q.trits[j * 4 + m][k] + 1) * 3 ** m for m in range(4))
                pairs.append(sym)
        body = 0
        for s in reversed(pairs):
            body = body * 90 + s
        return body, BODY, "4 same-lane features per char"
