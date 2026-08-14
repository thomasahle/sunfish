"""Pure base-3 mixed radix: the densest structure-blind packing.

3072 trits as one integer sum(t_i * 3^i) -- no char alignment, so the
base-90 rendering of it is pseudo-random and lzma can only entropy-code
it.  This is the reference point separating "alphabet compaction" from
"structure": base-3^4 wastes log2(90/81) = 0.152 bits/char vs this, but
keeps lzma's matches; the ledger (4850894) says structure wins on real
weights.  Re-measured here per net, both layouts, because that is the
whole point of the zoo.
"""
from . import register
from .. import entrysrc

BODY = """\
_T = []
for _f in range(3072):
    _w, _t = divmod(_w, 3)
    _T.append(_t - 1)
""" + entrysrc.SRC_HALF_FROM_T


@register
class MixedRadix3:
    name = "mr3"
    native_a = False

    def encode(self, q):
        body = 0
        for row in reversed(q.trits):
            for t in reversed(row):
                body = body * 3 + t + 1
        return body, BODY, "one base-3 integer, no alignment"
