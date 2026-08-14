"""(f) Sparse-index encoding: nonzero positions as gaps + signs.

Lane-major stream (zeros cluster per lane), nonzero count C and gap
radix R in the header, then C x (gap base R, sign base 2) interleaved.
Cost ~ C * (log2 R + 1) bits vs dense base-3's 3072 * 1.58: the
crossover sparsity is whatever the table says, not what the formula
says -- lzma sees both encodings differently.
"""
from . import register, mixed_pack
from .. import entrysrc

BODY = """\
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_c = _d + 90 * _e
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_R = _d + 90 * _e
_L = [0] * 3072
_i = -1
for _j in range(_c):
    _w, _d = divmod(_w, _R); _i += _d + 1
    _w, _s = divmod(_w, 2); _L[_i] = 2 * _s - 1
_T = [_L[_k * 768 + _f] for _f in range(768) for _k in range(NN)]
""" + entrysrc.SRC_HALF_FROM_T


@register
class SparseGap:
    name = "sparse_gap"
    native_a = False

    def encode(self, q):
        N = len(q.g)
        lane_major = [q.trits[f][k] for k in range(N) for f in range(768)]
        nz = [(i, t) for i, t in enumerate(lane_major) if t]
        gaps, prev = [], -1
        for i, _ in nz:
            gaps.append(i - prev - 1)
            prev = i
        R = max(gaps, default=0) + 1
        c = len(nz)
        pairs = [(90, c % 90), (90, c // 90), (90, R % 90), (90, R // 90)]
        for g, (_, t) in zip(gaps, nz):
            pairs += [(R, g), (2, (t + 1) // 2)]
        body = mixed_pack(pairs)
        return body, BODY, "%d nonzeros (%.1f%% zeros), gap radix %d" % (
            c, 100 * (1 - c / 3072), R)
