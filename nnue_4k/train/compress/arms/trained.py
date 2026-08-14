"""Arms that price a structure the net was TRAINED THROUGH.

Both structures were already measured POST HOC and both lost: cb8 +141 B,
lr_svd +326 B (bake-off 2026-08-14) -- a net trained for a dense trit
stream has no reason to repeat blocks or to be low-rank.  train/
structures.py moves them inside the training graph; these arms are how
the result gets PRICED, through the same real pack paths as everything
else.

Controls are kept deliberately: cb8 (post-hoc distinct-block dedupe)
shares this file's codebook decoder EXACTLY, so trained_cb vs cb8 on
their respective nets isolates the stored state; lr_svd (post-hoc SVD +
residual) is the low-rank control.

Neither arm invents structure.  A plain net carries none, and the arm
says so (NotApplicable -> a SKIP row) instead of fitting one -- that is
what makes the comparison a comparison.
"""
from collections import Counter

from . import register, mixed_pack, NotApplicable
from .codebook import book_body, book_pairs
from .. import qnet, entrysrc

LR_BODY = """\
_w, _nr = divmod(_w, 90)
_uq = []
for _q in range(_nr):
    _u2 = []
    for _f in range(768):
        _w, _t = divmod(_w, 3); _u2.append(_t - 1)
    _uq.append(_u2)
_sq = []
for _q in range(_nr):
    _s2 = []
    for _k in range(NN):
        _w, _t = divmod(_w, 3); _s2.append(_t - 1)
    _sq.append(_s2)
_T = []
for _f in range(768):
    for _k in range(NN):
        _T.append(sum(_uq[_q][_f] * _sq[_q][_k] for _q in range(_nr)))
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_c = _d + 90 * _e
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_R = _d + 90 * _e
_i = -1
for _j in range(_c):
    _w, _d = divmod(_w, _R); _i += _d + 1
    _w, _s = divmod(_w, 2); _T[_i] += 2 * _s - 1
_T = [-1 if _t < -1 else (1 if _t > 1 else _t) for _t in _T]
""" + entrysrc.SRC_HALF_FROM_T


def _struct(q, kind):
    st = q.struct
    if not st or st.get("kind") != kind:
        raise NotApplicable("net %s carries no trained '%s' structure "
                            "(train with model.arch=%s)"
                            % (q.name, kind, kind))
    return st


@register
class TrainedCodebook:
    """Trained product quantization: the book and the index stream are
    parameters of the net, not a dedupe of its output."""

    name = "trained_cb"
    native_a = False

    def encode(self, q):
        st = _struct(q, "cb")
        block, N = st["block"], st["N"]
        nblk = 768 // block
        book, assign = st["book"], st["assign"]
        if len(assign) != nblk:
            raise AssertionError("assignment stream is %d blocks, expected %d"
                                 % (len(assign), nblk))

        # encoder-side relabel: frequency order (small indices for common
        # codewords, what lzma likes) and DROP unused entries.  A bijection
        # on indices -- the reconstruction below re-checks it anyway.
        used = [k for k, _ in Counter(assign).most_common()]
        remap = {k: i for i, k in enumerate(used)}
        rows = [book[k] for k in used]
        index = [remap[k] for k in assign]

        # the round-trip, against the payload's own trits (not the arm's
        # arithmetic): decode == the exported quantization, or nothing ships
        flat = []
        for i in index:
            flat += rows[i]
        want = qnet.flat_trits(q)
        if flat != want:
            bad = sum(1 for a, b in zip(flat, want) if a != b)
            raise AssertionError("trained_cb: book+indices rebuild %d/%d trits "
                                 "wrong" % (bad, len(want)))

        syms = [[sum((row[j * N + k] + 1) * 3 ** k for k in range(N))
                 for j in range(block)] for row in rows]
        body = mixed_pack(book_pairs(syms, index))
        return body, book_body(block, nblk), \
            "K=%d trained (%d used of %d), %d-feature blocks x %d" % (
                len(rows), len(rows), st["K"], block, nblk)


@register
class TrainedLowRank:
    """Trained W = clip(U@V + R): the factors and the residual are
    parameters, so the residual is trained sparse instead of left over."""

    name = "trained_lr"
    native_a = False

    def encode(self, q):
        st = _struct(q, "lowrank")
        if st["wmax"] != 1:
            raise NotApplicable(
                "wmax=%d composite is CERTIFIED (grid ceiling 5) but leaves "
                "the shipped codec's representable set -- no b81 denominator "
                "for this net until a dense free-int baseline arm lands"
                % st["wmax"])
        U, V, R, r, N = st["U"], st["V"], st["R"], st["rank"], st["N"]

        # prune residual entries the clip makes invisible: bit-exactness is
        # preserved by the pruning rule itself, and re-checked below.
        res = []
        for f in range(768):
            for k in range(N):
                p = sum(U[f][j] * V[j][k] for j in range(r))
                t = max(-1, min(1, p + R[f][k]))
                res.append(0 if max(-1, min(1, p)) == t else R[f][k])

        flat = []
        for f in range(768):
            for k in range(N):
                p = sum(U[f][j] * V[j][k] for j in range(r))
                flat.append(max(-1, min(1, p + res[f * N + k])))
        want = qnet.flat_trits(q)
        if flat != want:
            bad = sum(1 for a, b in zip(flat, want) if a != b)
            raise AssertionError("trained_lr: U@V+R rebuilds %d/%d trits wrong"
                                 % (bad, len(want)))

        pairs = [(90, r)]
        for j in range(r):
            pairs += [(3, U[f][j] + 1) for f in range(768)]
        for j in range(r):
            pairs += [(3, V[j][k] + 1) for k in range(N)]
        nz = [(i, v) for i, v in enumerate(res) if v]
        gaps, prev = [], -1
        for i, _ in nz:
            gaps.append(i - prev - 1)
            prev = i
        radix = max(gaps, default=0) + 1
        c = len(nz)
        pairs += [(90, c % 90), (90, c // 90), (90, radix % 90), (90, radix // 90)]
        for gap, (_, v) in zip(gaps, nz):
            pairs += [(radix, gap), (2, (v + 1) // 2)]
        pruned = sum(1 for f in range(768) for k in range(N)
                     if R[f][k] and not res[f * N + k])
        return mixed_pack(pairs), LR_BODY, \
            "rank %d, residual nz %d (%.1f%%, %d clip-pruned), gap radix %d" % (
                r, c, 100 * c / (768 * N), pruned, radix)
