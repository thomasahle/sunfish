"""(e) Low-rank predictor + exact ternary residual.

torch SVD on the FLOAT weights (pre-quantization, rank swept), factors
ternarized, prediction P = clamp(sum_r uq_r sq_r^T, -1, +1); the stored
residual T - P (in {-2..2}) is sparse-coded.  Bit-exactness is by
construction: decode = rebuild P from the stored factors, add residual.
The factors cost ~768 trits per rank -- at N=4 the residual must lose a
LOT of nonzeros to pay for that, which is precisely the kind of claim
the zoo measures instead of arguing.
"""
from . import register, mixed_pack
from .. import entrysrc

BODY = """\
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
_T = [0] * 3072
for _f in range(768):
    for _k in range(NN):
        _p = 0
        for _q in range(_nr):
            _p += _uq[_q][_f] * _sq[_q][_k]
        _T[_f * NN + _k] = -1 if _p < -1 else (1 if _p > 1 else _p)
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_c = _d + 90 * _e
_w, _d = divmod(_w, 90); _w, _e = divmod(_w, 90)
_R = _d + 90 * _e
_i = -1
for _j in range(_c):
    _w, _d = divmod(_w, _R); _i += _d + 1
    _w, _v = divmod(_w, 4)
    _T[_i] += _v - 2 + (_v > 1)
""" + entrysrc.SRC_HALF_FROM_T


@register
class LowRank:
    name = "lr_svd"
    native_a = False

    def encode(self, q):
        import torch
        E32 = torch.tensor(q.Efloat, dtype=torch.float64) * 32
        U, S, Vt = torch.linalg.svd(E32, full_matrices=False)
        T = [t for row in q.trits for t in row]
        N = len(q.g)

        best = None
        for r in range(1, min(3, N + 1)):
            for pct in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                uqs, sqs = [], []
                for j in range(r):
                    u = U[:, j]
                    th = u.abs().quantile(pct).item()
                    uqs.append([int(torch.sign(x).item()) if abs(x.item()) > th else 0
                                for x in u])
                    sqs.append([int(torch.sign(v).item()) for v in Vt[j]])
                P = [max(-1, min(1, sum(uqs[j][f] * sqs[j][k] for j in range(r))))
                     for f in range(768) for k in range(N)]
                res = [t - p for t, p in zip(T, P)]
                nz = sum(1 for x in res if x)
                # stored-bits estimate ONLY for picking (r, pct) inside the
                # arm; the arm's reported number is still pack.sh's.
                gaps_bits = nz * 14
                cost = r * 768 * 1.585 + gaps_bits
                if best is None or cost < best[0]:
                    best = (cost, r, pct, uqs, sqs, res, nz)

        _, r, pct, uqs, sqs, res, nz = best
        pairs = [(90, r)]
        for j in range(r):
            pairs += [(3, t + 1) for t in uqs[j]]
        for j in range(r):
            pairs += [(3, t + 1) for t in sqs[j]]
        nzs = [(i, v) for i, v in enumerate(res) if v]
        gaps, prev = [], -1
        for i, _ in nzs:
            gaps.append(i - prev - 1)
            prev = i
        R = max(gaps, default=0) + 1
        c = len(nzs)
        pairs += [(90, c % 90), (90, c // 90), (90, R % 90), (90, R // 90)]
        for g, (_, v) in zip(gaps, nzs):
            pairs += [(R, g), (4, v + 2 - (v > 0))]
        body = mixed_pack(pairs)
        return body, BODY, "rank %d @ q%.2f, residual nz %d (%.1f%%)" % (
            r, pct, nz, 100 * nz / 3072)
