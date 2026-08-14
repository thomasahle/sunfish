"""(b) Weight reordering: help lzma find matches by permuting features.

Three honesty classes, all measured:

  b81_boustro / b81_filemajor -- FIXED square reorders (serpentine /
      file-major within each piece plane): no stored permutation at all,
      the decoder pays only an index expression.
  b81_pieceperm -- greedy plane chaining (12 piece planes ordered to
      maximize adjacent equal-symbol counts): the 12-permutation is
      stored in factorial base, 5 base-90 digits, plus an unrank loop.
  reorder_stored -- the full greedy 768-feature similarity chain with
      the WHOLE permutation stored (Lehmer code, log90(768!) ~ 1221
      digits).  The brief says be honest about stored-permutation cost;
      this measures it rather than argues about it.
"""
from . import register, mixed_pack
from .. import qnet, entrysrc

# ---------------------------------------------------------- fixed orders

# stream position (p, f) holds the symbol of square MAP[f] of plane p;
# the decoder writes stored digit into _S[p*64 + MAP[f]].
BOUSTRO = [r * 8 + (7 - c if r % 2 else c) for r in range(8) for c in range(8)]
FILEMAJ = [c * 8 + r for c in range(8) for r in range(8)]

BODY_SQMAP = """\
_mp = %s
_S = [0] * 768
for _f in range(768):
    _w, _d = divmod(_w, 90)
    _S[_f // 64 * 64 + _mp[_f %% 64]] = _d
"""


class _SqOrder:
    native_a = False

    def __init__(self, name, mp, mapexpr, note):
        self.name, self.mp, self.mapexpr, self.note = name, mp, mapexpr, note

    def encode(self, q):
        syms = qnet.symbols81(q)
        stored = [syms[p * 64 + self.mp[f]] for p in range(12) for f in range(64)]
        body = 0
        for s in reversed(stored):
            body = body * 90 + s
        return body, BODY_SQMAP % self.mapexpr + entrysrc.SRC_HALF_FROM_S, self.note


register(_SqOrder("b81_boustro", BOUSTRO,
                  "[_r * 8 + (7 - _c if _r % 2 else _c) for _r in range(8) for _c in range(8)]",
                  "serpentine squares, no stored perm"))
register(_SqOrder("b81_filemajor", FILEMAJ,
                  "[_c * 8 + _r for _c in range(8) for _r in range(8)]",
                  "file-major squares, no stored perm"))


# ------------------------------------------------------- greedy orderings

def _greedy_chain(items, sim):
    """Nearest-neighbour chain maximizing adjacent similarity; extends
    whichever end has the better next hop."""
    n = len(items)
    best = (-1, 0, 1)
    for a in range(n):
        for b in range(a + 1, n):
            s = sim(a, b)
            if s > best[0]:
                best = (s, a, b)
    chain = [best[1], best[2]]
    left = set(range(n)) - set(chain)
    while left:
        h = max(left, key=lambda x: sim(chain[0], x))
        t = max(left, key=lambda x: sim(chain[-1], x))
        if sim(chain[0], h) >= sim(chain[-1], t):
            chain.insert(0, h)
            left.remove(h)
        else:
            chain.append(t)
            left.remove(t)
    return chain


def _lehmer(perm):
    """Lehmer code of perm as decode pops: for b in (n..1): pop d selects
    perm's next element from the remaining list."""
    left = list(range(len(perm)))
    pops = []
    for p in perm:
        d = left.index(p)
        pops.append((len(left), d))
        left.pop(d)
    return pops


BODY_PIECEPERM = """\
_pp = []
_o = list(range(12))
for _b in range(12, 0, -1):
    _w, _d = divmod(_w, _b)
    _pp.append(_o.pop(_d))
_S = [0] * 768
for _p in _pp:
    for _f in range(64):
        _w, _d = divmod(_w, 90)
        _S[_p * 64 + _f] = _d
""" + entrysrc.SRC_HALF_FROM_S


@register
class PiecePerm:
    name = "b81_pieceperm"
    native_a = False

    def encode(self, q):
        syms = qnet.symbols81(q)
        planes = [syms[p * 64:(p + 1) * 64] for p in range(12)]

        def sim(a, b):
            return sum(x == y for x, y in zip(planes[a], planes[b]))
        chain = _greedy_chain(planes, sim)
        pops = _lehmer(chain)
        stored = [(90, s) for p in chain for s in planes[p]]
        body = mixed_pack(pops + stored)
        return body, BODY_PIECEPERM, "greedy plane chain %s, perm stored (12!)" % chain


BODY_STOREDPERM = """\
_pp = []
_o = list(range(768))
for _b in range(768, 0, -1):
    _w, _d = divmod(_w, _b)
    _pp.append(_o.pop(_d))
_S = [0] * 768
for _f in range(768):
    _w, _d = divmod(_w, 90)
    _S[_pp[_f]] = _d
""" + entrysrc.SRC_HALF_FROM_S


@register
class StoredPerm:
    name = "reorder_stored"
    native_a = False

    def encode(self, q):
        import numpy as np
        syms = np.array(qnet.symbols81(q))
        trits = np.array(q.trits)                      # 768 x N
        n = 768
        # greedy nearest-neighbour on trit-row similarity, vectorized
        used = np.zeros(n, bool)
        chain = [0]                                    # start anywhere; greedy from there
        used[chain[0]] = True
        for _ in range(n - 1):
            cur = trits[chain[-1]]
            s = (trits == cur).sum(1) + (syms == syms[chain[-1]])
            s[used] = -1
            nxt = int(np.argmax(s))
            chain.append(nxt)
            used[nxt] = True
        pops = _lehmer(chain)
        stored = [(90, int(syms[f])) for f in chain]
        body = mixed_pack(pops + stored)
        return body, BODY_STOREDPERM, "full 768-perm STORED (honest cost)"
