"""Negative control: the axis must be able to fail.

ctrl_shuffle stores the SAME 768 base-81 symbols in a seeded-random
order (storage-free: the decoder regenerates the permutation from the
seed).  All spatial structure lzma exploits is destroyed while entropy,
digit count and header stay identical -- so layout A must measure
clearly WORSE than b81, and layout B worse by about the unshuffle
decoder's cost.  If this arm ever ranks near the baseline, the
instrument is broken and no other row of the table can be trusted.
"""
import random

from . import register
from .. import qnet, entrysrc

SEED = 20260814

BODY = """\
import random
_L = []
for _f in range(768):
    _w, _d = divmod(_w, 90); _L.append(_d)
_pm = list(range(768))
random.Random(%d).shuffle(_pm)
_S = [0] * 768
for _f in range(768):
    _S[_pm[_f]] = _L[_f]
""" % SEED + entrysrc.SRC_HALF_FROM_S


@register
class CtrlShuffle:
    name = "ctrl_shuffle"
    native_a = False

    def encode(self, q):
        syms = qnet.symbols81(q)
        pm = list(range(768))
        random.Random(SEED).shuffle(pm)
        # decoder does _S[pm[i]] = stored[i]  =>  stored[i] = S[pm[i]]
        stored = [syms[pm[i]] for i in range(768)]
        body = 0
        for s in reversed(stored):
            body = body * 90 + s
        return body, BODY, "NEGATIVE CONTROL: seeded shuffle, must lose"
