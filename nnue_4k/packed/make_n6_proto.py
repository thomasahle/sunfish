#!/usr/bin/env python3
"""Derive the N=6 PRICING variant from replnet_proto.py, mechanically.

The capacity arm's registered width. The shipped proto is 4 lanes x 16 bits =
a 64-bit half and a 128-bit accumulator; N=6 is 6 lanes x 16 bits = a 96-bit
half and a 192-bit accumulator. The template's own comment anticipated the N=8
seam as "range(8), half 128, _R2 = 1|1<<128, and TWO payload chars per
feature"; this takes the same sites to 6 but does NOT take the two-chars-per
-feature route, because two base-90 chars carry 8100 values to encode 729 and
that wastes ~0.9 char per feature. Instead the feature decode pops BASE-3
digits straight out of the payload integer, which is the dense mixed-radix
form the container already is.

WHAT THIS PRICES, and what it does not. The incremental accumulator is NOT a
design to be added here -- it already exists in the shipped template
(Position.move does `acc + row[p][j] - row[p][i]` with capture, castling,
promotion and en-passant deltas, and rotate() leaves acc untouched because
both perspective blocks live in one int). So the -39.4% nps the N=4
replacement family already pays is ALREADY an incremental-accumulator number.
What N=6 changes is the WIDTH of every accumulator operation: 128 -> 192 bits,
which is 5 -> 7 limbs of Python big-int arithmetic per add and per nn_cp mask.
That is what this variant measures.

A string-edit derivation, not a fork: every hunk asserts it hit, so this file
cannot silently drift from the entry it prices (the make_ml2_proto.py
pattern). Weights are random with the measured sparsity -- pricing is a
property of the inference engine, not of the trained values.

usage: make_n6_proto.py [ENTRY] [OUT] [--N 6]
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
N = 6
for a in sys.argv[1:]:
    if a.startswith("--N"):
        N = int(a.split("=")[1]) if "=" in a else 6
src_path = argv[0] if argv else os.path.join(HERE, "..", "replnet_proto.py")
out_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "n6_proto_built.py")
HALF = 16 * N                      # bits per perspective block

with open(src_path) as f:
    s = f.read()

# ---- the base-90 codec, inverted -------------------------------------
# decode: _d = ord(c) - 35; v = _d - (_d > 4) - (_d > 56)
# the two collisions are how the codec skips the characters that would be
# unsafe inside the payload's string literal; take the lowest ord per value.
ENC = {}
for _o in range(35, 128):
    _d = _o - 35
    _v = _d - (_d > 4) - (_d > 56)
    if 0 <= _v <= 89 and _v not in ENC:
        ENC[_v] = chr(_o)
assert len(ENC) == 90, len(ENC)


def emit_payload(N, seed=20260817, zeros=0.50):
    """Mixed-radix payload for the N-lane variant, in the decoder's order.

    Popped first-to-last: SHIFT(90), N gains(90), N biases(90), then
    12 pieces x 64 squares x N trits(3).  So encode last-to-first.
    Gains and sparsity match the measured N=6 export
    (gains [79, 62, 62, 66, 68, 63], zeros 50.1%).
    """
    rng = random.Random(seed)
    fields = [(4, 90)]                                   # SHIFT
    fields += [(rng.randint(62, 79), 90) for _ in range(N)]      # gains
    fields += [(rng.randint(30, 58), 90) for _ in range(N)]      # biases + 44
    trits = []
    for _ in range(12 * 64 * N):
        trits.append(1 if rng.random() < zeros else rng.choice((0, 2)))
    fields += [(t, 3) for t in trits]
    w = 0
    for d, r in reversed(fields):
        w = w * r + d
    chars = []
    while w:
        w, d = divmod(w, 90)
        chars.append(ENC[d])
    return "".join(reversed(chars))


def sub(t):
    """Token substitution, so replacement bodies can contain literal '%'."""
    return (t.replace("@N@", str(N)).replace("@HALF@", str(HALF))
             .replace("@FULL@", str(2 * HALF)))


HUNKS = [(old, sub(new)) for old, new in [
    # 1. widths: the all-ones word, the block replicator
    ("""_U = ((1 << 128) - 1) // 65535
_R2 = 1 | 1 << 64                # replicate one block's word into both""",
     """_U = ((1 << @FULL@) - 1) // 65535
_R2 = 1 | 1 << @HALF@                # replicate one block's word into both"""),
    # 2. gain digits
    ("""for _k in range(4):
    _w, _d = divmod(_w, 90); _g.append(_d); MGP += _d * 32 << 16 * _k""",
     """for _k in range(@N@):
    _w, _d = divmod(_w, 90); _g.append(_d); MGP += _d * 32 << 16 * _k"""),
    # 3. bias digits
    ("""for _k in range(4):
    _w, _d = divmod(_w, 90); _B += _d - 44 << 16 * _k""",
     """for _k in range(@N@):
    _w, _d = divmod(_w, 90); _B += _d - 44 << 16 * _k"""),
    # 4. the feature decode: base-3 digits, not 4-trits-per-char
    ("""    for _f in range(64):
        _w, _d = divmod(_w, 90)
        _h[21 + _f // 8 * 10 + _f % 8] = sum(
            _g[_k] * (_d // 3 ** _k % 3 - 1) << 16 * _k for _k in range(4))""",
     """    for _f in range(64):
        _v = 0
        for _k in range(@N@):
            _w, _d = divmod(_w, 3); _v += _g[_k] * (_d - 1) << 16 * _k
        _h[21 + _f // 8 * 10 + _f % 8] = _v"""),
    # 5. the them-block offset in the shared row table
    ("""_rows0 = {_p: [_half[_p][_s] + (_half[_p.swapcase()][119 - _s] << 64)""",
     """_rows0 = {_p: [_half[_p][_s] + (_half[_p.swapcase()][119 - _s] << @HALF@)"""),
    # 6. the lane-sum fold in the read-out
    ("""    v = y % (1 << 64) % 65535 - (y >> 64) % 65535  # 2^16 == 1 (mod 2^16-1)""",
     """    v = y % (1 << @HALF@) % 65535 - (y >> @HALF@) % 65535  # 2^16 == 1 (mod 2^16-1)"""),
]]

for old, new in HUNKS:
    assert old in s, ("hunk not found -- replnet_proto.py drifted; fix ME, do not "
                      "guess:\n" + old[:160])
    s = s.replace(old, new, 1)

# 7. the payload itself: same field order, N lanes wide
i0 = s.index('for _c in "') + len('for _c in "')
i1 = s.index('"', i0)
s = s[:i0] + emit_payload(N) + s[i1:]

with open(out_path, "w") as f:
    f.write(s)
print(out_path)
