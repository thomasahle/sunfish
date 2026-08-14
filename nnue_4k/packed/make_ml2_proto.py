#!/usr/bin/env python3
"""Derive the ml2 PRICING variant from replnet_proto.py, mechanically.

The certified two-layer form (field_budget.certify_ml2, F2=32 m=4 umax=127
shift2=10; trained twin = packed_layers.LaneConv "circular"): after the
shipped F=16 crelu head, re-space each perspective block's 4 capped lanes
to 32-bit fields (two shift+mask steps), ONE squaring per block folded
mod 2^128-1 (the circular self-convolution), signed per-field u2 read-out
by mask+shift (the certificate's group-hsum precondition FAILS, so hsum is
illegal here), >> 10, added to the L1 cp before the clip.  Payload gains
4 u2 values as offset-4050 base-90 digit PAIRS between the biases and the
feature chars (make_proto_payload.py --u2 4 emits the same layout).

A string-edit derivation, not a fork: every hunk asserts it hit, so this
file cannot silently drift from the entry it prices (the make_variants.py
pattern).  Verification: packed/ml2_check.py (independent decode + the
packed_layers int-bridge as reference, bit-exact).  Priced 2026-08-15,
MEASUREMENTS.md: machinery +98 B over the round-2 code floor.

usage: make_ml2_proto.py [ENTRY] [OUT]     (defaults: ../replnet_proto.py,
                                            stdout path printed)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "replnet_proto.py")
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "ml2_proto_built.py")

with open(src_path) as f:
    s = f.read()

HUNKS = [
    # 1. layer-2 constants beside the layer-1 masks
    ("""MH, MVAL, MLO = _U << 15, _U * 32767, _U << 14
_PIECES = "PNBRQKpnbrqk"
""",
     """MH, MVAL, MLO = _U << 15, _U * 32767, _U << 14
# --- ml2 SECOND LAYER (certified F2=32, m=4, shift2=10; field_budget.certify_ml2):
# re-space the capped 16-bit lanes to 32-bit fields, ONE squaring per block
# folded mod 2^128-1 (circular self-convolution), signed per-field read-out
# (the certificate: group hsum EXCEEDS 2^32-2, so mask+shift, never hsum).
M32 = (1 << 32) - 1
MSP = 65535 | 65535 << 64        # fields 0 and 2 of the spread layout
MF = (1 << 128) - 1              # the circular fold: 2^(32*4) == 1
_PIECES = "PNBRQKpnbrqk"
"""),
    # 2. decode: u2 digit pairs between biases and features
    ("""for _k in range(4):
    _w, _d = divmod(_w, 90); _B += _d - 44 << 16 * _k
MGP *= _R2""",
     """for _k in range(4):
    _w, _d = divmod(_w, 90); _B += _d - 44 << 16 * _k
U2 = []
for _k in range(4):
    _w, _d = divmod(_w, 8100); U2.append(_d - 4050)   # layer-2 read-out, |u| <= 127
MGP *= _R2"""),
    # 3. nn_cp: the second multiply between the lane sums and the clip
    ("""    v = y % (1 << 64) % 65535 - (y >> 64) % 65535  # 2^16 == 1 (mod 2^16-1)
    if pf:
        v = -v
    # int(v / 2^s) is EXACT (|v| <= sum of lane caps 11392 << 2^53) and
    # truncates toward zero -- same result as the branchy shift pair.
    return max(-CLAMP, min(CLAMP, int(v / (1 << SHIFT))))""",
     """    v = y % (1 << 64) % 65535 - (y >> 64) % 65535  # 2^16 == 1 (mod 2^16-1)
    # --- layer 2: spread each block's lanes to 32-bit fields (two shift+mask
    # steps), square (= circular self-convolution after the mod-2^128-1 fold),
    # then the signed u2 read-out per field and the certified >> 10.
    a = y % (1 << 64)
    a = (a & M32) | (a >> 32) << 64
    a = (a & MSP) | (a & MSP << 16) << 16
    b = y >> 64
    b = (b & M32) | (b >> 32) << 64
    b = (b & MSP) | (b & MSP << 16) << 16
    a = a * a % MF
    b = b * b % MF
    w = (U2[0] * ((a & M32) - (b & M32))
         + U2[1] * ((a >> 32 & M32) - (b >> 32 & M32))
         + U2[2] * ((a >> 64 & M32) - (b >> 64 & M32))
         + U2[3] * ((a >> 96) - (b >> 96)))
    # two separate certified renorms (L1 >> SHIFT, L2 >> 10), both exact
    # trunc-toward-zero (values << 2^53), then one mover sign flip.
    v = int(v / (1 << SHIFT)) + int(w / 1024)
    if pf:
        v = -v
    return max(-CLAMP, min(CLAMP, v))"""),
]

for old, new in HUNKS:
    assert old in s, "hunk not found -- replnet_proto.py drifted; fix ME, do not guess:\n" + old[:120]
    s = s.replace(old, new, 1)

with open(out_path, "w") as f:
    f.write(s)
print(out_path)
