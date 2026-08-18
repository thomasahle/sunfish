#!/usr/bin/env python3
"""Derive the FACTORED pricing variant from replnet_proto.py, mechanically.

The shipped replacement net stores a free ternary weight per (feature, lane):
768 x N trits, one lane per nonlinear unit.  That format welds two numbers
together that do not have to be equal -- the number of INPUT DIRECTIONS the
net looks along, and the number of clipped-relu UNITS it spends on them.
This variant unwelds them:

    W[f, k] = sum_j U[f, j] * V[j, k]        f < 768, j < r, k < N

`U` is the same ternary 768 x r table the shipped format already stores (and
at r = 4 it is byte-for-byte the same stream); `V` is an r x N integer mixing
matrix costing r*N payload digits.  Because the product is folded into the
weight table AT LOAD, the accumulator, `Position.move`'s delta and `nn_cp` are
untouched -- the hot loop never learns that a factorisation happened.  The
shipped net is the special case r = N, V = diag(gains).

So the factored form is a strict generalisation reachable at N > r for the
price of r*N digits, and the question it exists to answer is whether N
clipped-relu units over r directions beat r units over r directions.

The reconstruction is PURE PYTHON and costs no extra code: the per-feature
work in the shipped decoder is already a function of one payload digit alone,
so hoisting it into a 3^r-entry lookup table both absorbs the mixing and
removes the inner loop.  numpy is not needed and is not imported.

Weights are RANDOM at a stated sparsity: this file prices the inference
engine and its container, not a trained net.  Every hunk asserts it hit, so
the variant cannot silently drift from the entry it prices (the
make_ml2_proto.py / make_n6_proto.py pattern).

usage: make_factor_proto.py --r 4 --N 32 [--lane-bits 16] [--zeros 0.43]
                            [--out PATH] [ENTRY]
"""
import argparse
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def enc(d):
    """Entry codec, inverted: digit -> byte, skipping only the backslash."""
    assert 0 <= d < 90
    return chr(35 + d + (35 + d >= 92))


def emit_payload(r, N, cap_bits, zeros, seed=20260817, nfeat=768):
    """LSB-first digit stream: shift, N caps, N biases, r*N mixing, 768*r trits.

    Trits are packed FOUR to a digit as a flat stream (values 0..80), which is
    what keeps the trit groups char-aligned -- the property the ledger's
    "base-3 and lzma COMPOSE" measurement depends on -- and it works for any
    r, unlike one-digit-per-feature which caps at r = 4.
    """
    rng = random.Random(seed)
    q = [4]                                                  # SHIFT
    q += [rng.randint(40, 80) for _ in range(N)]             # per-lane caps
    q += [rng.randint(30, 58) for _ in range(N)]             # biases + 44
    q += [rng.randint(44 - 12, 44 + 12) for _ in range(r * N)]   # V + 44
    # FEATURE-MAJOR, ceil(r/4) digits per feature (the last group zero-padded):
    # at r = 4 that is exactly one digit per feature and the stream is
    # byte-for-byte the shape the shipped N=4 payload already ships.
    for _f in range(nfeat):
        u = [0 if rng.random() < zeros else rng.choice((-1, 1)) for _ in range(r)]
        for c in range(0, r, 4):
            grp = u[c:c + 4]
            q.append(sum((t + 1) * 3 ** j for j, t in enumerate(grp)))
    assert all(0 <= d < 90 for d in q)
    return q


def build(src, r, N, lane_bits, zeros, seed=20260817, mirror=False):
    half = lane_bits * N
    full = 2 * half
    lmask = (1 << lane_bits) - 1          # 2^lane_bits == 1 (mod lmask)
    vbits = lane_bits - 1                 # sign bit is the top lane bit
    vmask = (1 << vbits) - 1
    lo = 1 << (vbits - 1)                 # offset-binary zero point
    # HORIZONTAL MIRRORING folds file f onto 7-f, so a piece has 32 feature
    # rows instead of 64 and U halves.  Chess is very nearly file-symmetric
    # (castling is the exception), and width is what this family pays for.
    nsq = 32 if mirror else 64
    nfeat = 12 * nsq
    ntrit = nfeat * r
    ndig = 1 + 2 * N + r * N + nfeat * ((r + 3) // 4)
    # Cap scale, and it MUST agree with the trainer.  The engine's cap is
    # G_k = cap_scale * g_k and the trainer's gvb() stores g_k = round(|v_k| *
    # 2^s / 32), i.e. it assumes G_k = 32 * g_k.  This was 1 << (vbits-7) =
    # 256 at lane_bits=16, chosen to let the cap reach the wider dynamic range
    # a factored table can produce -- which would have made every exported
    # cap 8x too large the first time a trained net met this decoder.  32 is
    # the shipped convention and costs nothing: a lane saturates at 32*g_k
    # either way, and the trained net simply scales U@V to suit.
    cap_scale = 32

    hunks = []

    def sub(old, new):
        nonlocal src
        assert src.count(old) == 1, "hunk missed (%d hits): %.60r" % (src.count(old), old)
        src = src.replace(old, new)
        hunks.append(old.split("\n")[0][:44])

    # --- lane geometry -------------------------------------------------
    sub("_U = ((1 << 128) - 1) // 65535",
        "_U = ((1 << %d) - 1) // %d" % (full, lmask))
    sub("_R2 = 1 | 1 << 64", "_R2 = 1 | 1 << %d" % half)
    sub("MH, MVAL, MLO = _U << 15, _U * 32767, _U << 14",
        "MH, MVAL, MLO = _U << %d, _U * %d, _U << %d" % (vbits, vmask, vbits - 1))

    # --- header decode: caps, biases, and the r x N mixing matrix ------
    sub("""_g, _B, MGP = _q[1:5], 0, 0
for _k in range(4):
    MGP += _g[_k] * 32 << 16 * _k
    _B += _q[5 + _k] - 44 << 16 * _k""",
        """_g, _B, MGP = _q[1:%d], 0, 0
for _k in range(%d):
    MGP += _g[_k] * %d << %d * _k
    _B += _q[%d + _k] - 44 << %d * _k""" % (1 + N, N, cap_scale, lane_bits,
                                            1 + N, lane_bits))

    # --- the factored reconstruction -----------------------------------
    # _T[d] is the packed lane word for one 4-trit payload digit's worth of
    # U-columns; a feature's row is the sum of its ceil(r/4) digit words, so
    # the mixing V is applied once per (digit value, lane) and never per
    # feature.  Exact integer arithmetic; no numpy.
    nchunk = (r + 3) // 4                # base-90 digits per feature
    voff, foff = 1 + 2 * N, 1 + 2 * N + r * N
    if nchunk == 1:
        # The whole affordable frontier lives here (r <= 4), so it gets the
        # short form: one digit per feature, one 81-entry lane-word table.
        body = """_V = _q[%d:%d]
_T = [sum(sum((_d // 3 ** _j %% 3 - 1) * (_V[_j * %d + _k] - 44)
              for _j in range(%d)) << %d * _k for _k in range(%d)) for _d in range(81)]
_half = {}
for _i, _p in enumerate(_PIECES):
    _half[_p] = _h = [0] * 120
    for _f in range(64):
        _h[21 + _f // 8 * 10 + _f %% 8] = _T[_q[%d + _i * %d + %s]]""" % (
            voff, foff, N, r, lane_bits, N, foff, nsq,
            "_f // 8 * 4 + min(_f % 8, 7 - _f % 8)" if mirror else "_f")
    else:
        body = """_V = _q[%d:%d]
_T = [[sum((_d // 3 ** _j %% 3 - 1) * (_V[(_c * 4 + _j) * %d + _k] - 44)
           for _j in range(min(4, %d - _c * 4))) << %d * _k for _k in range(%d)]
      for _c in range(%d) for _d in range(81)]
_T = [sum(_t) for _t in _T]
_half = {}
for _i, _p in enumerate(_PIECES):
    _half[_p] = _h = [0] * 120
    for _f in range(64):
        _h[21 + _f // 8 * 10 + _f %% 8] = sum(
            _T[_c * 81 + _q[%d + (_i * %d + %s) * %d + _c]]
            for _c in range(%d))""" % (
            voff, foff, N, r, lane_bits, N, nchunk, foff, nsq,
            "_f // 8 * 4 + min(_f % 8, 7 - _f % 8)" if mirror else "_f",
            nchunk, nchunk)
    sub("""_half = {}
for _i, _p in enumerate(_PIECES):
    _half[_p] = _h = [0] * 120
    for _f in range(64):
        _d = _q[9 + _i * 64 + _f]
        _h[21 + _f // 8 * 10 + _f % 8] = sum(
            _g[_k] * (_d // 3 ** _k % 3 - 1) << 16 * _k for _k in range(4))""", body)

    sub("(_half[_p.swapcase()][119 - _s] << 64)",
        "(_half[_p.swapcase()][119 - _s] << %d)" % half)

    # --- nn_cp: the read-out masks and the lane sum --------------------
    sub("""    m = ((acc & MLO) >> 14) * 32767             # lane >= 0 ?
    y = ((acc & m) | MLO) - MLO                 # relu
    m = (((MGH - y) & MH) >> 15) * 32767        # lane <= G_k ?
    y = (y & m) | (MGP & (m ^ MVAL))            # capped at G_k
    v = y % (1 << 64) % 65535 - (y >> 64) % 65535  # 2^16 == 1 (mod 2^16-1)""",
        """    m = ((acc & MLO) >> %d) * %d             # lane >= 0 ?
    y = ((acc & m) | MLO) - MLO                 # relu
    m = (((MGH - y) & MH) >> %d) * %d        # lane <= G_k ?
    y = (y & m) | (MGP & (m ^ MVAL))            # capped at G_k
    v = y %% (1 << %d) %% %d - (y >> %d) %% %d  # 2^%d == 1 (mod 2^%d-1)""" % (
            vbits - 1, vmask, vbits, vmask, half, lmask, half, lmask,
            lane_bits, lane_bits))

    # --- the payload itself --------------------------------------------
    q = emit_payload(r, N, vbits, zeros, seed, nfeat)
    # the trit stream is flat, but the feature loop above reads ceil(r/4)
    # digits per feature, so the two must agree on the digit count
    assert len(q) == ndig, (len(q), ndig)
    body = "".join(enc(d) for d in q)
    pat = re.compile(r'b"[^"]*"')
    m = max(pat.finditer(src), key=lambda m: len(m.group()))
    assert len(m.group()) > 400, "payload literal not found"
    src = src[:m.start()] + 'b"%s"' % body + src[m.end():]
    hunks.append("payload literal")
    return src, ndig, len(hunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?",
                    default=os.path.join(HERE, "..", "replnet_proto.py"))
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--lane-bits", type=int, default=16)
    ap.add_argument("--zeros", type=float, default=0.43)
    ap.add_argument("--mirror", action="store_true",
                    help="fold file f onto 7-f: 32 squares per piece, U halves")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=os.path.join(HERE, "factor_proto_built.py"))
    a = ap.parse_args()
    with open(a.entry) as f:
        src = f.read()
    out, ndig, nh = build(src, a.r, a.N, a.lane_bits, a.zeros, a.seed, a.mirror)
    with open(a.out, "w") as f:
        f.write(out)
    os.chmod(a.out, 0o755)
    print("r=%d N=%d lane_bits=%d: %d payload digits, %d hunks -> %s"
          % (a.r, a.N, a.lane_bits, ndig, nh, a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
