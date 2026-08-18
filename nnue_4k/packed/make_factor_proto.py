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


PHW = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
# Phase-bucket edges: the measured pool10m medians (PORTFOLIO REGISTRATION,
# 2026-08-19).  Constants, not fitted here -- the engine must compute the
# bucket the trainer did.
PH_EDGES = {2: (11,), 4: (4, 11, 20)}


def _selector(buckets, bkind):
    """Source for `_pick(board) -> (us_half, them_half)`, read at the root."""
    if bkind == "pb":
        # Material phase is position-global, so both perspectives take the
        # SAME bucket: B half-tables, not B**2 pairings.
        return ("_PH = %r\n"
                "def _pick(_b):\n"
                "    _v = sum(_PH[_c.upper()] for _c in _b if _c.isalpha())\n"
                "    _k = %s\n"
                "    return _HALVES[_k], _HALVES[_k]"
                % (PHW, " + ".join("(_v > %d)" % e
                                   for e in PH_EDGES[buckets])))
    if bkind == "kb":
        # Own-king RANK BAND, own frame, per side: the board at the root is
        # already in the side-to-move's frame, so "K" is us and "k" is them.
        assert buckets == 2, "kb selector is the rank band: 2 buckets"
        return ("def _pick(_b):\n"
                "    return (_HALVES[_b.index('K') // 10 <= 7],\n"
                "            _HALVES[(119 - _b.index('k')) // 10 <= 7])")
    raise ValueError("bkind must be pb or kb, got %r" % bkind)


def build(src, r, N, lane_bits, zeros, seed=20260817, mirror=False,
          buckets=1, bkind="pb"):
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
    # BUCKETS multiply the stored rows and nothing else: the read-out, the
    # accumulator and Position.move are untouched, because the bucket is
    # resolved into ONE pair of half-tables at the search root -- the same
    # mechanism the entry already uses to swap K_MID/K_END, and the reason
    # this costs no per-move time.  The payload is bucket-major, matching
    # structures.Factored's fold index ((b*12 + p)*nsq + rank*4 + file'),
    # so the trainer's U rows and the decoder's digits are the same order by
    # construction rather than by agreement.
    nrow = 12 * nsq
    nfeat = nrow * buckets
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
    bexpr = ("(_b * 12 + _i)" if buckets > 1 else "_i")
    if nchunk == 1:
        # The whole affordable frontier lives here (r <= 4), so it gets the
        # short form: one digit per feature, one 81-entry lane-word table.
        body = """_V = _q[%d:%d]
_T = [sum(sum((_d // 3 ** _j %% 3 - 1) * (_V[_j * %d + _k] - 44)
              for _j in range(%d)) << %d * _k for _k in range(%d)) for _d in range(81)]
_HALVES = []
for _b in range(%d):
    _half = {}
    for _i, _p in enumerate(_PIECES):
        _half[_p] = _h = [0] * 120
        for _f in range(64):
            _h[21 + _f // 8 * 10 + _f %% 8] = _T[_q[%d + %s * %d + %s]]
    _HALVES.append(_half)""" % (
            voff, foff, N, r, lane_bits, N, buckets, foff, bexpr, nsq,
            "_f // 8 * 4 + min(_f % 8, 7 - _f % 8)" if mirror else "_f")
    else:
        body = """_V = _q[%d:%d]
_T = [[sum((_d // 3 ** _j %% 3 - 1) * (_V[(_c * 4 + _j) * %d + _k] - 44)
           for _j in range(min(4, %d - _c * 4))) << %d * _k for _k in range(%d)]
      for _c in range(%d) for _d in range(81)]
_T = [sum(_t) for _t in _T]
_HALVES = []
for _b in range(%d):
    _half = {}
    for _i, _p in enumerate(_PIECES):
        _half[_p] = _h = [0] * 120
        for _f in range(64):
            _h[21 + _f // 8 * 10 + _f %% 8] = sum(
                _T[_c * 81 + _q[%d + (%s * %d + %s) * %d + _c]]
                for _c in range(%d))
    _HALVES.append(_half)""" % (
            voff, foff, N, r, lane_bits, N, nchunk, buckets, foff, bexpr, nsq,
            "_f // 8 * 4 + min(_f % 8, 7 - _f % 8)" if mirror else "_f",
            nchunk, nchunk)
    sub("""_half = {}
for _i, _p in enumerate(_PIECES):
    _half[_p] = _h = [0] * 120
    for _f in range(64):
        _d = _q[9 + _i * 64 + _f]
        _h[21 + _f // 8 * 10 + _f % 8] = sum(
            _g[_k] * (_d // 3 ** _k % 3 - 1) << 16 * _k for _k in range(4))""", body)

    if buckets > 1:
        # ROWS is rebuilt from the selected halves at the search root, so it
        # must be a LIST (assigned in place) rather than a tuple rebound
        # through `global` -- the slice assignment is cheaper in packed bytes
        # than the global statement it replaces.
        sub("""_rows0 = {_p: [_half[_p][_s] + (_half[_p.swapcase()][119 - _s] << 64)
               for _s in range(120)] for _p in _PIECES}
_rows1 = {_p: [_rows0[_p.swapcase()][119 - _s] for _s in range(120)] for _p in _PIECES}
ROWS = (_rows0, _rows1)""",
            """def _mkrows(_hu, _ht):
    _r0 = {_p: [_hu[_p][_s] + (_ht[_p.swapcase()][119 - _s] << %d)
                for _s in range(120)] for _p in _PIECES}
    return [_r0, {_p: [_r0[_p.swapcase()][119 - _s] for _s in range(120)]
                  for _p in _PIECES}]
%s
ROWS = _mkrows(_HALVES[0], _HALVES[0])""" % (half, _selector(buckets, bkind)))
        # the root already rebuilds the position after the K_MID/K_END swap;
        # the bucket choice rides that existing rebuild for free.
        sub("""        # The carried score was accumulated under the OTHER table.
        pos = self.r = from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp)""",
            """        # Bucket choice is a ROOT decision, exactly like the table swap
        # above, and it rides the rebuild that swap already pays for.
        ROWS[:] = _mkrows(*_pick(pos.board))
        # The carried score was accumulated under the OTHER table.
        pos = self.r = from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp)""")
    else:
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
    ap.add_argument("--buckets", type=int, default=1,
                    help="input buckets B: the first-layer table is B times "
                         "taller and the bucket is chosen at the search root")
    ap.add_argument("--bucket-kind", default="pb", choices=("pb", "kb"),
                    help="pb = material phase (position-global); "
                         "kb = own-king rank band (per side)")
    ap.add_argument("--mirror", action="store_true",
                    help="fold file f onto 7-f: 32 squares per piece, U halves")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=os.path.join(HERE, "factor_proto_built.py"))
    a = ap.parse_args()
    with open(a.entry) as f:
        src = f.read()
    out, ndig, nh = build(src, a.r, a.N, a.lane_bits, a.zeros, a.seed,
                          a.mirror, a.buckets, a.bucket_kind)
    with open(a.out, "w") as f:
        f.write(out)
    os.chmod(a.out, 0o755)
    print("r=%d N=%d lane_bits=%d B=%d(%s): %d payload digits, %d hunks -> %s"
          % (a.r, a.N, a.lane_bits, a.buckets, a.bucket_kind, ndig, nh, a.out),
          file=sys.stderr)


if __name__ == "__main__":
    main()
