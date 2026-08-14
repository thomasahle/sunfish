#!/usr/bin/env python3
"""Per-layer bit-exactness: every torch layer against the actual Python
big-int operation it mirrors, on probe inputs -- forward EXACT, no
tolerances -- plus an end-to-end two-layer probe and the certificate's
refuse/accept behaviour.  Runs standalone (python3 test_packed_layers.py)
and under pytest; it is part of the pipeline's own gate: no pipeline
change lands with this red.
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfgmod            # noqa: E402
import constraints                 # noqa: E402
import features                    # noqa: E402
import field_budget as fb          # noqa: E402
import packed_layers as pl         # noqa: E402
from model import Ml2Net, build_model  # noqa: E402

R = random.Random(20260814)


def rand_lanes(n, hi):
    return [R.randrange(0, hi + 1) for _ in range(n)]


def test_lane_conv_linear_vs_bigint():
    for _ in range(50):
        na, nb = R.randrange(1, 9), R.randrange(1, 9)
        F = 32
        # sized so every conv coefficient fits F bits: na*hi*hi < 2^32
        hi = int((((1 << F) - 1) / max(na, nb)) ** 0.5)
        a, b = rand_lanes(na, hi), rand_lanes(nb, hi)
        want = pl.bigint_linear_conv(a, b, F)
        conv = pl.LaneConv(na, nb, "linear")
        got = conv(torch.tensor(a, dtype=torch.float64),
                   torch.tensor(b, dtype=torch.float64))
        assert got.tolist() == want, (a, b, got.tolist(), want)


def test_lane_conv_circular_vs_bigint():
    for _ in range(50):
        m = R.choice((2, 4))
        n = R.randrange(m, 17)
        F = 32
        # sized BY THE ANALYZER: shrink until the conv fields are certified
        # (the first draft of this test overflowed a field by sizing on n
        # alone -- the exact bug class the certificate exists to refuse)
        hi = 1 << 15
        while not fb.check_field_nonneg(
                fb.conv_bounds(fb.LaneBounds.uniform(n, 0, hi),
                               fb.LaneBounds.uniform(n, 0, hi), "circular", m), F).ok:
            hi //= 2
        a, b = rand_lanes(n, hi), rand_lanes(n, hi)
        want = pl.bigint_circular_conv(a, b, m, F)
        conv = pl.LaneConv(n, n, "circular", m)
        got = conv(torch.tensor(a, dtype=torch.float64),
                   torch.tensor(b, dtype=torch.float64))
        assert got.tolist() == want, (a, b, m, got.tolist(), want)


def test_swar_clamp_vs_bigint():
    F = 16
    BIAS = 1 << (F - 2)
    for _ in range(200):
        n = R.randrange(1, 12)
        G = [R.randrange(1, 3000) for _ in range(n)]
        vals = [R.randrange(-BIAS, BIAS) for _ in range(n)]
        want = pl.bigint_swar_clamp([BIAS + v for v in vals], G, F)
        got = pl.SwarClamp()(torch.tensor(vals, dtype=torch.float64),
                             torch.tensor(G, dtype=torch.float64))
        assert got.tolist() == want, (vals, G, got.tolist(), want)


def test_hsum_vs_bigint():
    F = 16
    for _ in range(100):
        n = R.randrange(1, 10)
        # respect the modulus precondition, then equality must be exact
        vals = rand_lanes(n, ((1 << F) - 2) // max(n, 1))
        want = pl.bigint_hsum(vals, F)
        got = pl.HSum()(torch.tensor(vals, dtype=torch.float64))
        assert int(got.item()) == want


def test_shift_renorm_vs_engine():
    sr = pl.ShiftRenorm(4)
    vals = [R.randrange(-10 ** 6, 10 ** 6) for _ in range(500)] + [-1, 0, 1, -16, 15]
    want = [(v >> 4) if v >= 0 else -((-v) >> 4) for v in vals]
    got = sr(torch.tensor(vals, dtype=torch.float64))
    assert got.tolist() == want
    # the documented STE rule: backward is g / 2^s
    x = torch.tensor([100.0, -100.0], dtype=torch.float64, requires_grad=True)
    sr(x).sum().backward()
    assert x.grad.tolist() == [1.0 / 16, 1.0 / 16]


def test_conv_gradients_exact():
    """The conv backward is the true bilinear gradient: d/da_i sum_g w_g h_g
    = sum_j w_{(i+j)%m} b_j.  Integer inputs -> exact float64 equality."""
    m, n = 4, 8
    conv = pl.LaneConv(n, n, "circular", m)
    a = torch.tensor(rand_lanes(n, 1000), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(rand_lanes(n, 1000), dtype=torch.float64, requires_grad=True)
    w = torch.tensor(rand_lanes(m, 50), dtype=torch.float64)
    (conv(a, b) * w).sum().backward()
    ga = [sum(w[(i + j) % m].item() * b[j].item() for j in range(n)) for i in range(n)]
    gb = [sum(w[(i + j) % m].item() * a[i].item() for i in range(n)) for j in range(n)]
    assert a.grad.tolist() == ga and b.grad.tolist() == gb


def test_certificate_refuses_f16_accepts_f32():
    assert not fb.certify_ml2(F2=16).ok, "F2=16 must fail no-carry (the field-budget wall)"
    cert = fb.certify_ml2(F2=32)
    assert cert.ok, cert.report()
    depth, detail = fb.max_feasible_depth(
        F=32, m=4, start_hi=2848, shift_policy=lambda h: max(h.absmax().bit_length() - 12, 0))
    assert depth >= 8, detail   # renorm makes depth structural, not lucky


def random_board():
    sq = [r * 10 + f for r in range(2, 10) for f in range(1, 9)]
    R.shuffle(sq)
    b = ["."] * 120
    for i in range(120):
        if not (21 <= i <= 98 and 1 <= i % 10 <= 8):
            b[i] = " "
    pool = ["K", "k"] + R.sample(list("QRRBBNNPPPPPPPPqrrbbnnpppppppp"), R.randrange(0, 26))
    for p in pool:
        b[sq.pop()] = p
    return "".join(b)


def test_end_to_end_two_layer_bitexact():
    """Layer 1 (ternary rows, offset lanes, SWAR crelu) + layer 2 (big-int
    self-multiply, fold mod 2^(32*4)-1, integer read-out, shift) evaluated
    TWO ways on random boards: pure python big-ints (the engine side) and
    the torch layers in float64 (the training mirror).  Bit-exact, and the
    integer form is exactly antisymmetric."""
    N, m, F2, shift2, umax = 4, 4, 32, 10, 127
    trits = [[R.choice((-1, 0, 0, 1)) for _ in range(N)] for _ in range(768)]
    g = [R.randrange(40, 90) for _ in range(N)]
    bd = [R.randrange(0, 90) for _ in range(N)]
    u2 = [R.randrange(-umax, umax + 1) for _ in range(m)]
    conv = pl.LaneConv(N, N, "circular", m)
    clamp = pl.SwarClamp()
    shift = pl.ShiftRenorm(shift2)

    def int_lanes(board):
        us = [bd[k] - 44 for k in range(N)]
        them = [bd[k] - 44 for k in range(N)]
        for s, p in enumerate(board):
            if p in features.PIECES:
                fu, fm = features.feat(p, s), features.feat(p.swapcase(), 119 - s)
                for k in range(N):
                    us[k] += g[k] * trits[fu][k]
                    them[k] += g[k] * trits[fm][k]
        return us, them

    for _ in range(25):
        board = random_board()
        us, them = int_lanes(board)
        # ---- engine side, python big ints all the way
        BIAS = 1 << 14
        yu = pl.bigint_swar_clamp([BIAS + v for v in us], [32 * x for x in g], 16)
        yt = pl.bigint_swar_clamp([BIAS + v for v in them], [32 * x for x in g], 16)
        hu = pl.bigint_circular_conv(yu, yu, m, F2)
        ht = pl.bigint_circular_conv(yt, yt, m, F2)
        acc = sum(w * (a - b) for w, a, b in zip(u2, hu, ht))
        want = (acc >> shift2) if acc >= 0 else -((-acc) >> shift2)
        # ---- torch mirror, float64 exact-int semantics, same modules
        tu = torch.tensor(us, dtype=torch.float64)
        tt = torch.tensor(them, dtype=torch.float64)
        caps = torch.tensor([32 * x for x in g], dtype=torch.float64)
        cu, ct = clamp(tu, caps), clamp(tt, caps)
        assert cu.tolist() == yu and ct.tolist() == yt, "L1 crelu mismatch"
        h = conv(cu, cu) - conv(ct, ct)
        assert h.tolist() == [a - b for a, b in zip(hu, ht)], "L2 conv mismatch"
        out = shift((h * torch.tensor(u2, dtype=torch.float64)).sum())
        assert int(out.item()) == want, (int(out.item()), want)
        # ---- exact antisymmetry of the integer form: swap us/them
        acc_r = sum(w * (b - a) for w, a, b in zip(u2, hu, ht))
        want_r = (acc_r >> shift2) if acc_r >= 0 else -((-acc_r) >> shift2)
        assert want_r == -want, "trunc shift must commute with negation"


def test_ml2_model_antisymmetric_by_construction():
    cfg = cfgmod.ModelCfg(arch="ml2", N=4, base="mat", ternary=0.85)
    torch.manual_seed(0)
    net = build_model(cfg)
    assert isinstance(net, Ml2Net)
    with torch.no_grad():
        net.u2.copy_(torch.randn(4))          # wake the second layer for the probe
    fi = torch.tensor([features.feat("K", 95), features.feat("Q", 44),
                       features.feat("k", 25), features.feat("p", 35),
                       features.feat("K", 95), features.feat("k", 25)])
    mi = torch.tensor([features.feat(p, 119 - s) for p, s in
                       (("k", 95), ("q", 44), ("K", 25), ("P", 35), ("k", 95), ("K", 25))])
    fo = torch.tensor([0, 4])
    base = torch.tensor([37.0, -12.0])
    constraints.check_antisymmetry(net, fi, mi, fo, base)


def test_certified_bounds_are_reachable_not_loose():
    """The analyzer's conv bound is EXACT interval arithmetic: constant
    inputs at the bound achieve it (no hidden slack, so a certificate
    margin is a real margin)."""
    y = fb.LaneBounds.uniform(4, 0, 2848)
    h = fb.conv_bounds(y, y, "circular", 4)
    got = pl.LaneConv(4, 4, "circular", 4)(
        torch.full((4,), 2848.0, dtype=torch.float64),
        torch.full((4,), 2848.0, dtype=torch.float64))
    assert got.tolist() == h.hi


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS %s" % fn.__name__, flush=True)
    print("test_packed_layers: %d/%d bit-exactness tests PASS" % (len(fns), len(fns)))


if __name__ == "__main__":
    main()
