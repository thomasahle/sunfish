#!/usr/bin/env python3
"""Per-layer bit-exactness: every torch layer against the actual Python
big-int operation it mirrors, on probe inputs -- forward EXACT, no
tolerances -- plus an end-to-end two-layer probe and the certificate's
refuse/accept behaviour.  Runs standalone (python3 test_packed_layers.py)
and under pytest; it is part of the pipeline's own gate: no pipeline
change lands with this red.

The trained-structure parametrizations (structures.py) are held to the
same standard, because they are layer-1 weight tables and nothing else:
their STE rules are checked as GRADIENT IDENTITIES (not by eyeballing a
loss curve), their weights are checked on the payload grid, the head they
produce is checked bit-exact against python big-int evaluation, and their
bake-off arms are checked to decode -- inside a REAL spliced entry -- to
the same module data the trainer quantization implies.
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
import structures                  # noqa: E402
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


# ------------------------------------------- trained structure (structures.py)

def test_codebook_ste_rule_is_the_documented_one():
    """forward = hard argmin codeword; backward = identity to the shadow and
    softmax-weighted to the book.  Checked as an exact gradient identity, so
    the docstring cannot drift away from the code."""
    torch.manual_seed(0)
    cb = structures.CodebookWeight(64, 4, 5, 8, tau=0.85, temp=0.7)
    w = (torch.randn(64, 4) * constraints.WSCALE).requires_grad_(True)
    y, k = cb(w)
    B = cb.codewords()
    x = (w * 32.0).reshape(cb.nblk, cb.D)
    d = cb._dist(x.detach(), B.detach())
    assert torch.equal(k, d.argmin(-1))
    assert torch.equal(y.reshape(cb.nblk, cb.D) * 32.0, B[k].detach()), \
        "forward must BE the hard codeword"
    assert set(y.mul(32).flatten().tolist()) <= {-1.0, 0.0, 1.0}

    g = torch.randn(64, 4)
    (y * g).sum().backward()
    # identity route to the shadow: d(y)/d(w) = 1 (y and w share the /32)
    assert torch.allclose(w.grad, g, atol=0, rtol=0), "shadow route is not identity"
    # soft route to the book: codeword j collects sum_b p_bj * dL/dY_b in
    # TRIT units (Y = 32*weight, so dL/dY = g/32), through the ternary
    # grid's own 1/WSCALE
    p = torch.softmax(-d / (cb.temp * cb.D), -1)
    want = p.t() @ g.reshape(cb.nblk, cb.D) / (32 * constraints.WSCALE)
    assert torch.allclose(cb.book.grad, want, atol=1e-12), "book route is not soft"


def test_codebook_assignment_moves_with_the_shadow():
    """The only thing that re-assigns a block is the shadow crossing a
    Voronoi boundary -- so a shadow set EQUAL to a codeword must select it."""
    torch.manual_seed(1)
    cb = structures.CodebookWeight(64, 4, 4, 8, tau=0.85, temp=0.5)
    B = cb.codewords().detach()
    w = torch.zeros(64, 4)
    for b in range(cb.nblk):
        w.view(cb.nblk, cb.D)[b] = B[b % cb.K] / 32.0
    y, k = cb(w)
    assert k.tolist() == [b % cb.K for b in range(cb.nblk)]
    assert torch.equal(y, w)


def test_lowrank_epoch0_is_exactly_the_plain_net():
    """V zero-init => U@V = 0 => weight() is bit-identical to the plain
    ternary net at the same seed (the ml2 u2-silent precedent)."""
    kw = dict(N=4, base="mat", ternary=0.85)
    torch.manual_seed(0)
    plain = build_model(cfgmod.ModelCfg(arch="residual", **kw))
    torch.manual_seed(0)
    lr = build_model(cfgmod.ModelCfg(arch="lowrank", lr_rank=2, **kw))
    assert torch.equal(plain.weight(), lr.weight()), "epoch 0 must be the plain net"
    fi = torch.tensor([features.feat("K", 95), features.feat("q", 44)])
    mi = torch.tensor([features.feat("k", 24), features.feat("Q", 75)])
    fo, base = torch.tensor([0]), torch.tensor([12.0])
    assert torch.equal(plain(fi, mi, fo, base), lr(fi, mi, fo, base))
    constraints.check_antisymmetry(lr, fi, mi, fo, base)


def test_lowrank_composite_grid_and_gradients():
    """Composite stays on the certified grid; U and V receive gradient
    through the ternary STE even while V's forward is all zeros."""
    torch.manual_seed(0)
    net = build_model(cfgmod.ModelCfg(arch="lowrank", N=4, base="mat",
                                      ternary=0.85, lr_rank=2))
    w = net.weight()
    assert set((w * 32).flatten().tolist()) <= {-1.0, 0.0, 1.0}
    w.sum().backward()
    assert net.struct.V.grad is not None and net.struct.V.grad.abs().sum() > 0, \
        "a silent factor that cannot wake up is a dead parametrization"
    assert net.raw.grad is not None and net.raw.grad.abs().sum() > 0
    # with V awake the composite still lands on the grid, clipped
    with torch.no_grad():
        net.struct.V.copy_(torch.ones_like(net.struct.V))
        net.struct.U.copy_(torch.ones_like(net.struct.U))
    w = net.weight() * 32
    assert set(w.flatten().tolist()) <= {-1.0, 0.0, 1.0}
    assert (w == 1).all(), "U@V = rank>=1 plus residual, clipped to +wmax"


def _structured_qnet(arch, name, **model_kw):
    """A structured net as the bake-off sees it: real model -> real export
    quantization -> QNet, with the reconstruction asserted (this is the
    export round-trip, in-process)."""
    cfg = cfgmod.ModelCfg(arch=arch, N=4, base="mat", ternary=0.85, **model_kw)
    torch.manual_seed(7)
    net = build_model(cfg)
    with torch.no_grad():                     # wake the structure up
        net.raw.mul_(1.7)
        if arch == "lowrank":
            net.struct.V.copy_(torch.randn_like(net.struct.V) * 0.06)
            net.struct.U.mul_(1.4)
        else:
            net.struct.book.mul_(1.7)
    with torch.no_grad():
        E = net.weight()
    struct = net.export_struct()
    trits = tuple(tuple(int(round(x * 32)) for x in row) for row in E.tolist())
    flat = [t for row in trits for t in row]
    assert structures.reconstruct(struct) == flat, \
        "%s: structures.reconstruct != the exported trits" % arch
    from compress import qnet as qn
    g = [R.randrange(40, 90) for _ in range(4)]
    bd = [R.randrange(0, 90) for _ in range(4)]
    return qn.QNet(name, 3, g, bd, trits, 600, E.tolist(), struct)


def test_structured_head_bitexact_vs_bigint():
    """A structured net's rows, evaluated as the ENGINE does (python big
    ints: offset lanes, SWAR crelu, modular hsum) equal the torch mirror --
    the structure changed the weights, not the arithmetic."""
    for arch, kw in (("cb", {"cb_k": 12, "cb_block": 8}),
                     ("lowrank", {"lr_rank": 2})):
        q = _structured_qnet(arch, "probe_" + arch, **kw)
        clamp = pl.SwarClamp()
        caps = [32 * x for x in q.g]
        for _ in range(10):
            board = random_board()
            us = [q.bd[k] - 44 for k in range(4)]
            them = list(us)
            for s, p in enumerate(board):
                if p in features.PIECES:
                    fu, fm = features.feat(p, s), features.feat(p.swapcase(), 119 - s)
                    for k in range(4):
                        us[k] += q.g[k] * q.trits[fu][k]
                        them[k] += q.g[k] * q.trits[fm][k]
                        assert abs(us[k]) < 1 << 14 and abs(them[k]) < 1 << 14
            yu = pl.bigint_swar_clamp([(1 << 14) + v for v in us], caps, 16)
            cu = clamp(torch.tensor(us, dtype=torch.float64),
                       torch.tensor(caps, dtype=torch.float64))
            assert cu.tolist() == yu, (arch, us, yu)
            assert pl.bigint_hsum(yu, 16) == int(pl.HSum()(cu).item())


def _arm_roundtrip(q, armname, stock):
    """Encode -> splice into the REAL entry -> exec -> the decoded module
    data must equal the independent mirror of the trainer quantization."""
    from compress import qnet as qn, entrysrc
    from compress.arms import all_arms
    arm = next(a for a in all_arms() if a.name == armname)
    body, body_src, note = arm.encode(q)
    full = qn.header_int(q) + qn.HEADER_RADIX * body
    s90 = qn.int_to_s90(full)
    src = entrysrc.replace_region(
        stock, entrysrc.build_region(entrysrc.prologue_a(s90), body_src))
    got, _ = entrysrc.exec_entry(src)
    want = qn.expected_module_data(q)
    for key in ("SHIFT", "MGP", "MGH", "ACC_BASE", "ROWS"):
        assert qn.module_data_of(got)[key] == want[key], (armname, key)
    return len(s90), note


# The entry blob every recorded bake-off table was measured against
# (986fa96 = fb717214c3e2).  A round-trip test needs ONE denominator, and
# the working entry is golfed continuously by another lane -- the same
# reason the harness itself pins.  When compress/ is re-seamed against a
# newer entry, move this pin with it (entrysrc.require_seam names the
# drift; test_entry_seam below reports it without failing this suite).
RECORDED_ENTRY = "986fa96:nnue_4k/replnet_proto.py"


def test_entry_seam_drift_is_visible():
    """Not a gate on this suite -- a REPORT.  The entry belongs to the golf
    lane; if it has moved past the seam, say exactly how, here, where a
    pipeline change is being made."""
    from compress import entrysrc
    for spec in ("HEAD:nnue_4k/replnet_proto.py", RECORDED_ENTRY):
        src, prov = entrysrc.read_entry(spec)
        miss = entrysrc.seam_missing(src)
        print("    seam %s: %s" % (prov, "OK" if not miss else "DRIFTED -- " +
                                   "; ".join(miss)), flush=True)
    src, prov = entrysrc.read_entry(RECORDED_ENTRY)
    assert not entrysrc.seam_missing(src), \
        "the RECORDED entry pin lost the seam -- the pin is wrong, not the entry"


def test_trained_arms_export_roundtrip():
    from compress import entrysrc
    stock, _ = entrysrc.read_entry(RECORDED_ENTRY)
    for arch, armname, kw in (("cb", "trained_cb", {"cb_k": 12, "cb_block": 8}),
                              ("lowrank", "trained_lr", {"lr_rank": 2})):
        q = _structured_qnet(arch, "probe_" + arch, **kw)
        chars, note = _arm_roundtrip(q, armname, stock)
        print("    %s: %d payload chars -- %s" % (armname, chars, note), flush=True)


def test_trained_arms_skip_a_plain_net():
    """No structure, no arm: NotApplicable, never a fitted stand-in."""
    from compress import qnet as qn
    from compress.arms import all_arms, NotApplicable
    trits = tuple(tuple(R.choice((-1, 0, 0, 1)) for _ in range(4)) for _ in range(768))
    q = qn.QNet("plain", 3, [50] * 4, [44] * 4, trits, 600,
                [[t / 32 for t in row] for row in trits])
    assert q.struct is None
    for armname in ("trained_cb", "trained_lr"):
        arm = next(a for a in all_arms() if a.name == armname)
        try:
            arm.encode(q)
        except NotApplicable:
            continue
        raise AssertionError("%s priced a net that carries no structure" % armname)


def test_structure_certificates_refuse_and_accept():
    """The grid ceiling is a NUMBER, and it refuses above itself."""
    assert fb.max_certified_grid() == 5, "gmax 89: 32*5*89 + 45 = 14285 < 2^14"
    probe = fb.Certificate("c6")
    fb.certify_grid_head(probe, 6)
    assert not probe.ok, "|w| = 6*g must fail no-borrow (17133 > 16383)"
    assert fb.certify_trained_cb(32, 8).ok
    assert fb.certify_trained_cb(32, 8, cmax=5).ok
    assert not fb.certify_trained_cb(32, 8, cmax=6).ok
    assert not fb.certify_trained_cb(32, 7).ok, "7 does not divide 768"
    assert not fb.certify_trained_cb(200, 8).ok, "K > 96 blocks stores nothing"
    assert fb.certify_lowrank(1).ok and fb.certify_lowrank(8).ok
    assert not fb.certify_lowrank(1, wmax=6).ok
    # an uncertifiable config must never reach the optimizer
    cfg = cfgmod.ModelCfg(arch="lowrank", N=4, ternary=0.85, lr_wmax=6)
    try:
        fb.certify_or_raise(cfg)
    except SystemExit:
        pass
    else:
        raise AssertionError("certify_or_raise let an uncertifiable config train")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS %s" % fn.__name__, flush=True)
    print("test_packed_layers: %d/%d bit-exactness tests PASS" % (len(fns), len(fns)))


if __name__ == "__main__":
    main()
