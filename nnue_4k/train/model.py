"""The net family: 768 -> N -> 1 residual over a fixed base, exactly
antisymmetric BY CONSTRUCTION -- one shared first-layer table read from
both perspectives, one output weight serving both with opposite sign,
every extension odd (h, rff) or confined to the even slot (f in the tail).
No penalty enforces this; the architecture cannot express an asymmetric
evaluation (constraints.check_antisymmetry verifies the claim on probes).

This is train_packed.py's Net, made config-driven and modular.  Two things
are deliberately preserved bit-for-bit for ledger comparability:

  * the FORWARD math (same ops in the same order), and
  * the INIT RNG ORDER for the plain/ternary family: raw randn, v randn,
    then the ternary v re-init randn -- so torch.manual_seed(seed) gives
    the SAME initial weights as a historical train_packed.py run.
"""
import torch
import torch.nn as nn

import constraints
from constraints import WSCALE


def act_fn(segs):
    """Convex piecewise-linear activation, normalised so phi(0)=0, phi(1)=1.
    segs=(0.0,) is clipped ReLU; k=3 tracks squared clipped ReLU."""
    A = sum(1.0 - t for t in segs)

    def act(a):
        s = torch.relu(a)
        for t in segs[1:]:
            s = s + torch.relu(a - t)
        return s.clamp(max=A) / A
    return act


def circ(x, y, m):
    """Circular convolution of (batch, m) group sums -- what the packed
    fold modulo 2^(16m)-1 reads out of a lane product."""
    return torch.stack([sum(x[:, s] * y[:, (g - s) % m] for s in range(m))
                        for g in range(m)], -1)


class ResidualNet(nn.Module):
    """pred_cp = base_cp + clip(residual, +-clampcp).

    First layer is factorized (nnue-pytorch's feature factorizer at our
    scale): W_b[f] = raw[b*768+f] + shared[f] + typ[f//64]; weight() -- the
    folded (B*768, N) table -- is what gets exported, so factorization is
    free at run time.  The ternary path quantizes the EFFECTIVE weight by
    STE inside forward (constraints.ternary_ste)."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.ternary and (cfg.kb > 1 or cfg.nb or cfg.rff or cfg.phase or cfg.segs != 1):
            raise ValueError("ternary is the packed replnet path: kb=1, plain crelu, "
                             "no extensions -- the payload codec carries none of that")
        self.cfg = cfg
        N, B = cfg.N, cfg.kb
        self.segs = tuple(i / cfg.segs for i in range(cfg.segs))
        self.act = act_fn(self.segs)
        self.clampcp = float(cfg.clampcp)
        # --- init order matters for repro; see module docstring
        self.raw = nn.Parameter(torch.randn(B * 768, N) * WSCALE)
        self.shared = nn.Parameter(torch.zeros(768, N)) if B > 1 else None
        self.typ = nn.Parameter(torch.zeros(12, N)) if cfg.factor else None
        self.bias = nn.Parameter(torch.zeros(N) + 0.1)
        self.v = nn.Parameter(torch.randn(N) * (25.0 / N ** 0.5))
        if cfg.ternary:
            # v_k is the cp value of a SATURATED lane (cap = 32*g_k, g <= 89):
            # the default ~12cp init is a dead net AdamW takes thousands of
            # steps to wake.  Bias starts inside its exportable band.
            self.v = nn.Parameter(120.0 + torch.randn(N).abs() * 15.0)
            self.bias = nn.Parameter(torch.zeros(N) + 0.02)
        self.gridste = cfg.gridste
        self.nb, self.m, self.tailw = cfg.nb, cfg.bm, cfg.tailw
        self.phase, self.baff, self.nb2, self.rff = cfg.phase, cfg.baff, cfg.nb2, cfg.rff
        if cfg.phase:
            self.s = nn.Parameter(torch.ones(cfg.phase))
        if cfg.rff:
            self.theta = nn.Parameter(torch.randn(768, cfg.rff) * cfg.rffsigma)
            self.phb = nn.Parameter(torch.zeros(cfg.rff))
            self.rw = nn.Parameter(torch.zeros(cfg.rff))
        if cfg.nb:
            nu = cfg.nb * (2 if cfg.nb2 else 1)
            self.rawb = nn.Parameter(torch.randn(768, nu) * WSCALE)
            self.biasb = nn.Parameter(torch.zeros(nu) + 0.1)
            self.gb = nn.Parameter(torch.ones(nu))
            self.u = nn.Parameter(torch.zeros(cfg.bm))
            self.register_buffer("gidx", torch.arange(cfg.nb) % cfg.bm)
            if cfg.baff:
                self.w1 = nn.Parameter(torch.zeros(cfg.bm))
                self.w2 = nn.Parameter(torch.zeros(cfg.bm)) if cfg.nb2 else None
            if cfg.tailw:
                self.t1 = nn.Linear(1 + 2 * cfg.bm, cfg.tailw)
                self.t2 = nn.Linear(cfg.tailw, 1)
                nn.init.zeros_(self.t2.weight)
                nn.init.zeros_(self.t2.bias)

    @property
    def B(self):
        return self.cfg.kb

    def virt(self):
        v = 0
        if self.typ is not None:
            v = self.typ.repeat_interleave(64, 0)
        if self.shared is not None:
            v = v + self.shared
        return v

    def folded(self):
        """The EFFECTIVE float table (raw + virtual features), pre-grid."""
        v = self.virt()
        if isinstance(v, int):
            return self.raw
        N = self.raw.shape[1]
        return (self.raw.view(self.B, 768, N) + v).view(self.B * 768, N)

    def weight(self):
        w = self.folded()
        if self.cfg.ternary:
            w, self._u = constraints.ternary_ste(w, self.cfg.ternary)
        return w

    def clamp_weights(self, wclip):
        """Enforce the clip on the EFFECTIVE weight -- what is exported."""
        with torch.no_grad():
            v = self.virt()
            if isinstance(v, int):
                self.raw.clamp_(-wclip, wclip)
            else:
                N = self.raw.shape[1]
                r = self.raw.view(self.B, 768, N)
                r.copy_((r + v).clamp(-wclip, wclip) - v)
            if self.nb:
                self.rawb.clamp_(-wclip, wclip)
            if self.cfg.ternary:
                # PER-LANE exportable bias range, the bound the exporter
                # actually enforces: it stores bd_k = round(b_k*32*g_k) in
                # [-44, 45], so b_k lives in [-44/(32*g_k), 45/(32*g_k)] and
                # the bound MOVES with the lane's gain.
                #
                # This was a hardcoded +-0.019 for every lane, which is the
                # bound at g=72 and wrong everywhere else in BOTH directions:
                # too loose above g=72, so the exporter had to truncate (that
                # is what the "bias digits CLIPPED" notice reported), and too
                # tight below it, so representable range went unused.  Worse,
                # gains GROW during training, so the true bound shrinks while
                # the constant did not: a self-tightening squeeze.  The N=6
                # seed-0 run came out with all six biases at exactly
                # +-0.019000 -- every one of them pinned on the rail, with no
                # freedom left -- and its val moved 0.2% across six passes
                # over 10M positions while train loss ROSE 9.6%.
                #
                # Fixing it makes the trainer and the exporter agree on one
                # bound.  That is the same defect class as the payload codec
                # (an encoder grew a case its decoder lacked), closed here
                # from the other side.
                s = self.export_shift()
                g = torch.clamp(torch.round(self.v.abs() * (1 << s) / 32.0), 1.0, 89.0)
                lo, hi = -44.0 / (32.0 * g), 45.0 / (32.0 * g)
                self.bias.copy_(torch.min(torch.max(self.bias, lo), hi))

    def export_shift(self):
        """The L1 shift the exporter will pick -- export.export_ml2 /
        export_replnet's rule, verbatim, read off the RAW v the exporter is
        handed (`model.v`), never off a snapped copy: the payload carries one
        shift and both layers must be scaled by that same number."""
        with torch.no_grad():
            vmax = float(self.v.abs().max())
        for s in range(8, -1, -1):
            if vmax * (1 << s) / 32.0 <= 89.49:
                return s
        return 0

    def gvb(self):
        """(v, bias) AS THE PAYLOAD WILL CARRY THEM.

        The payload stores integers -- gain digits g_k = round(v_k*2^s/32)
        capped at 89, and bias digits bd_k = round(b_k*32*g_k) clipped to
        [-44, 45] -- so the values the ENGINE evaluates are v = 32*g/2^s and
        b = bd/(32*g), not the floats the optimizer holds.  With
        model.gridste the snap happens INSIDE forward (straight-through), the
        way the ternary weights already do it and the way `u2grid` now does
        the layer-2 read-out: after 2026-08-15 the campaign's standing rule is
        that a net trains under the resolution its artifact has, from step 1.

        Off (the default) this is the identity, so every historical config
        reproduces bit-for-bit."""
        vab = self.v.abs() if self.cfg.ternary else self.v
        if not (self.gridste and self.cfg.ternary):
            return vab, self.bias
        s = self.export_shift()
        g = torch.clamp(torch.round(vab * (1 << s) / 32.0), 0.0, 89.0)
        v_q = 32.0 * g / (1 << s)
        # a gain digit of 0 is a DEAD lane; the exporter shouts about it, and
        # the grid must not silently divide by it here
        gsafe = torch.clamp(g, min=1.0)
        bd = torch.clamp(torch.round(self.bias * 32.0 * gsafe), -44.0, 45.0)
        b_q = bd / (32.0 * gsafe)
        return (vab + (v_q - vab).detach(),
                self.bias + (b_q - self.bias).detach())

    def forward(self, fi, mi, fo, base_cp):
        E = self.weight()
        vab, bias = self.gvb()
        au = self.act(nn.functional.embedding_bag(fi, E, fo, mode="sum") + bias)
        at = self.act(nn.functional.embedding_bag(mi, E, fo, mode="sum") + bias)
        # ternary: folded gains are activation CAPS [0, 32*g_k] -- not
        # sign-symmetric, so the output weight must be positive
        d = ((au - at) * (vab if self.cfg.ternary else self.v)).sum(-1)
        if self.rff:
            fr, mr = (fi % 768, mi % 768) if self.B > 1 else (fi, mi)
            pu = torch.cos(nn.functional.embedding_bag(fr, self.theta, fo, mode="sum") + self.phb)
            pt = torch.cos(nn.functional.embedding_bag(mr, self.theta, fo, mode="sum") + self.phb)
            d = d + ((pu - pt) * self.rw).sum(-1)
        if self.nb:
            fib, mib = (fi % 768, mi % 768) if self.B > 1 else (fi, mi)
            bu = self.act(nn.functional.embedding_bag(fib, self.rawb, fo, mode="sum") + self.biasb)
            bt = self.act(nn.functional.embedding_bag(mib, self.rawb, fo, mode="sum") + self.biasb)
            g = self.gb.abs()
            su, st = bu * g, bt * g
            nb, m = self.nb, self.m
            if self.nb2:
                A1 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, su[:, :nb])
                A2 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, su[:, nb:])
                B1 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, st[:, :nb])
                B2 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, st[:, nb:])
                if self.baff:
                    w1 = self.w1.unsqueeze(0).expand_as(A1)
                    w2 = self.w2.unsqueeze(0).expand_as(A2)
                    h = circ(A1 + w1, A2 + w2, m) - circ(B1 + w1, B2 + w2, m)
                else:
                    h = circ(A1, A2, m) - circ(B1, B2, m)
                f_ab = (circ(A1, B2, m) + circ(B1, A2, m)) if self.tailw else None
            else:
                A = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, su)
                B = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, st)
                h = circ(A, A, m) - circ(B, B, m)
                if self.baff:
                    h = h + circ(A - B, self.w1.unsqueeze(0).expand_as(A), m)
                f_ab = circ(A, B, m) if self.tailw else None
            d = d + (h * self.u).sum(-1)
            if self.tailw:
                f = f_ab
                zp = torch.cat([d.unsqueeze(-1) / 300.0, h / 100.0, f / 100.0], -1)
                zn = torch.cat([-d.unsqueeze(-1) / 300.0, -h / 100.0, f / 100.0], -1)
                t = self.t2(torch.tanh(self.t1(zp))) - self.t2(torch.tanh(self.t1(zn)))
                d = d + 150.0 * t.squeeze(-1)   # odd-symmetrized: exact antisymmetry
        if self.phase:
            cnt = torch.diff(fo, append=fi.new_tensor([fi.shape[0]]))
            pb = ((cnt - 1) * self.phase // 32).clamp(0, self.phase - 1)
            d = d * self.s[pb]                  # piece count is swap-invariant
        self.pre = d                            # pre-clip residual (satpen reads this)
        return base_cp + d.clamp(-self.clampcp, self.clampcp)


class Ml2Net(ResidualNet):
    """Two packed layers: the ternary head plus a second layer that is the
    circular self-convolution of the clamped head lanes -- in the engine,
    ONE extra big-int multiply of the crelu output blocks, fields re-spaced
    to 32 bits, folded mod 2^(32m)-1 (packed_layers.LaneConv is that op).

    h = conv(A,A) - conv(B,B) is odd under perspective swap and the read-out
    u2 is shared, so exact antisymmetry survives by construction.  u2 starts
    silent (zeros): epoch 0 is exactly the one-layer net.

    Trains only under a field-budget certificate (train.py calls
    certify_or_raise first); the training-normalised h/100 keeps u2 in
    optimizer-friendly range, the export step owns the integer mapping."""

    SHIFT2, UMAX = 10, 127          # certified: field_budget.certify_ml2

    def __init__(self, cfg):
        super().__init__(cfg)
        import packed_layers
        self.conv = packed_layers.LaneConv(cfg.N, cfg.N, "circular", cfg.bm)
        self.u2 = nn.Parameter(torch.zeros(cfg.bm))
        self.u2grid = cfg.u2grid

    def _u2(self, vab, shift=None):
        """u2 as the ENGINE will carry it.

        Free float by default -- and that default is what MEASUREMENTS
        2026-08-15 caught: the engine stores a signed integer |U2| <= 127 and
        renormalises by 2^10, so the value it can actually carry is
        U2 = u2 * 2^SHIFT2 / (100 * 2^(2*shift)) with `shift` the export's L1
        shift, and the 0.01280 net's u2 landed at U2 = [0.17, 0.08, 0.14,
        0.11] -> ALL ZERO.  With u2grid=1 the snap happens INSIDE forward
        (straight-through), so the loss sees the resolution the artifact has
        -- the same discipline the ternary weights already get, and the cure
        for the quant-error-compounding wall packed_layers.py names."""
        if not self.u2grid:
            return self.u2
        if shift is None:
            shift = self.export_shift()
        scale = (1 << self.SHIFT2) / (100.0 * (1 << (2 * shift)))
        q = torch.clamp(torch.round(self.u2 * scale), -self.UMAX, self.UMAX) / scale
        return self.u2 + (q - self.u2).detach()      # STE: snap forward, pass grad

    def forward(self, fi, mi, fo, base_cp):
        E = self.weight()
        vab, bias = self.gvb()
        au = self.act(nn.functional.embedding_bag(fi, E, fo, mode="sum") + bias)
        at = self.act(nn.functional.embedding_bag(mi, E, fo, mode="sum") + bias)
        d = ((au - at) * vab).sum(-1)
        A, B = au * vab.abs(), at * vab.abs()       # cp-scaled lane values
        h = self.conv(A, A) - self.conv(B, B)       # odd: exact antisymmetry
        # ONE shift for both layers -- the payload carries a single one, so the
        # snapped gains and the layer-2 scale must be read off the same value
        d = d + (h * self._u2(vab, self.export_shift())).sum(-1) / 100.0
        self.pre = d
        return base_cp + d.clamp(-self.clampcp, self.clampcp)


class StructuredNet(ResidualNet):
    """The one-layer replnet whose WEIGHT TABLE is parametrized by the
    structure the payload stores (structures.py).  Forward, antisymmetry,
    export order and the engine's arithmetic are the plain net's -- only
    weight() changes, so a structured run is comparable to the plain family
    line for line (and its epoch-0 net is the plain one for lowrank)."""

    def __init__(self, cfg):
        if not cfg.ternary:
            raise ValueError("arch=%s is a replnet parametrization: it needs "
                             "the ternary grid (model.ternary > 0)" % cfg.arch)
        super().__init__(cfg)
        import structures
        if cfg.arch == "cb":
            self.struct = structures.CodebookWeight(
                768 * cfg.kb, cfg.N, cfg.cb_k, cfg.cb_block, cfg.ternary,
                cfg.cb_temp, init_from=self.raw)
        else:
            self.struct = structures.LowRankResidual(
                768 * cfg.kb, cfg.N, cfg.lr_rank, cfg.ternary, cfg.lr_wmax)

    def weight(self):
        w = self.folded()
        if self.cfg.arch == "cb":
            # l1/rate price the SHADOW: for a codebook net the payload is
            # book + indices, whose dial is (K, block), and pressure on the
            # shadow is what steers blocks onto the all-zero codeword.
            self._u = w / constraints.WSCALE
            w, self._assign = self.struct(w)
        else:
            w, self._u = self.struct(w)
        return w

    def clamp_weights(self, wclip):
        super().clamp_weights(wclip)
        self.struct.clamp_(wclip)

    def export_struct(self):
        with torch.no_grad():
            return self.struct.export_struct(self.folded())


def build_model(cfg):
    if cfg.arch == "ml2":
        return Ml2Net(cfg)
    if cfg.arch in ("cb", "lowrank"):
        return StructuredNet(cfg)
    return ResidualNet(cfg)


def lambda_loss(pred, y, outcome, K, p, lam):
    """|target - sigmoid(pred/K)|^p with target = lam*sigmoid(y/K) + (1-lam)*outcome.

    lam=1 reproduces sigmoid_loss EXACTLY (the control must be the incumbent
    experiment, not a near-copy of it), and at lam=1 the outcome tensor is
    never read -- so a corpus without results still trains the control."""
    q = torch.sigmoid(pred / K)
    t = torch.sigmoid(y / K)
    if lam < 1.0:
        t = lam * t + (1.0 - lam) * outcome
    return ((t - q).abs() ** p).mean()


def listwise_rank_loss(pred, k, temp):
    """-log softmax over each group of K siblings, best at local index 0.

    The net evaluates the CHILD, whose mover is the opponent, so the parent's
    preference for a move is -pred(child): ranking best-first means ranking
    child evals ascending, and the logits are -pred/temp.

    SHIFT-INVARIANT BY CONSTRUCTION -- softmax is unchanged by adding a
    constant to every logit in a group, so this objective cannot be satisfied
    by moving the eval's LEVEL, only its ORDER.  That is the whole point of
    the ranking arm, and test_arm10_rank.py asserts it rather than trusting
    it."""
    z = (-pred).view(-1, k) / temp
    return -torch.log_softmax(z, dim=1)[:, 0].mean()


def rank_top1(pred, k):
    """Expected fraction of groups whose searched move the net ranks first,
    UNDER RANDOM TIE-BREAKING.

    `argmax` returns the FIRST maximal index, and the searched move sits at
    local index 0 -- so the plain form scores every tie at the top as a hit.
    This is not a corner case: measured on a material-only net, which
    evaluates most siblings identically because a quiet move changes no
    material, it inflated top1 from 0.108 to 0.518 against a random baseline
    of 1/k.  Trained nets tie less (10-32% of groups) but in the SAME
    direction, so the plain form overstates every rank arm's per-epoch print
    -- ARM 10's first run reported 0.1889 where the honest number is 0.1788.

    A group therefore counts 1/(number tied at the top) when nothing beats
    the searched move, and 0 otherwise.  The -log softmax needs no such
    repair: equal logits give a uniform distribution and score exactly
    log(k), which is how the inflation was spotted."""
    z = (-pred).view(-1, k)
    zb = z[:, :1]
    better = (z > zb).sum(dim=1).float()
    tied = (z == zb).sum(dim=1).float()      # includes the move itself
    return ((better == 0).float() / tied).mean()


def sigmoid_loss(pred, y, K, p):
    """|sigmoid(pred/K) - sigmoid(y/K)|^p, mean.  K=400 house scale;
    p=2.6 is the wide-net house value (nnue-pytorch's finding), the
    replnet arms ran the 2.0 default."""
    return ((torch.sigmoid(pred / K) - torch.sigmoid(y / K)).abs() ** p).mean()
