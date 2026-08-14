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

    def weight(self):
        v = self.virt()
        if isinstance(v, int):
            w = self.raw
        else:
            N = self.raw.shape[1]
            w = (self.raw.view(self.B, 768, N) + v).view(self.B * 768, N)
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
                # exportable bias range is +-44 lane units of a 32*g_k cap
                self.bias.clamp_(-0.019, 0.019)

    def forward(self, fi, mi, fo, base_cp):
        E = self.weight()
        au = self.act(nn.functional.embedding_bag(fi, E, fo, mode="sum") + self.bias)
        at = self.act(nn.functional.embedding_bag(mi, E, fo, mode="sum") + self.bias)
        # ternary: folded gains are activation CAPS [0, 32*g_k] -- not
        # sign-symmetric, so the output weight must be positive
        d = ((au - at) * (self.v.abs() if self.cfg.ternary else self.v)).sum(-1)
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


def sigmoid_loss(pred, y, K, p):
    """|sigmoid(pred/K) - sigmoid(y/K)|^p, mean.  K=400 house scale;
    p=2.6 is the wide-net house value (nnue-pytorch's finding), the
    replnet arms ran the 2.0 default."""
    return ((torch.sigmoid(pred / K) - torch.sigmoid(y / K)).abs() ** p).mean()
