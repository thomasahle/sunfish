"""Compression STRUCTURE as a training parametrization, not a post-pass.

The house principle (Thomas, 2026-08-14: "we should train everything
end-to-end") with a measurement behind it: the bake-off priced both of
these structures POST HOC and both lost -- codebook-over-blocks (cb8) at
+141 B on an l1-trained ternary net, low-rank+residual (lr_svd) at +326 B,
because a net trained for a dense trit stream has no reason to repeat
blocks or to be low-rank.  So the structure moves INSIDE the graph: the
weight table is PARAMETRIZED by the thing the payload will actually
store, and the net is trained through it.

Both modules are weight parametrizations of layer 1 only -- they return
the effective (768, N) table on the payload's `/32` lane grid, and the
rest of model.ResidualNet is untouched.  Both must hold a field-budget
certificate before training (field_budget.certify_or_raise), and both
export their stored state for the bake-off arms that price them
('trained_cb', 'trained_lr'), never a composed byte estimate.

--------------------------------------------------------------- STE rules

CodebookWeight (product quantization).  Blocks of `block` consecutive
features form a vector x_b = 32*W_b in TRIT UNITS (dimension D =
block*N); the codebook B (K x D) is itself ternary via ternary_grid, so
every codeword is directly a piece of the payload's trit stream.  With
d_bk = ||x_b - B_k||^2 (computed under no_grad -- the responsibilities
are frozen constants) and p_b = softmax(-d_b / (temp * D)):

    forward    y_b = B_{k*},  k* = argmin_k d_bk        (HARD argmin)
    backward   dL/dx_b  = dL/dy_b                       (identity, exactly
                                                         ternary_ste's rule)
               dL/dB_k  = p_bk * dL/dy_b                (SOFT assignment)

realized as  y = exact_ste(B[k*], x + p @ B).  As temp -> 0 the soft
route becomes one-hot at k* and the
two routes coincide with the hard quantizer's own gradients; larger temp
spreads credit over near-miss codewords so an unused entry can still be
pulled into service.  The shadow x is a real parameter (model.raw): it
moves under the identity route and re-assigns when it crosses a Voronoi
boundary -- that is the only thing that changes an assignment, and it is
why the assignment stream is trained rather than fitted.

LowRankResidual.  W = clip(U@V + R, -wmax, +wmax) in trit units, with U
(768 x rank) and V (rank x N) ternary via the same STE and R the ordinary
ternary residual on model.raw.  V is ZERO-INITIALISED, so U@V = 0 and
epoch 0 is EXACTLY the plain ternary net (torch.equal, not approximately
-- the ml2 u2-silent precedent: training can only improve on a known
point).  The clip is the SwarClamp house position -- exact subgradient,
pass inside the band, zero outside -- and it is executed by the DECODER
too, so encoder and engine agree by construction.  Zero-init V still
receives gradient (dV = U^T g through the STE), which is what wakes the
factor up.

--------------------------------------------------------------- the grid

Codebook entries and the low-rank composite are TERNARY (cmax/wmax = 1),
not free-int, and field_budget says why: a weight of c lane-gain units
puts 32*c*gmax + 45 into an offset-binary lane bounded by 2^14, so the
certified ceiling is c = 5 (c = 6 REFUSES, 17133 > 16383).  Free-int is
therefore legal up to 5 but costs log2(11)/log2(3) = 2.19x per stored
element, drops the no-borrow margin from 13490 to 2098 lane units, and
leaves the shipped codec's representable set -- there is no b81 baseline
for such a net, so its bake-off table would have no denominator.  Ternary
is the cheaper certified form, and it is directly executable: the decoded
stream IS the trit stream the shipped entry already consumes.
"""
import torch
import torch.nn as nn

import constraints
from constraints import WSCALE


def exact_ste(value, surrogate):
    """Forward EXACTLY `value`, backward exactly `surrogate`'s gradient.

    `a - a.detach()` is exactly 0.0 for any finite a, so the forward here
    carries no rounding -- unlike `v + (hard - v).detach()`, which lands
    within an ulp of the grid instead of ON it.  These modules assert
    on-grid weights (structures are only exportable if the trained value
    IS the stored value), so they need the exact form.  constraints
    .ternary_ste keeps the older one on purpose: its exports round, and
    perturbing it would move --repro-arm1, another lane's instrument."""
    return value.detach() + (surrogate - surrogate.detach())


def ternary_grid(w, tau):
    """constraints.ternary_ste's grid and gradient, exactly valued:
    {-1,0,+1}/32 (a power-of-two divide, so exact) with backward u/32,
    u = w/WSCALE.  Returns (quantized, u)."""
    u = w / WSCALE
    hard = torch.sign(u) * (u.abs() > tau).to(u.dtype)
    return exact_ste(hard / 32.0, u / 32.0), u


class CodebookWeight(nn.Module):
    """Trained product-quantization codebook over blocks of features."""

    def __init__(self, nfeat, N, K, block, tau, temp, init_from=None):
        super().__init__()
        if nfeat % block:
            raise ValueError("block %d does not divide %d features" % (block, nfeat))
        self.nfeat, self.N, self.K, self.block = nfeat, N, K, block
        self.nblk, self.D = nfeat // block, block * N
        self.tau, self.temp = tau, float(temp)
        if K > self.nblk:
            raise ValueError("K=%d exceeds the %d blocks it must summarise -- "
                             "a book bigger than its data stores nothing"
                             % (K, self.nblk))
        if init_from is None:
            book = torch.randn(K, self.D) * WSCALE
        else:
            # RNG-FREE init: K evenly spaced blocks of the shadow's own init
            # (iid at init, so this samples the same distribution without
            # perturbing the seeded stream the plain family is compared to).
            blocks = init_from.detach().reshape(self.nblk, self.D)
            idx = torch.linspace(0, self.nblk - 1, K).round().long()
            book = blocks[idx].clone()
        self.book = nn.Parameter(book)

    def codewords(self):
        """The book on the payload trit grid: (K, D) in {-1, 0, +1}."""
        bq, _ = ternary_grid(self.book, self.tau)
        return bq * 32.0

    def assign(self, w):
        """Hard argmin assignment (no grad): (nblk,) long."""
        with torch.no_grad():
            x = (w * 32.0).reshape(self.nblk, self.D)
            B = self.codewords()
            return self._dist(x, B).argmin(-1)

    @staticmethod
    def _dist(x, B):
        return (x * x).sum(-1, keepdim=True) - 2.0 * x @ B.t() + (B * B).sum(-1)

    def forward(self, w):
        """(nfeat, N) float weights -> (quantized weights, assignment)."""
        x = (w * 32.0).reshape(self.nblk, self.D)
        B = self.codewords()
        with torch.no_grad():                      # frozen responsibilities
            d = self._dist(x, B)
            k = d.argmin(-1)
            p = torch.softmax(-d / (self.temp * self.D), -1)
        y = exact_ste(B[k], x + p @ B)      # hard forward, identity + soft back
        return y.reshape(self.nfeat, self.N) / 32.0, k

    def clamp_(self, wclip):
        with torch.no_grad():
            self.book.clamp_(-wclip, wclip)

    def export_struct(self, w):
        """Stored state for the 'trained_cb' arm.  Book rows are flat
        feature-major lane-minor trits, exactly the payload's order."""
        k = self.assign(w)
        B = self.codewords().detach().long()
        return {"kind": "cb", "K": self.K, "block": self.block, "N": self.N,
                "temp": self.temp, "book": B.tolist(), "assign": k.tolist(),
                "used": int(k.unique().numel())}


class LowRankResidual(nn.Module):
    """W = clip(U@V + R, +-wmax): trained factors, trained sparse residual."""

    def __init__(self, nfeat, N, rank, tau, wmax=1):
        super().__init__()
        self.nfeat, self.N, self.rank, self.tau, self.wmax = nfeat, N, rank, tau, wmax
        self.U = nn.Parameter(torch.randn(nfeat, rank) * WSCALE)
        self.V = nn.Parameter(torch.zeros(rank, N))   # silent at epoch 0

    def factors(self):
        uq, _ = ternary_grid(self.U, self.tau)
        vq, _ = ternary_grid(self.V, self.tau)
        return uq * 32.0, vq * 32.0

    def forward(self, w):
        """(nfeat, N) float residual weights -> quantized composite, u."""
        R, u = ternary_grid(w, self.tau)
        Uq, Vq = self.factors()
        T = (Uq @ Vq + R * 32.0).clamp(-self.wmax, self.wmax)
        return T / 32.0, u

    def clamp_(self, wclip):
        with torch.no_grad():
            self.U.clamp_(-wclip, wclip)
            self.V.clamp_(-wclip, wclip)

    def export_struct(self, w):
        """Stored state for the 'trained_lr' arm: U, V and the residual
        trits.  The arm additionally prunes residual entries the clip makes
        invisible (bit-exact, encoder-side)."""
        with torch.no_grad():
            R, _ = ternary_grid(w, self.tau)
            Uq, Vq = self.factors()
        return {"kind": "lowrank", "rank": self.rank, "wmax": self.wmax,
                "N": self.N, "U": Uq.long().tolist(), "V": Vq.long().tolist(),
                "R": (R * 32.0).long().tolist()}


def reconstruct(struct):
    """The DECODER's arithmetic, in python: stored state -> flat trits
    (feature-major, lane-minor).  export.py asserts the trained weights
    equal this, so a structure that cannot be rebuilt never ships."""
    if struct["kind"] == "cb":
        book, block = struct["book"], struct["block"]
        out = []
        for b in struct["assign"]:
            out += book[b]
        assert len(out) == len(struct["assign"]) * block * struct["N"]
        return out
    if struct["kind"] == "lowrank":
        U, V, R, wmax = struct["U"], struct["V"], struct["R"], struct["wmax"]
        r, N = struct["rank"], struct["N"]
        out = []
        for f, urow in enumerate(U):
            for k in range(N):
                p = sum(urow[q] * V[q][k] for q in range(r)) + R[f][k]
                out.append(max(-wmax, min(wmax, p)))
        return out
    raise ValueError("unknown struct kind %r" % struct["kind"])
