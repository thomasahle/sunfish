"""Training-time constraints as reusable modules.

Every rule here exists because a measurement paid for it (MEASUREMENTS.md):

  satpen     the kbbil collapse: the engine's hard clip passes zero gradient
             outside the band, so saturation is FREE capacity in the loss and
             ruinous in play (10x pegged evals, +27% tree inflation).
             Default-ON at 0.03 @ 480cp since the rehab arc.
  ternary STE  the replnet payload grid {-1,0,+1}/32-of-lane-saturation put
             INSIDE the forward pass -- rounding a float fit afterwards
             optimises the wrong function (distill_train.py's rule).
  l1         sparsity pressure on pre-ternarization magnitudes: the payload
             byte budget is a HARD gate at >= ~58% zeros; this term buys the
             bytes (~+0.0002 val per +14% zeros).
  wclip      the effective weight (what is exported) is clamped after every
             step, so training never leaves the exportable range.
  phasecap   the learned s~7 amplified quantization noise 7x at build time;
             capping forces v to carry the scale.

Exact antisymmetry is NOT here: it is not a penalty, it is the model's
construction (model.py) -- one shared table, us-minus-them, odd read-outs.
`check_antisymmetry` below verifies the construction on a probe batch.
"""
import torch

WSCALE = 0.05     # raw init std: u = raw/WSCALE is ~N(0,1) at init


def ternary_ste(w, tau):
    """Straight-through ternary on the EFFECTIVE weight: forward sees
    {-1,0,+1}/32 of lane saturation (the packed payload's exact grid),
    backward passes through u = w/WSCALE.  Returns (w_quantized, u);
    feed u to l1_pressure."""
    u = w / WSCALE
    hard = torch.sign(u) * (u.abs() > tau).float()
    return (u + (hard - u).detach()) / 32.0, u


def saturation_penalty(pre_clip, weight, thresh):
    """weight * mean(relu(|pre-clip residual| - thresh)/100)^2, in the loss.
    Val stays the plain metric for comparability."""
    return weight * (torch.relu(pre_clip.abs() - thresh) / 100).pow(2).mean()


def l1_pressure(u, weight):
    return weight * u.abs().mean()


def rate_penalty(u, tau, weight, T=8.0):
    """Differentiable payload-rate estimate: expected code length of the
    ternarized weights under a per-lane order-0 prior, in BYTES.

    Soft occupancy of the STE's own grid -- P(+1) = sig((u - tau) * T),
    P(-1) = sig((-u - tau) * T), P(0) the rest (T -> inf recovers the hard
    threshold) -- aggregated per lane; the expected bits are 768 * H(lane
    marginal), which is exactly what the zoo's rc_o0 arm realizes at
    decode time.  Calibration fact from the bake-off (2026-08-14, v1):
    this bound sits ~35% ABOVE what the shipped base-3^4+lzma path
    achieves on the same net (519 B order-0 vs 382 B measured in
    context), because lzma also captures run/match structure.  So this
    is a steering signal whose unit is honest (bytes-ish) but whose
    absolute value is an upper bound -- size arms by the MEASURED table,
    never by this term.  Unlike l1 it prices the whole distribution (a
    lane pushed
    toward uniform +/-1 costs log2(3) bits/trit even at zero mean), so
    it is the principled dial for the c1024 capacity family: rate 0.001
    charges ~0.1 val-loss-units per 100 estimated payload bytes.
    """
    sp = torch.sigmoid((u - tau) * T)
    sn = torch.sigmoid((-u - tau) * T)
    probs = torch.stack([sn.mean(0), (1 - sp - sn).mean(0), sp.mean(0)], -1)
    H = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log2()).sum(-1)
    return weight * H.sum() * u.shape[0] / 8.0


def phasecap_(s, cap):
    """Project phase scales into [1/cap, cap] in place (call under no_grad
    is not needed: clamp_ on a leaf param outside the graph)."""
    with torch.no_grad():
        s.clamp_(1.0 / cap, cap)


def check_antisymmetry(model, fi, mi, fo, base):
    """The construction check: swapping perspectives (us <-> them feature
    lists) and negating the base must exactly negate the prediction.  Run
    on a probe batch in tests and after resume; a failure is a model bug,
    never a tolerance."""
    with torch.no_grad():
        a = model(fi, mi, fo, base)
        b = model(mi, fi, fo, -base)
    if not torch.equal(a, -b):
        worst = (a + b).abs().max().item()
        raise AssertionError("antisymmetry broken by construction: max |s(p)+s(~p)| = %g" % worst)
