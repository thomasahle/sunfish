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
