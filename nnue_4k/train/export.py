#!/usr/bin/env python3
"""Export + pricing: checkpoint -> payload/.sfnn -> entry -> pack.sh bytes.

Three export families, dispatched on the config (same rules as
train_packed.py -- float-only for anything whose packed build the val loss
has not earned yet):

  ternary    replnet payload: base-90 string in replnet_proto.py's exact
             extraction order (LSB first: shift, gains, bias digits, 768
             chars of 4 trits).  Bias clips and dead lanes print loudly.
  plain      pnet.pick_shift + pnet.build: the folded-gain packed net
             (G_k = C*|v_k| folded into the rows, sum(G) <= 65534 asserted).
  float      kb>1 / bilinear / rff / phase: float pickle, engine-side
             packing is a separate earned step (build_kb.py, build_ext.py).

Pricing is MEASURED, never composed: `price` splices the payload into a
copy of the entry, runs the invariant suite on the spliced module, runs
tools/build/pack.sh on it, and reports pack.sh's own byte count.  The
gate-ladder invocations are STAGED (printed), never launched -- gates and
screens stay coordinator-dispatched.

usage:
  export.py RUN_DIR_OR_PICKLE --price [--entry PATH]
"""
import argparse
import json
import os
import re
import subprocess
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import features                    # noqa: E402
import pnet                        # noqa: E402  (via features' path bootstrap)

REPO = os.path.dirname(os.path.dirname(_here))
PIECES = pnet.PIECES


def enc90(e):
    """Inverse of the entry codec's digit map (skips '\\' and '\"')."""
    d = e + (e >= 5)
    d += d >= 57
    return chr(35 + d)


def export_replnet(path, E, b, v, clampcp, base_kind, train_meta, struct=None):
    """Ternary payload export, replnet_proto.py's exact extraction order.
    A port of train_packed.export_replnet -- same digits, same warnings.

    `struct` is a trained-structure record (structures.export_struct): the
    payload string stays the SHIPPED codec's (so the b81 baseline remains
    this net's denominator), and the structure rides along in the pickle
    for the arms that price it.  The reconstruction is asserted here --
    bit-exact against the very trits the payload carries -- so a structure
    that cannot be rebuilt never reaches the bake-off."""
    v = [abs(x) for x in v]
    trits = (E * 32).round().long().clamp(-1, 1)
    zeros = float((trits == 0).float().mean())
    shift = 0
    for s in range(8, -1, -1):
        if max(v) * (1 << s) / 32.0 <= 89.49:
            shift = s
            break
    g = [max(0, min(89, round(x * (1 << shift) / 32.0))) for x in v]
    if 0 in g:
        print("export_replnet: DEAD LANES (gain rounded to 0): g=%s" % g, flush=True)
    bd, clip = [], 0
    for k in range(len(g)):
        d = round(b[k] * 32 * g[k])
        clip += not -44 <= d <= 45
        bd.append(max(-44, min(45, d)) + 44)
    if clip:
        print("export_replnet: %d/%d bias digits CLIPPED to the payload range"
              % (clip, len(g)), flush=True)
    digits = [shift] + g + bd
    for f in range(768):
        digits.append(sum((int(trits[f, k]) + 1) * 3 ** k for k in range(len(g))))
    s90 = "".join(enc90(d) for d in reversed(digits))
    assert "\\" not in s90 and '"' not in s90
    extra, note = {}, ""
    if struct is not None:
        import structures
        want = [int(trits[f, k]) for f in range(768) for k in range(len(g))]
        got = structures.reconstruct(struct)
        if got != want:
            bad = sum(1 for a, c in zip(got, want) if a != c)
            raise AssertionError(
                "trained structure (%s) does not rebuild the exported trits: "
                "%d/%d elements differ -- the parametrization and the decoder "
                "disagree" % (struct["kind"], bad, len(want)))
        extra["struct"] = struct
        note = "  struct %s OK" % struct["kind"]
    with open(path + ".payload", "w") as f:
        f.write(s90 + "\n")
    pnet.save(path, {"kind": "replnet-ternary", "B": 1, "N": len(g),
                     "shift": shift, "g": g, "bias_digits": bd, "zeros": zeros,
                     "clampcp": clampcp, "base_kind": base_kind,
                     "train": train_meta, "E": E.tolist(), "bias": b, "v": v,
                     **extra})
    print("export_replnet: zeros %.1f%%  shift %d  gains %s  bias %s  -> %s.payload%s"
          % (100 * zeros, shift, g, bd, path, note), flush=True)
    return "zeros %.1f%% shift %d%s" % (100 * zeros, shift, note)


def export_model(model, cfg, path):
    """Dispatch a live model to its export family, then run the
    knowledge-class probe suite (diagnostics, never gates) and ledger it
    beside the artifact.  Returns a short info string for the epoch log."""
    info = _export_model(model, cfg, path)
    import probes
    scores = probes.report(model, cfg, compact=True)
    with open(path + ".probes.json", "w") as f:
        json.dump(scores, f, sort_keys=True)
    return info


def _export_model(model, cfg, path):
    """The per-family export dispatch."""
    m = cfg.model
    with torch.no_grad():
        E = model.weight().detach()
        b = model.bias.detach().tolist()
        v = model.v.detach().tolist()
    meta = {"config": __import__("config").to_dict(cfg)}
    if m.arch == "ml2":
        # float-only, one rule for every extension: the packed build is
        # engine-side work the val loss has to earn first (and the ml2
        # payload/machinery is PRICE-FIRST per its queue entry)
        pnet.save(path, {"kind": "float-ml2", "B": 1, "N": m.N, "m": m.bm,
                         "E": E.tolist(), "bias": b, "v": v,
                         "u2": model.u2.detach().tolist(), "clampcp": m.clampcp,
                         "base_kind": m.base, "ternary": m.ternary, "train": meta})
        return "float export (ml2)"
    if m.ternary:
        struct = model.export_struct() if m.arch in ("cb", "lowrank") else None
        return export_replnet(path, E, b, v, m.clampcp, m.base, meta, struct)
    extras = {}
    if m.phase:
        extras.update(phase=m.phase, phase_s=model.s.detach().tolist())
    if m.rff:
        extras["rff"] = {"theta": model.theta.detach().tolist(),
                         "phb": model.phb.detach().tolist(),
                         "rw": model.rw.detach().tolist()}
    if m.nb:
        extras.update(nb=m.nb, m=m.bm, nb2=m.nb2,
                      Eb=model.rawb.detach().tolist(),
                      biasb=model.biasb.detach().tolist(),
                      gb=model.gb.detach().abs().tolist(),
                      u=model.u.detach().tolist())
        if m.baff:
            extras["waff"] = [model.w1.detach().tolist()] + \
                ([model.w2.detach().tolist()] if m.nb2 else [])
        if m.tailw:
            extras["tail"] = {n_: q.detach().tolist() for n_, q in
                              (("t1w", model.t1.weight), ("t1b", model.t1.bias),
                               ("t2w", model.t2.weight), ("t2b", model.t2.bias))}
    segs = tuple(i / m.segs for i in range(m.segs))
    if m.kb > 1 or m.nb or m.phase or m.rff:
        kind = "float-kb" if m.kb > 1 else (
            "float-bil" if m.nb else ("float-phase" if m.phase else "float-rff"))
        pnet.save(path, {"kind": kind, "B": m.kb, "N": m.N, "E": E.tolist(),
                         "bias": b, "v": v, "clampcp": m.clampcp, "segs": segs,
                         "base_kind": m.base, "train": meta, **extras})
        return "float export (%s)" % kind
    # plain family: folded-gain packed build
    W = [{c: [0.0] * 120 for c in PIECES} for _ in range(m.N)]
    for c in PIECES:
        for s in pnet.SQUARES:
            col = E[features.feat(c, s)].tolist()
            for k in range(m.N):
                W[k][c][s] = col[k]
    shift, worst, sabs = pnet.pick_shift(W, b, v, segs=segs)
    d = pnet.build(W, b, v, shift, clampcp=m.clampcp, segs=segs)
    d["base_kind"] = m.base
    d["train"] = meta
    pnet.save(path, d)
    return "shift %d sum|v| %.0f excursion %d" % (shift, sabs, d["excursion"])


# ------------------------------------------------------------------ pricing
PAYLOAD_RE = re.compile(r'^for _c in "(.*)":$', re.M)


def splice_entry(payload_path, entry_path, out_path):
    """Replace the entry's payload string with the trained one.  The entry
    keeps its own codec, decoder and machinery -- pricing measures the REAL
    composed artifact, nothing synthetic."""
    with open(payload_path) as f:
        s90 = f.read().strip()
    with open(entry_path) as f:
        src = f.read()
    m = PAYLOAD_RE.search(src)
    if not m:
        raise ValueError("no payload string found in %s" % entry_path)
    src = src[:m.start(1)] + s90 + src[m.end(1):]
    with open(out_path, "w") as f:
        f.write(src)
    return out_path


def price(payload_path, entry_path=None, out_dir=None):
    entry_path = entry_path or os.path.join(REPO, "nnue_4k", "replnet_proto.py")
    out_dir = out_dir or os.path.dirname(os.path.abspath(payload_path))
    spliced = os.path.join(out_dir, "entry_spliced.py")
    packed = os.path.join(out_dir, "entry_spliced.packed")
    splice_entry(payload_path, entry_path, spliced)

    # invariant suite + bit-exactness triangle on the spliced module
    netpath = payload_path[:-len(".payload")]
    check = os.path.join(_here, "verify_export.py")
    r = subprocess.run([sys.executable, check, netpath, spliced],
                       capture_output=True, text=True, cwd=out_dir)
    print(r.stdout, end="", flush=True)
    if r.returncode:
        print(r.stderr, end="", flush=True)
        raise SystemExit("verify_export FAILED on the spliced entry")

    # measured bytes: pack.sh's own count, never composed arithmetic
    pack = os.path.join(REPO, "tools", "build", "pack.sh")
    r = subprocess.run(["bash", pack, spliced, packed], capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode:
        print(r.stderr, end="", flush=True)
        raise SystemExit("pack.sh failed")
    nbytes = os.path.getsize(packed)
    verdict = "IN BUDGET (%d spare)" % (4096 - nbytes) if nbytes <= 4096 \
        else "OVER by %d" % (nbytes - 4096)
    print("MEASURED artifact: %d bytes -- %s" % (nbytes, verdict), flush=True)

    print("\nGate ladder, STAGED (coordinator-dispatched, in order; each gates "
          "the next):", flush=True)
    for cmd in (
            "python3 tools/build/legality_gate.py %s" % packed,
            "python3 tools/build/mate_gate.py %s" % packed,
            "python3 tools/build/mate_conversion_gate.py %s" % packed,
            "python3 tools/build/first_yield_gate.py %s" % packed,
            "# fixed-node SPRT vs pst_entry @ HEAD (elo0=0 elo1=10), then timed "
            "confirmation -- request the slot from the coordinator"):
        print("  " + cmd, flush=True)
    return nbytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="run dir, .pickle, or .payload")
    p.add_argument("--price", action="store_true")
    p.add_argument("--entry", default=None)
    p.add_argument("--bakeoff", action="store_true",
                   help="run the compress/ encoder zoo (all arms x both "
                        "container layouts, measured through the real pack "
                        "paths) and report the per-net winner")
    a = p.parse_args()
    t = a.target
    if os.path.isdir(t):
        t = os.path.join(t, "best.pickle")
    payload = t if t.endswith(".payload") else t + ".payload"
    if not os.path.exists(payload):
        raise SystemExit("%s not found -- only ternary exports have payloads; "
                         "float/kb exports price via build_kb.py + pack_entry.sh" % payload)
    if a.bakeoff:
        from compress import bakeoff
        kw = {"entry": a.entry} if a.entry else {}
        res = bakeoff.run_net(payload[:-len(".payload")], **kw)
        print()
        print(bakeoff.format_table(res))
        w = res["ranked"][0]
        print("\nbakeoff winner: %s layout %s at %d bytes (baseline %d)"
              % (w["arm"], w["layout"], w["bytes"], res["baseline_bytes"]))
        if not all(res["checks"].values()) or any("FAILED" in r for r in res["rows"]):
            raise SystemExit("bakeoff instrument checks or arms FAILED -- see table")
    if a.price:
        price(payload, a.entry)
    if not (a.price or a.bakeoff):
        print(payload)


if __name__ == "__main__":
    main()
