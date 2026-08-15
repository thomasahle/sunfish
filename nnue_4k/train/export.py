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

# torch is imported lazily by the live-model paths only: re-exporting a
# SAVED checkpoint (lists on disk) and pricing it through pack.sh are pure
# python, and the box that prices is not the box that trains.

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


# ------------------------------------------------------------------ ml2
# The engine's layer-2 read-out is FIXED by field_budget.certify_ml2 and by
# the landed derivation packed/make_ml2_proto.py: signed per-field u2 with
# |u2| <= 127, then >> 10.  Training normalises the conv by 100 and keeps u2
# a free float ("the export step owns the integer mapping", model.Ml2Net),
# so the mapping is this and only this:
#
#   engine lane y_k = A_k * 2^shift      (A = au*|v| in cp; cap 32*g_k = v_k*2^shift)
#   engine conv     = 2^(2*shift) * conv(A,A)
#   engine L2 cp    = 2^(2*shift) * sum U2.H / 2^SHIFT2   ==   sum u2.H / 100
#   =>  U2_k = u2_k * 2^SHIFT2 / (100 * 2^(2*shift))
#
# The L1 shift is therefore the export's ONLY free knob for layer 2: every
# unit it drops multiplies U2 by 4 and coarsens the L1 gains g_k by 2.
ML2_SHIFT2 = 10
ML2_UMAX = 127


def ml2_readout(u2, shift, shift2=ML2_SHIFT2, umax=ML2_UMAX):
    """(integer u2 digits, exact float pre-image) for a payload at `shift`."""
    scale = (1 << shift2) / (100.0 * (1 << (2 * shift)))
    exact = [x * scale for x in u2]
    return [max(-umax, min(umax, int(round(x)))) for x in exact], exact


def ml2_shift_table(v, u2, shift2=ML2_SHIFT2, umax=ML2_UMAX):
    """The price sheet for that knob: per legal shift, the gains it leaves
    and the integer read-out it can carry.  Printed on refusal so the
    trade-off is on the table instead of in someone's head."""
    rows = []
    for s in range(8, -1, -1):
        g = [max(0, min(89, round(x * (1 << s) / 32.0))) for x in v]
        if max(g) > 89 or max(v) * (1 << s) / 32.0 > 89.49:
            continue
        U2, exact = ml2_readout(u2, s, shift2, umax)
        rows.append("  shift %d  gains %-18s U2 %-18s (exact %s)%s"
                    % (s, g, U2, ["%.3f" % x for x in exact],
                       "  DEAD LANES" if 0 in g else ("  SILENT L2" if not any(U2) else "")))
    return "\n".join(rows)


def export_ml2(path, E, b, v, u2, clampcp, base_kind, train_meta, shift=None):
    """Ternary ml2 payload: the single-layer body with the certified
    layer-2 seam spliced in -- 4 offset-4050 base-90 digit PAIRS between the
    bias digits and the feature chars (packed/make_proto_payload.py --u2 4
    emits the same layout; packed/ml2_check.py decodes it independently).

    Everything the single-layer export prints, this prints too, plus the
    integer read-out.  A read-out that rounds to all zeros is announced as
    DEAD: the payload is still written (it is a legitimate single-layer net)
    but nothing downstream may quietly price it as a two-layer one."""
    v = [abs(x) for x in v]
    trits = [[max(-1, min(1, int(round(x * 32)))) for x in row] for row in E]
    N = len(v)
    zeros = sum(t == 0 for row in trits for t in row) / float(len(trits) * N)
    if shift is None:
        shift = 0
        for s in range(8, -1, -1):
            if max(v) * (1 << s) / 32.0 <= 89.49:
                shift = s
                break
    elif max(v) * (1 << shift) / 32.0 > 89.49:
        raise SystemExit("export_ml2: shift %d overflows the gain digit "
                         "(max v %.2f) -- refused" % (shift, max(v)))
    g = [max(0, min(89, round(x * (1 << shift) / 32.0))) for x in v]
    if 0 in g:
        print("export_ml2: DEAD LANES (gain rounded to 0): g=%s" % g, flush=True)
    bd, clip = [], 0
    for k in range(N):
        d = round(b[k] * 32 * g[k])
        clip += not -44 <= d <= 45
        bd.append(max(-44, min(45, d)) + 44)
    if clip:
        print("export_ml2: %d/%d bias digits CLIPPED to the payload range" % (clip, N), flush=True)
    U2, exact = ml2_readout(u2, shift)
    digits = [shift] + g + bd
    for x in U2:                       # LSB pair first, offset 4050 (certify_ml2 layout)
        d = x + 4050
        digits += [d % 90, d // 90]
    for f in range(len(trits)):
        digits.append(sum((trits[f][k] + 1) * 3 ** k for k in range(N)))
    s90 = "".join(enc90(d) for d in reversed(digits))
    assert "\\" not in s90 and '"' not in s90
    with open(path + ".payload", "w") as f:
        f.write(s90 + "\n")
    pnet.save(path, {"kind": "replnet-ml2", "B": 1, "N": N, "m": len(U2),
                     "shift": shift, "g": g, "bias_digits": bd, "zeros": zeros,
                     "u2_digits": U2, "u2": u2, "shift2": ML2_SHIFT2,
                     "clampcp": clampcp, "base_kind": base_kind,
                     "train": train_meta, "E": E, "bias": b, "v": v})
    dead = not any(U2)
    print("export_ml2: zeros %.1f%%  shift %d  gains %s  bias %s  u2 %s (exact %s)  -> %s.payload"
          % (100 * zeros, shift, g, bd, U2, ["%.3f" % x for x in exact], path), flush=True)
    if dead:
        print("export_ml2: DEAD LAYER-2 READ-OUT -- every u2 digit rounds to 0 at shift %d, so the "
              "exported net's second layer is SILENT (it evaluates as the single-layer net while "
              "paying ml2's code and nps).  The knob is the L1 shift:\n%s"
              % (shift, ml2_shift_table(v, u2)), flush=True)
    return "zeros %.1f%% shift %d u2 %s%s" % (100 * zeros, shift, U2, "  DEAD L2" if dead else "")


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
    import torch
    m = cfg.model
    with torch.no_grad():
        E = model.weight().detach()
        b = model.bias.detach().tolist()
        v = model.v.detach().tolist()
    meta = {"config": __import__("config").to_dict(cfg)}
    if m.arch == "ml2":
        # the certified two-layer payload (engine form: packed/make_ml2_proto.py).
        # Its ternary body is the shipped codec's, so this net's bytes stay
        # comparable with every single-layer number in the ledger.
        return export_ml2(path, E.tolist(), b, v, model.u2.detach().tolist(),
                          m.clampcp, m.base, meta)
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
    netpath = payload_path[:-len(".payload")]
    ml2 = _load_meta(netpath).get("kind") == "replnet-ml2"
    out_dir = out_dir or os.path.dirname(os.path.abspath(payload_path))
    if entry_path is None and ml2:
        # the ml2 arm prices the ml2 ENTRY: derived from the shipped one by
        # the landed generator, never a fork (every hunk asserts it hit).
        gen = os.path.join(REPO, "nnue_4k", "packed", "make_ml2_proto.py")
        entry_path = os.path.join(out_dir, "ml2_entry.py")
        subprocess.run([sys.executable, gen, os.path.join(REPO, "nnue_4k", "replnet_proto.py"),
                        entry_path], check=True, capture_output=True, text=True)
    entry_path = entry_path or os.path.join(REPO, "nnue_4k", "replnet_proto.py")
    spliced = os.path.join(out_dir, "entry_spliced.py")
    packed = os.path.join(out_dir, "entry_spliced.packed")
    splice_entry(payload_path, entry_path, spliced)

    # invariant suite + bit-exactness on the spliced module.  ml2's checker
    # is packed/ml2_check.py (verify_export's triangle is single-layer: it
    # has no second layer to mirror), and it self-derives its reference from
    # packed_layers' int bridge, so it never trusts this file's arithmetic.
    check = os.path.join(REPO, "nnue_4k", "packed", "ml2_check.py") if ml2 \
        else os.path.join(_here, "verify_export.py")
    argv = [sys.executable, check, spliced] if ml2 \
        else [sys.executable, check, netpath, spliced]
    r = subprocess.run(argv, capture_output=True, text=True, cwd=out_dir)
    print(r.stdout, end="", flush=True)
    if r.returncode:
        print(r.stderr, end="", flush=True)
        raise SystemExit("%s FAILED on the spliced entry" % os.path.basename(check))

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


def _load_meta(netpath):
    import pickle
    try:
        with open(netpath, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="run dir, .pickle, or .payload")
    p.add_argument("--price", action="store_true")
    p.add_argument("--entry", default=None)
    p.add_argument("--shift", type=int, default=None,
                   help="ml2 only: override the L1 shift.  It is the export's "
                        "only knob on the certified layer-2 read-out (U2 scales "
                        "4x per unit dropped, gains halve) -- a payload-scale "
                        "decision, so it is explicit, never inferred")
    p.add_argument("--bakeoff", action="store_true",
                   help="run the compress/ encoder zoo (all arms x both "
                        "container layouts, measured through the real pack "
                        "paths) and report the per-net winner")
    a = p.parse_args()
    t = a.target
    if os.path.isdir(t):
        t = os.path.join(t, "best.pickle")
    payload = t if t.endswith(".payload") else t + ".payload"
    net = payload[:-len(".payload")]
    d = _load_meta(net)
    if d.get("kind") in ("float-ml2", "replnet-ml2") and (a.shift is not None
                                                          or not os.path.exists(payload)):
        # re-export a SAVED ml2 checkpoint: the floats on disk are the whole
        # net, so this needs no torch and no retraining
        export_ml2(net, d["E"], d["bias"], d["v"], d["u2"], d["clampcp"],
                   d["base_kind"], d.get("train", {}), shift=a.shift)
        d = _load_meta(net)
    if not os.path.exists(payload):
        raise SystemExit("%s not found -- only ternary exports have payloads; "
                         "float/kb exports price via build_kb.py + pack_entry.sh" % payload)
    if d.get("kind") == "replnet-ml2" and not any(d["u2_digits"]) and (a.price or a.bakeoff):
        raise SystemExit(
            "REFUSED to price a SILENT second layer: every u2 digit rounds to 0 at shift %d, so "
            "this artifact carries ml2's +98 B of code and its ~0.90x nps and evaluates as the "
            "single-layer net.  Choose the L1 shift deliberately (--shift) or retrain u2 on the "
            "certified grid -- this file will not pick for you:\n%s"
            % (d["shift"], ml2_shift_table(d["v"], d["u2"])))
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
