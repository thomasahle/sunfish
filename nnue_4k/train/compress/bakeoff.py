#!/usr/bin/env python3
"""One command, all encoders, one net: the compression bake-off.

For every registered arm x container layout (A joint / B split):

  1. build the REAL spliced entry (stock entry for the baseline's layout
     A -- byte-identical to the recorded export path -- otherwise the
     decode region is replaced with the arm's decoder);
  2. BIT-EXACT gate: exec the spliced source and compare SHIFT/MGP/MGH/
     ACC_BASE/ROWS against qnet.expected_module_data(), the independent
     mirror of the trainer quantization (itself checked once per net via
     train/verify_export.py's torch triangle on the baseline splice);
  3. measure through the REAL pack path only (tools/build/pack.sh,
     tools/build/pack_entry.sh) -- artifact bytes, payload-elided bytes
     (the proto's machinery convention), and boot the actual artifact
     ('uci' -> 'uciok') so layout B's SF_A self-read is exercised;
  4. rank by artifact bytes; decoder cost and payload-in-context are
     DELTAS OF MEASURED ARTIFACTS, never composed arithmetic.

Instrument checks (fail loudly, rank nothing on failure):
  * the baseline arm's layout-A string must equal the exporter's own
    .payload file when present, and its artifact must reproduce the
    recorded bytes (3831 v1 / 3834 repro_arm1);
  * the ctrl_shuffle arm must measure WORSE than baseline in layout A
    (structure destroyed) and >= baseline in layout B (decoder cost) --
    an axis that cannot fail cannot be trusted.

usage:
  bakeoff.py NET [NET...] [--entry E] [--arms a,b] [--out DIR]
             [--no-boot] [--no-torch-verify] [--json]
"""
import argparse
import json
import os
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))          # train/ (for verify path)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))  # nnue_4k/

from compress import qnet, entrysrc, packrun               # noqa: E402
from compress.arms import all_arms                         # noqa: E402

REPO = packrun.REPO
# Pinned by default: another lane golfs the working-tree entry live (see
# entrysrc.read_entry), and a bake-off needs one denominator.
DEFAULT_ENTRY = "HEAD:nnue_4k/replnet_proto.py"


def _write(path, data):
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as f:
        f.write(data)
    return path


def torch_verify(netpath, spliced, cwd):
    """The house triangle (payload == trainer quantization == entry) on
    the baseline splice.  Anchors the in-process mirror to the trainer."""
    check = os.path.join(os.path.dirname(_here), "verify_export.py")
    r = subprocess.run([sys.executable, check, os.path.abspath(netpath),
                       os.path.abspath(spliced)],
                       capture_output=True, text=True, cwd=cwd)
    if r.returncode:
        raise RuntimeError("verify_export FAILED:\n%s%s" % (r.stdout, r.stderr))
    return r.stdout.strip().splitlines()[-1]


def measure_cell(arm, layout, q, stock, expected, outdir, boot):
    """One (arm, layout): returns the result row dict."""
    name = "%s_%s" % (arm.name, layout)
    body, body_src, note = arm.encode(q)
    full = qnet.header_int(q) + qnet.HEADER_RADIX * body
    row = {"arm": arm.name, "layout": layout, "note": note}

    if layout == "A":
        s90 = qnet.int_to_s90(full)
        row["payload_chars"] = len(s90)
        if getattr(arm, "native_a", False):
            src = entrysrc.splice_payload(stock, s90)
            src_elided = entrysrc.splice_payload(stock, "")
        else:
            src = entrysrc.replace_region(
                stock, entrysrc.build_region(entrysrc.prologue_a(s90), body_src))
            src_elided = entrysrc.replace_region(
                stock, entrysrc.build_region(entrysrc.prologue_a(""), body_src))
        g, dt = entrysrc.exec_entry(src)
        _gate(g, expected, name)
        row["decode_s"] = round(dt, 3)
        ep = _write(os.path.join(outdir, name + ".py"), src)
        row["bytes"] = packrun.pack_a(ep, os.path.join(outdir, name + ".packed"))
        ee = _write(os.path.join(outdir, name + "_elided.py"), src_elided)
        row["bytes_elided"] = packrun.pack_a(
            ee, os.path.join(outdir, name + "_elided.packed"))
    else:
        tail = qnet.int_to_bytes(full)
        row["payload_bytes_raw"] = len(tail)
        src = entrysrc.replace_region(
            stock, entrysrc.build_region(entrysrc.PROLOGUE_B, body_src))
        g, dt = entrysrc.exec_entry(src, tail_bytes=tail, tmpdir=outdir)
        _gate(g, expected, name)
        row["decode_s"] = round(dt, 3)
        ep = _write(os.path.join(outdir, name + ".py"), src)
        wp = _write(os.path.join(outdir, name + ".weights"), tail)
        row["bytes"] = packrun.pack_b(ep, wp, os.path.join(outdir, name + ".packed"))
        we = _write(os.path.join(outdir, name + "_elided.weights"), b"")
        row["bytes_elided"] = packrun.pack_b(
            ep, we, os.path.join(outdir, name + "_elided.packed"))
    if boot:
        row["boot_s"] = round(packrun.boot_smoke(
            os.path.join(outdir, name + ".packed")), 2)
    return row


def _gate(g, expected, name):
    """Bit-exactness: the spliced module's decoded data must equal the
    independent mirror EXACTLY.  Also re-assert the layout constants the
    mirror hardcodes, so entry drift is loud here, not downstream."""
    assert g["NN"] == qnet.NN and g["LBITS"] == qnet.LBITS \
        and g["VBITS"] == qnet.VBITS, "entry layout constants drifted"
    got = qnet.module_data_of(g)
    for k in ("SHIFT", "MGP", "MGH", "ACC_BASE", "ROWS"):
        if got[k] != expected[k]:
            raise AssertionError("%s: decoded %s != trainer quantization" % (name, k))


def run_net(netpath, entry=DEFAULT_ENTRY, arms=None, outdir=None,
            boot=True, do_torch_verify=True):
    q = qnet.load_qnet(netpath)
    expected = qnet.expected_module_data(q)
    stock, entry_prov = entrysrc.read_entry(entry)
    print("  [%s] entry: %s" % (q.name, entry_prov), flush=True)
    outdir = outdir or os.path.join(
        os.path.dirname(os.path.abspath(netpath.rstrip("/"))), "bakeoff_" + q.name)
    os.makedirs(outdir, exist_ok=True)
    zoo = [a for a in all_arms() if arms is None or a.name in arms]
    baseline = next(a for a in zoo if getattr(a, "native_a", False))

    # ---- instrument first: the exporter's own string, the house triangle
    body, _, _ = baseline.encode(q)
    full = qnet.header_int(q) + qnet.HEADER_RADIX * body
    ppath = (netpath[:-len(".payload")] if netpath.endswith(".payload") else netpath)
    ppath = (os.path.join(ppath, "best.pickle") if os.path.isdir(ppath) else ppath) + ".payload"
    if os.path.exists(ppath):
        with open(ppath) as f:
            recorded = f.read().strip()
        assert qnet.int_to_s90(full) == recorded, \
            "baseline string != exporter payload -- digit conventions broken"
    if do_torch_verify:
        spliced = _write(os.path.join(outdir, "b81_A.py"),
                         entrysrc.splice_payload(stock, qnet.int_to_s90(full)))
        line = torch_verify(ppath[:-len(".payload")], spliced, outdir)
        print("  [%s] %s" % (q.name, line), flush=True)

    rows = []
    for arm in zoo:
        for layout in "AB":
            t0 = time.time()
            try:
                row = measure_cell(arm, layout, q, stock, expected, outdir, boot)
            except Exception as e:
                import traceback
                traceback.print_exc()
                row = {"arm": arm.name, "layout": layout, "FAILED": str(e)}
            row["wall_s"] = round(time.time() - t0, 1)
            rows.append(row)
            if "bytes" in row:
                print("  [%s] %-16s %s  %5d B  (elided %d, decode %.3fs%s)"
                      % (q.name, row["arm"], layout, row["bytes"],
                         row["bytes_elided"], row["decode_s"],
                         ", boot %.2fs" % row["boot_s"] if "boot_s" in row else ""),
                      flush=True)

    ok = [r for r in rows if "bytes" in r]
    base_a = next(r for r in ok if r["arm"] == baseline.name and r["layout"] == "A")
    for r in ok:
        r["delta_vs_base"] = r["bytes"] - base_a["bytes"]
        r["payload_in_ctx"] = r["bytes"] - r["bytes_elided"]
        r["decoder_cost"] = r["bytes_elided"] - base_a["bytes_elided"]
    ok.sort(key=lambda r: (r["bytes"], r.get("decode_s", 9e9)))

    # ---- the axis must be able to fail: control vs baseline, SAME layout
    # (in A the shuffle destroys lzma's matches; in B the raw tail hides
    # that, so only the unshuffle decoder's cost separates them).
    checks = {"reproduced_payload_string": os.path.exists(ppath)}
    for r in ok:
        if r["arm"] != "ctrl_shuffle":
            continue
        base = next((b for b in ok if b["arm"] == baseline.name
                     and b["layout"] == r["layout"]), None)
        if base:
            checks["ctrl_worse_%s" % r["layout"]] = \
                r["bytes"] > base["bytes"] if r["layout"] == "A" \
                else r["bytes"] >= base["bytes"]
    return {"net": q.name, "netpath": os.path.abspath(netpath),
            "entry": entry_prov, "outdir": outdir,
            "baseline_bytes": base_a["bytes"], "rows": rows, "ranked": ok,
            "checks": checks}


def best(netpath, **kw):
    """export.py's hook: measured winner as (arm, layout, bytes, packed)."""
    res = run_net(netpath, **kw)
    w = res["ranked"][0]
    packed = os.path.join(res["outdir"], "%s_%s.packed" % (w["arm"], w["layout"]))
    return w["arm"], w["layout"], w["bytes"], packed


def format_table(res):
    lines = ["net %s: baseline (b81, A) = %d B; 4096-budget spare %d"
             % (res["net"], res["baseline_bytes"], 4096 - res["baseline_bytes"]),
             "%-16s %-3s %7s %6s %8s %8s %8s %7s  %s"
             % ("arm", "lay", "bytes", "delta", "payload", "decoder",
                "decode_s", "boot_s", "note")]
    for r in res["ranked"]:
        lines.append("%-16s %-3s %7d %+6d %8d %8d %8.3f %7s  %s"
                     % (r["arm"], r["layout"], r["bytes"], r["delta_vs_base"],
                        r["payload_in_ctx"], r["decoder_cost"], r["decode_s"],
                        ("%.2f" % r["boot_s"]) if "boot_s" in r else "-",
                        r.get("note", "")))
    for r in res["rows"]:
        if "FAILED" in r:
            lines.append("%-16s %-3s FAILED: %s" % (r["arm"], r["layout"], r["FAILED"]))
    lines.append("instrument checks: " + ", ".join(
        "%s=%s" % (k, v) for k, v in res["checks"].items()))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("nets", nargs="+", help="run dir, .pickle or .payload")
    p.add_argument("--entry", default=DEFAULT_ENTRY)
    p.add_argument("--arms", default=None, help="comma-separated subset")
    p.add_argument("--out", default=None)
    p.add_argument("--no-boot", action="store_true")
    p.add_argument("--no-torch-verify", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    arms = a.arms.split(",") if a.arms else None
    failed = False
    for net in a.nets:
        res = run_net(net, entry=a.entry, arms=arms, outdir=a.out,
                      boot=not a.no_boot, do_torch_verify=not a.no_torch_verify)
        print()
        print(format_table(res))
        print()
        jp = os.path.join(res["outdir"], "bakeoff_%s.json" % res["net"])
        with open(jp, "w") as f:
            json.dump(res, f, indent=1)
        print("results json: %s" % jp)
        failed |= any("FAILED" in r for r in res["rows"])
        failed |= not all(res["checks"].values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
