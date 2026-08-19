#!/usr/bin/env python3
"""Trained arch=factor checkpoint -> a packed, certified 4k artifact.

Stage 2 of the factored-compression lane.  `make_factor_proto.py` builds the
ENGINE for a given (r, N, mirror) and certifies its decoder against an
independent reconstruction (`factor_check.py`); this fills that engine with a
TRAINED payload instead of a random one, and re-runs the same certification
plus an end-to-end round-trip.

The payload layout is the decoder's, LSB-first, and it is asserted rather
than assumed at every step:

    q[0]                     SHIFT
    q[1 .. N]                per-lane cap digits g_k          (G_k = 32*g_k)
    q[N+1 .. 2N]             bias digits + 44
    q[2N+1 .. 2N+r*N]        V + 44, row-major j*N + k
    q[2N+r*N+1 ...]          ceil(r/4) trit digits per feature row,
                             feature-major (32 rows/piece when mirrored),
                             BUCKET-MAJOR: row = (b*12 + piece)*nsq + rk*4 + fl'

    B (= kb*pb) is read from the checkpoint -- len(U)/768 -- and the bucket
    KIND from its own config, so a kb net cannot be built with a phase
    selector by a mistyped flag.

    build_factor_entry.py CKPT.pickle [--out entry.py] [--packed entry.packed]
"""
import argparse
import os
import pickle
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import make_factor_proto as M                                    # noqa: E402


def enc(d):
    assert 0 <= d < 90, d
    return chr(35 + d + (35 + d >= 92))


def shift_for(v):
    """model.export_shift()'s rule, VERBATIM, so the artifact and the trainer
    pick the same shift.

    Both bounds, in the trainer's order: the largest shift whose gains still
    fit 89, AND whose worst-case cap sum fits nn_cp's lane-sum fold.  Missing
    the second clause here while the trainer enforces it would make this
    builder refuse a net that trained perfectly correctly -- the same
    train/ship divergence in the other direction.
    """
    vmax = max(abs(x) for x in v) or 1.0
    for s in range(8, -1, -1):
        if vmax * (1 << s) / 32.0 > 89.49:
            continue
        g = [min(89, max(0, int(round(abs(x) * (1 << s) / 32.0)))) for x in v]
        if 32 * sum(g) <= 65534:
            return s
    return 0


def shape_of(ck):
    """(B, bucket-kind) from the checkpoint alone -- never from a flag."""
    B = len(ck["U"]) // 768
    assert B * 768 == len(ck["U"]), ("U has %d rows, not a multiple of 768"
                                     % len(ck["U"]))
    cfg = (ck.get("meta") or {}).get("config", {}).get("model", {})
    kb, pb = int(cfg.get("kb", 1)), int(cfg.get("pb", 1))
    if B > 1 and kb * pb != B:
        raise SystemExit("checkpoint disagrees with itself: U implies B=%d, "
                         "config says kb=%d pb=%d" % (B, kb, pb))
    return B, ("kb" if kb > 1 else "pb")


def payload(ck, force_shift=None):
    """Trained state -> the decoder's digit stream, with every field checked."""
    r, N, mirror = ck["rank"], ck["N"], ck["mirror"]
    B, _ = shape_of(ck)
    U, V, b, v = ck["U"], ck["V"], ck["bias"], ck["v"]
    s = shift_for(v) if force_shift is None else int(force_shift)
    g = [min(89, max(1, int(round(abs(x) * (1 << s) / 32.0)))) for x in v]
    capsum = 32 * sum(g)
    if capsum > 65534:
        # g_k = round(|v_k|*2^s/32) GROWS with s, so the fix is a SMALLER
        # shift, which halves every gain and costs one bit of cp resolution.
        # (An earlier version of this message said "raise", which is exactly
        # backwards and would have sent the next reader the wrong way.)
        fits = next((t for t in range(s, -1, -1)
                     if 32 * sum(min(89, max(1, int(round(abs(x) * (1 << t) / 32.0))))
                                 for x in v) <= 65534), 0)
        raise SystemExit(
            "REFUSED: sum_k G_k = %d exceeds nn_cp's lane-sum fold (65534) by "
            "%d (%.1f%% over, N=%d, mean gain %.1f). A saturating position "
            "would wrap the lane sum and the eval would be garbage.\n"
            "  The shift must go DOWN, not up: shift %d fits (%d), costing "
            "%d bit(s) of cp resolution.\n"
            "  Do not re-derive a packed net at a shift it was not trained "
            "at -- that is a train/ship divergence. Retrain with the "
            "cap-sum-aware export_shift and rebuild."
            % (capsum, capsum - 65534, 100.0 * (capsum - 65534) / 65534, N,
               sum(g) / N, fits,
               32 * sum(min(89, max(1, int(round(abs(x) * (1 << fits) / 32.0))))
                        for x in v), s - fits))
    bd = [min(45, max(-44, int(round(b[k] * 32.0 * g[k])))) for k in range(N)]
    clipped = sum(1 for k in range(N)
                  if abs(round(b[k] * 32.0 * g[k])) > 45)
    # U as STORED: the folded rows when mirrored.  Row i of the folded table
    # is any full-board row that maps to it, and file fl < 4 maps to itself.
    nsq = 32 if mirror else 64
    rows = []
    for i in range(12 * nsq * B):
        b, rem = divmod(i, 12 * nsq)
        pc, sq = divmod(rem, nsq)
        f = b * 768 + pc * 64 + (sq // 4 * 8 + sq % 4 if mirror else sq)
        rows.append(U[f])
    q = [s] + g + [d + 44 for d in bd]
    for j in range(r):
        for k in range(N):
            d = V[j][k]
            if not -44 <= d <= 45:
                raise SystemExit("REFUSED: V[%d][%d] = %d is outside the "
                                 "payload digit range [-44, 45]" % (j, k, d))
            q.append(d + 44)
    for row in rows:
        for c in range(0, r, 4):
            grp = row[c:c + 4]
            assert all(t in (-1, 0, 1) for t in grp), grp
            q.append(sum((t + 1) * 3 ** j for j, t in enumerate(grp)))
    assert len(q) == 1 + 2 * N + r * N + 12 * nsq * B * ((r + 3) // 4), len(q)
    return q, s, g, bd, clipped, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="/tmp/factor_entry.py")
    ap.add_argument("--packed", default="/tmp/factor_entry.packed")
    ap.add_argument("--force-shift", type=int, default=None,
                    help="WHAT-IF ONLY: override the export shift to price a "
                         "shape before retraining at it. The artifact this "
                         "produces is a TRAIN/SHIP DIVERGENCE and must never be "
                         "screened or shipped -- it evaluates a net that was "
                         "trained against a different resolution.")
    a = ap.parse_args()
    ck = pickle.load(open(a.ckpt, "rb"))
    if ck.get("kind") != "factor":
        raise SystemExit("not an arch=factor checkpoint: kind=%r" % ck.get("kind"))
    r, N, mirror = ck["rank"], ck["N"], ck["mirror"]
    q, s, g, bd, clipped, rows = payload(ck, a.force_shift)
    if a.force_shift is not None:
        print("*** WHAT-IF BUILD at forced shift %d: this artifact is a "
              "TRAIN/SHIP DIVERGENCE and prices bytes ONLY. Do not screen it."
              % a.force_shift)

    B, bkind = shape_of(ck)
    src = open(os.path.join(REPO, "nnue_4k", "replnet_proto.py")).read()
    eng, ndig, _ = M.build(src, r, N, 16, 0.43, mirror=mirror,
                           buckets=B, bkind=bkind)
    assert ndig == len(q), ("the engine expects %d digits, the payload is %d"
                            % (ndig, len(q)))
    body = "".join(enc(d) for d in q)
    pat = re.compile(r'b"[^"]*"')
    m = max(pat.finditer(eng), key=lambda m: len(m.group()))
    eng = eng[:m.start()] + 'b"%s"' % body + eng[m.end():]
    open(a.out, "w").write(eng)
    os.chmod(a.out, 0o755)

    # -- certify: the engine's own decode must reproduce the trained state
    import importlib.util
    spec = importlib.util.spec_from_file_location("fe", a.out)
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    assert e._q == q, "the base-90 literal did not round-trip"
    assert e.SHIFT == s and list(e._g) == g, (e.SHIFT, s)
    Vd = [[q[1 + 2 * N + j * N + k] - 44 for k in range(N)] for j in range(r)]
    assert Vd == ck["V"], "V did not survive the payload"
    nsq = 32 if mirror else 64
    sqs = [21 + f // 8 * 10 + f % 8 for f in range(64)]
    halves = e._HALVES if B > 1 else [e._half]
    assert len(halves) == B, ("engine built %d half-tables, checkpoint has %d"
                              % (len(halves), B))
    for b in range(B):
        for i, p in enumerate(e._PIECES):
            for f in range(64):
                fl = f % 8
                j = (b * 12 + i) * nsq + (f // 8 * 4 + min(fl, 7 - fl)
                                          if mirror else f)
                want = sum(sum(rows[j][t] * Vd[t][k] for t in range(r)) << 16 * k
                           for k in range(N))
                assert halves[b][p][sqs[f]] == want, ("row mismatch", b, p, f)

    sz = subprocess.run(["bash", os.path.join(REPO, "tools/build/pack.sh"),
                         a.out, a.packed], capture_output=True, text=True)
    n = int([l for l in sz.stdout.splitlines() if "Total" in l][0].split()[-1])
    run = subprocess.run([a.packed], input="uci\nposition startpos\ngo nodes 3000\nquit\n",
                         capture_output=True, text=True, timeout=120)
    bm = [l for l in run.stdout.splitlines() if l.startswith("bestmove")]
    zeros = 100.0 * sum(1 for row in rows for t in row if t == 0) / max(
        1, sum(len(row) for row in rows))

    print("FACTORED ENTRY BUILT AND CERTIFIED")
    print("  shape          r=%d N=%d mirror=%s B=%d(%s)   %d stored rows, "
          "U zeros %.1f%%" % (r, N, mirror, B, bkind, len(rows), zeros))
    print("  payload        %d base-90 digits" % len(q))
    print("  shift %d  caps sum %d/65534  bias digits clipped %d/%d"
          % (s, 32 * sum(g), clipped, N))
    print("  decode         literal round-trips; V exact; all %d rows == U@V "
          "in every bucket" % (768 * B))
    print("  PACKED         %d B   (%d spare against 4096)" % (n, 4096 - n))
    print("  runs           %s" % (bm[0] if bm else "NO BESTMOVE -- BROKEN"))
    if n > 4096:
        print("  OVER BUDGET by %d B" % (n - 4096))


if __name__ == "__main__":
    main()
