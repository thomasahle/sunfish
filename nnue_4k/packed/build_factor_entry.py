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
                             feature-major (32 rows/piece when mirrored)

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
    """export.py's own rule, so the artifact and the trainer pick one shift."""
    vmax = max(abs(x) for x in v) or 1.0
    for s in range(8, -1, -1):
        if vmax * (1 << s) / 32.0 <= 89.49:
            return s
    return 0


def payload(ck):
    """Trained state -> the decoder's digit stream, with every field checked."""
    r, N, mirror = ck["rank"], ck["N"], ck["mirror"]
    U, V, b, v = ck["U"], ck["V"], ck["bias"], ck["v"]
    s = shift_for(v)
    g = [min(89, max(1, int(round(abs(x) * (1 << s) / 32.0)))) for x in v]
    capsum = 32 * sum(g)
    if capsum > 65534:
        raise SystemExit(
            "REFUSED: sum_k G_k = %d exceeds nn_cp's lane-sum fold (65534) by "
            "%d. A saturating position would wrap and the eval would be "
            "garbage. Raise the export shift (halves every gain) and retrain "
            "or re-derive -- do NOT pack this net." % (capsum, capsum - 65534))
    bd = [min(45, max(-44, int(round(b[k] * 32.0 * g[k])))) for k in range(N)]
    clipped = sum(1 for k in range(N)
                  if abs(round(b[k] * 32.0 * g[k])) > 45)
    # U as STORED: the folded rows when mirrored.  Row i of the folded table
    # is any full-board row that maps to it, and file fl < 4 maps to itself.
    nsq = 32 if mirror else 64
    rows = []
    for i in range(12 * nsq):
        f = (i // nsq) * 64 + (i % nsq) // 4 * 8 + (i % 4) if mirror else i
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
    assert len(q) == 1 + 2 * N + r * N + 12 * nsq * ((r + 3) // 4), len(q)
    return q, s, g, bd, clipped, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="/tmp/factor_entry.py")
    ap.add_argument("--packed", default="/tmp/factor_entry.packed")
    a = ap.parse_args()
    ck = pickle.load(open(a.ckpt, "rb"))
    if ck.get("kind") != "factor":
        raise SystemExit("not an arch=factor checkpoint: kind=%r" % ck.get("kind"))
    r, N, mirror = ck["rank"], ck["N"], ck["mirror"]
    q, s, g, bd, clipped, rows = payload(ck)

    src = open(os.path.join(REPO, "nnue_4k", "replnet_proto.py")).read()
    eng, ndig, _ = M.build(src, r, N, 16, 0.43, mirror=mirror)
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
    for i, p in enumerate(e._PIECES):
        for f in range(64):
            fl = f % 8
            row = rows[i * nsq + (f // 8 * 4 + min(fl, 7 - fl) if mirror else f)]
            want = sum(sum(row[j] * Vd[j][k] for j in range(r)) << 16 * k
                       for k in range(N))
            assert e._half[p][sqs[f]] == want, ("row mismatch", p, f)

    sz = subprocess.run(["bash", os.path.join(REPO, "tools/build/pack.sh"),
                         a.out, a.packed], capture_output=True, text=True)
    n = int([l for l in sz.stdout.splitlines() if "Total" in l][0].split()[-1])
    run = subprocess.run([a.packed], input="uci\nposition startpos\ngo nodes 3000\nquit\n",
                         capture_output=True, text=True, timeout=120)
    bm = [l for l in run.stdout.splitlines() if l.startswith("bestmove")]
    zeros = 100.0 * sum(1 for row in rows for t in row if t == 0) / max(
        1, sum(len(row) for row in rows))

    print("FACTORED ENTRY BUILT AND CERTIFIED")
    print("  shape          r=%d N=%d mirror=%s   %d stored rows, U zeros %.1f%%"
          % (r, N, mirror, len(rows), zeros))
    print("  payload        %d base-90 digits" % len(q))
    print("  shift %d  caps sum %d/65534  bias digits clipped %d/%d"
          % (s, 32 * sum(g), clipped, N))
    print("  decode         literal round-trips; V exact; all 768 rows == U@V")
    print("  PACKED         %d B   (%d spare against 4096)" % (n, 4096 - n))
    print("  runs           %s" % (bm[0] if bm else "NO BESTMOVE -- BROKEN"))
    if n > 4096:
        print("  OVER BUDGET by %d B" % (n - 4096))


if __name__ == "__main__":
    main()
