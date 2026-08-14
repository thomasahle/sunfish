"""Standalone artifact smoke AND time-manager assay, in one instrument.

Smoke: the artifact runs ALONE (empty cwd, SF_NET and PYTHONPATH unset, so no
sunfish_ui/ can be resolved and the builtin loop is what answers) and returns a
LEGAL move -- checked with python-chess, not by eye.

Assay: the same run times each `go`, which is the only black-box way to see
WHICH budget line is live inside a packed artifact. At wtime=60000 the two arms
must differ by construction -- /40 -> ~1.5 s, /12 -> ~5.0 s, both taken at
0.8x by the soft break -- and at winc>0 they must AGREE, because the fix is
byte-identical to the old policy for every winc > 0. An arm that does not show
this has not got the mod that its name claims.

usage: tm_smoke.py ARTIFACT

Shipped to the bench-box arena as tm_smoke.py for the stage-1 run
(~/sunfish-bench/tmfix60-20260814/); this is the archived canonical copy.
"""
import os
import subprocess
import sys
import tempfile
import time

import chess

ENGINE = os.path.abspath(sys.argv[1])
ENV = {k: v for k, v in os.environ.items() if k not in ("SF_NET", "PYTHONPATH")}
CASES = [
    ("wtime 60000 winc 0", "sudden death, full clock"),
    ("wtime 60000 winc 1000", "increment, full clock"),
    ("wtime 1900 winc 0", "sub-2s clock (the negative-cap regime)"),
]

d = tempfile.mkdtemp(prefix="smoke_")
p = subprocess.Popen([ENGINE], cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1, env=ENV)


def send(s):
    p.stdin.write(s)
    p.stdin.flush()


def until(prefix):
    while True:
        o = p.stdout.readline()
        if not o:
            raise SystemExit("ENGINE DIED waiting for %r; stderr: %s"
                             % (prefix, p.stderr.read()[-400:]))
        if o.startswith(prefix):
            return o.strip()


send("uci\n")
banner = until("uciok")
send("isready\n")
until("readyok")
print("%-18s cwd=%s  uciok+readyok OK" % (os.path.basename(ENGINE), d))

bad = 0
for go, why in CASES:
    send("position startpos\ngo %s\n" % go)
    t0 = time.time()
    line = until("bestmove")
    dt = time.time() - t0
    mv = line.split()[1] if len(line.split()) > 1 else None
    legal = mv is not None and mv not in ("(none)", "0000") and \
        chess.Move.from_uci(mv) in chess.Board().legal_moves
    bad += not legal
    print("  go %-24s %6.2f s   bestmove %-6s %s   [%s]"
          % (go, dt, mv, "LEGAL" if legal else "*** NOT LEGAL ***", why))

send("quit\n")
p.wait(timeout=10)
print("  SMOKE %s" % ("PASSED" if not bad else "FAILED"))
sys.exit(bool(bad))
