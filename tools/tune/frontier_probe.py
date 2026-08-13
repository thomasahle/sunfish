"""What does the engine actually reach in a game? The teacher must go past it.

Distillation only teaches something the student does not already know, so the
teacher's budget has to be read against the budget the engine spends on a move
in the matches that decide the question -- 30+1 on the bench box, where the
entry's own rule spends wtime/12 + 0.9*inc.

Timed, so it is a measurement of THIS BOX UNDER THIS LOAD, and the load is
printed with it. That is the honest frame: the same contention applies to the
games these labels are meant to win.

usage: frontier_probe.py ENTRY.py FENS.txt [WTIME_MS] [INC_MS]
"""
import os
import subprocess
import sys

ENGINE, FENS = os.path.abspath(sys.argv[1]), sys.argv[2]
WT = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
INC = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = []
for fen in [ln.strip() for ln in open(FENS) if ln.strip()]:
    p = subprocess.Popen([sys.executable, ENGINE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1,
                         env=dict(os.environ, PYTHONPATH=REPO))
    p.stdin.write("uci\nisready\nposition fen %s\ngo wtime %d btime %d winc %d binc %d\n"
                  % (fen, WT, WT, INC, INC))
    p.stdin.flush()
    depth = nodes = ms = 0
    driver = None
    while True:
        o = p.stdout.readline()
        if not o: break
        if o.startswith("info string driver"): driver = o.strip()
        if o.startswith("info depth"):
            t = o.split()
            depth, ms, nodes = int(t[2]), int(t[4]), int(t[6])
        if o.startswith("bestmove"): break
    p.stdin.write("quit\n"); p.stdin.flush(); p.wait(timeout=10)
    if driver is None or " fen" not in driver:
        raise SystemExit("WRONG DRIVER: %s -- the builtin loop ignores `position fen`" % (driver,))
    rows.append((depth, nodes, ms))

d = sorted(r[0] for r in rows); n = sorted(r[1] for r in rows); t = sorted(r[2] for r in rows)
mid = len(rows) // 2
print("%d positions at %d+%.0f  (think = wtime/12 + 0.9*inc = %.2f s)"
      % (len(rows), WT // 1000, INC / 1000, (WT / 12 + 0.9 * INC) / 1000))
print("depth  median %d   min %d   max %d" % (d[mid], d[0], d[-1]))
print("nodes  median %d   min %d   max %d" % (n[mid], n[0], n[-1]))
print("nps    median %d" % (1000 * n[mid] / max(t[mid], 1)))
print("load: %s" % open("/proc/loadavg").read().split(" ", 3)[:3] if os.path.exists("/proc/loadavg") else "")
