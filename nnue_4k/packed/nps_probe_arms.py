"""Relative nps of the built checkouts, INTERLEAVED.

The registered prediction: a pb2 bucket is chosen once at the search ROOT, so
Position.move, the accumulator delta and nn_cp are byte-identical to B=1 and
the per-move cost of buckets is ZERO.  arm10 (B=1) vs arm11 (pb2) is that
prediction's clean A/B -- same r, same N, same everything but the bucket.

Interleaved round-robin over engines so a drifting box load hits every arm
equally; absolutes are load-confounded and only the RATIOS are quoted.
"""
import os, subprocess, sys, time

ARENA = "/home/thomas-ahle/sunfish-bench/screens"
POS = ["", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
       "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6"]
NODES = 30000
ROUNDS = 5
engines = sys.argv[1:] or ["arm10", "arm11", "arm15", "entryd0"]
tot = {e: 0.0 for e in engines}
cnt = {e: 0 for e in engines}
for _ in range(ROUNDS):
    for e in engines:                      # interleaved, not blocked
        cmds = "uci\n" + "".join(
            "position startpos%s\ngo nodes %d\n" % ((" moves " + p) if p else "", NODES)
            for p in POS) + "quit\n"
        t0 = time.perf_counter()
        subprocess.run(["%s/w_%s.sh" % (ARENA, e)], input=cmds,
                       capture_output=True, text=True, timeout=600)
        tot[e] += time.perf_counter() - t0
        cnt[e] += 1
base = tot[engines[0]] / cnt[engines[0]]
print("interleaved x%d, %d positions, %d nodes each  (load-confounded in "
      "ABSOLUTE terms; ratios only)" % (ROUNDS, len(POS), NODES))
for e in engines:
    s = tot[e] / cnt[e]
    print("  %-9s %7.2f s/round   %6.3f x %s   (nps index %5.1f%%)"
          % (e, s, s / base, engines[0], 100.0 * base / s))
