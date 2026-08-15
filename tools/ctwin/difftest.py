#!/usr/bin/env python3
"""Differential harness: sunfish.c vs the Python reference (sunfish.py).

Drives both engines over the same protocol (see pyref.py) and compares
transcripts BYTE FOR BYTE:
  phase 1  movegen: gen_moves() order and value() of every move, at the
           position and (with --walk) one ply below each move;
  phase 2  search: every MTD-bi probe up to --depth: (depth, gamma, score,
           killer move, node count) -- node-identity, not just bestmove.

Any mismatch prints the position, the phase, and the first divergent line,
then stops that position and counts it.  Exit status 1 on any divergence.

Usage:
  python3 difftest.py --quick            # 10 positions, depth 4, walk
  python3 difftest.py --n 40 --depth 6   # wide sweep
  python3 difftest.py --bench            # speed ratio at identical nodes
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FILES = os.path.join(REPO, "tests", "files")


class Engine:
    def __init__(self, argv, name):
        self.name = name
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=HERE)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("%s: engine died (rc=%s)" % (self.name, self.proc.poll()))
        return line.rstrip("\n")

    def cmd_ok(self, cmd):
        self.send(cmd)
        r = self.readline()
        if r != "ok":
            raise RuntimeError("%s: %r -> %r" % (self.name, cmd, r))

    def cmd_lines(self, cmd, stop_prefix):
        """Send cmd, collect lines until one starts with stop_prefix
        (that line included)."""
        self.send(cmd)
        out = []
        while True:
            line = self.readline()
            out.append(line)
            if line.startswith(stop_prefix):
                return out

    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def load_positions(n):
    """Deterministic position sample: startpos, then fixture FENs/EPDs
    spread evenly through each file (no randomness, stable across runs)."""
    specs = [("startpos", "position startpos")]
    sources = [
        ("chessathome_openings.fen", 6),
        ("bratko_kopec_test.epd", 4),
        ("win_at_chess_test.epd", 4),
        ("mate1.fen", 2),
        ("mate2.fen", 2),
        ("stalemate1.fen", 2),
        ("nullmove_mates.fen", 1),
        ("perft.epd", 3),
        ("queen.fen", 2),
    ]
    for fname, take in sources:
        path = os.path.join(FILES, fname)
        if not os.path.exists(path):
            continue
        lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        step = max(1, len(lines) // take)
        for k, line in enumerate(lines[::step][:take]):
            fen = line.split(";")[0].strip()
            fields = fen.split()
            if len(fields) < 4 or "/" not in fields[0]:
                continue
            # keep placement side castling ep only: EPD has no clocks and
            # sunfish ignores them anyway -- both engines parse 4 fields.
            fen4 = " ".join(fields[:4])
            specs.append(("%s:%d" % (fname, k), "position fen " + fen4))
    return specs[:n + 1] if n else specs


def compare_lists(a, b, label, name):
    if a == b:
        return None
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return ("MISMATCH %s at %s line %d:\n  py: %s\n  c : %s"
            % (label, name, k,
               a[k] if k < len(a) else "<missing>",
               b[k] if k < len(b) else "<missing>"))


def movegen_phase(py, cc, name, walk):
    py_moves = py.cmd_lines("moves", "end")
    c_moves = cc.cmd_lines("moves", "end")
    err = compare_lists(py_moves, c_moves, "movegen", name)
    if err:
        return err, 1
    checked = 1
    if walk:
        for line in py_moves[:-1]:
            _, mv, _ = line.split()
            i, j, prom = mv.split(",")
            push = "push %s %s %s" % (i, j, prom)
            py.cmd_ok(push)
            cc.cmd_ok(push)
            pm = py.cmd_lines("moves", "end")
            cm = cc.cmd_lines("moves", "end")
            err = compare_lists(pm, cm, "movegen+1 (after %s)" % mv, name)
            py.cmd_ok("pop")
            cc.cmd_ok("pop")
            checked += 1
            if err:
                return err, checked
    return None, checked


def search_phase(py, cc, name, depth):
    pl = py.cmd_lines("go depth %d" % depth, "done")
    cl = cc.cmd_lines("go depth %d" % depth, "done")
    err = compare_lists(pl, cl, "search depth<=%d" % depth, name)
    return err, len(pl) - 1, pl[-1]


def run_diff(args):
    specs = load_positions(args.n)
    py = Engine(["pypy3", os.path.join(HERE, "pyref.py")], "pyref")
    cc = Engine([os.path.join(HERE, "sunfish_c"),
                 os.path.join(HERE, "tables_classic.txt")], "ctwin")
    fails, probes, gencmp = 0, 0, 0
    try:
        for name, poscmd in specs:
            for e in (py, cc):
                e.cmd_ok("reset")
                for kv in args.set:
                    e.cmd_ok("set " + kv.replace("=", " "))
                if e is cc:
                    for kv in args.cset:
                        e.cmd_ok("set " + kv.replace("=", " "))
                e.cmd_ok(poscmd)
            err, checked = movegen_phase(py, cc, name, args.walk)
            gencmp += checked
            if err:
                print(err)
                print("  pos: %s (%s)" % (name, poscmd))
                fails += 1
                continue
            err, nprobes, done = search_phase(py, cc, name, args.depth)
            probes += nprobes
            if err:
                print(err)
                print("  pos: %s (%s)" % (name, poscmd))
                fails += 1
            elif args.verbose:
                print("  ok %-32s %3d probes, %s" % (name, nprobes, done))
    finally:
        py.quit()
        cc.quit()
    print("coverage: %d positions x depth 1..%d, %d probes compared, "
          "%d movegen lists compared, %d mismatches"
          % (len(specs), args.depth, probes, gencmp, fails))
    return 1 if fails else 0


def run_bench(args):
    """Wall-time ratio at identical nodes (identity assumed pre-verified;
    node equality is still checked from the done lines)."""
    specs = load_positions(args.n or 6)
    results = []
    for engv, name in (
            (["pypy3", os.path.join(HERE, "pyref.py")], "pyref"),
            ([os.path.join(HERE, "sunfish_c"),
              os.path.join(HERE, "tables_classic.txt")], "ctwin")):
        e = Engine(engv, name)
        tot_t, tot_n = 0.0, 0
        try:
            for pname, poscmd in specs:
                e.cmd_ok("reset")
                e.cmd_ok(poscmd)
                t0 = time.time()
                out = e.cmd_lines("go depth %d" % args.depth, "done")
                tot_t += time.time() - t0
                tot_n += int(out[-1].split()[2])   # "done nodes N gen G"
        finally:
            e.quit()
        results.append((name, tot_t, tot_n))
        print("%s: %.3fs for %d nodes over %d positions at depth %d"
              % (name, tot_t, tot_n, len(specs), args.depth))
    (n1, t1, nn1), (n2, t2, nn2) = results
    if nn1 != nn2:
        print("NODE MISMATCH in bench: %s=%d %s=%d" % (n1, nn1, n2, nn2))
        return 1
    print("speed ratio %s/%s: %.2fx at identical %d nodes" % (n1, n2, t1 / t2, nn1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="positions beyond startpos (0=all sampled)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--walk", action="store_true", help="depth-2 movegen walk")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=V",
                    help="tuning knob applied to BOTH engines (repeatable); "
                         "shared knobs only: QS QS_A LMR EVAL_ROUGHNESS TABLE_SIZE")
    ap.add_argument("--cset", action="append", default=[], metavar="NAME=V",
                    help="knob applied to the C SIDE ONLY (repeatable). For "
                         "PR-service knobs where the Python side IS the "
                         "checked-out reference (its behavior is not a knob), "
                         "e.g. QS_TAIL=1 against a pr171 worktree, plus any "
                         "flavor knobs the PR base needs (IID_MIN_DEPTH, "
                         "MATE_DIST). pyref stays strict: an unknown --set "
                         "still fails loudly.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n, args.depth, args.walk = 10, 4, True
    if args.bench:
        sys.exit(run_bench(args))
    sys.exit(run_diff(args))


if __name__ == "__main__":
    main()
