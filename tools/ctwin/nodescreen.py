#!/usr/bin/env python3
"""C-only node + movegen-call screen over the battery matrix.

For each cell (battery.json, or --cells NAME,NAME,...) runs the difftest
position sample through sunfish_c at fixed depth and reports total nodes
and gen_moves() walks vs the master cell.  This is the cheap first pass
that prunes the matrix before any games: a cell that blows up nodes on
the same probes is dominated before it plays a move.

Identity per cell is difftest's job (--set of the same knobs), not this
script's; run the identity suites before trusting a cell's numbers.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import difftest


def run_cell(knobs, positions, depth, binary, tables):
    argv = [binary, tables] + ["%s=%d" % (k, v) for k, v in sorted(knobs.items())]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    nodes = gens = 0
    try:
        for _, poscmd in positions:
            proc.stdin.write("reset\n%s\ngo depth %d\n" % (poscmd, depth))
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("engine died: %s" % argv)
                if line.startswith("done"):
                    f = line.split()
                    nodes += int(f[2])
                    gens += int(f[4])
                    break
                if line.startswith("err"):
                    raise RuntimeError("engine: %s (%s)" % (line.strip(), argv))
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
    return nodes, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=os.path.join(HERE, "battery.json"))
    ap.add_argument("--cells", default=None, help="comma list; default = all")
    ap.add_argument("--n", type=int, default=0, help="positions beyond startpos (0=all)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--binary", default=os.path.join(HERE, "sunfish_c"))
    ap.add_argument("--tables", default=os.path.join(HERE, "tables_classic.txt"))
    args = ap.parse_args()

    cells = json.load(open(args.battery))["cells"]
    names = args.cells.split(",") if args.cells else list(cells)
    positions = difftest.load_positions(args.n)

    base = None
    print("cell                      nodes        gen_calls   vs master")
    for name in names:
        nodes, gens = run_cell(cells[name], positions, args.depth, args.binary, args.tables)
        if name == "master":
            base = (nodes, gens)
        rel = ""
        if base and name != "master":
            rel = "nodes %+.2f%%  gen %+.2f%%" % (100.0 * (nodes - base[0]) / base[0],
                                                  100.0 * (gens - base[1]) / base[1])
        print("%-24s %10d %12d   %s" % (name, nodes, gens, rel), flush=True)
    if base is None:
        print("(no master cell in the list: absolute numbers only)")


if __name__ == "__main__":
    main()
