"""Per-node cost of the pdbl accumulator. Trees DIFFER (the eval changed), so
this compares nps at a fixed node budget, not wall time at a fixed tree --
nps is the per-node quantity, which is what a maintenance cost shows up in.
PALINDROME order, 8 reps, median."""
import importlib.util, statistics, sys, time
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec)
    sys.modules[n] = m; spec.loader.exec_module(m); return m
mods = {"base": load("bin/e_base.py", "b"), "pdbl": load("bin/e_pdbl.py", "p")}
FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8"]
def b120(fb):
    import re
    b = re.sub(r"\d", lambda x: "." * int(x.group(0)), fb)
    b = list(21 * " " + "  ".join(b.split("/")) + 21 * " "); b[9::10] = ["\n"] * 12
    return "".join(b)
res = {k: [] for k in mods}
nodes = {k: [] for k in mods}
for rep in range(8):
    order = ["base", "pdbl"] if rep % 2 == 0 else ["pdbl", "base"]
    for name in order:
        mod = mods[name]; tot_n = 0; t0 = time.perf_counter()
        for fb in FENS:
            pos = mod.from_board(b120(fb))
            s = mod.Searcher(); s.node_cap = 25000
            try:
                for d, g, sc, mv in s.search([pos]):
                    if s.nodes >= 25000: break
            except mod.Stop: pass
            tot_n += s.nodes
        res[name].append(time.perf_counter() - t0); nodes[name].append(tot_n)
for name in ("base", "pdbl"):
    t = statistics.median(res[name]); n = statistics.median(nodes[name])
    print("%-5s  %6d nodes  %.4f s  = %7.0f nps" % (name, n, t, n / t))
rb = statistics.median(nodes["base"]) / statistics.median(res["base"])
rp = statistics.median(nodes["pdbl"]) / statistics.median(res["pdbl"])
print("pdbl / base nps ratio: %.4f  (%.1f%% of nps)" % (rp / rb, 100 * rp / rb))
