"""ROOT-FREEZE AGREEMENT GATE (registered binding, 2026-08-19).

The engine picks the bucket ONCE at the search root and freezes it; the
trainer saw every position's TRUE bucket.  So on any line where the bucket
would have changed, the engine evaluates with the wrong table.  That is the
kbbil failure mode by name -- a constraint the loss cannot see -- and it does
not get to be waved through with "the crossing rate is small".

The gate measures the two factors separately and multiplies them:
  RATE      already measured: 2.16% of generated moves cross a king rank
            band, 0.07% cross the material-phase cut.
  MAGNITUDE measured here: |cp(true bucket) - cp(frozen/wrong bucket)|,
            which is what the wrongness is actually worth.

usage: rootfreeze.py CKPT.pickle [NFENS]
"""
import pickle, sys
sys.path.insert(0, ".")
import numpy as np, torch
import features, config, model as M

ck = pickle.load(open(sys.argv[1], "rb"))
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 400
cfgd = ck["meta"]["config"]["model"]
mc = config.ModelCfg(**{k: v for k, v in cfgd.items()
                        if k in config.ModelCfg.__dataclass_fields__})
B, r, N = len(ck["U"]) // 768, ck["rank"], ck["N"]
if B == 1:
    print("B=1: no bucket, gate not applicable"); sys.exit(0)
kind = "kb" if mc.kb > 1 else "pb"
RATE = {"kb": 0.0216, "pb": 0.0007}[kind]
ext = features.extractor_for(mc.kb, mc.pb)
CLAMP = int(ck["clampcp"])

net = M.build_model(mc)
with torch.no_grad():
    net.v.copy_(torch.tensor(ck["v"], dtype=torch.float32))
    net.bias.copy_(torch.tensor(ck["bias"], dtype=torch.float32))
    g = [int(x) for x in net.lane_gains()]; s = int(net.export_shift())
bd = [min(45, max(-44, int(round(ck["bias"][k] * 32.0 * g[k])))) for k in range(N)]
U, V = ck["U"], ck["V"]
ROW = [[sum(U[f][j] * V[j][k] for j in range(r)) for k in range(N)]
       for f in range(768 * B)]
CAP = [32 * g[k] for k in range(N)]
MIR = features.mirror_map()

def cp(feats, ow, ob):
    us, them = list(bd), list(bd)
    for f in feats:
        ru, rt = ROW[f + 768 * ow], ROW[int(MIR[f]) + 768 * ob]
        for k in range(N):
            us[k] += ru[k]; them[k] += rt[k]
    v = sum(min(max(us[k], 0), CAP[k]) - min(max(them[k], 0), CAP[k])
            for k in range(N))
    return max(-CLAMP, min(CLAMP, int(v / (1 << s))))

d = np.load("pool10m.npz", allow_pickle=False)
fens = d["fens"]; step = max(1, len(fens) // NF)
deltas = []
for raw in fens[::step][:NF]:
    f = raw.decode() if isinstance(raw, bytes) else raw
    parts = f.split()
    board = features.fen_to_board120(parts[0])
    if parts[1] == "b": board = board[::-1].swapcase()
    feats, _, kbs = features.extract(board)
    ow, ob = ext.codes(*kbs, features.phase_of(board))
    deltas.append(abs(cp(feats, ow, ob) - cp(feats, (ow + 1) % B, ob)))

a = np.array(deltas, dtype=float)
print("ROOT-FREEZE GATE  %s  B=%d  n=%d" % (kind, B, len(a)))
print("  |cp(true) - cp(wrong bucket)|   mean %.1f  median %.1f  p90 %.1f  max %.1f cp"
      % (a.mean(), np.median(a), np.percentile(a, 90), a.max()))
print("  crossing rate (measured)        %.2f%% of generated moves" % (100 * RATE))
print("  EXPECTED eval error from freezing = rate x mean = %.3f cp" % (RATE * a.mean()))
print("  identical evals: %d/%d (%.0f%%)" % (int((a == 0).sum()), len(a),
                                             100.0 * (a == 0).mean()))
