"""verify_export, two claims separated.

(A) BIT-EXACT: an independent INTEGER reference of nn_cp -- built from the
    checkpoint's U, V, bias digits, gains and shift, in pure python, with no
    big-int tricks -- must equal the spliced engine's nn_cp exactly. This is
    the claim the registration demands and it admits no tolerance.
(B) QUANTIZATION RESIDUAL: the trained FLOAT model vs the engine. This is
    expected to be nonzero (the bias band, the gain grid and the truncating
    shift all round), and it is reported as a magnitude, never as a pass.

usage: verify_cp2.py CKPT.pickle ENGINE.py [NFENS]
"""
import importlib.util, io, pickle, sys
sys.path.insert(0, ".")
import numpy as np, torch
import features, config, model as M

ck = pickle.load(open(sys.argv[1], "rb"))
eng_path = sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 200
cfgd = ck["meta"]["config"]["model"]
mc = config.ModelCfg(**{k: v for k, v in cfgd.items()
                        if k in config.ModelCfg.__dataclass_fields__})
B, r, N = len(ck["U"]) // 768, ck["rank"], ck["N"]
ext = features.extractor_for(mc.kb, mc.pb)
CLAMP = int(ck["clampcp"])

net = M.build_model(mc)
with torch.no_grad():
    net.v.copy_(torch.tensor(ck["v"], dtype=torch.float32))
    net.bias.copy_(torch.tensor(ck["bias"], dtype=torch.float32))
    g = [int(x) for x in net.lane_gains()]
    s = int(net.export_shift())
bd = [min(45, max(-44, int(round(ck["bias"][k] * 32.0 * g[k])))) for k in range(N)]
U, V = ck["U"], ck["V"]
ROW = [[sum(U[f][j] * V[j][k] for j in range(r)) for k in range(N)]
       for f in range(768 * B)]
CAP = [32 * g[k] for k in range(N)]

def ref_cp(feats, ow, ob):
    """The engine's arithmetic, in plain integers."""
    us = list(bd); them = list(bd)
    MIR = features.mirror_map()
    for f in feats:
        ru, rt = ROW[f + 768 * ow], ROW[int(MIR[f]) + 768 * ob]
        for k in range(N):
            us[k] += ru[k]; them[k] += rt[k]
    v = sum(min(max(us[k], 0), CAP[k]) - min(max(them[k], 0), CAP[k])
            for k in range(N))
    return max(-CLAMP, min(CLAMP, int(v / (1 << s))))

with torch.no_grad():
    E = (torch.tensor(U, dtype=torch.float64) @ torch.tensor(V, dtype=torch.float64))
    E = E / (32.0 * torch.tensor(g, dtype=torch.float64))
    bq = torch.tensor([bd[k] / (32.0 * g[k]) for k in range(N)], dtype=torch.float64)
    vv = torch.tensor(ck["v"], dtype=torch.float64)

spec = importlib.util.spec_from_file_location("eng", eng_path)
eng = importlib.util.module_from_spec(spec)
old = sys.stdin; sys.stdin = io.StringIO("")
try: spec.loader.exec_module(eng)
finally: sys.stdin = old

d = np.load("pool10m.npz", allow_pickle=False)
fens = d["fens"]; step = max(1, len(fens) // NF)
MIR = features.mirror_map()
exact, resid, n = 0, [], 0
for raw in fens[::step][:NF]:
    f = raw.decode() if isinstance(raw, bytes) else raw
    parts = f.split()
    board = features.fen_to_board120(parts[0])
    if parts[1] == "b": board = board[::-1].swapcase()
    feats, _, kbs = features.extract(board)
    ow, ob = ext.codes(*kbs, features.phase_of(board))
    if B > 1: eng.ROWS[:] = eng._mkrows(*eng._pick(board))
    pos = eng.from_board(board)
    cp_e = int(eng.nn_cp(pos.acc, pos.pf))
    cp_r = ref_cp(feats, ow, ob)
    exact += (cp_e == cp_r)
    ti = torch.tensor(feats, dtype=torch.long)
    acc = E[ti + 768 * ow].sum(0) + bq
    accm = E[MIR[ti] + 768 * ob].sum(0) + bq
    cp_f = float(((acc.clamp(0, 1) - accm.clamp(0, 1)) * vv).sum())
    resid.append(cp_f - cp_e); n += 1

resid = np.abs(np.array(resid))
print("%-34s A) integer ref == engine: %d/%d %s" % (
    eng_path.split("/")[-1], exact, n, "BIT-EXACT" if exact == n else "*** MISMATCH ***"))
print("%-34s B) float model vs engine: mean %.3f cp  max %.2f cp (quantization)"
      % ("", resid.mean(), resid.max()))
sys.exit(0 if exact == n else 1)
