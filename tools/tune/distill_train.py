"""Train the 384-parameter student on the teacher's labels, in PyTorch.

ONE VARIABLE. This is deliberately the same 384-parameter model, the same
feature matrix, the same positions, the same 80/20 split and the same
loss FORM as the C1/C2 fit that measured -57.7 and -93.8. The only thing
that changes is WHERE THE LABEL COMES FROM: our own search's converged value
instead of Stockfish's depth-8 score. If the distilled student converts and C2
did not, the teacher is what did it, and no other explanation is available.

Optimisers come from `torch.optim`; nothing here is hand-rolled. The ARTIFACT
keeps zero dependencies -- torch is trainer-side only and never appears in the
entry, which stores integers and decodes them with the shipped 13-byte
decoder. numpy is left in charge of the closed-form solve that the linear arm
is checked against.

ARMS
  linear   float weights, LBFGS. The sanity arm: it must land on the same
           optimum the closed-form/scipy fit finds, or the torch harness is
           wrong and every number after it is too.
  q8 / q16 QUANTISATION-AWARE, straight-through. The table that ships is
           integers on a step-8 (or step-16) grid; rounding a float fit
           afterwards optimises the wrong function. STE puts the rounding
           INSIDE the forward pass so the fit is over tables we can actually
           store. Step 8 is already measured to cost ~180 B less than exact
           and to play the exact fit's move.

The king table is frozen throughout: its value is a 60000 sentinel and the
K_MID/K_END pair is the landed kend fix, which a fit must not overwrite.

usage: distill_train.py DATA.npz OUTDIR [ARMS]
"""
import hashlib
import json
import os
import pathlib
import re
import sys

import chess
import numpy as np
import torch

DATA = sys.argv[1]
OUTDIR = sys.argv[2]
ARMS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["linear", "q8", "q16"])
REPO = str(pathlib.Path(__file__).resolve().parents[2])
PIECES = "PNBRQK"
PHASE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}   # 4ku's weights
K = 350.0                                                  # cp -> win-prob scale
KING = slice(5 * 64, 6 * 64)
SEED = 20260813
os.makedirs(OUTDIR, exist_ok=True)

# Determinism, pinned and recorded. A training run that cannot be reproduced
# cannot be compared with its own successor.
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)          # the laptop is running a timed league
DEV = "cpu"                       # 384 parameters; a GPU is pure overhead here

# ---- data -------------------------------------------------------------------
d = np.load(DATA, allow_pickle=False)
X_np = d["X"].astype(np.float64)
y_np = d["y"].astype(np.float64)
fens = [str(f) for f in d["fens"]]
n = len(y_np)
meta = json.loads(str(d["meta"]))
print("data: %s | %d positions | teacher %s"
      % (os.path.basename(DATA), n, meta.get("teacher", meta.get("engine"))))

ph = np.zeros(n)
for i, fen in enumerate(fens):
    ph[i] = min(sum(PHASE[p.symbol().upper()] for _, p in chess.Board(fen).piece_map().items()), 24)
print("mean phase %.2f/24" % ph.mean())

# THE SPLIT IS KEYED ON THE POSITION, NOT ON ITS ROW NUMBER. An index
# permutation is only stable while the row count is: the distilled set drops
# the positions the teacher saw a mate in, so a `default_rng(SEED).permutation(n)`
# split puts DIFFERENT positions in the held-out set for each teacher, and the
# comparison that this whole experiment rests on -- same positions, same split,
# same model, different label -- quietly stops being true. Hashing the FEN
# gives every position a fixed side of the split in every set it appears in.
va = np.array([i for i, f in enumerate(fens)
               if int(hashlib.sha256((str(SEED) + f).encode()).hexdigest()[:8], 16) % 5 == 0])
tr = np.array([i for i in range(n) if i not in set(va.tolist())])
print("split: %d train / %d held out (by FEN hash, seed %d -- identical assignment "
      "for every set built from these positions)" % (len(tr), len(va), SEED))

t_all = 1.0 / (1.0 + np.exp(-y_np / K))

# ---- warm start: classic's tables, piece values folded in -------------------
src = open(REPO + "/sunfish.py").read()
piece = eval(re.search(r"^piece = (\{[^}]*\})", src, re.M).group(1))
pst0 = eval(re.search(r"^pst = (\{.*?^\})", src, re.M | re.S).group(1))
w0 = np.zeros(384)
for pi, p in enumerate(PIECES):
    tab = np.array(pst0[p], dtype=np.float64) + piece[p]
    # classic writes rank 8 first; our features index chess squares (A1=0).
    w0[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)

Xt = torch.tensor(X_np, dtype=torch.float64, device=DEV)
tt = torch.tensor(t_all, dtype=torch.float64, device=DEV)
TR = torch.tensor(tr, device=DEV)
VA = torch.tensor(va, device=DEV)
W0 = torch.tensor(w0, dtype=torch.float64, device=DEV)
FREE = torch.ones(384, dtype=torch.bool, device=DEV)
FREE[KING] = False


def full(v):
    """Free parameters back into the 384-vector, king frozen at classic's."""
    w = W0.clone()
    return w.masked_scatter(FREE, v)


def loss_of(w, idx):
    p = torch.sigmoid(Xt[idx] @ w / K)
    return ((p - tt[idx]) ** 2).mean()


class Round(torch.autograd.Function):
    """Snap to a step grid forward, pass the gradient through unchanged.

    The straight-through estimator. Without it the gradient of a rounded
    weight is zero almost everywhere and the fit cannot move; with it the
    forward pass evaluates the table we will actually ship.
    """
    @staticmethod
    def forward(ctx, v, step):
        return torch.round(v / step) * step

    @staticmethod
    def backward(ctx, g):
        return g, None


def train(arm, step=None, iters=400):
    v = W0[FREE].clone().requires_grad_(True)
    if step is None:
        opt = torch.optim.LBFGS([v], max_iter=iters, history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-16,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = loss_of(full(v), TR)
            l.backward()
            return l
        opt.step(closure)
        return full(v).detach()
    # Quantisation-aware: the rounding is not differentiable, so LBFGS's line
    # search has nothing to work with. Adam on the straight-through gradient.
    opt = torch.optim.Adam([v], lr=2.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters * 6)
    for _ in range(iters * 6):
        opt.zero_grad()
        loss_of(full(Round.apply(v, step)), TR).backward()
        opt.step()
        sched.step()
    return full(Round.apply(v, step)).detach()


# ---- emit / unemit: the shipped integer form, and the fit re-read from it ---
def emit(w, step=None):
    """Integer tables in classic's shape (rank 8 first), piece value factored
    out -- byte-for-byte the form the generator pastes.

    The piece value is snapped to the same grid as the table when there is
    one. It is subtracted from every entry, so a base off the grid shifts the
    whole table half a step off it and the codec has to round a second time --
    silently undoing the quantisation-aware training that produced the values.
    """
    w = w.cpu().numpy() if torch.is_tensor(w) else w
    out = {}
    for pi, p in enumerate(PIECES):
        tab = w[pi * 64:(pi + 1) * 64]
        if p == "K":
            out[p] = list(pst0["K"])
            out["_value_K"] = piece["K"]
            continue
        base = float(np.median(tab))
        if step: base = round(base / step) * step
        out[p] = np.round(tab - base).astype(int).reshape(8, 8)[::-1].reshape(64).tolist()
        out["_value_" + p] = int(round(base))
    return out


def unemit(out):
    """Rebuild the weights FROM the emitted integers, indexed exactly as the
    engine indexes them. A mirrored or dropped table shows up here as a loss
    explosion instead of as -67 Elo in a match."""
    w = np.zeros(384)
    for pi, p in enumerate(PIECES):
        tab = np.array(out[p], dtype=np.float64) + out["_value_" + p]
        w[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)
    return torch.tensor(w, dtype=torch.float64, device=DEV)


def bands(w):
    """Held-out loss per phase band -- DIAGNOSTIC, and a weak one.

    C2's failure was explained on the record by this table ("the middlegame
    band is slightly worse"). Refitting on 12 splits, that band reads
    -2.96 +/- 2.36 and its SIGN FLIPS (`band_stability.py`): it is the least
    improved band, not a worse one. Read these rows for shape, never as a
    mechanism, and never as a prediction of Elo."""
    out = {}
    for lo, hi in ((0, 5), (6, 11), (12, 17), (18, 24)):
        idx = va[(ph[va] >= lo) & (ph[va] <= hi)]
        if len(idx) < 30: continue
        b = loss_of(W0, torch.tensor(idx, device=DEV)).item()
        c = loss_of(w, torch.tensor(idx, device=DEV)).item()
        out["%d-%d" % (lo, hi)] = (len(idx), b, c, 100 * (c - b) / b)
    return out


base_tr, base_va = loss_of(W0, TR).item(), loss_of(W0, VA).item()
print("\nclassic (no fit)      train %.6f   HELD-OUT %.6f" % (base_tr, base_va))

# The linear arm is the harness check: run on the Stockfish-labelled set it
# has to reproduce the scipy fit that became C2 -- which it does, on held-out
# loss, on the phase bands, on packed bytes, and on C2's own first-yield
# failure node for node. See the ledger entry.
results = {}
for arm in ARMS:
    step = {"linear": None, "q8": 8, "q16": 16}[arm]
    w = train(arm, step)
    tr_l, va_l = loss_of(w, TR).item(), loss_of(w, VA).item()
    out = emit(w, step)
    e_l = loss_of(unemit(out), VA).item()
    ok = "EMIT OK" if abs(e_l - va_l) < 2e-5 else "EMIT MISMATCH"
    print("%-7s (384p, step %-4s) train %.6f   HELD-OUT %.6f  (%+.2f%%)  emit %.6f  %s"
          % (arm, step or "free", tr_l, va_l, 100 * (va_l - base_va) / base_va, e_l, ok))
    for k, (m, b, c, pct) in bands(w).items():
        print("        phase %-6s n=%-5d classic %.6f  student %.6f  (%+.2f%%)" % (k, m, b, c, pct))
    results[arm] = {"heldout": va_l, "train": tr_l, "emit": e_l, "emit_ok": ok == "EMIT OK",
                    "step": step, "tables": out,
                    "bands": {k: list(v) for k, v in bands(w).items()}}

prov = {
    "torch": torch.__version__,
    "numpy": np.__version__,
    "python": sys.version.split()[0],
    "seed": SEED,
    "deterministic_algorithms": True,
    "threads": 1,
    "device": DEV,
    "data": os.path.basename(DATA),
    "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest(),
    "data_meta": meta,
    "split": "20% held out by sha256(seed+fen), stable across datasets",
    "objective": "MSE on sigmoid(cp/%g) -- the same form C1/C2 were fitted with" % K,
}
json.dump({"provenance": prov, "classic_heldout": base_va, "arms": results},
          open(os.path.join(OUTDIR, "students.json"), "w"))
print("\nwrote %s/students.json" % OUTDIR)
print("CANDIDATES ONLY. Held-out loss is the metric that MIS-RANKED C2 by 5.9%%")
print("while it lost 94 Elo. Nothing above is an Elo claim; only games decide.")
