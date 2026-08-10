#!/usr/bin/env python3
"""Train the 768 -> N -> 1 residual net that the packed engine evaluates.

The architecture is forced by what the packed head can compute cheaply
(packed/pnet.py explains why):

    pred_cp = pst_cp(position)                      # classic sunfish, FIXED
            + clip(sum_k v_k*(crelu(a_k^us) - crelu(a_k^them)), -CLAMP, CLAMP)

  * ONE hidden layer.  A second matrix is what made the previous campaign's
    per-node cost O(L1*L2) python multiplies.
  * SYMMETRIC perspectives: the same v_k serves both halves with opposite
    sign.  That is what lets both halves share one lane layout, and it makes
    the evaluation exactly antisymmetric for free.
  * The piece-square part is not trained.  It is classic's own table, so
    `value(move)` in the engine stays an exact delta of it and move
    ordering, the QS gate and futility pruning behave as they do in classic.
    The net only learns the residual, weighted where the loss says it
    matters.
  * clip(), not a tanh squash: the engine applies a hard clip, and training
    with the operator you will actually run is worth more than a smooth
    gradient at the boundary.

Data is the lichess Stockfish-evaluated dump, filtered exactly as the
previous campaign's best net was -- the quiet-position filter was worth
about +130 Elo there and costs nothing here.
"""
import argparse, json, os, pickle, random, subprocess, sys, time
from array import array

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnet

p = argparse.ArgumentParser()
p.add_argument("--data", default="eval_slice.zst")
p.add_argument("--N", type=int, default=256)
p.add_argument("--out", default="net.pickle")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=8192)
p.add_argument("--limit", type=int, default=2_000_000)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--sigK", type=float, default=400.0)
p.add_argument("--cpmax", type=int, default=1000)
p.add_argument("--clampcp", type=int, default=600)
p.add_argument("--wclip", type=float, default=1.0)
p.add_argument("--segs", type=int, default=1,
               help="activation segments: 1 = clipped ReLU, 3 ~ squared clipped ReLU")
p.add_argument("--quiet", type=int, default=1)
p.add_argument("--factor", type=int, default=1,
               help="train with virtual per-piece-type features (folded into "
                    "the real table at export; helps generalisation, free at "
                    "run time)")
p.add_argument("--losspow", type=float, default=2.0,
               help="loss exponent p in |sig(pred)-sig(y)|^p; nnue-pytorch "
                    "found ~2.6 better than plain MSE")
p.add_argument("--cache", default="")
p.add_argument("--workers", type=int, default=0,
               help="parse the dump with this many processes (0 = in-process)")
p.add_argument("--threads", type=int, default=0,
               help="torch CPU threads (0 = torch default)")
args = p.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

PIECES = pnet.PIECES
PIDX = {c: i for i, c in enumerate(PIECES)}

# classic sunfish's piece-square tables, verbatim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sunfish as classic
PST = classic.pst


def sq64(i):
    return (i // 10 - 2) * 8 + (i % 10 - 1)


def feat(p_, i):
    return PIDX[p_] * 64 + sq64(i)


def fen_to_board120(fen_board):
    board = [" "] * 20
    for row in fen_board.split("/"):
        line = [" "]
        for ch in row:
            line += ["."] * int(ch) if ch.isdigit() else [ch]
        board += line + ["\n"]
    return "".join(board + [" "] * 20)


def parse_lines(lines):
    """One worker's share: JSON lines -> (feats, lens, pstc, y) arrays."""
    FEATS, LENS, PSTC, Y = array("i"), array("i"), array("i"), array("i")
    for line in lines:
        d = json.loads(line)
        fen = d["fen"].split()
        ev = max(d["evals"], key=lambda e: e["depth"])
        pv = ev["pvs"][0]
        if "cp" not in pv or abs(pv["cp"]) > args.cpmax:
            continue
        cp = pv["cp"]
        board = fen_to_board120(fen[0])
        if args.quiet:
            # the engine evaluates quiescence leaves; training on tactical
            # positions teaches averages over unresolved captures
            mv = pv["line"].split()[0]
            dst = (8 - int(mv[3])) * 10 + 21 + (ord(mv[2]) - 97)
            if board[dst].isalpha() or len(mv) > 4:
                continue
        if fen[1] == "b":
            board, cp = board[::-1].swapcase(), -cp
        ps = n = 0
        for i, c in enumerate(board):
            if c.isalpha():
                FEATS.append(feat(c, i))
                n += 1
                ps += PST[c][i] if c.isupper() else -PST[c.upper()][119 - i]
        LENS.append(n)
        PSTC.append(ps)
        Y.append(cp)
    return FEATS, LENS, PSTC, Y


def parse(path, limit, workers=0):
    proc = subprocess.Popen(["zstd", "-d", "-c", path], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    FEATS, OFFS, PSTC, Y = array("i"), array("i"), array("i"), array("i")
    off = 0

    def chunks():
        while True:
            lines = proc.stdout.readlines(1 << 23)      # ~8 MB of lines
            if not lines:
                return
            yield lines

    def gather(parts):
        nonlocal off
        for f, l, ps, y in parts:
            FEATS.extend(f)
            for n in l:
                OFFS.append(off)
                off += n
            PSTC.extend(ps)
            Y.extend(y)
            if len(Y) % 1_000_000 < len(y):
                print("  ...%dM positions" % (len(Y) // 1_000_000), flush=True)
            if len(Y) >= limit:
                return True
        return False

    if workers <= 1:
        gather(parse_lines(ls) for ls in chunks())
    else:
        # fork, explicitly: spawn would re-execute this script in every
        # worker (there is deliberately no __main__ guard)
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            if gather(pool.imap(parse_lines, chunks())):
                pool.terminate()
    proc.kill()
    return FEATS, OFFS, PSTC, Y


t0 = time.time()
if args.cache and os.path.exists(args.cache):
    with open(args.cache, "rb") as f:
        FEATS, OFFS, PSTC, Y = pickle.load(f)
    print("loaded %d cached positions in %.0fs" % (len(Y), time.time() - t0), flush=True)
else:
    print("parsing data...", flush=True)
    FEATS, OFFS, PSTC, Y = parse(args.data, args.limit, args.workers)
    print("%d positions in %.0fs" % (len(Y), time.time() - t0), flush=True)
    if args.cache:
        with open(args.cache, "wb") as f:
            pickle.dump((FEATS, OFFS, PSTC, Y), f, protocol=4)

feats = torch.tensor(FEATS, dtype=torch.long)
offs = torch.tensor(OFFS, dtype=torch.long)
pstc = torch.tensor(PSTC, dtype=torch.float32)
ys = torch.tensor(Y, dtype=torch.float32)
lens = torch.diff(offs, append=torch.tensor([len(FEATS)]))
del FEATS, OFFS, PSTC, Y

# the mirrored feature of index f: swap colour, flip the square
IDX = torch.arange(768)
MIRROR = ((IDX // 64 + 6) % 12) * 64 + (63 - IDX % 64)
mfeats = MIRROR[feats]

N = args.N
CLAMP = float(args.clampcp)
K = args.sigK
# breakpoints of the convex piecewise-linear activation; A normalises it
# so phi(0)=0 and phi(1)=1 whatever the number of segments
SEGS = tuple(i / args.segs for i in range(args.segs))
AA = sum(1.0 - t for t in SEGS)


def act(a):
    s = torch.relu(a)
    for t in SEGS[1:]:
        s = s + torch.relu(a - t)
    return s.clamp(max=AA) / AA


class Net(nn.Module):
    """768 -> N -> 1 with optional virtual piece-type features.

    The effective first-layer weight is  W[f] = raw[f] + typ[f // 64]:  the
    virtual row is shared by all 64 squares of a piece type, so the sparse
    per-square rows only learn the DEVIATION from the type's average.  This
    is nnue-pytorch's feature factorizer at our scale; it costs nothing at
    run time because `weight()` is what gets exported.
    """

    def __init__(self, N, factor):
        super().__init__()
        self.raw = nn.Parameter(torch.randn(768, N) * 0.05)
        self.typ = nn.Parameter(torch.zeros(12, N)) if factor else None
        self.bias = nn.Parameter(torch.zeros(N) + 0.1)
        self.v = nn.Parameter(torch.randn(N) * (25.0 / N ** 0.5))

    def weight(self):
        if self.typ is None:
            return self.raw
        return self.raw + self.typ.repeat_interleave(64, 0)

    def clamp_weights(self, w):
        # enforce the clip on the EFFECTIVE weight, which is what is exported
        with torch.no_grad():
            if self.typ is None:
                self.raw.clamp_(-w, w)
            else:
                t = self.typ.repeat_interleave(64, 0)
                self.raw.copy_((self.raw + t).clamp(-w, w) - t)

    def forward(self, fi, mi, fo, ps):
        E = self.weight()
        au = act(nn.functional.embedding_bag(fi, E, fo, mode="sum") + self.bias)
        at = act(nn.functional.embedding_bag(mi, E, fo, mode="sum") + self.bias)
        return ps + ((au - at) * self.v).sum(-1).clamp(-CLAMP, CLAMP)


model = Net(N, args.factor)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

n = len(ys)
perm = list(range(n))
random.seed(0)
random.shuffle(perm)
nval = min(200_000, n // 20)
val_ids, train_ids = perm[:nval], perm[nval:]


def batches(ids, bs, shuffle=True):
    ids = ids[:]
    if shuffle:
        random.shuffle(ids)
    for s in range(0, len(ids), bs):
        c = torch.tensor(ids[s:s + bs], dtype=torch.long)
        l = lens[c]
        o = torch.cat([torch.zeros(1, dtype=torch.long), l.cumsum(0)[:-1]])
        # gather this batch's variable-length feature lists
        base = offs[c]
        gidx = torch.repeat_interleave(base - o, l) + torch.arange(int(l.sum()))
        yield feats[gidx], mfeats[gidx], o, pstc[c], ys[c]


def export(path):
    with torch.no_grad():
        E = model.weight().detach()                        # (768, N)
        b = model.bias.detach().tolist()
        v = model.v.detach().tolist()
        W = [{c: [0.0] * 120 for c in PIECES} for _ in range(N)]
        for c in PIECES:
            for s in pnet.SQUARES:
                col = E[feat(c, s)].tolist()
                for k in range(N):
                    W[k][c][s] = col[k]
    shift, worst, sabs = pnet.pick_shift(W, b, v, segs=SEGS)
    d = pnet.build(W, b, v, shift, clampcp=args.clampcp, segs=SEGS)
    d["train"] = vars(args)
    pnet.save(path, d)
    return shift, worst, sabs, d


# anchor rows: what the val split costs with no net at all
with torch.no_grad():
    vy, vp = ys[torch.tensor(val_ids)], pstc[torch.tensor(val_ids)]
    sy = torch.sigmoid(vy / K)
    print("val anchors: zero %.5f  pst %.5f  (val %d positions)"
          % (((torch.sigmoid(vy * 0) - sy) ** 2).mean(),
             ((torch.sigmoid(vp / K) - sy) ** 2).mean(), len(val_ids)), flush=True)

best = float("inf")
for epoch in range(args.epochs):
    model.train()
    tl = tn = 0
    for fi, mi, fo, ps, y in batches(train_ids, args.batch):
        pred = model(fi, mi, fo, ps)
        loss = ((torch.sigmoid(pred / K) - torch.sigmoid(y / K)).abs()
                ** args.losspow).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.clamp_weights(args.wclip)
        tl += loss.item() * len(y)
        tn += len(y)
    sched.step()
    model.eval()
    vl = vn = mae = sat = 0
    with torch.no_grad():
        for fi, mi, fo, ps, y in batches(val_ids, args.batch, shuffle=False):
            pred = model(fi, mi, fo, ps)
            vl += ((torch.sigmoid(pred / K) - torch.sigmoid(y / K)) ** 2).mean().item() * len(y)
            mae += (pred - y).abs().clamp(max=1000).mean().item() * len(y)
            sat += ((pred - ps).abs() >= CLAMP - 0.5).sum().item()
            vn += len(y)
    tag = ""
    if vl / vn < best:
        best = vl / vn
        shift, worst, sabs, d = export(args.out)
        tag = "  -> wrote %s (shift %d sum|v| %.0f excursion %d)" % (
            args.out, shift, sabs, d["excursion"])
    print("epoch %d: train %.5f  val %.5f  val-MAE %.0f cp  clip-saturated %.2f%%%s"
          % (epoch, tl / tn, vl / vn, mae / vn, 100.0 * sat / vn, tag), flush=True)

print("best val %.5f -> %s" % (best, args.out))
