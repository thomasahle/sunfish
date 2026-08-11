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
p.add_argument("--kb", type=int, default=1,
               help="own-king buckets per perspective (HalfKP-lite).  1 = "
                    "plain 768 features and packed export; >1 trains "
                    "bucket-conditioned first-layer rows and exports FLOAT "
                    "weights only (packed export for kb nets is a separate, "
                    "engine-side step -- do not build it before the val loss "
                    "has earned it).  4 = pnet.kbucket, 8 = pnet.kbucket8")
p.add_argument("--nb", type=int, default=0,
               help="dedicated bilinear lanes per perspective (0 = off).  "
                    "Each lane is a crelu unit allocated to group "
                    "lane%%m; the packed engine folds the lane product "
                    "modulo 2^(16m)-1, which reads out the circular "
                    "convolution of the m grouped lane sums (ledger "
                    "8c4eda5).  The antisymmetric read-out is "
                    "h = conv(A,A)-conv(B,B); the cross features "
                    "f = conv(A,B) are swap-invariant and only enter "
                    "through the (even-in-f) tail.  Exports FLOAT only")
p.add_argument("--bm", type=int, default=4,
               help="bilinear groups m; m=4 folds mod 2^64-1, the overflow-"
                    "safe cheap choice")
p.add_argument("--baff", type=int, default=0,
               help="affine bilinear: h += conv(A-B, w) with a trained "
                    "per-group offset w shared across perspectives (the "
                    "sharing keeps the read-out structurally odd).  In "
                    "single-block mode only w1+w2 is identifiable, so one "
                    "vector is trained; with --nb2 both offsets are")
p.add_argument("--nb2", type=int, default=0,
               help="two-block bilinear: conv(A1+w1, A2+w2) with two "
                    "DIFFERENT lane blocks per perspective instead of "
                    "squaring one -- non-symmetric quadratic forms.  "
                    "Doubles the bilinear lane count (2*nb units)")
p.add_argument("--tailw", type=int, default=0,
               help="width of the odd-symmetrized narrow tail over "
                    "[d, h, f] (0 = plain linear read-out d + u.h).  "
                    "Never a second wide layer")
p.add_argument("--base", choices=("pst", "mat"), default="pst",
               help="the fixed base under the net residual.  pst = classic "
                    "piece-square tables (positional prior baked in); mat = "
                    "MATERIAL only (piece values incl. the 60000cp kings) -- "
                    "the net owns all positional knowledge, including what "
                    "the hand-coded K_MID/K_END phase swap encoded.  The "
                    "material base is recomputed from the cached features, "
                    "so every existing cache works unchanged.  Val stays "
                    "comparable across bases: the prediction target is the "
                    "same total eval")
p.add_argument("--satpen", type=float, default=0.0,
               help="saturation penalty weight: training loss adds "
                    "satpen * mean(relu(|pre-clip residual| - satthresh)/100)^2. "
                    "The clamp passes zero gradient outside the band, so "
                    "without this, +/-600 saturation is FREE capacity -- the "
                    "kbbil collapse (ledger b45e48d) showed play pays for it "
                    "(10x pegged evals, +27%% tree inflation).  Val stays the "
                    "plain metric for comparability")
p.add_argument("--satthresh", type=float, default=480.0,
               help="where the saturation penalty starts, in cp")
p.add_argument("--phasecap", type=float, default=0.0,
               help="project the phase scales into [1/cap, cap] after every "
                    "step (0 = off).  The learned s~7 amplified everything "
                    "composed before it -- quantization noise, tail output -- "
                    "by 7x at build time; capping forces v to carry the scale")
p.add_argument("--phase", type=int, default=0,
               help="material-phase output buckets: the net residual is "
                    "scaled by a learned per-bucket scalar s_b, bucket = "
                    "(piece_count-1)*phase//32.  Piece count is swap-"
                    "invariant so antisymmetry is untouched; engine cost "
                    "is one small multiply in nn_cp.  0 = off")
p.add_argument("--factor", type=int, default=1,
               help="train with virtual per-piece-type features (folded into "
                    "the real table at export; helps generalisation, free at "
                    "run time)")
p.add_argument("--losspow", type=float, default=2.0,
               help="loss exponent p in |sig(pred)-sig(y)|^p; nnue-pytorch "
                    "found ~2.6 better than plain MSE")
p.add_argument("--valn", type=int, default=0,
               help="draw the val split from a permutation of the FIRST N0 "
                    "positions (parse order is deterministic, so with the "
                    "same seed the val set is IDENTICAL to a run whose "
                    "dataset was those N0 positions -- val numbers stay "
                    "directly comparable when scaling the data).  Positions "
                    "beyond N0 all become training data.  0 = permute all")
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
# king-bucket scheme; the multiplier packs (own, opp) into one byte
KBF = pnet.kbucket8 if args.kb == 8 else pnet.kbucket
KBMUL = args.kb if args.kb > 1 else 4

# classic sunfish's piece-square tables, verbatim.  In the repo, classic
# lives at the root (nnue_4k/packed/ is two below); on the bench box the
# training dir is flat (packed/ one below) -- put both parents on the path.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
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
    """One worker's share: JSON lines -> (feats, lens, pstc, y, kb) arrays.

    kb packs both king buckets as 4*kb(white) + kb(black), each side's
    bucket taken from its OWN frame (the board here is already normalised
    to white-to-move)."""
    FEATS, LENS, PSTC, Y = array("i"), array("i"), array("i"), array("i")
    KB = array("b")
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
        KB.append(KBMUL * KBF(board.index("K"))
                  + KBF(119 - board.index("k")))
    return FEATS, LENS, PSTC, Y, KB


def parse(path, limit, workers=0):
    proc = subprocess.Popen(["zstd", "-d", "-c", path], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    # OFFS is 64-bit: cumulative feature offsets pass 2^31 just beyond 100M
    # positions (~21 features each) -- array("i") overflowed there
    FEATS, OFFS, PSTC, Y = array("i"), array("q"), array("i"), array("i")
    KB = array("b")
    off = 0

    def chunks():
        while True:
            lines = proc.stdout.readlines(1 << 23)      # ~8 MB of lines
            if not lines:
                return
            yield lines

    def gather(parts):
        nonlocal off
        for f, l, ps, y, kb in parts:
            FEATS.extend(f)
            for n in l:
                OFFS.append(off)
                off += n
            PSTC.extend(ps)
            Y.extend(y)
            KB.extend(kb)
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
    return FEATS, OFFS, PSTC, Y, KB


t0 = time.time()
if args.cache and os.path.exists(args.cache):
    with open(args.cache, "rb") as f:
        loaded = pickle.load(f)
    if len(loaded) == 6:             # scheme-tagged cache
        FEATS, OFFS, PSTC, Y, KB, scheme = loaded
        if args.kb > 1 and scheme != args.kb:
            raise ValueError("%s was parsed with kb scheme %d, not %d; use a "
                             "different --cache file" % (args.cache, scheme, args.kb))
    elif len(loaded) == 5:           # kb4-era cache, untagged
        FEATS, OFFS, PSTC, Y, KB = loaded
        if args.kb not in (1, 4):
            raise ValueError("%s is a kb4-era cache; use a different --cache "
                             "file to train with --kb %d" % (args.cache, args.kb))
    elif args.kb == 1:               # pre-king-bucket cache: buckets unused
        (FEATS, OFFS, PSTC, Y), KB = loaded, array("b", bytes(len(loaded[3])))
    else:
        raise ValueError("%s is a pre-king-bucket cache; re-parse the dump "
                         "(use a fresh --cache file) to train with --kb > 1" % args.cache)
    print("loaded %d cached positions in %.0fs" % (len(Y), time.time() - t0), flush=True)
else:
    print("parsing data...", flush=True)
    FEATS, OFFS, PSTC, Y, KB = parse(args.data, args.limit, args.workers)
    print("%d positions in %.0fs" % (len(Y), time.time() - t0), flush=True)
    if args.cache:
        with open(args.cache, "wb") as f:
            pickle.dump((FEATS, OFFS, PSTC, Y, KB, args.kb), f, protocol=4)

feats = torch.tensor(FEATS, dtype=torch.long)
offs = torch.tensor(OFFS, dtype=torch.long)
pstc = torch.tensor(PSTC, dtype=torch.float32)
ys = torch.tensor(Y, dtype=torch.float32)
if args.base == "mat":
    # material-only base, recomputed from the cached features: feature
    # f is piece PIECES[f//64] (white 0-5 positive, black 6-11 negative)
    _mv = torch.tensor([classic.piece[PIECES[i].upper()] * (1 if i < 6 else -1)
                        for i in range(12) for _ in range(64)],
                       dtype=torch.float32)
    pstc = nn.functional.embedding_bag(
        feats, _mv.unsqueeze(1), offs, mode="sum").squeeze(1)
_kbt = torch.tensor(KB, dtype=torch.long)
kbw = _kbt // KBMUL                                # white's own-frame bucket
kbb = _kbt % KBMUL                                 # black's
lens = torch.diff(offs, append=torch.tensor([len(FEATS)]))
del FEATS, OFFS, PSTC, Y, KB

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


def circ(x, y, m):
    """Circular convolution of (batch, m) group sums -- exactly what the
    packed fold modulo 2^(16m)-1 reads out of a lane product."""
    return torch.stack([sum(x[:, s] * y[:, (g - s) % m] for s in range(m))
                        for g in range(m)], -1)


class Net(nn.Module):
    """768 -> N -> 1 with optional virtual features and king buckets.

    The effective first-layer weight of bucket b is

        W_b[f] = raw[b*768 + f] + shared[f] + typ[f // 64]

    `typ` is shared by all 64 squares of a piece type, `shared` by all
    buckets of a feature: the sparse per-square (and per-bucket) rows only
    learn DEVIATIONS from the coarser averages.  This is nnue-pytorch's
    feature factorizer at our scale; it costs nothing at run time because
    `weight()` -- the folded (B*768, N) table -- is what gets exported.
    The caller offsets feature indices by 768*bucket (each perspective by
    its OWN king's bucket).
    """

    def __init__(self, N, factor, B, nb=0, m=4, tailw=0, phase=0,
                 baff=0, nb2=0):
        super().__init__()
        self.B = B
        self.raw = nn.Parameter(torch.randn(B * 768, N) * 0.05)
        self.shared = nn.Parameter(torch.zeros(768, N)) if B > 1 else None
        self.typ = nn.Parameter(torch.zeros(12, N)) if factor else None
        self.bias = nn.Parameter(torch.zeros(N) + 0.1)
        self.v = nn.Parameter(torch.randn(N) * (25.0 / N ** 0.5))
        # ---- dedicated bilinear lanes (packed: extra 2*nb lanes, group =
        # lane index mod m; one cropped square per perspective block reads
        # out conv(A,A) and conv(B,B) via the mod 2^(16m)-1 fold)
        self.nb, self.m, self.tailw, self.phase = nb, m, tailw, phase
        self.baff, self.nb2 = baff, nb2
        if phase:
            self.s = nn.Parameter(torch.ones(phase))  # per-bucket residual scale
        if nb:
            # bilinear lanes are bucket-FREE even when B > 1: their rows are
            # identical across bucket tables, so king-bucket rebuilds cost
            # nothing extra and the feature stays a pure second-order add-on
            nu = nb * (2 if nb2 else 1)   # two-block mode doubles the lanes
            self.rawb = nn.Parameter(torch.randn(768, nu) * 0.05)
            self.biasb = nn.Parameter(torch.zeros(nu) + 0.1)
            self.gb = nn.Parameter(torch.ones(nu))    # lane gains, |.| at use
            self.u = nn.Parameter(torch.zeros(m))     # read-out of h, starts silent
            self.register_buffer("gidx", torch.arange(nb) % m)
            if baff:
                # affine offsets, SHARED across perspectives (that sharing is
                # what keeps h structurally odd).  Single-block: only w1+w2
                # is identifiable, so one vector; two-block: both.
                self.w1 = nn.Parameter(torch.zeros(m))
                self.w2 = nn.Parameter(torch.zeros(m)) if nb2 else None
            if tailw:
                self.t1 = nn.Linear(1 + 2 * m, tailw)
                self.t2 = nn.Linear(tailw, 1)
                nn.init.zeros_(self.t2.weight)
                nn.init.zeros_(self.t2.bias)

    def virt(self):
        """The bucket-independent part of the effective weight, (768, N)."""
        v = 0
        if self.typ is not None:
            v = self.typ.repeat_interleave(64, 0)
        if self.shared is not None:
            v = v + self.shared
        return v

    def weight(self):
        v = self.virt()
        if isinstance(v, int):
            return self.raw
        N = self.raw.shape[1]
        return (self.raw.view(self.B, 768, N) + v).view(self.B * 768, N)

    def clamp_weights(self, w):
        # enforce the clip on the EFFECTIVE weight, which is what is exported
        with torch.no_grad():
            v = self.virt()
            if isinstance(v, int):
                self.raw.clamp_(-w, w)
            else:
                N = self.raw.shape[1]
                r = self.raw.view(self.B, 768, N)
                r.copy_((r + v).clamp(-w, w) - v)
            if self.nb:
                self.rawb.clamp_(-w, w)

    def forward(self, fi, mi, fo, ps):
        E = self.weight()
        au = act(nn.functional.embedding_bag(fi, E, fo, mode="sum") + self.bias)
        at = act(nn.functional.embedding_bag(mi, E, fo, mode="sum") + self.bias)
        d = ((au - at) * self.v).sum(-1)
        if self.nb:
            # strip king-bucket offsets: bilinear rows are bucket-free
            fib, mib = (fi % 768, mi % 768) if self.B > 1 else (fi, mi)
            bu = act(nn.functional.embedding_bag(fib, self.rawb, fo, mode="sum") + self.biasb)
            bt = act(nn.functional.embedding_bag(mib, self.rawb, fo, mode="sum") + self.biasb)
            g = self.gb.abs()
            su, st = bu * g, bt * g
            nb, m = self.nb, self.m
            if self.nb2:
                # two different lane blocks: non-symmetric quadratic forms
                A1 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, su[:, :nb])
                A2 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, su[:, nb:])
                B1 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, st[:, :nb])
                B2 = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, st[:, nb:])
                if self.baff:
                    w1 = self.w1.unsqueeze(0).expand_as(A1)
                    w2 = self.w2.unsqueeze(0).expand_as(A2)
                    h = circ(A1 + w1, A2 + w2, m) - circ(B1 + w1, B2 + w2, m)
                else:
                    h = circ(A1, A2, m) - circ(B1, B2, m)
                # symmetrized cross features stay swap-invariant
                f_ab = (circ(A1, B2, m) + circ(B1, A2, m)) if self.tailw else None
            else:
                A = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, su)
                B = torch.zeros(su.shape[0], m).index_add_(1, self.gidx, st)
                # h flips sign under perspective swap (A <-> B); f is
                # invariant, so it may only enter the tail's even slot
                h = circ(A, A, m) - circ(B, B, m)
                if self.baff:
                    # expansion identity: only w1+w2 is identifiable, and the
                    # constant conv(w1,w2) cancels in the A/B difference
                    h = h + circ(A - B, self.w1.unsqueeze(0).expand_as(A), m)
                f_ab = circ(A, B, m) if self.tailw else None
            d = d + (h * self.u).sum(-1)
            if self.tailw:
                f = f_ab
                zp = torch.cat([d.unsqueeze(-1) / 300.0, h / 100.0, f / 100.0], -1)
                zn = torch.cat([-d.unsqueeze(-1) / 300.0, -h / 100.0, f / 100.0], -1)
                t = self.t2(torch.tanh(self.t1(zp))) - self.t2(torch.tanh(self.t1(zn)))
                d = d + 150.0 * t.squeeze(-1)   # odd-symmetrized: exact antisymmetry
        if self.phase:
            cnt = torch.diff(fo, append=fi.new_tensor([fi.shape[0]]))
            pb = ((cnt - 1) * self.phase // 32).clamp(0, self.phase - 1)
            d = d * self.s[pb]                  # piece count is swap-invariant
        self.pre = d                            # pre-clip residual (satpen)
        return ps + d.clamp(-CLAMP, CLAMP)


model = Net(N, args.factor, args.kb, args.nb, args.bm, args.tailw, args.phase,
            args.baff, args.nb2)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

n = len(ys)
random.seed(0)
if args.valn and args.valn < n:
    # comparable-val mode: the permutation (and so the val set) is exactly
    # the one a run over the first valn positions would draw; the tail is
    # pure extra training data
    perm = list(range(args.valn))
    random.shuffle(perm)
    nval = min(200_000, args.valn // 20)
    val_ids, train_ids = perm[:nval], perm[nval:] + list(range(args.valn, n))
else:
    perm = list(range(n))
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
        fi, mi = feats[gidx], mfeats[gidx]
        if args.kb > 1:
            # each perspective's rows come from its OWN king's bucket
            fi = fi + 768 * torch.repeat_interleave(kbw[c], l)
            mi = mi + 768 * torch.repeat_interleave(kbb[c], l)
        yield fi, mi, o, pstc[c], ys[c]


def export(path):
    with torch.no_grad():
        E = model.weight().detach()                        # (B*768, N)
        b = model.bias.detach().tolist()
        v = model.v.detach().tolist()
        phs = ({"phase": args.phase, "phase_s": model.s.detach().tolist()}
               if args.phase else {})
        bil = {}
        if args.nb:
            bil = {"nb": args.nb, "m": args.bm, "nb2": args.nb2,
                   "Eb": model.rawb.detach().tolist(),
                   "biasb": model.biasb.detach().tolist(),
                   "gb": model.gb.detach().abs().tolist(),
                   "u": model.u.detach().tolist()}
            if args.baff:
                bil["waff"] = [model.w1.detach().tolist()] + \
                    ([model.w2.detach().tolist()] if args.nb2 else [])
            if args.tailw:
                bil["tail"] = {n_: q.detach().tolist() for n_, q in
                               (("t1w", model.t1.weight), ("t1b", model.t1.bias),
                                ("t2w", model.t2.weight), ("t2b", model.t2.bias))}
        if args.kb > 1 or args.nb or args.phase:
            # float-only export, one rule for every extension: the packed
            # build is engine-side work that the val loss has to earn first
            kind = "float-kb" if args.kb > 1 else (
                "float-bil" if args.nb else "float-phase")
            pnet.save(path, {"kind": kind, "B": args.kb, "N": N,
                             "E": E.tolist(), "bias": b, "v": v,
                             "clampcp": args.clampcp, "segs": SEGS,
                             "base_kind": args.base,
                             "train": vars(args), **bil, **phs})
            return 0, 0.0, sum(abs(x) for x in v), {"excursion": 0}
        W = [{c: [0.0] * 120 for c in PIECES} for _ in range(N)]
        for c in PIECES:
            for s in pnet.SQUARES:
                col = E[feat(c, s)].tolist()
                for k in range(N):
                    W[k][c][s] = col[k]
    shift, worst, sabs = pnet.pick_shift(W, b, v, segs=SEGS)
    d = pnet.build(W, b, v, shift, clampcp=args.clampcp, segs=SEGS)
    d["base_kind"] = args.base
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
        if args.satpen:
            loss = loss + args.satpen * (
                torch.relu(model.pre.abs() - args.satthresh) / 100).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.clamp_weights(args.wclip)
        if args.phasecap and args.phase:
            with torch.no_grad():
                model.s.clamp_(1.0 / args.phasecap, args.phasecap)
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
