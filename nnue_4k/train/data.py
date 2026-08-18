"""Unified Dataset over every corpus the lane trains on.

Sources (data.kind, sniffed from the extension when "auto"):
  dump        lichess Stockfish-eval .jsonl.zst (the primary corpus).
              Parsed exactly as train_packed.py parses it (same filters,
              same white-to-move normalisation), PLUS a per-position FEN
              hash so splits can be keyed on the position.
  legacy-pkl  train_packed.py --cache pickles (FEATS/OFFS/PSTC/Y/KB[,kb]).
              READ-ONLY: these caches carry no FENs, so only the
              legacy-perm split applies to them -- which is exactly what
              reproducing a historical run needs.
  npz         labeled FEN sets (distill160k.npz-style: fens, y white-POV,
              meta).  Re-featurised here; X is ignored (384-space).
  npz-cache   this module's own cache format (all tensors + fen hashes +
              all kb schemes, one parse serves every extractor).
  binpack     Stockfish binpack via the flag-gated reader adapter
              (train/binpack.py; reader ONLY, after nnue-pytorch).

THE SPLIT IS KEYED ON THE POSITION, NOT ON ITS ROW NUMBER (the house rule,
distill_train.py): fenhash = sha256(str(split_seed) + fen)[:8 hex] as u32,
val iff fenhash %% val_mod == 0.  Byte-identical val membership for every
corpus built from the same positions.  legacy-perm reproduces
train_packed.py's random.seed(seed) permutation split (including --valn
pinning) for runs that must be compared against the historical ledger.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from array import array

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402


class Dataset:
    """Tensor core: feats/offs/lens (ragged sparse features), pstc, matc,
    y, kb4/kb8/kb16, fenhash (None for legacy caches)."""

    def __init__(self, feats, offs, pstc, y, kb4, kb8, kb16, fenhash, meta,
                 outcome=None):
        self.feats = feats                     # long (total_feats,)
        self.offs = offs                       # long (n,)
        self.pstc = pstc                       # float32 (n,) classic pst base
        self.y = y                             # float32 (n,) cp, white-to-move POV
        self.outcome = outcome                 # float32 (n,) game result in the
        #                                        SAME white-to-move POV as y, or
        #                                        None for corpora without results
        self.kb4, self.kb8, self.kb16 = kb4, kb8, kb16   # long (n,) packed own/opp
        self.fenhash = fenhash                 # uint32 np array or None
        self.meta = meta
        self.lens = torch.diff(offs, append=torch.tensor([len(feats)]))
        self.mfeats = features.mirror_map()[feats]
        self._matc = None
        self._phasec = None

    def __len__(self):
        return len(self.y)

    @property
    def matc(self):
        """Material-only base, recomputed from the features (train_packed's
        --base mat), cached."""
        if self._matc is None:
            mv = features.material_vector().unsqueeze(1)
            self._matc = torch.nn.functional.embedding_bag(
                self.feats, mv, self.offs, mode="sum").squeeze(1)
        return self._matc

    @property
    def phasec(self):
        """Material phase 0..24, recomputed from the features, cached.

        Same construction as `matc` (an embedding_bag over a per-feature
        weight vector), so a phase bucket needs no new cached column and no
        CACHE_VERSION bump -- which is the whole reason pb is cheap to try."""
        if self._phasec is None:
            pv = features.phase_vector().unsqueeze(1)
            self._phasec = torch.nn.functional.embedding_bag(
                self.feats, pv, self.offs, mode="sum").squeeze(1)
        return self._phasec

    def phase_bucket(self, pb):
        """Phase -> bucket index, on features.PHASE_EDGES (fixed constants)."""
        p, out = self.phasec, torch.zeros(len(self.y), dtype=torch.long)
        for e in features.PHASE_EDGES[pb]:
            out = out + (p > e).long()
        return out

    def base(self, kind):
        return self.pstc if kind == "pst" else self.matc


def fen_hash(fen, split_seed):
    """The house split key: sha256(seed + fen), first 8 hex digits."""
    return int(hashlib.sha256((str(split_seed) + fen).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------- parsing
def _parse_lines(lines, cpmax, quiet, split_seed):
    """One worker's share of the lichess dump.  train_packed.parse_lines
    with the fen hash added (hashed on the RAW fen field, so the key is
    stable across parser versions)."""
    FEATS, LENS, PSTC, Y = array("i"), array("i"), array("i"), array("i")
    KB4, KB8, KB16, FH = array("h"), array("h"), array("h"), array("L")
    for line in lines:
        d = json.loads(line)
        fen = d["fen"].split()
        ev = max(d["evals"], key=lambda e: e["depth"])
        pv = ev["pvs"][0]
        if "cp" not in pv or abs(pv["cp"]) > cpmax:
            continue
        cp = pv["cp"]
        board = features.fen_to_board120(fen[0])
        if quiet:
            mv = pv["line"].split()[0]
            dst = (8 - int(mv[3])) * 10 + 21 + (ord(mv[2]) - 97)
            if board[dst].isalpha() or len(mv) > 4:
                continue
        if fen[1] == "b":
            board, cp = board[::-1].swapcase(), -cp
        fe, ps, kbs = features.extract(board)
        FEATS.extend(fe)
        LENS.append(len(fe))
        PSTC.append(ps)
        Y.append(cp)
        KB4.append(kbs[0]); KB8.append(kbs[1]); KB16.append(kbs[2])
        FH.append(fen_hash(d["fen"], split_seed))
    return FEATS, LENS, PSTC, Y, KB4, KB8, KB16, FH


def parse_dump(path, limit, cpmax=1000, quiet=0, workers=0, split_seed=20260813):
    proc = subprocess.Popen(["zstd", "-d", "-c", path], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    FEATS, OFFS, PSTC, Y = array("i"), array("q"), array("i"), array("i")
    KB4, KB8, KB16, FH = array("h"), array("h"), array("h"), array("L")
    off = 0

    def chunks():
        while True:
            lines = proc.stdout.readlines(1 << 23)
            if not lines:
                return
            yield lines

    def gather(parts):
        nonlocal off
        for f, l, ps, y, k4, k8, k16, fh in parts:
            FEATS.extend(f)
            for n in l:
                OFFS.append(off)
                off += n
            PSTC.extend(ps); Y.extend(y)
            KB4.extend(k4); KB8.extend(k8); KB16.extend(k16); FH.extend(fh)
            if len(Y) % 1_000_000 < len(y):
                print("  ...%dM positions" % (len(Y) // 1_000_000), flush=True)
            if len(Y) >= limit:
                return True
        return False

    if workers <= 1:
        gather(_parse_lines(ls, cpmax, quiet, split_seed) for ls in chunks())
    else:
        import functools
        import multiprocessing
        work = functools.partial(_parse_lines, cpmax=cpmax, quiet=quiet,
                                 split_seed=split_seed)
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            if gather(pool.imap(work, chunks())):
                pool.terminate()
    proc.kill()
    return FEATS, OFFS, PSTC, Y, KB4, KB8, KB16, FH


def parse_labeled_npz(path, split_seed=20260813, limit=0):
    """distill160k.npz-style: fens + y (white POV).  Positions with black to
    move are flipped to white-to-move and y negated, matching the dump."""
    d = np.load(path, allow_pickle=False)
    fens = [str(f) for f in d["fens"]]
    ys = d["y"].astype(np.int64)
    if limit:
        fens, ys = fens[:limit], ys[:limit]
    FEATS, OFFS, PSTC, Y = array("i"), array("q"), array("i"), array("i")
    KB4, KB8, KB16, FH = array("h"), array("h"), array("h"), array("L")
    off = 0
    for fen, cp in zip(fens, ys):
        parts = fen.split()
        board = features.fen_to_board120(parts[0])
        if parts[1] == "b":
            board, cp = board[::-1].swapcase(), -cp
        fe, ps, kbs = features.extract(board)
        FEATS.extend(fe)
        OFFS.append(off); off += len(fe)
        PSTC.append(ps); Y.append(int(cp))
        KB4.append(kbs[0]); KB8.append(kbs[1]); KB16.append(kbs[2])
        FH.append(fen_hash(fen, split_seed))
    return FEATS, OFFS, PSTC, Y, KB4, KB8, KB16, FH


MATVAL = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900}
GATE_FLOOR = 0.15       # each half must beat this
GATE_SPREAD = 0.15      # and the two halves must agree this closely


def frame_gate(mat, y, wtm, n_max=200000):
    """Refuse a corpus whose labels and base live in different frames.

    SPLIT-HALF form, and it replaces a single corr(base, label) > 0.5
    threshold.  That threshold was calibrated on twin depth-8 labels, which
    are nearly linear in material (measured 0.89).  Stockfish depth-28 labels
    are not: a CORRECTLY framed SF corpus measures 0.31, so the old gate
    would have refused good data -- a false alarm that pushes the next person
    to weaken the gate, which is how a real frame bug gets through.

    A frame error does not shift both halves together; it inverts exactly the
    black-to-move half.  So test the halves separately against the same-frame
    material base and require both to be positive and to AGREE.  Label
    linearity moves both halves together and cannot trip it; a frame error
    splits them and always does.  Measured:

        dump, correct    wtm +0.3266  btm +0.3012  spread 0.025  PASS
        dump, white-POV  wtm +0.3266  btm -0.3012  spread 0.628  FAIL
        self-play, ok    wtm +0.8948  btm +0.8955  spread 0.001  PASS
        self-play, bug   wtm +0.8948  btm -0.8955  spread 1.790  FAIL

    the last row being the exact bug that voided three lambda arms.
    """
    m = np.asarray(mat[:n_max], dtype=np.float64)
    yy = np.asarray(y[:n_max], dtype=np.float64)
    w = np.asarray(wtm[:n_max], dtype=bool)
    rs = []
    for name, sel in (("wtm", w), ("btm", ~w)):
        if int(sel.sum()) < 1000:
            raise SystemExit("FRAME GATE: only %d %s positions, too few to "
                             "verify the frame.  Refusing to train." % (int(sel.sum()), name))
        rs.append(float(np.corrcoef(m[sel], yy[sel])[0, 1]))
    rw, rb = rs
    spread = abs(rw - rb)
    print("frame gate: corr(material, label)  wtm %+.4f  btm %+.4f  spread %.4f"
          % (rw, rb, spread), flush=True)
    if not (rw > GATE_FLOOR and rb > GATE_FLOOR and spread < GATE_SPREAD):
        raise SystemExit(
            "FRAME GATE FAILED: wtm %+.4f, btm %+.4f, spread %.4f (need both "
            "> %.2f and spread < %.2f).  One side-to-move half is in the wrong "
            "frame.  Our twin and the self-play outcomes are ALREADY "
            "side-to-move -- flip the board, never the label -- while the "
            "Lichess dump's evals are WHITE-POV and must be negated for black. "
            "Mixing those two conventions breaks exactly one half.  Refusing "
            "to train." % (rw, rb, spread, GATE_FLOOR, GATE_SPREAD))


def parse_lambda_npz(path, split_seed=20260813, limit=0):
    """The lambda corpus (build_lambda_corpus.py): fens + cp + outcome.

    FRAME, and it is the opposite of parse_labeled_npz's -- read this before
    editing.  That function takes y in WHITE POV and negates it for black, to
    land in the mover frame the features use.  OUR cp comes from the twin,
    which scores SIDE TO MOVE, and our outcome is stored side-to-move too, so
    both channels are ALREADY in the mover frame: the board is flipped, the
    labels are NOT touched.

    Applying the white-POV flip here (which the first version did, by analogy)
    puts the label in the opposite frame from the base on exactly the
    black-to-move half of the corpus.  Measured cost: corr(matc, y) fell from
    0.834 to 0.002 and all three lambda arms early-killed at epoch 2 with the
    control dying identically.  The `corr` gate at the end of load() exists so
    that failure can never be silent again."""
    d = np.load(path, allow_pickle=False)
    fens = [str(f) for f in d["fens"]]
    ys = (d["y"] if "y" in d.files else d["cp"]).astype(np.int64)
    ocs = d["outcome"].astype(np.float32)
    # LOUD TRUNCATION.  DataCfg.limit defaults to 4.1M, which would quietly
    # train on 41% of a 10M corpus while the run's record claimed all of it.
    # A corpus size is a headline number; never let it change in silence.
    if limit and limit < len(fens):
        print("lambda corpus: TRUNCATING %d available positions to limit=%d "
              "(%.1f%%) -- set data.limit: 0 to use the whole corpus"
              % (len(fens), limit, 100.0 * limit / len(fens)), flush=True)
    else:
        print("lambda corpus: using all %d positions (limit=%s)"
              % (len(fens), limit or "0/unset"), flush=True)
    if limit:
        fens, ys, ocs = fens[:limit], ys[:limit], ocs[:limit]
    FEATS, OFFS, PSTC, Y = array("i"), array("q"), array("i"), array("i")
    KB4, KB8, KB16, FH, OUT = array("h"), array("h"), array("h"), array("L"), array("f")
    MAT, WTM = array("i"), array("b")
    off = 0
    for fen, cp, oc in zip(fens, ys, ocs):
        parts = fen.split()
        board = features.fen_to_board120(parts[0])
        oc = float(oc)
        wtm = parts[1] != "b"
        if not wtm:
            # board only: cp and oc are already mover-relative (see docstring)
            board = board[::-1].swapcase()
        # material AFTER the flip, so it is mover-relative like the label
        MAT.append(sum(v * (board.count(c) - board.count(c.lower()))
                       for c, v in MATVAL.items()))
        WTM.append(1 if wtm else 0)
        fe, ps, kbs = features.extract(board)
        FEATS.extend(fe)
        OFFS.append(off)
        off += len(fe)
        PSTC.append(ps)
        Y.append(int(cp))
        OUT.append(oc)
        KB4.append(kbs[0])
        KB8.append(kbs[1])
        KB16.append(kbs[2])
        FH.append(fen_hash(fen, split_seed))
    frame_gate(MAT, Y, WTM)
    return (FEATS, OFFS, PSTC, Y, KB4, KB8, KB16, FH), OUT


# --------------------------------------------------------------- caches
CACHE_VERSION = 1


def save_cache(path, arrays, meta):
    F, O, P, Y, K4, K8, K16, FH = arrays
    np.savez_compressed(path,
                        feats=np.asarray(F, dtype=np.int32),
                        offs=np.asarray(O, dtype=np.int64),
                        pstc=np.asarray(P, dtype=np.int32),
                        y=np.asarray(Y, dtype=np.int32),
                        kb4=np.asarray(K4, dtype=np.int16),
                        kb8=np.asarray(K8, dtype=np.int16),
                        kb16=np.asarray(K16, dtype=np.int16),
                        fenhash=np.asarray(FH, dtype=np.uint32),
                        meta=json.dumps({"cache_version": CACHE_VERSION, **meta}))


def _from_arrays(F, O, P, Y, K4, K8, K16, FH, meta):
    return Dataset(torch.tensor(np.asarray(F, dtype=np.int64)),
                   torch.tensor(np.asarray(O, dtype=np.int64)),
                   torch.tensor(np.asarray(P, dtype=np.float32)),
                   torch.tensor(np.asarray(Y, dtype=np.float32)),
                   torch.tensor(np.asarray(K4, dtype=np.int64)),
                   torch.tensor(np.asarray(K8, dtype=np.int64)),
                   torch.tensor(np.asarray(K16, dtype=np.int64)),
                   None if FH is None else np.asarray(FH, dtype=np.uint32),
                   meta)


def load_cache(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    return _from_arrays(d["feats"], d["offs"], d["pstc"], d["y"],
                        d["kb4"], d["kb8"], d["kb16"], d["fenhash"], meta)


def load_legacy_pkl(path, kb=1):
    """train_packed.py --cache pickles, READ-ONLY.  No FENs, so no fenhash:
    only the legacy-perm split is valid on these -- load() enforces it."""
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    if len(loaded) == 6:
        FEATS, OFFS, PSTC, Y, KB, scheme = loaded
    elif len(loaded) == 5:
        FEATS, OFFS, PSTC, Y, KB = loaded
        scheme = 4
    else:
        (FEATS, OFFS, PSTC, Y), KB, scheme = loaded, array("b", bytes(len(loaded[3]))), 1
    kbt = torch.tensor(np.asarray(KB, dtype=np.int64))
    zeros = torch.zeros(len(Y), dtype=torch.long)
    kbs = {4: zeros, 8: zeros, 16: zeros}
    if scheme in (4, 8, 16):
        kbs[scheme] = kbt
    if kb > 1 and scheme != kb:
        raise ValueError("%s was parsed with kb scheme %s, not %d" % (path, scheme, kb))
    return Dataset(torch.tensor(np.asarray(FEATS, dtype=np.int64)),
                   torch.tensor(np.asarray(OFFS, dtype=np.int64)),
                   torch.tensor(np.asarray(PSTC, dtype=np.float32)),
                   torch.tensor(np.asarray(Y, dtype=np.float32)),
                   kbs[4], kbs[8], kbs[16], None,
                   {"source": path, "legacy": True, "kb_scheme": scheme})


# --------------------------------------------------------------- loading
def sniff(path, kind="auto"):
    if kind != "auto":
        return kind
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return "legacy-pkl"
    if path.endswith(".npz"):
        return "npz"
    if path.endswith(".binpack"):
        return "binpack"
    return "dump"


def load(cfg):
    """DataCfg -> Dataset, cache-aware.  The new .npz cache is keyed by
    parse parameters recorded in its meta; a mismatch is an error, never a
    silent re-use (never hide errors)."""
    kind = sniff(cfg.source, cfg.kind)
    if kind == "legacy-pkl":
        if cfg.split != "legacy-perm":
            raise ValueError("legacy .pkl caches carry no FENs: split must be "
                             "'legacy-perm', got %r" % cfg.split)
        return load_legacy_pkl(cfg.source)
    if cfg.cache and os.path.exists(cfg.cache):
        ds = load_cache(cfg.cache)
        want = dict(source=os.path.basename(cfg.source), limit=cfg.limit,
                    cpmax=cfg.cpmax, quiet=cfg.quiet, split_seed=cfg.split_seed)
        got = {k: ds.meta.get(k) for k in want}
        if got != want:
            raise ValueError("cache %s was built with %s, config wants %s -- "
                             "use a different cache file" % (cfg.cache, got, want))
        print("loaded %d cached positions from %s" % (len(ds), cfg.cache), flush=True)
        return ds
    meta = dict(source=os.path.basename(cfg.source), limit=cfg.limit,
                cpmax=cfg.cpmax, quiet=cfg.quiet, split_seed=cfg.split_seed,
                kind=kind)
    if kind == "dump":
        arrays = parse_dump(cfg.source, cfg.limit, cfg.cpmax, cfg.quiet,
                            cfg.workers, cfg.split_seed)
    elif kind == "lambda-npz":
        # the lambda corpus carries a second label channel (game outcome);
        # everything else in the pipeline is identical to an npz set
        arrays, outc = parse_lambda_npz(cfg.source, cfg.split_seed, cfg.limit)
    elif kind == "npz":
        arrays = parse_labeled_npz(cfg.source, cfg.split_seed, cfg.limit)
    elif kind == "binpack":
        import binpack
        arrays = binpack.parse(cfg.source, cfg.limit, cfg.cpmax, cfg.split_seed)
    else:
        raise ValueError("unknown data kind %r" % kind)
    if cfg.cache:
        if kind == "lambda-npz":
            raise ValueError("the lambda corpus carries a second label channel "
                             "(outcome) that save_cache's format does not hold; "
                             "caching it would silently drop the channel and turn "
                             "every lambda<1 arm into the lam=1 control -- run it "
                             "uncached")
        save_cache(cfg.cache, arrays, meta)
        print("cached -> %s" % cfg.cache, flush=True)
    ds = _from_arrays(*arrays, meta)
    if kind == "lambda-npz":
        ds.outcome = torch.tensor(outc, dtype=torch.float32)
        # The FRAME GATE itself now runs inside parse_lambda_npz, in split-half
        # form, because only that scope still knows each position's side to
        # move.  It is strictly stronger than the single corr(base, label) >
        # 0.5 threshold that used to live here: it catches the same bug (see
        # frame_gate's docstring) without refusing correctly-framed corpora
        # whose labels are simply less linear in material than the twin's.
        # A lambda corpus is never cached, so that gate cannot be bypassed.
        n = min(20000, len(ds.y))
        r = float(np.corrcoef(ds.matc[:n].numpy(), ds.y[:n].numpy())[0, 1])
        print("lambda corpus: %d positions, outcome channel present, "
              "corr(base, label) = %.3f (pooled; the gate is per-half)"
              % (len(ds.outcome), r), flush=True)
    return ds


# --------------------------------------------------------------- splits
def split_fenkey(ds, val_mod):
    """val iff sha256(seed+fen) %% val_mod == 0: position-keyed, stable
    across corpus growth, re-slicing and teacher changes (the house rule)."""
    if ds.fenhash is None:
        raise ValueError("dataset has no fen hashes (legacy cache?)")
    va = np.nonzero(ds.fenhash % val_mod == 0)[0]
    tr = np.nonzero(ds.fenhash % val_mod != 0)[0]
    return tr.tolist(), va.tolist()


def split_legacy(n, rng, valn=0):
    """train_packed.py's split, bit for bit: random.seed(opt.seed) has just
    initialised `rng`, the permutation is the FIRST shuffle drawn from it.
    valn > 0 pins the val set to the one a run over the first valn
    positions would draw (--valn precedent: val stays byte-identical when
    the data scales)."""
    if valn and valn < n:
        perm = list(range(valn))
        rng.shuffle(perm)
        nval = min(200_000, valn // 20)
        return perm[nval:] + list(range(valn, n)), perm[:nval]
    perm = list(range(n))
    rng.shuffle(perm)
    nval = min(200_000, n // 20)
    return perm[nval:], perm[:nval]


def make_split(ds, cfg, rng):
    if cfg.split == "fenkey":
        return split_fenkey(ds, cfg.val_mod)
    if cfg.split == "legacy-perm":
        return split_legacy(len(ds), rng, cfg.valn)
    raise ValueError("unknown split %r" % cfg.split)


def val_sha(ds, val_ids):
    """sha256 over the val rows' identity -- printed and pinned in
    PROVENANCE so 'same val set' is checkable, not asserted."""
    h = hashlib.sha256()
    if ds.fenhash is not None:
        h.update(np.sort(ds.fenhash[np.asarray(val_ids)]).tobytes())
    else:
        h.update(np.asarray(sorted(val_ids), dtype=np.int64).tobytes())
        h.update(np.asarray(ds.y.numpy()[sorted(val_ids)], dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------- batches
def batches(ds, ids, bs, extractor, rng=None):
    """Deterministic batch iterator (a port of train_packed.batches).
    `rng` shuffles in place when given; extractor applies bucket offsets."""
    ids = ids[:]
    if rng is not None:
        rng.shuffle(ids)
    buckets = extractor.buckets(ds)
    for s in range(0, len(ids), bs):
        c = torch.tensor(ids[s:s + bs], dtype=torch.long)
        l = ds.lens[c]
        o = torch.cat([torch.zeros(1, dtype=torch.long), l.cumsum(0)[:-1]])
        base = ds.offs[c]
        gidx = torch.repeat_interleave(base - o, l) + torch.arange(int(l.sum()))
        fi, mi = ds.feats[gidx], ds.mfeats[gidx]
        if buckets is not None:
            kbw, kbb = buckets
            fi = fi + 768 * torch.repeat_interleave(kbw[c], l)
            mi = mi + 768 * torch.repeat_interleave(kbb[c], l)
        yield fi, mi, o, c
