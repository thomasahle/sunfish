"""Feature extractors: board -> sparse 768-base features, plus the pluggable
index transforms (king buckets) applied at batch time.

The cached tensor core is EXTRACTOR-INDEPENDENT: one parse of a corpus
stores the 768-base feature indices, the pst/material bases, the label and
ALL king-bucket codes (kb4/kb8/kb16, own-frame, both sides).  Extractors
are then pure index maps over that cache, so adding a net family never
re-parses data:

  ps768      identity (B = 1)
  kb4/8/16   fi + 768 * own_bucket per perspective (B = 4/8/16)
  bilinear / rff  model-side modules over the ps768 indices (bucket-free
                  by design -- see train_packed.py's ledger notes)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_here), "packed"))  # pnet
# classic sunfish: repo root when running in the repo (train/ is two below),
# the flat training dir on the box (train/ one below) -- both on the path,
# same dance as train_packed.py.
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
import pnet                                                          # noqa: E402
import sunfish as classic                                            # noqa: E402

PIECES = pnet.PIECES
PIDX = {c: i for i, c in enumerate(PIECES)}
PST = classic.pst
piece = classic.piece
KBF = {4: pnet.kbucket, 8: pnet.kbucket8, 16: pnet.kbucket16}


def sq64(i):
    return (i // 10 - 2) * 8 + (i % 10 - 1)


def feat(p, i):
    return PIDX[p] * 64 + sq64(i)


def mirror_map():
    """MIRROR[f]: swap colour, flip the square (torch LongTensor, 768)."""
    import torch
    idx = torch.arange(768)
    return ((idx // 64 + 6) % 12) * 64 + (63 - idx % 64)


def fen_to_board120(fen_board):
    board = [" "] * 20
    for row in fen_board.split("/"):
        line = [" "]
        for ch in row:
            line += ["."] * int(ch) if ch.isdigit() else [ch]
        board += line + ["\n"]
    return "".join(board + [" "] * 20)


def extract(board):
    """One white-to-move 120-board -> (feats, pst_cp, kb4, kb8, kb16).

    kbX packs both sides' OWN-frame buckets as X*own + opp, matching
    train_packed's convention (white own-frame, black own-frame)."""
    feats, ps = [], 0
    for i, c in enumerate(board):
        if c.isalpha():
            feats.append(feat(c, i))
            ps += PST[c][i] if c.isupper() else -PST[c.upper()][119 - i]
    wk, bk = board.index("K"), 119 - board.index("k")
    kbs = tuple(B * KBF[B](wk) + KBF[B](bk) for B in (4, 8, 16))
    return feats, ps, kbs


def phase_vector():
    """Per-feature MATERIAL-PHASE weight (torch float32, 768), unsigned.

    Classic's phase weights, both colours positive, so an embedding_bag sum
    over a position's features is the standard 0..24 phase.  Bucket edges are
    the measured medians of `pool10m` (PORTFOLIO REGISTRATION, 2026-08-19):
    pb2 splits 50.4/49.6 at 11, pb4 splits 26.6/23.9/26.3/23.3 at 4/11/20.
    They are CONSTANTS, not fitted per corpus -- the engine has to compute the
    same bucket the trainer did, so a data-dependent edge would be a
    train/deploy divergence of exactly the kind this ledger keeps finding."""
    import torch
    W = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
    return torch.tensor([W[PIECES[i].upper()] for i in range(12) for _ in range(64)],
                        dtype=torch.float32)


PHASE_EDGES = {2: (11.5,), 4: (4.5, 11.5, 20.5)}
PHASE_W = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}


def phase_of(board):
    """Material phase 0..24 of a 120-board -- the SCALAR twin of
    phase_vector().  Same weights, so the batch path and the probe path
    cannot disagree about which bucket a position is in."""
    return sum(PHASE_W[c.upper()] for c in board if c.isalpha())


def material_vector():
    """Per-feature material value (torch float32, 768): white positive."""
    import torch
    return torch.tensor([piece[PIECES[i].upper()] * (1 if i < 6 else -1)
                         for i in range(12) for _ in range(64)], dtype=torch.float32)


class Extractor:
    """An index transform over the cached ps768 base.  B is the bucket
    count (first-layer table is B*768 rows); buckets(ds) returns the
    per-position row offsets for (own, opp) perspectives.

    Two axes, and they compose as a product space (B = kb * pb):

      kb  OWN-KING buckets.  kb2 is the RANK BAND alone -- own king on its
          back two ranks vs advanced -- derived from the cached kb4 code by
          dropping the file bit, so it costs no re-parse.  It is the
          file-INVARIANT king code, which is the only kind that is coherent
          inside a file-mirrored table; a file-split bucket in a mirrored
          table is self-contradictory, and the data kills it anyway (63.4%
          of positions sit in one kb4 bucket).
      pb  MATERIAL-PHASE buckets, position-global, so both perspectives take
          the same one.  This is the learned form of the entry's hand
          K_MID/K_END taper.
    """

    def __init__(self, kb=1, pb=1):
        self.kb, self.pb = int(kb), int(pb)
        self.B = self.kb * self.pb
        self.name = "ps768" if self.B == 1 else "kb%d.pb%d" % (self.kb, self.pb)
        if self.kb not in (1, 2, 4, 8, 16):
            raise ValueError("kb must be 1/2/4/8/16, got %r" % kb)
        if self.pb not in (1, 2, 4):
            raise ValueError("pb must be 1/2/4, got %r" % pb)

    def codes(self, kb4, kb8, kb16, phase):
        """(own, opp) bucket codes for ONE position.

        THE definition.  `buckets()` is its vectorised form over a Dataset and
        probes.evalcp calls it directly, so there is exactly one place where a
        bucket index is decided.  They used to be two, and the second copy is
        what broke when kb=2 and pb arrived: it crashed on kb2 (loudly) and
        silently ignored pb (quietly), which is the worse of the two.
        """
        if self.B == 1:
            return 0, 0
        if self.kb > 1:
            src, div = {2: (kb4, 4), 4: (kb4, 4),
                        8: (kb8, 8), 16: (kb16, 16)}[self.kb]
            w, b = src // div, src % div
            if self.kb == 2:
                w, b = w // 2, b // 2
        else:
            w = b = 0
        if self.pb > 1:
            p = sum(phase > e for e in PHASE_EDGES[self.pb])
            w, b = w * self.pb + p, b * self.pb + p
        return w, b

    def buckets(self, ds):
        """(own_frame_white, own_frame_black) bucket tensors, or None."""
        if self.B == 1:
            return None
        import torch
        if self.kb > 1:
            # kb2 rides the cached kb4 code: own = kb4 // 4, and the rank
            # band is that code's high bit.
            src, div = ({2: (ds.kb4, 4), 4: (ds.kb4, 4),
                         8: (ds.kb8, 8), 16: (ds.kb16, 16)})[self.kb]
            w, b = src // div, src % div
            if self.kb == 2:
                w, b = w // 2, b // 2
        else:
            w = b = torch.zeros(len(ds.y), dtype=torch.long)
        if self.pb > 1:
            p = ds.phase_bucket(self.pb)
            w, b = w * self.pb + p, b * self.pb + p
        return w, b


def extractor_for(kb, pb=1):
    return Extractor(kb, pb)
