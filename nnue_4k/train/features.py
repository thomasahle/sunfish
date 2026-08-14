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


def material_vector():
    """Per-feature material value (torch float32, 768): white positive."""
    import torch
    return torch.tensor([piece[PIECES[i].upper()] * (1 if i < 6 else -1)
                         for i in range(12) for _ in range(64)], dtype=torch.float32)


class Extractor:
    """An index transform over the cached ps768 base.  B is the bucket
    count (first-layer table is B*768 rows); offsets(ds, ids) returns the
    per-position row offsets for (own, opp) perspectives."""

    def __init__(self, name):
        self.name = name
        self.B = {"ps768": 1, "kb4": 4, "kb8": 8, "kb16": 16}[name]

    def buckets(self, ds):
        """(own_frame_white, own_frame_black) bucket tensors, or None."""
        if self.B == 1:
            return None
        kb = {4: ds.kb4, 8: ds.kb8, 16: ds.kb16}[self.B]
        return kb // self.B, kb % self.B


def extractor_for(kb):
    return Extractor({1: "ps768", 4: "kb4", 8: "kb8", 16: "kb16"}[kb])
