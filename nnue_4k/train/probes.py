#!/usr/bin/env python3
"""Knowledge-class probes: does a trained net KNOW things, and which things?

The family objective (Thomas, 2026-08-14): one net general enough to learn
endgames, king protection, midgame, pawn structure, mobility — instead of
hand terms that spend bytes on code. These probes turn the loss taxonomy's
classes into per-net diagnostics run at export, so the ledger can watch
each class of knowledge arrive as capacity axes land.

Each probe is a CONTRAST of white-to-move positions with IDENTICAL
material, evaluated with base_cp = 0: the model output is pure net signal
(the material base cancels by construction, and phase probes difference
two such contrasts across phase contexts). Values are cp, mover's view,
positive = the net agrees with the expectation.

NOT gates: val does not gate and neither do probes — play does. A probe
reading is ledger evidence, not a pass/fail.
"""
import torch

import features

# (class, name, boards, combine) where boards are FEN board fields (white
# to move) and combine maps their cp list to one signed value expected > 0.
_P = "4k3/p7/8/4P3/8/8/8/4K3"          # e5 passer, black a7 pawn
_P6 = "4k3/p7/4P3/8/8/8/8/4K3"         # same, pawn advanced to e6
_PM = "r3k2r/p2q4/8/4P3/8/8/3Q4/R3K2R"  # e5 passer, heavy pieces on
_PM6 = "r3k2r/p2q4/4P3/8/8/8/3Q4/R3K2R"
_KG1 = "rq2k3/8/8/8/8/8/5PPP/RQ4K1"    # Kg1 sheltered, queens on
_KE4 = "rq2k3/8/8/8/4K3/8/5PPP/RQ6"    # Ke4 centralized, queens on
_KG1E = "4k3/8/8/8/8/8/5PPP/6K1"       # Kg1, bare ending
_KE4E = "4k3/8/8/8/4K3/8/5PPP/8"       # Ke4, bare ending

PROBES = [
    ("pawn", "passed_vs_opposed",
     ["4k3/p7/8/4P3/8/8/8/4K3", "4k3/4p3/8/4P3/8/8/8/4K3"],
     lambda c: c[0] - c[1]),
    ("pawn", "split_vs_doubled",
     ["4k3/8/8/8/8/3P4/4P3/4K3", "4k3/8/8/8/8/4P3/4P3/4K3"],
     lambda c: c[0] - c[1]),
    ("phase", "pawn_advance_end_vs_mid",
     [_P6, _P, _PM6, _PM],
     lambda c: (c[0] - c[1]) - (c[2] - c[3])),
    ("phase", "king_activity_end_vs_mid",
     [_KE4E, _KG1E, _KE4, _KG1],
     lambda c: (c[0] - c[1]) - (c[2] - c[3])),
    ("king", "centralization_penalty_mid",
     [_KG1, _KE4],
     lambda c: c[0] - c[1]),
    ("king", "shelter_intact_vs_lifted",
     [_KG1, "rq2k3/8/8/8/8/5PPP/8/RQ4K1"],
     lambda c: c[0] - c[1]),
    ("mobility", "knight_center_vs_rim",
     ["4k3/8/8/8/3N4/8/8/4K3", "4k3/8/8/8/8/8/8/N3K3"],
     lambda c: c[0] - c[1]),
    ("mobility", "rook_open_vs_closed_file",
     ["2k5/2p5/8/8/8/2P5/8/4R1K1", "2k5/2p5/8/8/8/2P5/8/2R3K1"],
     lambda c: c[0] - c[1]),
    ("second-order", "bishop_pair_marker",
     ["4k3/8/8/8/8/8/8/1BB1K3", "4k3/8/8/8/8/8/8/1BN1K3"],
     lambda c: c[0] - c[1]),
]


def evalcp(model, cfg, boards):
    """Model cp for a list of FEN board fields, base_cp = 0 (net only)."""
    MIRROR = features.mirror_map()
    ext = features.extractor_for(cfg.model.kb)
    fis, mis, offs = [], [], [0]
    for fb in boards:
        b = features.fen_to_board120(fb)
        feats, _, kbs = features.extract(b)
        f = torch.tensor(feats, dtype=torch.long)
        m = MIRROR[f]
        if ext.B > 1:
            kb = dict(zip((4, 8, 16), kbs))[ext.B]
            f = f + 768 * (kb // ext.B)
            m = m + 768 * (kb % ext.B)
        fis.append(f)
        mis.append(m)
        offs.append(offs[-1] + len(feats))
    fi, mi = torch.cat(fis), torch.cat(mis)
    fo = torch.tensor(offs[:-1], dtype=torch.long)
    was = model.training
    model.eval()
    with torch.no_grad():
        out = model(fi, mi, fo, torch.zeros(len(boards))).tolist()
    if was:
        model.train()
    return out


def run(model, cfg):
    """All probes -> list of (class, name, value_cp)."""
    boards, spans = [], []
    for _, _, bs, _ in PROBES:
        spans.append((len(boards), len(boards) + len(bs)))
        boards.extend(bs)
    cps = evalcp(model, cfg, boards)
    return [(cls, name, comb(cps[a:b]))
            for (cls, name, _, comb), (a, b) in zip(PROBES, spans)]


def report(model, cfg, compact=False):
    rows = run(model, cfg)
    if compact:
        print("probes: " + "  ".join("%s %+d" % (n, round(v))
                                     for _, n, v in rows), flush=True)
    else:
        print("knowledge-class probes (cp, + agrees with expectation; "
              "diagnostics, never gates):", flush=True)
        for cls, name, v in rows:
            print("  %-12s %-28s %+7.1f" % (cls, name, v), flush=True)
    return {n: v for _, n, v in rows}
