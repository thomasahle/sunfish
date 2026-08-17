#!/usr/bin/env python3
"""Export-time verification: the quantized torch model, an independent
integer reference, and the PACKED BIG-INT ENTRY must agree BIT-EXACTLY.

The invariant style is packed/replnet_check.py's (itself from
proto_check.py), extended with the quantization chain.  For a ternary
export the check is a triangle, every leg exact:

  (a) payload decode == trainer quantization: the base-90 string, decoded
      with the entry's own codec, yields exactly the trits/gains/bias
      digits that (E*32).round().clamp(-1,1) + the gain schedule produce.
  (b) integer reference == entry: a plain-python integer evaluation
      (per-lane ints, no packing) equals the entry's packed nn_cp -- SWAR
      relu/cap, modular horizontal sum, shift and clamp included -- on
      every probe position.
  (c) torch mirror == integer reference: the quantized model evaluated in
      float64 (every intermediate an integer < 2^53, so float64 IS integer
      arithmetic) equals the reference.  This is the same exactness
      argument train/packed_layers.py rests on.

Plus the entry invariants on the spliced module: mirror identity, a random
walk with incremental == from-scratch acc/ps/score, exact antisymmetry,
net-fires, king-gone sentinel margin.

usage: verify_export.py BEST.pickle SPLICED_ENTRY.py [nfens] [nplies]
"""
import importlib.util
import os
import pickle
import random
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import features  # noqa: E402


def load_entry(path):
    spec = importlib.util.spec_from_file_location("spliced_entry", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spliced_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_payload(s90, N):
    """The entry codec, standalone: LSB-first shift, N gains, N bias digits,
    768 chars of N trits each."""
    w = 0
    for c in s90:
        d = ord(c) - 35
        w = w * 90 + d - (d > 4) - (d > 56)
    w, shift = divmod(w, 90)
    g = []
    for _ in range(N):
        w, d = divmod(w, 90)
        g.append(d)
    bd = []
    for _ in range(N):
        w, d = divmod(w, 90)
        bd.append(d)
    # Feature radix mirrors export_replnet: one base-90 digit holds a whole
    # feature's N lanes only while 3^N <= 90, i.e. N <= 4.  Above that each
    # trit is its own base-3 field.  These two must never drift apart -- the
    # N=6 export shipped a payload no reader could decode precisely because
    # the encoder grew a case the decoder did not.
    trits = []
    for _ in range(768):
        row = []
        if N <= 4:
            w, d = divmod(w, 90)
            for _ in range(N):
                d, t = divmod(d, 3)
                row.append(t - 1)
        else:
            for _ in range(N):
                w, t = divmod(w, 3)
                row.append(t - 1)
        trits.append(row)
    assert w == 0, "payload had %d leftover digits" % w.bit_length()
    return shift, g, bd, trits


def int_ref_eval(trits, g, bd, shift, clampcp, board):
    """Independent integer evaluation, plain python ints, no packing.
    `board` is mover-oriented (uppercase = us); the entry's pf only selects
    which lane BLOCK encodes us, which an unpacked reference has no need
    of -- from_board(board, pf) evaluates identically for both pf."""
    N = len(g)
    us = [bd[k] - 44 for k in range(N)]
    them = [bd[k] - 44 for k in range(N)]
    for s, p in enumerate(board):
        if p in features.PIECES:
            fu = features.feat(p, s)
            fm = features.feat(p.swapcase(), 119 - s)
            for k in range(N):
                us[k] += g[k] * trits[fu][k]
                them[k] += g[k] * trits[fm][k]
    v = sum(min(max(a, 0), 32 * g[k]) for k, a in enumerate(us)) \
        - sum(min(max(a, 0), 32 * g[k]) for k, a in enumerate(them))
    v = (v >> shift) if v >= 0 else -((-v) >> shift)
    return -clampcp if v < -clampcp else (clampcp if v > clampcp else v)


def torch_mirror_eval(trits_t, g_t, bd_t, shift, clampcp, board):
    """The quantized model in float64: integer arithmetic by exactness
    (every value an integer far below 2^53)."""
    N = g_t.shape[0]
    us = (bd_t - 44).clone()
    them = (bd_t - 44).clone()
    for s, p in enumerate(board):
        if p in features.PIECES:
            us += g_t * trits_t[features.feat(p, s)]
            them += g_t * trits_t[features.feat(p.swapcase(), 119 - s)]
    caps = 32 * g_t
    v = (us.clamp(min=torch.zeros(N, dtype=torch.float64), max=caps).sum()
         - them.clamp(min=torch.zeros(N, dtype=torch.float64), max=caps).sum())
    v = int(v.item())
    v = (v >> shift) if v >= 0 else -((-v) >> shift)
    return max(-clampcp, min(clampcp, v))


def fen_probe_boards(nfens):
    path = os.path.join(os.path.dirname(_here), "packed", "shapecheck_fens.txt")
    boards = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            board = features.fen_to_board120(parts[0])
            if len(parts) > 1 and parts[1] == "b":
                board = board[::-1].swapcase()
            boards.append(board)
            if len(boards) >= nfens:
                break
    return boards


def main():
    netpath, entrypath = sys.argv[1], sys.argv[2]
    nfens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    nplies = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    e = load_entry(entrypath)
    with open(netpath, "rb") as f:
        d = pickle.load(f)
    assert d["kind"] == "replnet-ternary", d["kind"]
    N, clampcp = d["N"], d["clampcp"]

    # (a) payload decode == trainer quantization
    with open(netpath + ".payload") as f:
        s90 = f.read().strip()
    shift, g, bd, trits = decode_payload(s90, N)
    E = torch.tensor(d["E"], dtype=torch.float64)
    want_trits = (E * 32).round().long().clamp(-1, 1).tolist()
    assert trits == want_trits, "payload trits != trainer quantization"
    assert shift == d["shift"] and g == d["g"] and bd == d["bias_digits"], \
        "payload header != trainer export"
    print("verify_export (a): payload decode == trainer quantization "
          "(768x%d trits, shift %d, gains %s)" % (N, shift, g), flush=True)

    # entry invariants (replnet_check style) on the SPLICED module
    for p in e._PIECES:
        for s in range(120):
            assert e.ROWS[1][p][s] == e.ROWS[0][p.swapcase()][119 - s], (p, s)
    army = 9 * e.piece["Q"] + 2 * (e.piece["R"] + e.piece["B"] + e.piece["N"])
    slack = (e.piece["K"] - army) - e.MATE_LOWER
    assert slack > e.CLAMP, (slack, e.CLAMP)

    trits_t = torch.tensor(trits, dtype=torch.float64)
    g_t = torch.tensor(g, dtype=torch.float64)
    bd_t = torch.tensor(bd, dtype=torch.float64)

    def triple_check(board, pf, where):
        ref = int_ref_eval(trits, g, bd, shift, clampcp, board)
        # evaluate via the entry's own accumulator for this exact board/pf
        pos = e.from_board(board, pf=pf)
        got = e.nn_cp(pos.acc, pos.pf)
        assert got == ref, ("entry != int-ref", where, got, ref)
        tm = torch_mirror_eval(trits_t, g_t, bd_t, shift, clampcp, board)
        assert tm == ref, ("torch-mirror != int-ref", where, tm, ref)
        return ref

    # (b)+(c) over FEN probes, both perspectives, antisymmetry recomputed
    fired = 0
    for i, board in enumerate(fen_probe_boards(nfens)):
        r0 = triple_check(board, 0, ("fen", i, 0))
        r1 = triple_check(board, 1, ("fen", i, 1))
        assert r0 == r1, ("perspective flag changed the eval", i, r0, r1)
        rr = triple_check(board[::-1].swapcase(), 0, ("fen-rot", i))
        assert rr == -r0, ("antisymmetry", i, r0, rr)
        fired += r0 != 0

    # random walk: incremental == from-scratch == references
    random.seed(20260814)
    pos = e.from_board(e.initial)
    assert pos.ps == 0 and pos.score == e.nn_cp(pos.acc, pos.pf)
    for step in range(nplies):
        moves = [m for m in pos.gen_moves() if not pos.move(m).k()]
        if not moves:
            break
        pos = pos.move(random.choice(moves))
        fresh = e.from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp, pos.pf)
        assert pos.acc == fresh.acc and pos.ps == fresh.ps and pos.score == fresh.score, step
        r = pos.rotate()
        fr = e.from_board(r.board, r.wc, r.bc, r.ep, r.kp, r.pf)
        assert fr.score == -pos.score, (step, "antisymmetry")
        # the walk position itself, mover-oriented, against the reference
        assert e.nn_cp(pos.acc, pos.pf) == \
            int_ref_eval(trits, g, bd, shift, clampcp, pos.board), (step, "walk int-ref")
        fired += e.nn_cp(pos.acc, pos.pf) != 0
    assert fired, "net never fired -- rows are dead"
    print("verify_export (b,c): entry == int-ref == torch-mirror BIT-EXACT on "
          "%d fens x3 views + %d-ply walk; mirror + antisymmetry + sentinel "
          "margin (%d > %d) PASS" % (nfens, step + 1, slack, e.CLAMP), flush=True)


if __name__ == "__main__":
    main()
