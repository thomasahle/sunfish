#!/usr/bin/env python3
"""Retroactive audit: which harvested tournaments actually DREW a corrupt FEN,
and of those, which games exercised the phantom castling right.

Drawn-FEN intersection, not priors.  A corrupt FEN merely DRAWN is harmless --
the bogus rights bit is only live if an engine castles on it.  So each
affected game gets an individual determination, printed, never assumed.

usage: audit_books.py CORRUPT.epd PGN [PGN...]
"""
import os, re, sys


def bogus(fen):
    p = fen.split()
    board, rights = p[0], (p[2] if len(p) > 2 else "-")
    sq = {}
    for ri, row in enumerate(board.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                sq["abcdefgh"[f] + str(8 - ri)] = ch
                f += 1
    bad = []
    for flag, k, ksq, r, rsq in (("K", "K", "e1", "R", "h1"), ("Q", "K", "e1", "R", "a1"),
                                 ("k", "k", "e8", "r", "h8"), ("q", "k", "e8", "r", "a8")):
        if flag in rights and (sq.get(ksq) != k or sq.get(rsq) != r):
            bad.append(flag)
    return bad


CORRUPT = {}
for l in open(sys.argv[1]):
    l = l.strip()
    if not l:
        continue
    b = bogus(l)
    if b:
        CORRUPT[" ".join(l.split()[:4])] = b     # board stm rights ep

TAG = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
SAN = re.compile(r'(O-O-O|O-O)\b')
MOVETOK = re.compile(r'(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?')

rows = []
for path in sys.argv[2:]:
    if not os.path.exists(path):
        rows.append((path, 0, 0, 0, "MISSING", ""))
        continue
    txt = open(path, errors="replace").read()
    chunks = re.split(r'\n\s*\n(?=\[Event)', txt)
    ngames = drawn = affected = 0
    notes = []
    for g in chunks:
        tags = dict(TAG.findall(g))
        if "Result" not in tags:
            continue
        ngames += 1
        fen = tags.get("FEN")
        if not fen:
            continue
        key = " ".join(fen.split()[:4])
        if key not in CORRUPT:
            continue
        drawn += 1
        flags = CORRUPT[key]
        stm = fen.split()[1]
        body = g.split("]")[-1]
        body = re.sub(r'\{[^}]*\}', ' ', g[g.rfind("]") + 1:])
        toks = [m.group(1) for m in MOVETOK.finditer(body)]
        exercised = []
        for i, t in enumerate(toks):
            if t not in ("O-O", "O-O-O"):
                continue
            side = ("w" if stm == "w" else "b") if i % 2 == 0 else ("b" if stm == "w" else "w")
            flank = "K" if t == "O-O" else "Q"
            fl = flank if side == "w" else flank.lower()
            if fl in flags:
                exercised.append("%s by %s (ply %d)" % (t, side, i + 1))
        verdict = "DESYNC-VOID" if exercised else "drawn, right NOT exercised -> game fine"
        if exercised:
            affected += 1
        notes.append("      R%-5s %-9s vs %-9s  bogus=%-4s  %s%s"
                     % (tags.get("Round", "?"), tags.get("White", "?")[:9],
                        tags.get("Black", "?")[:9], "".join(flags), verdict,
                        ("  [" + "; ".join(exercised) + "]") if exercised else ""))
    v = "CLEAN" if drawn == 0 else ("CLEAN (drawn, none exercised)" if affected == 0
                                    else "AFFECTED -> RECOMPUTE")
    rows.append((path, ngames, drawn, affected, v, "\n".join(notes)))

print("%-52s %7s %7s %8s  %s" % ("arena / pgn", "games", "drawn", "affected", "verdict"))
print("-" * 108)
for path, n, d, a, v, notes in rows:
    print("%-52s %7d %7d %8d  %s" % (path.replace(os.path.expanduser("~/sunfish-bench/"), ""), n, d, a, v))
    if notes:
        print(notes)
