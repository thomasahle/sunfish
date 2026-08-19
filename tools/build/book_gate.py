#!/usr/bin/env python3
"""Refuse a book whose castling rights are impossible for its own board.

Why this exists: rr_confirm11 was VOIDED at 1,212/1,212 games by ONE illegal
move, and the cause was book line 395 --

    r1bq1rk1/ppp1bppp/3p1n2/4p3/4P3/1BNP4/PPP1QPPP/R1B2RK1 w Qq - 0 10

both kings castled SHORT, both QUEENSIDE rights still set.  An engine that
trusts the flags (every engine in this field does) generates the phantom
castle as a two-square king move `g8e8`; fastchess reads a two-square king
move as castling and puts the king on c8; the engine's own board says e8.
Four moves later the engine moves a king from e8, and the zero-tolerance gate
fires -- on a CORRECT engine handed an impossible position.

Verified both ways on the entry: with `Qq` it answers g8e8; with the rights
corrected to `-` it answers f7f8, a normal legal move.

usage: book_gate.py BOOK.epd [--write CLEAN.epd]
"""
import sys


def bogus(fen):
    p = fen.split()
    board = p[0]
    rights = p[2] if len(p) > 2 else "-"
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
    for flag, k, ksq, r, rsq in (("K", "K", "e1", "R", "h1"),
                                 ("Q", "K", "e1", "R", "a1"),
                                 ("k", "k", "e8", "r", "h8"),
                                 ("q", "k", "e8", "r", "a8")):
        if flag in rights and (sq.get(ksq) != k or sq.get(rsq) != r):
            bad.append(flag)
    return bad


def main():
    path = sys.argv[1]
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    bad = [(i, bogus(l), l) for i, l in enumerate(lines)]
    bad = [(i, b, l) for i, b, l in bad if b]
    print("book %s: %d positions, %d with impossible castling rights"
          % (path, len(lines), len(bad)))
    for i, b, l in bad:
        print("  line %-5d bogus=%-4s %s" % (i, "".join(b), l[:66]))
    if "--write" in sys.argv:
        out = sys.argv[sys.argv.index("--write") + 1]
        keep = [l for l in lines if not bogus(l)]
        with open(out, "w") as f:
            f.write("\n".join(keep) + "\n")
        print("wrote %s with %d clean positions (%d dropped)"
              % (out, len(keep), len(lines) - len(keep)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
