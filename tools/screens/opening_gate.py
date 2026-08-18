"""MEASURED opening-diversity gate for the round-robin.

THE DEFECT THIS CATCHES (rr6_sigk, 2026-08-17, 450 games VOID). fastchess
deals openings by the FLATTENED pair index: opening = (round*NPAIR + pairing)
mod NOPENINGS, with NOPENINGS = -rounds. So each pairing sees exactly

    rounds / gcd(rounds, pairings)

distinct openings. With 6 engines (15 pairings) and -rounds 15 that is ONE:
every pairing replayed a single opening 15 times, and because these engines
are deterministic at a fixed node budget, all 15 repetitions were the SAME
GAME. The tournament reported 450 games and contained 31 distinct ones.

Every gate already in the harness passed -- legality, count, forfeits,
dormancy -- because none of them asks whether the games differ from each
other. A replayed game inflates n without adding information, and the
pentanomial error bars then read 15 copies of one result as 15 independent
pairs, which is how a 30-game sample printed "+/- 0.00".

So this gate does not reason about the schedule, it MEASURES the artifact:
every ordered (White, Black) cell must show as many distinct opening FENs as
it played games, and no game may be a byte-repeat of another in its cell.

usage: opening_gate.py GAMES.pgn ROUNDS
"""
import collections
import re
import sys

TAG = re.compile(r'\[(\w+) "([^"]*)"\]')


def main():
    path = sys.argv[1]
    rounds = int(sys.argv[2])
    txt = open(path, errors="replace").read()
    cells = collections.defaultdict(list)
    movetexts = collections.Counter()
    for blk in txt.split("[Event ")[1:]:
        tags = dict(TAG.findall(blk))
        body = blk.split("\n\n", 1)[1] if "\n\n" in blk else ""
        mv = re.sub(r"\s+", " ", re.sub(r"\{[^}]*\}", "", body)).strip()
        key = (tags.get("White"), tags.get("Black"))
        cells[key].append(tags.get("FEN"))
        movetexts[(key[0], key[1], mv)] += 1

    bad = []
    for (w, b), fens in sorted(cells.items()):
        d = len(set(fens))
        if d != len(fens):
            bad.append("    %-9s(W) vs %-9s(B): %d games but only %d distinct opening(s)"
                       % (w, b, len(fens), d))
    dup = sum(c - 1 for c in movetexts.values() if c > 1)
    ngames = sum(len(v) for v in cells.values())
    distinct = len(movetexts)
    print("opening gate   %d games, %d distinct games, %d duplicate replays"
          % (ngames, distinct, dup))
    if bad or dup:
        print("OPENING-DIVERSITY VOID: openings were REPLAYED, so the reported game")
        print("  count is NOT the sample size -- effective n is the distinct count above.")
        print("  Cause: gcd(rounds, pairings) > 1 aliases the opening to the pairing.")
        print("  Fix: choose -rounds coprime to the pairing count, then re-run.")
        for line in bad[:12]:
            print(line)
        return 9
    print("opening gate   PASS: every pairing saw its full %d distinct openings, "
          "no duplicate games" % rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
