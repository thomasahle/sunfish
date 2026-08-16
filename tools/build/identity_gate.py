"""NODE-IDENTITY gate: does this engine search exactly the tree the last one did?

The gate a speed change needs, and the only one it needs. A change that makes
the artifact faster without touching the search is worth landing on the meter's
warrant alone -- the entry's whole edge over classic is nps and time management
(fixed-node -1.74 +/- 27.93, clean-clock +244.47 +/- 39.23 at 60+1), so under a
clock strictly-faster is strictly-better by construction. That argument is only
as good as the word "identical", which is what this measures.

BESTMOVE IS NOT THE TEST. Two engines can agree on the move at every position
and search entirely different trees to get there. This compares the whole MTD
transcript: every probe the driver yields -- (depth, gamma, score, killer move)
-- and the node count at each, for every position, to a fixed depth. If one
node moves anywhere, one line differs.

The search is driven through Searcher.search() directly: no clock, no driver,
no time manager, so the work is a deterministic function of (position, depth)
and the transcript is reproducible across machines. It is -- the reference for
the 2026-08-16 speed landing hashes the same on an arm64 laptop and on the
x86-64 bench box.

Positions: tools/ctwin/difftest.py's sources and even-stride sampling, with the
per-file takes scaled up so the battery is 60 positions rather than 26.

usage:
    python3 tools/build/identity_gate.py ENGINE.py                 # print
    python3 tools/build/identity_gate.py ENGINE.py --ref REF.txt   # compare
    python3 tools/build/identity_gate.py ENGINE.py > ref.txt       # record
exit status 1 on any divergence.
"""
import argparse
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "..", "..", "tests", "files")

# Same files and the same deterministic even stride as the C-twin difftest.
SOURCES = [
    ("chessathome_openings.fen", 16),
    ("bratko_kopec_test.epd", 10),
    ("win_at_chess_test.epd", 10),
    ("mate1.fen", 4),
    ("mate2.fen", 4),
    ("stalemate1.fen", 4),
    ("nullmove_mates.fen", 3),
    ("perft.epd", 6),
    ("queen.fen", 4),
]
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine_under_gate", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_fens(files_dir):
    out = [("startpos", STARTPOS)]
    for fname, take in SOURCES:
        path = os.path.join(files_dir, fname)
        if not os.path.exists(path):
            raise SystemExit("missing fixture: %s" % path)
        lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        step = max(1, len(lines) // take)
        for k, line in enumerate(lines[::step][:take]):
            fields = line.split(";")[0].strip().split()
            if len(fields) < 4 or "/" not in fields[0]:
                continue
            out.append(("%s:%d" % (fname, k), " ".join(fields[:4])))
    return out


def from_fen(eng, fen):
    """FEN -> Position, the same construction sunfish_ui/uci.py uses."""
    board, color, castling, enpas = fen.split()[:4]
    board = re.sub(r"\d", (lambda m: "." * int(m.group(0))), board)
    b = list(21 * " " + "  ".join(board.split("/")) + 21 * " ")
    b[9::10] = ["\n"] * 12
    pos = eng.from_board("".join(b), ("Q" in castling, "K" in castling),
                         ("k" in castling, "q" in castling),
                         eng.parse(enpas) if enpas != "-" else 0, 0)
    return pos if color == "w" else pos.rotate()


def transcript(eng, pos, depth):
    """Every MTD probe up to `depth`, with the node count at each.

    The first yield of depth+1 is how we learn the depth finished, so it is
    read and then dropped: it belongs to work we are not comparing.
    """
    s = eng.Searcher()
    out, nodes = [], 0
    for d, gamma, score, move in s.search([pos]):
        if d > depth:
            break
        out.append((d, gamma, score, move and tuple(move)))
        nodes = s.nodes
    return out, nodes


def check_derived(eng, fens, depth):
    """DERIVED-FIELD INVARIANT, for engines that carry one.

    The mirrored board `r` is a function of `board`, kept incrementally so the
    rotation is never recomputed. Nothing in the search re-derives it, so if
    the incremental maintenance is ever wrong the engine searches a corrupted
    mirror and the node transcript may still match -- the two are independent
    failures. This checks the contract itself, on every position the search
    actually visits, not on a sample.

    A no-op for engines without the field, so the gate stays universal.
    """
    probe = from_fen(eng, STARTPOS)
    if not hasattr(probe, "r"):
        return None
    bad = [0, 0]
    orig = eng.Searcher.bound

    def spy(self, pos, gamma, d, root=False):
        bad[1] += 1
        if pos.r != pos.board[::-1].swapcase():
            bad[0] += 1
        return orig(self, pos, gamma, d, root)

    eng.Searcher.bound = spy
    for name, fen in fens:
        s = eng.Searcher()
        for dd, g, sc, mv in s.search([from_fen(eng, fen)]):
            if dd > depth:
                break
    eng.Searcher.bound = orig
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--files", default=FILES)
    ap.add_argument("--ref", help="reference transcript to compare against")
    ap.add_argument("--invariant", action="store_true",
                    help="also check the derived mirrored-board field on every "
                         "position visited (slow; run it when the field changes)")
    args = ap.parse_args()

    eng = load_engine(args.engine)

    if args.invariant:
        r = check_derived(eng, load_fens(args.files), min(args.depth, 4))
        if r is None:
            print("derived-field invariant: N/A (engine carries no mirrored board)")
        elif r[0]:
            print("DERIVED-FIELD INVARIANT VIOLATED on %d of %d positions" % tuple(r))
            sys.exit(1)
        else:
            print("derived-field invariant HOLDS on all %d positions visited" % r[1])
        eng = load_engine(args.engine)   # fresh module: the spy mutated the class
    lines = []
    total = 0
    for name, fen in load_fens(args.files):
        tr, nodes = transcript(eng, from_fen(eng, fen), args.depth)
        total += nodes
        lines.append("%s nodes=%d n_probes=%d trace=%s" % (name, nodes, len(tr), tr))
    lines.append("TOTAL nodes=%d positions=%d" % (total, len(lines)))

    if not args.ref:
        print("\n".join(lines))
        return
    ref = open(args.ref).read().splitlines()
    if ref == lines:
        print("NODE-IDENTICAL: %d positions, depth %d, %d nodes, transcripts match"
              % (len(lines) - 1, args.depth, total))
        return
    k = next((i for i in range(min(len(ref), len(lines))) if ref[i] != lines[i]),
             min(len(ref), len(lines)))
    print("DIVERGED at line %d" % k)
    print("  ref: %s" % (ref[k][:200] if k < len(ref) else "<missing>"))
    print("  new: %s" % (lines[k][:200] if k < len(lines) else "<missing>"))
    sys.exit(1)


if __name__ == "__main__":
    main()
