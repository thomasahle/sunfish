"""Build a local Texel-tuning set: positions from our own games, SF labels.

Zero bytes are at stake here -- the tuned tables replace the existing ones
value for value -- so this is free Elo if it works at all. Classic's PSTs
are ~2014 vintage and have never been fitted to anything we measured.

Positions come from our own game pgns (the distribution the engine
actually plays), sampled sparsely to decorrelate, and are labelled with
local Stockfish. Shallow labels are fine for Texel tuning: the fit is
dominated by having many positions, not by per-position depth.

The games directory is an ARGUMENT, not a fixed path inside the repo. The
first version globbed `tools/tune/arena/*.pgn`, which was never committed and
was gitignored along with the .npz, so the 15,328-position set and the games
behind it were purged together with a scratchpad and the whole training track
stalled on it. Source games and labelled sets now live under
`~/repos/sunfish-data/` and are passed in, so the recipe outlives its
workspace. Stockfish is an argument for the same reason: the labeller must be
runnable somewhere other than one laptop.

usage: texel_data.py OUT.npz [NPOS] [DEPTH] [PGNDIR] [STOCKFISH] [THREADS]
"""
import glob
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import chess
import chess.pgn
import numpy as np

OUT = sys.argv[1]
NPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ARENA = sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser("~/repos/sunfish-data/pgn")
SF = sys.argv[5] if len(sys.argv) > 5 else "/opt/homebrew/bin/stockfish"
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 2

pgns = sorted(glob.glob(os.path.join(ARENA, "*.pgn")))
# An empty games directory used to mean "0 positions collected" and a valid
# but empty .npz. Say so and stop instead.
assert pgns, "no *.pgn in %s -- pass the games directory as argv[4]" % ARENA
assert os.path.exists(SF), "no stockfish at %s -- pass its path as argv[5]" % SF
print("games: " + ", ".join("%s (%.1f MB)" % (os.path.basename(p), os.path.getsize(p) / 1e6)
                            for p in pgns), flush=True)

# ---- collect FENs -----------------------------------------------------------
rng = random.Random(20260812)
fens = set()
for path in pgns:
    with open(path) as f:
        while len(fens) < NPOS * 3:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            board = g.board()
            for ply, mv in enumerate(g.mainline_moves()):
                board.push(mv)
                # sparse sampling past the book, skip captures-in-progress
                if ply >= 10 and ply % 7 == 0 and not board.is_check():
                    if len(board.piece_map()) >= 6:
                        fens.add(board.fen())
    if len(fens) >= NPOS * 3:
        break
fens = sorted(fens)
rng.shuffle(fens)
fens = fens[:NPOS]
print("collected %d unique positions" % len(fens), flush=True)

# ---- label with Stockfish ---------------------------------------------------
sf = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      text=True, bufsize=1)
def cmd(c):
    sf.stdin.write(c + "\n"); sf.stdin.flush()
def wait(tok):
    while True:
        ln = sf.stdout.readline()
        if not ln:
            raise RuntimeError("stockfish died waiting for " + tok)
        if ln.startswith(tok):
            return ln.rstrip()
        if ln.startswith("id name "):
            globals()["SFNAME"] = ln.split("id name ", 1)[1].strip()

SFNAME = "unknown"
cmd("uci")
wait("uciok")
assert SFNAME != "unknown", "stockfish never announced an id name"
cmd("setoption name Threads value %d" % THREADS)
# 64 MB, not 256: the hash is CLEARED between positions (below), and a bigger
# table only makes each clear slower. Depth 8 does not fill 64 MB.
cmd("setoption name Hash value 64")
cmd("isready")
wait("readyok")
print("labeller: %s | threads %d | depth %d" % (SFNAME, THREADS, DEPTH), flush=True)

labels = []
keep = []
t0 = time.time()
for n, fen in enumerate(fens):
    # A label must be a property of the POSITION, not of where it sat in the
    # list. Without this the transposition table carries over and the same FEN
    # gets a different number depending on what preceded it -- measured on the
    # box at depth 8: the same FEN scored -14 in one slot and -22 in another,
    # and two other positions moved 83 -> 97 and -90 -> -149. Both modes are
    # run-to-run reproducible, which is exactly why this was invisible. With
    # the clear, the label is a function of (fen, depth, engine version) alone.
    cmd("ucinewgame")
    cmd("isready")
    wait("readyok")
    cmd("position fen " + fen)
    cmd("go depth %d" % DEPTH)
    val = None
    while True:
        ln = sf.stdout.readline()
        if " score cp " in ln:
            val = int(ln.split(" score cp ")[1].split()[0])
        elif " score mate " in ln:
            val = None                      # skip decided positions
        if ln.startswith("bestmove"):
            break
    if val is not None and abs(val) < 1500:
        # SF reports from side-to-move POV; store WHITE POV
        white = fen.split()[1] == "w"
        labels.append(val if white else -val)
        keep.append(fen)
    if n and n % 2000 == 0:
        rate = n / (time.time() - t0)
        print("  labelled %d/%d  (%.0f pos/s, ETA %.1f min)"
              % (n, len(fens), rate, (len(fens) - n) / rate / 60), flush=True)
cmd("quit")
sf.wait()
print("kept %d labelled positions" % len(keep), flush=True)

# ---- features: 6x64 piece-square counts, white minus mirrored black ---------
PIECES = "PNBRQK"
X = np.zeros((len(keep), 384), dtype=np.int8)
for i, fen in enumerate(keep):
    b = chess.Board(fen)
    for sq, pc in b.piece_map().items():
        idx = PIECES.index(pc.symbol().upper())
        if pc.color == chess.WHITE:
            X[i, idx * 64 + sq] += 1
        else:
            X[i, idx * 64 + (sq ^ 56)] -= 1     # mirror rank for black
# The provenance travels INSIDE the file. A labelled set separated from the
# engine version, depth and games that produced it cannot be compared with
# anything or regenerated, and that is how a set becomes unusable long before
# it is deleted. `fens` is kept for the same reason -- any later fit can
# reweight by phase, or relabel at another depth, without touching the PGNs.
meta = {
    "engine": SFNAME,
    "engine_sha256": hashlib.sha256(open(SF, "rb").read()).hexdigest(),
    "depth": DEPTH,
    "threads": THREADS,
    "hash_cleared_per_position": True,
    "pgn_dir": ARENA,
    "pgns": [(os.path.basename(p), os.path.getsize(p),
              hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]) for p in pgns],
    "sampling": "ply>=10, every 7th ply, not in check, >=6 pieces, dedup by FEN",
    "filter": "|cp| < 1500, mate scores dropped",
    "pov": "white",
    "collected": len(fens),
    "kept": len(keep),
    "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "host": platform.node().split(".")[0],
}
np.savez_compressed(OUT, X=X, y=np.array(labels, dtype=np.int16),
                    fens=np.array(keep), meta=json.dumps(meta, indent=1))
print("wrote %s: X %s, y %s" % (OUT, X.shape, len(labels)))
print(json.dumps(meta, indent=1))
