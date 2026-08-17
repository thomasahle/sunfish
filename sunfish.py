#!/bin/sh

################################################################################
# Sunfish - a minimalist Python chess engine by Thomas Dybdahl Ahle and contributors.
# Copyright (c) Thomas Dybdahl Ahle. Licensed under the GNU General Public License v3.
# Original project: github.com/thomasahle/sunfish
################################################################################

# Python polyglot trick: find the best available python interpreter:
""":"
for cmd in pypy3 python3; do
   command -v "$cmd" > /dev/null && exec "$cmd" "$0" "$@"
done
echo "Error: sunfish requires pypy3 or python3" >&2
exit 1
":"""

import time
from itertools import count
from collections import namedtuple

version = "sunfish 2026"

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################

# With xz compression this whole section takes 652 bytes.
# That's pretty good given we have 64*6 = 384 values.
# Though probably we could do better...
# For one thing, they could easily all fit into int8.
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

# We make a special table for the king in the end game, which encourages
# central positioning. This is sufficient to play KRK and KQK endgames correctly.
# -70 -50 ... -50 -70
# -50 -30 ... -30 -50
# ...     ...     ...
# -50 -30 ... -30 -50
# -70 -50 ... -50 -70
K_MID, K_END = pst["K"], tuple(piece["K"] + 70
   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
#
# The band and what each landmark MEANS - the search reads these as three
# different kinds of thing, and only the first kind is compared for equality:
#
#   +-MATE_UPPER  RESERVED TOKENS, never an evaluation. -MATE_UPPER is the
#                 fold's init and the illegal-move sentinel: "score >
#                 -MATE_UPPER" is the legality test, so a king-capturable
#                 child must report the EXACT +MATE_UPPER (the same token,
#                 seen from the parent) or a mated node looks alive.
#   +-MATE_LOWER  band admission edges: |x| >= MATE_LOWER says "mate", below
#                 says "evaluation". The 13-queen margin puts them out of
#                 reach of any both-kings score and any non-capture move
#                 value, which is what lets both tests be score tests.
#   the gap       mate DISTANCE, strictly between the two: a mated node is
#                 -MATE_LOWER - depth*EVAL_ROUGHNESS, floored at 1-MATE_UPPER
#                 so it can never come back up the tree as the sentinel.
#                 Only ever compared by size, never for equality.
#
# A mate found with `depth` still to spend scores MATE_LOWER + depth*EVAL_ROUGHNESS,
# so a mate delivered near the root outscores one delivered near the horizon:
# among winning lines the search takes the SHORTEST, and the losing side
# drags the mate out as long as it can (issue #11). One ply is worth a whole
# EVAL_ROUGHNESS because that is the width the MTD-bi bracket stops at - at
# one point per ply the driver's last window could not tell two mates apart.
MATE_LOWER = piece["K"] - 13 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

# Constants for tuning search
QS = 40
QS_A = 140
LMR = 75
# Two jobs, deliberately one number: the width the MTD-bi bracket stops at,
# and what one ply of mate distance is worth. Distances must be more than a
# bracket apart or the driver's last window could not order two mates.
EVAL_ROUGHNESS = 15
# Target margin of the deep-null fuel probe (depth >= 6): the pass must
# beat pos.score + NULL_MARGIN for real moves to burn two plies. Its own
# parameter, not tied to EVAL_ROUGHNESS - the two knobs tune different
# things (driver convergence vs reduction aggression).
NULL_MARGIN = -200

# Max entries kept in each transposition table, roughly 1GB per million.
# Python dicts keep insertion order, so we cheaply evict the oldest entry
# when full (see issue #95).
TABLE_SIZE = 10**6

# minifier-hide start
opt_ranges = dict(
    QS = (0, 300),
    QS_A = (0, 300),
    LMR = (-200, 200),
    EVAL_ROUGHNESS = (0, 50),
    NULL_MARGIN = (-400, 800),
    TABLE_SIZE = (10**4, 10**8),
)
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        # NB: `in <literal-str>` is ~30% faster than the equivalent .isupper() /
        # .isspace() / .islower() method calls in CPython; this matters because
        # these checks run millions of times per search.
        for i, p in enumerate(self.board):
            if p not in "PNBRQK":
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q in " \nPNBRQK": break
                    # Pawn move, double move and capture
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if d in (N + W, N + E) and q == "." and j != self.ep and abs(j - self.kp) > 1: break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q in "pnbrqk": break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]: yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]: yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove"""
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # Castling rights, we move the rook or capture the opponent's
        wc = (wc[0] and i != A1, wc[1] and i != H1)
        bc = (bc[0] and j != H8, bc[1] and j != A8)
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:  board = put(board, j, prom)
            if j - i == 2 * N: ep = i + N
            if j == self.ep:   board = put(board, j + S, ".")
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q in "pnbrqk": score += pst[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2: score += pst["K"][119 - j]
        # Castling
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        # Special pawn stuff
        if p == "P":
            if A8 <= j <= H8: score += pst[prom][j] - pst["P"][j]
            if j == self.ep: score += pst["P"][119 - (j + S)]
        return score

    def king_capture(self):
        """The move that takes the opponent king, if any - i.e. the proof
        that this position was reached by an illegal move. Same test as
        gen_moves/value: the target is the king, or within one of the
        king-passant square (kp == 0 is safe: targets are >= A8 > 1).
        Serves double duty: found from a position it is the sentinel
        witness the search substitutes for a virtual cutoff; found from
        the null-rotation it says the side to move is in check."""
        return next((m for m in self.gen_moves()
                     if self.board[m.j] == "k" or abs(m.j - self.kp) < 2), None)


###############################################################################
# Search logic
###############################################################################

# Raised inside the search when the wall-clock deadline passes
class Stop(Exception): pass


# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score, self.tp_move, self.history = {}, {}, set()
        self.nodes, self.deadline, self.soft = 0, 1 << 63, 1 << 63

    def bound(self, pos, gamma, depth, root=False):
        """ Let s* be the score of the sub-tree from pos at this depth, as
            a function of (pos, depth) alone. This includes null moves, QS,
            futility and the reductions, and global parameters like
            self.history that don't change during search. (Things that
            change, like tp_move or gamma, are not allowed to change the
            sub-tree and value of s*.)

            It is assumed 1 - MATE_UPPER < gamma <= MATE_UPPER.

            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound)

            Note, bound() is not guaranteed to be deterministic: stored values
            in self.tp_score may be used to return a bound that is not the best
            possible, but it is guaranteed to be valid according to the rules above.

            On top of the bound, four exact promises. The first two are
            EXACT VALUES and not band membership: +-MATE_UPPER are reserved
            tokens the fold compares for equality, never scores;
            formal/Sunfish/BandContract.lean records what breaks if they
            are weakened to |r| >= MATE_LOWER.
            - our own king already captured: r = -MATE_UPPER.
            - if depth >= 1:
                - if the opponent king capturable: r = MATE_UPPER
                  (note this is stronger than just gamma <= r <= s*.) No
                  searched move can reach MATE_UPPER, since its child is
                  floored at 1 - MATE_UPPER, so an exact MATE_UPPER proves a
                  king capture - which is what makes score > -MATE_UPPER a
                  legality test one ply up. Only a searched real move sets
                  live; null, stand pat and futility estimates never do.
                - if mate/stalemate returns the exact
                  max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
                  / 0. The mate value carries the unspent depth, so s* is
                  still a function of (pos, depth) - see the constants.
            - every move in tp_move is legal. When a searched real move causes
              a fail-high at depth >= 1, it is written as the score witness;
              a virtual cutoff need not have one.
            - a nonterminal root fail-high leaves its real score witness in
              tp_move[root].
            """

        self.nodes += 1
        # Enforce the time budget inside the search: iteration boundaries can
        # be seconds apart on slow hardware, this is checked every ~2k nodes.
        if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # Driver probes (the search root) are UNSTORED: they
        # skip the table in both directions, the repetition-0 and the null
        # option, and store nothing - so every entry in the table describes
        # ONE value function, determined by (pos, depth) alone, and the key
        # needs no flag. (The root's own entry was provably dead weight: the
        # driver picks each gamma strictly inside its bracket, which is the
        # same two numbers the entry held.) At the root 'entry' stays
        # unbound - its only other reader is the store below, also skipped.
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma: return entry.lower
            if entry.upper < gamma: return entry.upper

            # Let's not repeat positions. We don't chat
            # - at the root (a driver probe) since it is in history, but not a draw.
            # - at depth=0, since it would be expensive and break "futility pruning".
            if depth > 0 and pos in self.history: return 0

        # Look for the strongest move from earlier searches of this position.
        # Read it before null-move in case the recursive probe evicts it.
        killer = self.tp_move.get(pos)

        # A fixed-target null probe reduces hot nodes. Its static guard also
        # limits intrinsic LMR to positions where passing is meaningful.
        d = depth
        guard = depth >= 6 and abs(pos.score) < 750 and any(c in pos.board for c in "RBNQ")
        if guard:
            t = pos.score + NULL_MARGIN
            d -= int(-self.bound(pos.rotate(nullmove=True), 1 - t, depth - 7) >= t)

        # Moves whose fixed cap is below gamma share one maximum-cap report.
        base = QS if depth == 0 else -MATE_UPPER
        margin = max(depth - 1, 0) * QS_A
        val_lower = max(base, min(MATE_LOWER, gamma - pos.score - margin)) if depth <= 3 else base

        # Yield resolved virtual reports or unresolved real moves lazily.
        def moves():

            # First try not moving at all, i.e. the null move.
            # See https://chessprogramming.org/Null_Move for details.
            # The idea is that "doing nothing" is a lower bound on the score
            # of the position, but we have to be careful with zugzwang, where
            # passing is better than any move - the piece test guards that
            # (K+P endings). Capping the pass at static evaluation plus one
            # score bucket also keeps its value monotone and below the positive
            # mate band. A sub-window cap needs no child report; otherwise one
            # is enough. No null at root, so we can always return a move. From
            # depth 6 on the pass is never a score candidate (see below).
            if (not root and 2 < depth < 6 and abs(pos.score) < 750
                    and any(c in pos.board for c in "RBNQ")):
                score = cap if (cap := pos.score + EVAL_ROUGHNESS) < gamma else min(cap,
                    -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))
                # A king capture substitutes the exact MATE_UPPER for a virtual fail-high.
                proof = score >= gamma and pos.king_capture()
                yield (proof, MATE_UPPER) if proof else (None, score)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else. (Note depth at root is always > 0.)
            if depth == 0:
                yield None, pos.score

            if killer and (val := pos.value(killer)) >= val_lower:
                yield killer, MATE_UPPER if val >= MATE_LOWER else val

            # Search by decreasing intrinsic value. The cap is monotone in that
            # value, so the sub-window tail is a suffix and the first move of it
            # carries the maximum cap for all of it - one report, emitted last so
            # a prefix cutoff skips it entirely.
            values = sorted(((v, m) for m in pos.gen_moves() if (v := pos.value(m)) >= base), reverse=True)
            n = sum(v >= val_lower for v, m in values)
            yield from ((m, MATE_UPPER if v >= MATE_LOWER else v) for v, m in values[:n])
            if n < len(values): yield None, min(MATE_LOWER - 1, pos.score + values[n][0] + margin)

        # Run through the moves, shortcutting when score >= gamma.
        # live is True if we saw a legal (not null, score > -MATE_UPPER) move
        best, live = -MATE_UPPER, False
        for move, score in moves():
            if move is not None and score < MATE_LOWER:
                val = score
                cap = (MATE_UPPER if depth > 3 else
                    min(MATE_LOWER - 1, pos.score + val + margin))
                move_depth = d - 1 - (not root and guard and val < LMR)
                score = min(cap, -self.bound(pos.move(move), 1 - gamma, move_depth))
            best = max(best, score)
            live |= move is not None and score > -MATE_UPPER
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None and depth:
                    self.tp_move[pos] = move
                    # Never evict the current search root: its killer is the
                    # answer go_loop plays, and once the table churns more
                    # than TABLE_SIZE stores in one deep probe, FIFO would
                    # age it out MID-SEARCH - a later deep fail-low probe
                    # then stores whatever capture sorts first and a timeout
                    # plays it (three -5ish queen/piece giveaways in 145
                    # production games).
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break

        # If no legal real move was witnessed, classify terminality exactly.
        if depth and not live and all(
                pos.move(m).king_capture() for m in pos.gen_moves()):
            # We can't move, but is it a checkmate or stalemate?
            # The mate carries its DISTANCE: the depth we still had left when
            # we found it, one EVAL_ROUGHNESS per ply, so the winner picks the
            # fastest mate and the loser the slowest (issue #11) and the gap
            # survives the driver's final bracket. Nothing but (pos, depth)
            # enters, which is why the table needs no store/probe adjustment
            # and keeps its one value per key: measuring the distance from the
            # ROOT is what would have poisoned it. The floor is 1 - MATE_UPPER
            # and not -MATE_UPPER: one ply up this value is negated, and one
            # more it is back, so -MATE_UPPER here would reach a grandparent
            # as exactly the illegal-move sentinel and "score > -MATE_UPPER"
            # would leave `live` unset for a legal move.
            mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
            best = mate if pos.rotate(nullmove=True).king_capture() else 0

        # Table part 2. Every search decision is gamma-independent, so all
        # bounds target one value function determined by the key and stored
        # entries can never contradict each other (lower > upper). A change
        # that lets gamma (or any key-external state) select between
        # incomparable evaluations of a move breaks this - that is a bug,
        # not a configuration; see formal/README.md.
        if not root:
            self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        if len(self.tp_score) > TABLE_SIZE:
            del self.tp_score[next(iter(self.tp_score))]

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes, self.history = 0, set(history)
        self.tp_score.clear()
        # Table choice is fixed for the whole search (and tp_score is
        # cleared above), so every bound targets one value function.
        pos = self.root = history[-1]

        # When queens come off, the kings can start to move to the center.
        # This is important to win KRK/KQK endings. Both directions every
        # search: table state must never outlive the condition (reused
        # processes start new games with this module state).
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            lower, upper = 1 - MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma: lower = score
                if score < gamma: upper = score
                yield depth, gamma, score, self.tp_move.get(pos)
                gamma = (lower + upper + 1) // 2
            if time.time() > self.soft: return


###############################################################################
# UCI User interface
###############################################################################

# parse/render/hist live at module level: sunfish_ui/uci.py (and the tests)
# reach them as engine-module attributes, and main() uses hist before its
# own body would define it.
def parse(c): return A1 + ord(c[0]) - ord("a") - 10 * (int(c[1]) - 1)
def render(i): return chr((i - A1) % 10 + ord("a")) + str(1 - (i - A1) // 10)

hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]


def main():
    # minifier-hide start
    # The real UCI interface: pondering, Hash option, spec-complete go
    # parsing, and FEN positions. It ships in the wheel, so a checkout and
    # an installed sunfish both reach it, and the import is deliberately
    # unconditional - an engine that cannot find its interface must say so
    # and stop, not play on with the reduced one (issue #156).
    #
    # Only the packed build runs the loop below, and it never reaches this
    # line: pack.sh deletes everything between the minifier-hide markers,
    # taking the import and this return with it.
    import sys, sunfish_ui.uci
    return sunfish_ui.uci.run(sys.modules[__name__], hist[-1])
    # minifier-hide end

    searcher = Searcher()
    while True:
        args = input().split()
        if args[0] == "uci":
            print("id name", version)
            print("uciok")

        elif args[0] == "isready":
            print("readyok")

        elif args[0] == "quit":
            break

        elif args[:2] == ["position", "startpos"]:
            del hist[1:]
            for ply, move in enumerate(args[3:]):
                i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
                if ply % 2 == 1:
                    i, j = 119 - i, 119 - j
                hist.append(hist[-1].move(Move(i, j, prom)))

        elif args[0] == "go":
            # The times may come in any order and combination, e.g. "go wtime 100 btime 100"
            times = dict(zip(args[1::2], map(int, args[2::2])))
            side = "wb"[len(hist) % 2 == 0]
            wtime, winc = times.get(side + "time", 60000), times.get(side + "inc", 0)
            # A whole-game POOL, not a divisor, in milliseconds. Over an
            # M = 40 move horizon the pool is what the clock still holds once
            # all M+2 remaining moves have paid the ~200ms of lag between our
            # bestmove and the clock stopping; `soft` is one move's share of
            # it, and the wall is five shares or half of what this move can
            # safely reach. Two limits, because one number cannot answer both
            # "what is this move worth" and "how long may one iteration run".
            # THE WALL CANNOT GO NEGATIVE, which a wtime/2 - 1s cap can:
            # lichess.org/EAThUL0P was lost that way, ~16 moves at no search
            # on an already-expired deadline. sunfish_ui/uci.py's pool_budget
            # is this same arithmetic in seconds, asserted equal on a grid.
            # THE ONE MEASURED HOLE, disclosed rather than fixed: with no
            # increment the pool is empty below 42*200ms of clock, so `soft`
            # falls to the floor and the quarter-clock clamp is unreachable -
            # a sudden-death endgame one depth shallower than the old
            # divisor's, and -209.91 +/- 60.11 at a 1s clock in the driver's
            # own arm. It never flags. Scoping the pool to P > 0 would fix it
            # and is a design change with its own screen; what ships is what
            # was measured.
            soft = min(max(0, wtime + 39 * winc - 42 * 200) / 40, max(0, wtime - 400) / 4)
            think = max(times.get("movetime", min(5 * soft, (wtime - 400) / 2)) / 1000, .05)
            soft = min(max(soft / 1000, .05), think)

            start = time.time()
            searcher.deadline, searcher.soft = start + think, start + soft
            # A fail high gives the move that achieved it, but only a
            # COMPLETED depth's last fail-high is trustworthy - a stop
            # inside a depth can catch a probe at a nonsense window.
            # Searcher owns the exact MTD bracket, so it reads the soft limit
            # only after that bracket closes; an unsettled move may use the wall.
            best, cand, d0 = None, None, 1
            try:
                for depth, gamma, score, move in searcher.search(hist):
                    if depth > d0:
                        best, d0 = cand or best, depth
                    if score >= gamma:
                        if move is None: print("info depth", depth, "score cp", score); break
                        i, j = move.i, move.j
                        if len(hist) % 2 == 0:
                            i, j = 119 - i, 119 - j
                        cand = render(i) + render(j) + move.prom.lower()
                        print("info depth", depth, "score cp", score, "pv", cand)
            except Stop:
                cand = best or cand

            print("bestmove", cand or best or '(none)')


if __name__ == "__main__":
    main()
