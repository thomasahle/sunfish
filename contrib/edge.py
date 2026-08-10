#!/bin/sh
""":"
# Polyglot header: run with pypy3 when available, else python3 (issue #102).
# No -u needed: all UCI output is flushed explicitly (tools/uci.py).
for cmd in pypy3 python3; do
   command -v "$cmd" > /dev/null && exec "$cmd" "$0" "$@"
done
echo "Error: sunfish requires pypy3 or python3" >&2
exit 1
":"""

import time
from itertools import count
from collections import namedtuple

__version__ = "2026"
version = "sunfish " + __version__

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

# Mop-up: once one side is down to a bare king, the midgame king table
# gives the search no progress signal - every shuffle scores alike and
# won KRK/KQK endings drift to the 50-move horizon. Swap in a formulaic
# centralization gradient (value falls with distance from the center):
# because both kings share the table (zero-sum via rotation), the same
# swap simultaneously rewards driving the bare king to the edge and
# marching our own king up - the two halves of classical mop-up.
K_MID = pst["K"]
K_END = (0,) * 20 + sum(
    ((0,) + tuple(
        piece["K"] + 70 - 10 * (abs(2 * rank - 7) + abs(2 * file - 7))
        for file in range(8)) + (0,)
     for rank in range(8)), ()) + (0,) * 20

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
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
# The margin must cover the largest army a kingless side can face: nine
# queens (8 promotions + original) plus 2R+2B+2N with piece-square
# bonuses sums to 11749, which the old 10-queen margin (9290) missed by
# 2459 - a kingless position could evade the king-gone check below.
# 13 queens covers it with slack (formal/Sunfish/EvalBounds.lean proves
# both the leak and the repair).
MATE_LOWER = piece["K"] - 13 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]
# Every static evaluation must stay within [-MATE_UPPER, MATE_UPPER]: the
# transposition table's fresh entries assume it (formal/Sunfish/Tricks.lean,
# `Bounded`). The tables above guarantee it; keep it true if you change them.

# Constants for tuning search
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15
# Max entries kept in each transposition table, roughly 1GB per million.
# Python dicts keep insertion order, so we cheaply evict the oldest entry
# when full (see issue #95).
TABLE_SIZE = 10**6

# minifier-hide start
opt_ranges = dict(
    QS = (0, 300),
    QS_A = (0, 300),
    EVAL_ROUGHNESS = (0, 50),
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
                    if q in " \nPNBRQK":
                        break
                    # Pawn move, double move and capture
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                            #and j != self.ep and abs(j - self.kp) >= 2
                        ):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q in "pnbrqk":
                        break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

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
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        # Castling
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        # Special pawn stuff
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
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
        self.nodes, self.deadline = 0, None

    def bound(self, pos, gamma, depth, root=False):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) """
        self.nodes += 1
        # Enforce the time budget inside the search: iteration boundaries can
        # be seconds apart on slow hardware, this is checked every ~2k nodes.
        if self.deadline is not None and self.nodes % 2048 == 0 \
                and time.time() > self.deadline:
            raise Stop

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
        # Driver probes (the search root, and IID below) are UNSTORED: they
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
        if not root and depth > 0 and pos in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        # If depth == 0 we only try moves with high intrinsic score (captures and
        # promotions). Otherwise we do all moves. This is called quiescent search.
        val_lower = QS - depth * QS_A

        def moves():
            # Look for the strongest move from last time, the hash-move.
            # tp_move stores only real fail-high winners (including the
            # substituted king capture below), so a stored move is always
            # a move gen_moves yields at this position.
            killer = self.tp_move.get(pos)

            # First try not moving at all, i.e. the null move. Zugzwang -
            # passing may be an un-chess-like free tempo - remains a measured,
            # accepted approximation (formal/README.md); the score guard
            # limits exposure. The raw score is yielded: a virtual (None)
            # fail-high is validated in the consumer below before it may
            # cut, which subsumes the mate-band fold and the null-stalemate
            # verifier that used to live here.
            if depth > 2 and not root and abs(pos.score) < 500 and any(
                    c in pos.board for c in "RBNQ"):
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # If there isn't one, try to find one with a more shallow search.
            # This is known as Internal Iterative Deepening (IID). The probe
            # runs as a driver probe (root=True): no null cutoff that would
            # end it without storing a move, no repetition truncation, and
            # no table entry under deviant semantics.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, root=True)
                killer = self.tp_move.get(pos)

            # Only play the move if it would be included at the current val-limit,
            # since otherwise we'd get search instability.
            # We will search it again in the main loop below, but the tp will fix
            # things for us.
            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Then all the other moves
            for val, move in sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True):
                # Quiescent search
                if val < val_lower:
                    break

                # If the new score is less than gamma, the opponent will for sure just
                # stand pat, since ""pos.score + val < gamma === -(pos.score + val) >= 1-gamma""
                # This is known as futility pruning.
                if depth <= 1 and pos.score + val < gamma:
                    # Need special case for MATE, since it would normally be caught
                    # before standing pat. A sub-mate futility yield estimates
                    # the child's stand-pat without searching it, so it is
                    # value evidence only, never legality evidence: it goes
                    # out as a virtual (None) yield - it can never cut (its
                    # score is below gamma by construction), and it must not
                    # set 'live' and mask the terminality correction below.
                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)
                    # We can also break, since we have ordered the moves by value,
                    # so it can't get any better than this.
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        # Run through the moves, shortcutting when possible. The fold also
        # collects a legality certificate: by KingCapturableReportsExact
        # and its converse (the sentinel is produced ONLY at capturable
        # children), a searched real move's score is two-way evidence -
        # exactly -MATE_UPPER proves the move left our king capturable,
        # anything above proves it legal. live records the second case:
        # a searched move is proven legal, so the node cannot be mate or
        # stalemate. Virtual scores (null / stand-pat / sub-mate futility
        # estimates) are value evidence only and never touch it.
        #
        # A virtual (None) fail-high is validated before it is allowed to
        # cut, restoring KingCapturableReportsExact - "if we can capture
        # the opponent king, bound() returns exactly MATE_UPPER":
        # - the stored killer, when present, decides capturability O(1):
        #   it is legal (KillerLegal), and at a capturable node it IS a
        #   king capture (value tops the mate band iff the move takes the
        #   king) - a quiet killer also certifies a legal move, letting
        #   the terminal arm skip its scan;
        # - if a real king capture exists, substitute it: the node
        #   reports MATE_UPPER and tp_move stores the true capture, so a
        #   stand-pat or null cutoff can never mask the sentinel again;
        # - a mate-band claim without a capture is vacuous (if passing wins
        #   the king, capturing it is a real move too): fold identity;
        # - a positive claim at a verified-terminal node (every generated
        #   move loses the king to the legality oracle - a plain QS probe:
        #   at window MATE_UPPER an entry is decisive only with a bound the
        #   invariant reserves for capturable nodes, so the probe stays a
        #   complete decision procedure warm, and the correction's rare
        #   re-scan hits the entries it stores) would outscore the exact
        #   draw the correction stores: fold identity. This arm is
        #   depth-gated like the correction itself: at depth 0 QS
        #   evaluates the fold, stand-pat included, and folding a
        #   terminal stand-pat with no correction to rescue it would make
        #   the node RETURN the reserved -MATE_UPPER sentinel.
        best, live = -MATE_UPPER, False
        for move, score in moves():
            if move is None and score >= gamma:
                king = self.tp_move.get(pos) or pos.king_capture()
                if king and pos.value(king) >= MATE_LOWER:
                    move, score = king, MATE_UPPER
                elif depth and (score >= MATE_LOWER or 0 < score and not king and all(
                        self.bound(pos.move(m), MATE_UPPER, 0) == MATE_UPPER
                        for m in pos.gen_moves())):
                    score = -MATE_UPPER
                # Band-edge verification: a sub-band pass report can
                # straddle the mate band (true pass value >= MATE_LOWER
                # under a loose child bound). One probe at the band
                # boundary is decisive both ways: a fail-low says the
                # pass is really a mate-band claim - vacuous without a
                # capture (fold identity) - a fail-high says the cutoff
                # is sound as reported.
                elif depth > 2 and self.bound(pos.rotate(nullmove=True),
                        1 - MATE_LOWER, depth - 3) < 1 - MATE_LOWER:
                    score = -MATE_UPPER
            best = max(best, score)
            live = live or move is not None and score > -MATE_UPPER
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos] = move
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(iter(self.tp_move))]
                break

        # Mate/stalemate correction, verify-on-suspicion: if we failed low
        # and no searched move was proven legal, the node is SUSPECT -
        # either no legal move exists, or the only legal moves were never
        # searched. 'not live' already certifies every SEARCHED move
        # illegal (two-way evidence above), and a fail-low loop ran to
        # exhaustion, so above depth 1 the searched moves are exactly
        # those the QS threshold admits: only the filtered remainder
        # still needs the legality oracle, and from depth 3 up the
        # threshold admits everything and the scan is probe-free. At
        # depth 1 futility also skips admitted moves, so every move is
        # probed. probe == MATE_UPPER iff the move leaves our king
        # capturable, because a king capture always tops the child's move
        # order and outranks every QS threshold (and warm entries cannot
        # lie: see the oracle note above). If the scan holds, no legal
        # move exists: an exact draw, or mated if the pass leaves our
        # king en prise. Depth 0 is excluded: QS evaluates the fold
        # (stand-pat included) and never claims an exact terminal value -
        # mates are found from depth 1 up. Mated scores as -MATE_LOWER,
        # not -MATE_UPPER: the latter stays a reserved sentinel meaning
        # "the king is literally capturable", and a parent whose child is
        # merely mated must not look king-capturable itself.
        if depth and best < gamma and not live and all(
                depth > 1 and pos.value(m) >= val_lower
                or self.bound(pos.move(m), MATE_UPPER, 0) == MATE_UPPER
                for m in pos.gen_moves()):
            best = -MATE_LOWER if pos.rotate(nullmove=True).king_capture() else 0

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
        pos = history[-1]
        bare = sum(c.isupper() for c in pos.board) == 1 or sum(c.islower() for c in pos.board) == 1
        pst["K"] = K_END if bare else K_MID

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            # The bracket keeps every window computed at this depth inside
            # (-MATE_LOWER, MATE_LOWER], where the terminal corrections in
            # bound() are sound (formal/Sunfish/Stalemate.lean proves both
            # directions) - but gamma CARRIES across depths, and a mate-band
            # score at the previous depth parks it outside the band
            # (formal/Sunfish/Driver.lean, carried_gamma_escapes_band), so
            # clamp it back in before the first probe.
            gamma = min(max(gamma, 1 - MATE_LOWER), MATE_LOWER)
            lower, upper = -MATE_LOWER, MATE_LOWER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma: lower = score
                if score < gamma: upper = score
                yield depth, gamma, score, self.tp_move.get(pos)
                gamma = (lower + upper + 1) // 2


###############################################################################
# UCI User interface
###############################################################################

# parse/render/hist live at module level: tools/uci.py (and the tests)
# reach them as engine-module attributes, and main() uses hist before its
# own body would define it.
def parse(c): return A1 + ord(c[0]) - ord("a") - 10 * (int(c[1]) - 1)
def render(i): return chr((i - A1) % 10 + ord("a")) + str(1 - (i - A1) // 10)

hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]


def main():
    # minifier-hide start
    # Development checkout: use the full-featured UCI interface in
    # tools/ (pondering, Hash option, spec-complete go parsing). An
    # installed or packed sunfish has no tools/ and falls through to
    # the built-in loop below, which is all a GUI needs.
    try:
        import sys, tools.uci
        return tools.uci.run(sys.modules[__name__], hist[-1])
    except ImportError:
        pass
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
            think = min(wtime / 40 + winc, wtime / 2 - 1000)
            think = times.get("movetime", think) / 1000

            start = time.time()
            move_str = None
            for depth, gamma, score, move in searcher.search(hist):
                # The only way we can be sure to have the real move in tp_move,
                # is if we have just failed high.
                if score >= gamma:
                    i, j = move.i, move.j
                    if len(hist) % 2 == 0:
                        i, j = 119 - i, 119 - j
                    move_str = render(i) + render(j) + move.prom.lower()
                    print("info depth", depth, "score cp", score, "pv", move_str)
                if move_str and time.time() - start > think * 0.8:
                    break

            print("bestmove", move_str or '(none)')


if __name__ == "__main__":
    main()
