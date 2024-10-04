#!/usr/bin/env pypy3

import time, math
from itertools import count
from collections import namedtuple, defaultdict

# If we could rely on the env -S argument, we could just use "pypy3 -u"
# as the shebang to unbuffer stdout. But alas we have to do this instead:
# TODO: I we really want to save bytes, maybe wrap this in minifier-hide,
# and put pypy3 directly in pack.sh instead of using exec on the .py file.
from functools import partial
print = partial(print, flush=True)

version = "sunfish 2"

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
MATE_LOWER = piece["K"] - 10 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

# TODO: It seems my formula B - A * depth doesn't actually work well,
# and BayesOpt prefers to set A >= 250, making it meaningless.
# Maybe I should try three values B_0, B_1, B_2 to get a better idea of the
# right formula
# Constants for tuning search
#QS_B = 219
#QS_A = 500
#EVAL_ROUGHNESS = 13
QS_B = 50
QS_A = 250
EVAL_ROUGHNESS = 17

# Constants to be removed later
USE_BOUND_FOR_CHECK_TEST = 1
IID_LIMIT = 2 # depth > 2
IID_REDUCE = 3 # depth reduction in IID
REPEAT_NULL = 1 # Whether a null move can be responded too by another null move
NULL_LIMIT = 2 # Only null-move if depth > NULL_LIMIT
STALEMATE_LIMIT = 0 # Only null-move if depth > NULL_LIMIT



# There is some issue with combining "bound for check test" with a high "null_limit".

# minifier-hide start
opt_ranges = dict(
    QS_A = (0, 500),
    QS_B = (0, 500),
    EVAL_ROUGHNESS = (0, 50),
    USE_BOUND_FOR_CHECK_TEST = (0, 2),
    IID_LIMIT = (0, 5),
    IID_REDUCE = (1, 5),
    REPEAT_NULL = (0, 1),
    NULL_LIMIT = (0, 5),
    STALEMATE_LIMIT = (0, 5),
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
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper():
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
                    if p in "PNK" or q.islower():
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


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
LO, UP = range(2)
UpperEntry = namedtuple("UE", "depth score", defaults=(0, MATE_UPPER))
LowerEntry = namedtuple("LE", "depth score move", defaults=(0, -MATE_UPPER, None))

class Searcher:
    def __init__(self):
        self.tt_old = (defaultdict(LowerEntry), defaultdict(UpperEntry))
        self.tt_new = (defaultdict(LowerEntry), defaultdict(UpperEntry))
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) """
        self.nodes += 1

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
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        lo_entry, up_entry = self.tt_old[LO][pos, root], self.tt_old[UP][pos, root]
        # score <= upper < gamma
        if up_entry.depth >= depth and up_entry.score < gamma: return up_entry.score
        # gamma <= lower <= score
        if lo_entry.depth >= depth and lo_entry.score >= gamma: return lo_entry.score

        # Kinda like IID?
        if lo_entry.move is None:
            depth -= IID_REDUCE

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            # TODO: Since we nearly always null-move anyway, can't we just save the value
            # and use it to detect checks? It would interfer a bit with the zugswang detection
            # (the RBNQ check), but in Micromax he just does null-move and then checks that
            # afterwards, before he returns the score.
            if depth > NULL_LIMIT and not root and any(c in pos.board for c in "RBNQ"):
                #yield None, null_score
                # TODO: Make NULL_REDUCE argument
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3, root=not REPEAT_NULL)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # Then killer move. We search it twice, but the tp will fix things for us.
            # Note, we don't have to check for legality, since we've already done it
            # before.
            # We no longer have to worry about search instability
            if lo_entry.move:
                # tt_score = -self.bound(pos.move(hi_entry.move), 1 - gamma, depth - 1, root=False)
                # Singular extension: Check that all other moves are much worse
                #if depth > 3 and tt_score >= gamma:
                extend = False
                if False and depth > 3:
                    g = lo_entry.score - QS_B - 3 * depth
                    if all(-self.bound(pos.move(m), 1 - g, (depth - 2), root=False) < g
                           for m in pos.gen_moves() if m != lo_entry.move):
                        extend = True
                        #actual = -self.bound(pos.move(lo_entry.move), 1 - gamma, depth-2, root=False)
                        #print('extending', depth, render_move(lo_entry.move, True), f'gamma={gamma}, lo_entry.score={lo_entry.score}, g={g}, m={m}, actual={actual}')
                        #print(pos.board)
                d1 = depth - 1 + int(extend)
                tt_score = -self.bound(pos.move(lo_entry.move), 1 - gamma, d1, root=False)
                yield lo_entry.move, tt_score

            # Basically just QS if QS_A is high enough
            val_lower = QS_B - QS_A * depth

            # Then all the other moves
            val_moves = sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True)
            for i, (val, move) in enumerate(val_moves):
                if val < val_lower:
                    break

                # Late move reduction
                d1 = depth - 1 - int(i > 5)
                #d1 = depth - 1 - i.bit_length() // 2

                # A very safe form of futility pruning
                if d1 <= 0 and pos.score + val < gamma:
                    yield move, pos.score + val
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, d1, root=False)

        # Run through the moves, shortcutting when possible
        best, best_move = -MATE_UPPER, None
        for move, score in moves():
            if score > best:
                best, best_move = score, move
            if best >= gamma:
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.

        # We will fix this problem another way: We add the requirement to bound, that
        # it always returns MATE_UPPER if the king is capturable. Even if another move
        # was also sufficient to go above gamma. If we see this value we know we are either
        # mate, or stalemate. It then suffices to check whether we're in check.

        # Note that at low depths, this may not actually be true, since maybe we just pruned
        # all the legal moves. So sunfish may report "mate", but then after more search
        # realize it's not a mate after all. That's fair.

        # This is too expensive to test at depth = 0
        #if depth > STALEMATE_LIMIT and best == -MATE_UPPER:
        if depth > 0 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            #in_check = self.bound(flipped, MATE_UPPER, 0, root=True) == MATE_UPPER
            in_check = any(flipped.value(m) >= MATE_LOWER for m in flipped.gen_moves())
            best = 0 if not in_check else -MATE_LOWER

        if lo_entry.move is None:
            depth += IID_REDUCE

        # gamma <= best <= score (new lower bound)
        if best >= gamma and self.tt_new[LO][pos, root].depth <= depth:
            self.tt_new[LO][pos, root] = LowerEntry(depth, best, best_move)
        # score <= best < gamma (new upper bound)
        if best < gamma and self.tt_new[UP][pos, root].depth <= depth:
            self.tt_new[UP][pos, root] = UpperEntry(depth, best)

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0

        # Clear tt since history has changed
        self.tt_new = (defaultdict(LowerEntry), defaultdict(UpperEntry))

        # Store known position so we don't repeat them
        for pos in history:
            self.tt_new[LO][pos, False] = LowerEntry(1000, 0, None)
            self.tt_new[UP][pos, False] = UpperEntry(1000, 0)

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        best_move = None
        for depth in range(1, 1000):
            self.tt_old = (self.tt_new[0].copy(), self.tt_new[1].copy())
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            lower, upper = -MATE_LOWER, MATE_LOWER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(history[-1], gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                gamma = (lower + upper + 1) // 2
                # Could try to yield upperbound / lowerbound, but it seems no
                # UCI interfaces support it very well.
                yield depth, self.tt_new[LO][pos, True].move, score


###############################################################################
# UCI User interface
###############################################################################


def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

def render_move(move, white_pov):
    if move is None:
        return "(none)"
    i, j = move.i, move.j
    if not white_pov:
        i, j = 119 - i, 119 - j
    return render(i) + render(j) + move.prom.lower()

# minifier-hide start
import sys, uci
uci.run(sys.modules[__name__])
sys.exit()
# minifier-hide end


hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
while True:
    args = input().split()
    if args[0] == "uci":
        print(f"id name {version}")
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
        wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
        if len(hist) % 2 == 0:
            wtime, winc = btime, binc
        think = min(wtime / 40 + winc, wtime / 2 - 1)

        start = time.time()
        best_move = None
        for depth, move, score in Searcher().search(hist):
            if move:
                print(f"info depth {depth} score cp {score}")
                best_move = move or best_move
            if best_move and time.time() - start > think * 0.8:
                break

        i, j = best_move.i, best_move.j
        if len(hist) % 2 == 0:
            i, j = 119 - i, 119 - j
        move_str = render(i) + render(j) + move.prom.lower()
        print("bestmove", move_str)

