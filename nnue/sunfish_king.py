#!/usr/bin/env pypy

import re, sys, time, pickle
from itertools import count, product
from collections import namedtuple, defaultdict
import numpy as np
from functools import partial
print = partial(print, flush=True)

###############################################################################
# Load and expand the Piece-Square tables.
###############################################################################

ars, scale = pickle.load(open(sys.argv[1], "br"))
# emb(64, c), pieces(c, c, 6), comb(c, c, c, 1)
emb, pieces, comb = [np.frombuffer(ar, dtype=np.int8) / 127 for ar in ars]
#import matplotlib.pyplot as plt
#for b in emb.reshape(64,8).T.reshape(8,8,8):
#    plt.imshow(b)
#    plt.show()
emb, pieces, comb = emb.reshape(8,8,8), pieces.reshape(8,8,6), comb.reshape(8,8,8)

#emb = emb.reshape(64,8)
#test = 0
#wk, bk = 4, 60
#for i, p in enumerate('RNBQKBNR'+'P'*8):
#    p = 'PNBRQK'.find(p)
#    w = np.einsum('dwb,dc,c,w,b->', comb, pieces[:,:,p], emb[i], emb[wk], emb[bk])
#    print(i, p, w)
#    test += w
#print(test)
#emb = emb.reshape(8,8,8)

# Pad to use 12x10 notation
emb = np.pad(emb[::-1], ((2,2),(1,1),(0,0))).reshape(120,8)
pst = np.einsum('sc,dcp,def,we,bf->wbps', emb, pieces, comb, emb, emb, optimize=True)
pst = (360 * scale * pst).round().astype(int)
#pst = scale * pst

# The king is worth a lot, to make mates visible
pst[:,:,5,:] += 10**5
# Mate values will be be around that value
max_val = pst[:,:,:5,:].max()
MATE_LOWER = pst[:,:,5,:].min() - 16 * max_val
MATE_UPPER = pst[:,:,5,:].max() + 16 * max_val
print(f'{MATE_LOWER=}, {MATE_UPPER=}')

# We want to access the piece types using their letter
kpst = [[dict(zip('PNBRQK', pst[wk,bk])) for bk in range(120)] for wk in range(120)]


###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

def calc_score(board, wk, bk):
    # print(board, wk, bk)
    w = sum(kpst[wk][bk][p][k] for k, p in enumerate(board) if p.isupper())
    b = sum(kpst[119-bk][119-wk][p.upper()][119-k] for k, p in enumerate(board) if p.islower())
    return w - b
wk, bk = initial.find('K'), initial.find('k')
#print(f'{wk=}, {bk=}')
#print('first score', calc_score(initial, wk, bk))
#print('rot score', calc_score(initial[::-1].swapcase(), 119-bk, 119-wk))
def flip(board):
    top = '         \n'*2
    mid = ''.join(' '+row+'\n' for row in board.split()[::-1])
    return top + mid + top
#print(calc_score('\n'.join(initial.split('\n')[::-1]).swapcase(), 119-bk, 119-wk))
#print('flip score', calc_score(flip(initial).swapcase(), 119-bk, 119-wk))


# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13


###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")

class Position(namedtuple('Position', 'board score wk bk wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wk -- position of white/our king
    bk -- position of black/opponent king
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
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P':
                        if d in (N, N+N) and q != '.': break
                        if d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                        if d in (N+W, N+E) and q == '.' \
                            and j not in (self.ep, self.kp, self.kp-1, self.kp+1): break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    # Move it
                    yield Move(i, j, '')
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield Move(j+E, j+W, '')
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield Move(j+W, j+E, '')

    def rotate(self, nullmove=False):
        ''' Rotates the board, preserving enpassant, unless nullmove '''
        pos1 = Position(
            self.board[::-1].swapcase(), -self.score,
            119-self.bk, 119-self.wk, self.bc, self.wc,
            119-self.ep if self.ep and not nullmove else 0,
            119-self.kp if self.kp and not nullmove else 0)
        # new_score = calc_score(pos1.board, pos1.wk, pos1.bk)
        # assert new_score == pos1.score, (new_score, pos1.score)
        return pos1

    def move(self, move):
        i, j, prom = move
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board, score, wk, bk, wc, bc, ep, kp = self
        ep, kp = 0, 0
        p, q = board[i], board[j]
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2*N:
                ep = i + N
            if j == self.ep:
                board = put(board, j+S, '.')
        # We rotate the returned position, so it's ready for the next player
        if p != 'K':
            score = self.score + self.value(move)
        else:
            wk = j
            score = calc_score(board, wk, bk)
        # if score != calc_score(board, wk, bk):
        #     print('Move:', move)
        #     print('Old board, score =', self.score)
        #     print(self.board)
        #     print('New board, score', score, calc_score(board,wk,bk))
        #     print(board)
        return Position(board, score, wk, bk, wc, bc, ep, kp).rotate()

    def value(self, move):
        """ Returns the value change to the position after applying the move.
            Except it doesn't recompute when the king moves. """
        # Look up current pst for both players
        wpst, bpst = kpst[self.wk][self.bk], kpst[119-self.bk][119-self.wk]
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move: Remove score from old location, and add for new location
        score = wpst[p][j] - wpst[p][i]
        # If capture, remove opponent piece
        if q.islower():
            score += bpst[q.upper()][119-j]
        # Castling check detection. Only check in elif, since otherwise
        # we might capture the king twice(!)
        elif abs(j-self.kp) < 2:
            # Now we actually know where the opponent king is hiding!
            score += bpst['K'][119-self.bk]
        # Castling
        if p == 'K' and abs(i-j) == 2:
            score += wpst['R'][(i+j)//2] - wpst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += wpst[prom][j] - wpst['P'][j]
            if j == self.ep:
                score += bpst['P'][119-(j+S)]
        return score

    def is_capture(self, move):
        # We can get rid of this again later when we've tuned QS. Maybe.
        return (
            self.board[move.j] != "."
            or abs(move.j - self.kp) < 2
            or self.board[move.i] == "P" and (A8 <= move.j <= H8 or move.j == self.ep)
        )

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.hist_move = defaultdict(int)
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
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
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma:
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # Here extensions may be added
        # Such as 'if in_check: depth += 1'

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            if depth > 2 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.rotate(nullmove=True), 1-gamma, depth-3, root=False)
            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score
            # Then killer move. We search it twice, but the tp will fix things for us.
            # Note, we don't have to check for legality, since we've already done it
            # before. Also note that in QS the killer must be a capture, otherwise we
            # will be non deterministic.
            killer = self.tp_move.get(pos)
            #if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
            if killer and (depth > 0 or pos.is_capture(killer)):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            # Then all the other moves
            def combined(move):
                return self.hist_move[move] + pos.value(move)
            for move in sorted(pos.gen_moves(), key=combined, reverse=True):
                # If depth == 0 we only try moves with high intrinsic score (captures and
                # promotions). Otherwise we do all moves.
                # if depth > 0 or pos.value(move) >= QS_LIMIT:
                if depth > 0 or pos.is_capture(move):
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                self.tp_move[pos] = move
                #self.hist_move[move] += depth**2
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.
        # This doesn't prevent sunfish from making a move that results in stalemate,
        # but only if depth == 1, so that's probably fair enough.
        # (Btw, at depth 1 we can also mate without realizing.)
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.rotate(nullmove=True))
                best = -MATE_UPPER if in_check else 0

        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)

        self.hist_move[self.tp_move.get(pos)] += depth**2
        return best

    def search(self, history):
        """ Iterative deepening MTD-bi search """
        self.nodes = 0
        self.history = set(history)
        self.tp_score.clear()

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays
            # better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(history[-1], gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                gamma = (lower + upper + 1) // 2
                yield depth, self.tp_move.get(history[-1]), score


###############################################################################
# UCI User interface
###############################################################################

def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank

def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

def render_move(move, white_pov):
    if move is None:
        return '0000'
    i, j = move.i, move.j
    if not white_pov:
        i, j = 119 - i, 119 - j
    return render(i) + render(j) + move.prom.lower()

hist = [Position(initial, calc_score(initial, 95, 25), 95, 25, (True,True), (True,True), 0, 0)]
while True:
    args = input().split()
    if args[0] == "uci":
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
        if len(args) == 9:
            _, wtime, _, btime, _, winc, _, binc = args[1:]
            think = int(wtime) / 1000 / 40 + int(winc) / 1000
            think = min(think, int(wtime)/2)
        else:
            think = 2
        start = time.time()
        best_move = None
        searcher = Searcher()
        for depth, move, score in searcher.search(hist):
            print(f"info depth {depth} score cp {score} nodes {searcher.nodes}")
            if move is not None:
                best_move = move
            if think > 0 and time.time() - start > think * 0.8:
                break
        move_str = render_move(best_move, white_pov=len(hist) % 2 == 1)
        print("bestmove", move_str)

