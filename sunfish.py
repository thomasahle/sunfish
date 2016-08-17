#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re, sys, time
from itertools import count
from collections import OrderedDict, namedtuple

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e2

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = 60000 - 8*2700
MATE_UPPER = 60000 + 8*2700

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

###############################################################################
# Move and evaluation tables
###############################################################################

N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

pst = {
    'P': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 198, 198, 198, 198, 198, 198, 198, 198, 0,
        0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
        0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
        0, 178, 198, 208, 218, 218, 208, 198, 178, 0,
        0, 178, 198, 218, 238, 238, 218, 198, 178, 0,
        0, 178, 198, 208, 218, 218, 208, 198, 178, 0,
        0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
        0, 198, 198, 198, 198, 198, 198, 198, 198, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'B': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 797, 824, 817, 808, 808, 817, 824, 797, 0,
        0, 814, 841, 834, 825, 825, 834, 841, 814, 0,
        0, 818, 845, 838, 829, 829, 838, 845, 818, 0,
        0, 824, 851, 844, 835, 835, 844, 851, 824, 0,
        0, 827, 854, 847, 838, 838, 847, 854, 827, 0,
        0, 826, 853, 846, 837, 837, 846, 853, 826, 0,
        0, 817, 844, 837, 828, 828, 837, 844, 817, 0,
        0, 792, 819, 812, 803, 803, 812, 819, 792, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'N': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 627, 762, 786, 798, 798, 786, 762, 627, 0,
        0, 763, 798, 822, 834, 834, 822, 798, 763, 0,
        0, 817, 852, 876, 888, 888, 876, 852, 817, 0,
        0, 797, 832, 856, 868, 868, 856, 832, 797, 0,
        0, 799, 834, 858, 870, 870, 858, 834, 799, 0,
        0, 758, 793, 817, 829, 829, 817, 793, 758, 0,
        0, 739, 774, 798, 810, 810, 798, 774, 739, 0,
        0, 683, 718, 742, 754, 754, 742, 718, 683, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'R': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'Q': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'K': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 60098, 60132, 60073, 60025, 60025, 60073, 60132, 60098, 0,
        0, 60119, 60153, 60094, 60046, 60046, 60094, 60153, 60119, 0,
        0, 60146, 60180, 60121, 60073, 60073, 60121, 60180, 60146, 0,
        0, 60173, 60207, 60148, 60100, 60100, 60148, 60207, 60173, 0,
        0, 60196, 60230, 60171, 60123, 60123, 60171, 60230, 60196, 0,
        0, 60224, 60258, 60199, 60151, 60151, 60199, 60258, 60224, 0,
        0, 60287, 60321, 60262, 60214, 60214, 60262, 60321, 60287, 0,
        0, 60298, 60332, 60273, 60225, 60225, 60273, 60332, 60298, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
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
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' and j not in (self.ep, self.kp): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 119-self.ep, 119-self.kp)

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights
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
                board = put(board, j, 'Q')
            if j - i == 2*N:
                ep = i + N
            if j - i in (N+W, N+E) and q == '.':
                board = put(board, j+S, '.')
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst['K'][j]
        # Castling
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][j+S]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

# The normal OrderedDict doesn't update the position of a key in the list,
# when the value is changed.
class LRUCache:
    '''Store items in the order the keys were last added'''
    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try: self.od.move_to_end(key)
        except KeyError: return default
        return self.od[key]

    def __setitem__(self, key, value):
        try: del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last=False)
        self.od[key] = value

class Searcher:
    def __init__(self):
        self.tp_score = LRUCache(TABLE_SIZE)
        self.tp_move = LRUCache(10**8)
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=False):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1

        # Stop searching if we have lost.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER
        assert pos.score < MATE_LOWER, "We should never get here if we've won"
        #assert abs(pos.score) < MATE_LOWER, "We no longer allow illegal moves"

        # Look in the table if we have already searched this position before.
        # We use the table value if it was done with at least as deep a search
        # as ours, and the gamma value is compatible.
        # TODO: Maybe only use table for larger depth?
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper


        # Null move. Is also used for stalemate checking
        # Note this means we may return a wrong value, and thus break our guarantee
        # At root we still need the null score for stalemate testing
        # But do we really need the extra case? Does it win us a lot not to do the deep null move test at root?
        # Apparantly it makes a huge difference... Maybe the depth-3 version is wrong somehow?
        # The depth 0 is basically just a 'is in check' check.
        if depth > 0:
            in_check = any(pos.board[119-move[1]] == 'K' for move in pos.rotate().gen_moves())
            nullscore = -self.bound(pos.rotate(), 1-gamma, 0)
            # The mate finder should find checks to kp, which they other version doesn't!
            #mate_finder = -self.bound(pos.rotate(), 1-MATE_LOWER, 0)
            #if in_check != (mate_finder <= -MATE_LOWER):
                #import xboard
                #print ('hvad da?', __file__, in_check, mate_finder, nullscore, xboard.renderFEN(pos, 0))
            #mate_finder = self.bound(pos.rotate(), MATE_UPPER, 0)
            #if in_check != (mate_finder >= MATE_LOWER):
            #    import xboard
            #    print ('hvad da?', __file__, in_check, (mate_finder >= MATE_LOWER), mate_finder, nullscore, xboard.renderFEN(pos, 0))
            #in_check = nullscore <= -MATE_LOWER
        # In qs we just use the pos.score for standing pat
        else:
            nullscore = pos.score
            in_check = False
        if depth > 3 and not root and not in_check:
            nullscore = -self.bound(pos.rotate(), 1-gamma, depth-3)
            in_danger = nullscore <= -MATE_LOWER
            # TODO: If null score gets mated, we may still want to do a 
        if not root and nullscore >= gamma:
            return nullscore

        # TODO: test this check extension
        # Test 1: Seems to make mate finding a LOT slower... odly
        #if in_check:
        #    depth += 1


        # We generate all possible, pseudo legal moves and order them to provoke
        # cuts. At the next level of the tree we are going to minimize the score.
        # This can be shown equal to maximizing the negative score, with a slightly
        # adjusted gamma value.
        best, bmove = -MATE_UPPER, None
        any_moves = False
        # Putting killer first actually means that king-captures are no longer first
        # Well: only if king captures are not in the table, but why wouldn't they be?
        # For some reason this still doesn't improve performance much...
        # It does make the time to search ply=8 go from 3 seconds to 2.5 though
        # But it also makes the quickmate2 run slower
        killer = self.tp_move.get(pos)
        killer = [killer] if killer is not None else []
        for move in killer + sorted(pos.gen_moves(), key=pos.value, reverse=True):
        #for move in killer + list(pos.gen_moves()):
            # We check captures with the value function, as it also contains ep and kp
            # Note that we will always search king-captures first, so they aren't hidden
            # by fail-high
            if depth <= 0 and pos.value(move) < 150:
                break
            pos1 = pos.move(move)
            is_legal = depth <= 0 or not any(pos1.board[j] == 'k' for i,j in pos1.gen_moves())
            if is_legal:
                any_moves = True
                # Do the actual recursive check
                score = -self.bound(pos1, 1-gamma, depth-1)
                if score > best:
                    best = score
                if score >= gamma:
                    bmove = move
                    break

        # Check for stalemate. If best move (immediately) loses king,
        # but not we are not in check.
        # But is best really -MATE_UPPER when legal moves?
        # Does't it actually depend on the gamma used?
        # We can't just check best==-MATE_UPPER, at least not until we have delay
        if depth > 0 and not any_moves and not in_check:
            #import xboard
            #print('looks like stalemate to', __file__, xboard.renderFEN(pos,0))
            return 0

        # Since we fail-high with nullscore, we should also use it for fail-low
        # If root, the null score is only used for the draw check
        if best < nullscore and not root:
            best = nullscore
            bmove = None

        # Delay
        # This actually made us go from 14s to 13.5s in quickmate2
        if best >= MATE_UPPER-100: best -= 1
        if best <= -(MATE_UPPER-100): best += 1

        # We don't want fail-low moves in tp_move, which is supposed to give the pv.
        # However then we need to guarantee that we always have a fail-high move at root.
        # This should be easy, but search instability (from null move) can screw us.
        if best >= gamma:
            # We save the found move together with the score, so we can retrieve it in
            # the play loop. We also trim the transposition table in FIFO order.
            self.tp_move[pos] = bmove
            self.tp_score[(pos, depth, root)] = Entry(best, entry.upper)

        #if root:
        #    print('Root {} fail {}. Move {}'.format(depth,
        #        ('high' if best >= gamma else 'low'), self.tp_move.get(pos)))

        # If we find an upper bound, it should always be better than the current upper bound
        # since if it weren't, the current upper bound would have caused a table-return much earlier
        if best < gamma:
            #pass
            self.tp_score[(pos, depth, root)] = Entry(entry.lower, best)
        return best

    #TODO: Try speeding up hash table using score for key

    # secs over maxn is a breaking change. Can we do this?
    # I guess I could send a pull request to deep pink
    def _search(self, pos, secs):
        """ Iterative deepening MTD-bi search """
        self.nodes = 0
        start = time.time()

        # We limit the depth to some constant, so we don't get a stack overflow in
        # the end game.
        for depth in range(1, 99):
            self.depth = depth
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # However this may be broken by values from the transposition table,
            # as they don't have the same concept of p(score). Hence we just use
            # 'lower < upper - margin' as the loop condition.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - 20:
                gamma = (lower+upper+1)//2
                # TODO: Check if allowing null-move in this search has a positive effect
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma:
                    if score > upper:
                        print(__file__, 'wtf1', lower, upper, 'gamma score', gamma, score, 'depth', depth)
                        import tools
                        print('pos', tools.renderFEN(pos, 0))
                    lower = score
                if score < gamma:
                    if score < lower:
                        print(__file__, 'wtf2', lower, upper, 'gamma score', gamma, score, 'depth', depth)
                    upper = score
                #assert lower <= upper
            #while pos not in tp_move.get(pos):
            #    lower = bound(pos, lower, depth, root=True)
            # We do this to ensure there is a fail-high move in the table that we can return
            #if not score >= lower:
            #    print('sunfish.py wtf', lower, upper, score, 'depth', depth)
            #assert score >= lower
            # TODO: What about draws?
            score = self.bound(pos, lower, depth, root=True)
            while score < lower:
                print('weird', score)
                lower = score
                score = self.bound(pos, lower, depth, root=True)
            assert score == self.tp_score.get((pos, depth, True)).lower
            assert self.tp_move.get(pos) is not None or score == 0 or abs(score) >= MATE_LOWER, \
                    "{}, {} <= {} <= {}".format(self.tp_move.get(pos), lower, score, upper)

            yield
            # We stop deepening if we have spent too long, or if we have already won the game.
            if time.time()-start > secs and depth >= 2 or abs(score) >= MATE_LOWER:
                break
        #print('depth', depth)

        # If the game hasn't finished we can retrieve our move from the
        # transposition table.
        #return self.tp_move.get(pos), score

    def search(self, pos, secs):
        list(self._search(pos, secs))
        # If the game hasn't finished we can retrieve our move from the
        # transposition table.
        return self.tp_move.get(pos), self.tp_score.get((pos, self.depth, True)).lower


###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input
    class NewOrderedDict(OrderedDict):
        def move_to_end(self, key):
            value = self.pop(key)
            self[key] = value
    OrderedDict = NewOrderedDict


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


def print_pos(pos):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    for i, row in enumerate(pos.board.strip().split('\n ')):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')


def main():
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    searcher = Searcher()
    while True:
        print_pos(pos)

        if pos.score <= -MATE_LOWER:
            print("You lost")
            break

        # We query the user until she enters a (pseudo) legal move.
        move = None
        while move not in pos.gen_moves():
            match = re.match('([a-h][1-8])'*2, input('Your move: '))
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
        pos = pos.move(move)

        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.
        print_pos(pos.rotate())

        if pos.score <= -MATE_LOWER:
            print("You won")
            break

        # Fire up the engine to look for a move.
        move, score = searcher.search(pos, secs=2)

        if score == MATE_UPPER:
            print("Checkmate!")

        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.
        print("My move:", render(119-move[0]) + render(119-move[1]))
        pos = pos.move(move)


if __name__ == '__main__':
    main()
