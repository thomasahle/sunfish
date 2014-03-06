import sys
import time

from itertools import count
from collections import Counter, OrderedDict, namedtuple

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# This constant controls how much time we spend on looking for optimal moves.
NODES_SEARCHED = 2e4

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
MATE_VALUE = 10000

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 21, 28, 91, 98
initial = (
	'          ' + #   0 -  9
	'          ' + #  10 - 19
	' RNBQKBNR ' + #  20 - 29
	' PPPPPPPP ' + #  30 - 39
	' ........ ' + #  40 - 49
	' ........ ' + #  50 - 59
	' ........ ' + #  60 - 69
	' ........ ' + #  70 - 79
	' pppppppp ' + #  80 - 89
	' rnbqkbnr ' + #  90 - 99
	'          ' + # 100 -109
	'          ')  # 110 -119


###############################################################################
# Move and evaluation tables
###############################################################################

directions = {
	'P': (10, 20, 9, 11),
	'N': (-19, -8, 12, 21, 19, 8, -12, -21),
	'B': (-9, 11, 9, -11),
	'R': (-10, 1, 10, -1),
	'Q': (-9, 11, 9, -11, -10, 1, 10, -1),
	'K': (-9, 11, 9, -11, -10, 1, 10, -1)
}

weight = {
	'P': 100,
	'N': 320,
	'B': 330,
	'R': 500,
	'Q': 900,
	'K': 2*MATE_VALUE
}

pst = {
	'P': (
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0, -20,   0,   0,   0,   0,   0,   0, -20,   0,
		0, -20,   0,  10,  20,  20,  10,   0, -20,   0,
		0, -20,   0,  20,  40,  40,  20,   0, -20,   0,
		0, -20,   0,  10,  20,  20,  10,   0, -20,   0,
		0, -20,   0,   5,  10,  10,   5,   0, -20,   0,
		0, -20,   0,   0,   0,   0,   0,   0, -20,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	),
	'B': (
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0, -44, -17, -24, -33, -33, -24, -17, -44,   0,
		0, -19,   8,   1,  -8,  -8,   1,   8, -19,   0,
		0, -10,  17,  10,   1,   1,  10,  17, -10,   0,
		0,  -9,  18,  11,   2,   2,  11,  18,  -9,   0,
		0, -12,  15,   8,  -1,  -1,   8,  15, -12,   0,
		0, -18,   9,   2,  -7,  -7,   2,   9, -18,   0,
		0, -22,   5,  -2, -11, -11,  -2,   5, -22,   0,
		0, -39, -12, -19, -28, -28, -19, -12, -39,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	),
	'N': (
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,-134, -99, -75, -63, -63, -75, -99,-134,   0,
		0, -78, -43, -19,  -7,  -7, -19, -43, -78,   0,
		0, -59, -24,   0,  12,  12,   0, -24, -59,   0,
		0, -18,  17,  41,  53,  53,  41,  17, -18,   0,
		0, -20,  15,  39,  51,  51,  39,  15, -20,   0,
		0,   0,  35,  59,  71,  71,  59,  35,   0,   0,
		0, -54, -19,   5,  17,  17,   5, -19, -54,   0,
		0,-190, -55, -31, -19, -19, -31, -55,-190,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	),
	'R': (
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0, -24, -24,  -2,   2,   2,  -2, -24, -24,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0, -12,  -7,  -2,   2,   2,  -2,  -7, -12,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	),
	'Q': (
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,  -5,  -5,   0,   0,   0,   0,
		0,   0,   0,   0,  -5,  -5,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	),
	'K': (
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0, 298, 332, 273, 175, 225, 175, 332, 298,   0,
		0, 287, 321, 214, 175, 175, 175, 321, 287,   0,
		0, 224, 258, 199, 151, 151, 199, 258, 224,   0,
		0, 196, 230, 171, 123, 123, 171, 230, 196,   0,
		0, 173, 207, 148, 100, 100, 148, 207, 173,   0,
		0, 146, 180, 121,  73,  73, 121, 180, 146,   0,
		0, 119, 153,  94,  46,  46,  94, 153, 119,   0,
		0,  98, 132,  73,  25,  25,  73, 132,  98,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	)
}


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
	""" A state of a chess game
	board -- a 120 char representation of the board
	score -- the board evaluation
	wc -- the castling rights
	bc -- the opponent castling rights
	ep - the en passant square
	kp - the king passant square
	"""

	def genMoves(self):
		# For each of our pieces, iterate through each possible 'ray' of moves,
		# as defined in the 'directions' map. The rays are broken e.g. by
		# captures or immediately in case of pieces such as knights.
		piecelist = ((i,p) for i,p in enumerate(self.board) if p.isupper())
		for i, p in piecelist:
			for d in directions[p]:
				for j in count(i+d, d):
					q = self.board[j]
					# Stay inside the board
					if self.board[j] == ' ': break
					# Castling
					if i == A1 and q == 'K' and self.wc[0]: yield (j, j-2)
					if i == H1 and q == 'K' and self.wc[1]: yield (j, j+2)
					# No friendly captures
					if q.isupper(): break
					# Special pawn stuff
					if p == 'P' and d in (9, 11) and q == '.' and j not in (self.ep, self.kp): break
					if p == 'P' and d in (10, 20) and q != '.': break
					if p == 'P' and d == 20 and (i > 40 or self.board[j-10] != '.'): break
					# Move it
					yield (i, j)
					# Stop crawlers from sliding
					if p in ('P','N','K'): break
					# No sliding after captures
					if q.islower(): break

	def rotate(self):
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
		# TODO: tempo via auto decreasing of score?
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
		# Special pawn stuff
		if p == 'P':
			if 90 < j < 100:
				board = put(board, j, 'Q')
			if j - i == 20:
				ep = i + 10
			if j - i in (9, 11) and q == '.':
				board = put(board, j-10, '.')
		# We rotate the returned position, so it's ready for the next player
		return Position(board, score, wc, bc, ep, kp).rotate()

	def value(self, move):
		i, j = move
		p, q = self.board[i], self.board[j]
		# Actual move
		score = pst[p][j] - pst[p][i]
		# Capture
		if q.islower():
			score += pst[q.upper()][j]
			score += weight[q.upper()]
		# Castling check detection
		if abs(j-self.kp) < 2:
			score += weight['K']
		# Castling
		if p == 'K':
			score += pst['R'][(i+j)//2]
			score -= pst['R'][A1 if j < i else H1]
		# Special pawn stuff
		if p == 'P':
			if 80 < j < 90:
				score += weight['Q'] - weight['P']
				score += pst['Q'][j] - pst['P'][j]
			if j == self.ep:
				score += pst['P'][j-10]
				score += weight['P']
		return score

Entry = namedtuple('Entry', 'depth score gamma move')
tp = OrderedDict()


###############################################################################
# Search logic
###############################################################################

N = 0
def bound(pos, gamma, depth):
	""" returns s(pos) <= r < gamma    if s(pos) < gamma
		returns s(pos) >= r >= gamma   if s(pos) >= gamma """
	global N; N += 1

	# Look in the table if we have already searched this position before.
	# We use the table value if it was done with at least as deep a search
	# as ours, and the gamma value is compatible.
	entry = tp.get(pos)
	if entry is not None and entry.depth >= depth and (
			entry.score < entry.gamma and entry.score < gamma or
			entry.score >= entry.gamma and entry.score >= gamma):
		return entry.score

	# Stop searching if we have won/lost.
	if abs(pos.score) >= MATE_VALUE:
		return pos.score

	# Null move. Is also used for stalemate checking
	nullscore = -bound(pos.rotate(), 1-gamma, depth-3) if depth > 0 else pos.score
	if nullscore >= gamma:
		# We need a pv
		if entry is None:
			tp[pos] = Entry(0, nullscore, gamma, next(pos.genMoves()))
		return nullscore

	# Try the TT move, it's usually good enough to get a free cutoff, or
	# otherwise get a good score.
	best, bmove, movecount = -3*MATE_VALUE, None, 0
	if entry is not None:
		movecount += 1
		score = -bound(pos.move(entry.move), 1-gamma, depth-1)
		best = score
		bmove = entry.move
		if score >= gamma:
			if depth >= entry.depth:
				tp[pos] = Entry(depth, best, gamma, bmove)
				if len(tp) > TABLE_SIZE:
					tp.pop()
			return best
	# We generate all possible, pseudo legal moves and order them to provoke
	# cuts. At the next level of the tree we are going to minimize the score.
	# This can be shown equal to maximizing the negative score, with a slightly
	# adjusted gamma value.
	for move in sorted(pos.genMoves(), key=pos.value, reverse=True):
		# QSearch
		if depth <= 0 and pos.value(move) < 100:
			break
		# Don't try the TT move twice
		if entry is not None and move == entry.move:
			continue
		# LMR
		movecount += 1
		reductions = 0
		if movecount > 5 and 3 > depth > 0:
			reductions = 1
		score = -bound(pos.move(move), 1-gamma, depth-1-reductions)
		if score > best and reductions == 1:
			score = -bound(pos.move(move), 1-gamma, depth-1)
		if score > best:
			best = score
			bmove = move
		if score >= gamma:
			break
	# If there are no captures, or just not any good ones, stand pat
	if depth <= 0 and best < nullscore:
		return nullscore
	# Check for stalemate
	if depth > 0 and bmove is None and nullscore > -MATE_VALUE:
		print("Stalemate detected")
		print(pos.board)
		best = 0

	# We save the found move together with the score, so we can retrieve it in
	# the play loop. We also trim the transposition table in FILO order.
	# We prefer fail-high moves, as they are the ones we can build our pv from.
	if entry is None or depth >= entry.depth and score >= gamma:
		tp[pos] = Entry(depth, best, gamma, bmove)
		if len(tp) > TABLE_SIZE:
			tp.pop()
	return best

def search(pos, maxn=NODES_SEARCHED):
	""" Iterative deepening MTD-bi search """
	global N; N = 0

	starttime = time.time()
	print("Depth\tScore\tTime\tNodes\tPV")
	# We limit the depth to some constant, so we don't get a stack overflow in
	# the end game.
	for depth in range(1, 99):
		# The inner loop is a binary search on the score of the position.
		# Inv: lower <= score <= upper
		# However this may be broken by values from the transposition table,
		# as they don't have the same concept of p(score). Hence we just use
		# 'lower < upper - margin' as the loop condition.
		lower, upper = -3*MATE_VALUE, 3*MATE_VALUE
		while lower < upper - 3:
			gamma = (lower+upper+1)//2
			score = bound(pos, gamma, depth)
			if score >= gamma:
				lower = score
			if score < gamma:
				upper = score
		now = int((time.time() - starttime) * 100)
		print("%d\t%d\t%d\t%d\t%s" % (depth, score, now, N, ' '.join(pv(1,pos))))
		# We stop deepening if the global N counter shows we have spent too
		# long, or if we have already won the game.
		if N >= maxn or abs(score) >= MATE_VALUE:
			break

	# If the game hasn't finished we can retrieve our move from the
	# transposition table.
	entry = tp.get(pos)
	if entry is not None:
		return entry.move, score
	return None, score


###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
	input = raw_input

def parse(c):
	return 21 + 10*(ord(c[1]) - ord('1')) + (ord(c[0]) - ord('a'))

def render(i):
	return chr(i%10 + ord('a') - 1) + str(i//10 - 1)

def mrender(color,pos, m):
	# Sunfish always assumes promotion to queen
	p = 'q' if A8 <= m[1] <= H8 and pos.board[m[0]] == 'P' else ''
	m = (119-m[0],119-m[1]) if color == 1 else m
	return render(m[0]) + render(m[1]) + p
def pv(color, pos):
	res = []
	while True:
		entry = tp.get(pos)
		if entry is None:
			break
		move = mrender(color,pos,entry.move)
		if move in res:
			res.append(move)
			res.append('loop')
			break
		res.append(move)
		pos, color = pos.move(entry.move), 1-color
	return res

def main():
	pos = Position(initial,0,(True,True),(True,True),0,0)
	while True:
		# We mirror the board before printing it, so it looks more natural when
		# playing.
		print('\n'.join(pos.board[i:i+10] for i in range(100,0,-10)))
		print('')

		# We don't do stallmates, so losing is the same as not having any
		# possible moves.
		legalMoves = list(pos.genMoves())
		if not legalMoves:
			print("You lost")
			break

		# We query the user until she enters a legal move.
		move = None
		while move not in legalMoves:
			smove = input("Your move: ")
			move = parse(smove[0:2]), parse(smove[2:4])
		pos = pos.move(move)

		# After our move we rotate the board and print it again.
		# This allows us to see the effect of our move.
		print('\n'.join(pos.rotate().board[i:i+10] for i in range(100,0,-10)))
		print('')

		# Fire up the engine to look for a move.
		m, _ = search(pos)
		if m is None:
			print("You won")
			break

		# The black player moves from a rotated position, so we have to
		# 'back rotate' the move before printing it.
		print("My move: %s%s" % (render(119-m[0]),render(119-m[1])))
		print("Visited %d nodes." % N)
		pos = pos.move(m)

if __name__ == '__main__':
	main()

