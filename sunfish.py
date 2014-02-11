from itertools import count
from collections import Counter, OrderedDict, namedtuple

TABLE_SIZE = 1e7
NODES_SEARCHED = 2e4

initial = (
	'          ' +
	'          ' +
	' RNBQKBNR ' +
	' PPPPPPPP ' +
	' ........ ' +
	' ........ ' +
	' ........ ' +
	' ........ ' +
	' pppppppp ' +
	' rnbqkbnr ' +
	'          ' +
	'          ')

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
	'K': 20000
}
MATE_VALUE = 10000

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
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
		0,   8,   8,   8,   8,   8,   8,   8,   8,   0,
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

class Position(namedtuple('Position','board score wc bc ep kp')):
	""" A state of a chess game
	board -- a 120 char representation of the board
	score -- the board evaluation
	wc -- the castling rights
	bc -- the opponent castling rights
	ep - the en passant square
	kp - the king passant square
	"""

	def genMoves(self):
		board = self.board
		for i, p in enumerate(board):
			if p.isupper():
				for d in directions[p]:
					for j in count(i+d, d):
						# Stay inside the board
						if board[j] == ' ': break
						# Castling
						if i == 21 and board[j] == 'K' and self.wc[0]: yield (j, j-2)
						if i == 28 and board[j] == 'K' and self.wc[1]: yield (j, j+2)
						# No friendly captures
						if board[j].isupper(): break
						# Special pawn stuff
						if p == 'P' and d in (9, 11) and board[j] == '.' and j not in (self.ep, self.kp): break
						if p == 'P' and d in (10, 20) and board[j] != '.': break
						if p == 'P' and d == 20 and (i > 40 or board[j-10] != '.'): break

						yield (i,j)

						# Stop crawlers from sliding
						if p in ('P','N','K'): break
						# No sliding after captures
						if board[j].islower(): break

	def flip(self):
		return Position(
			self.board[::-1].swapcase(), -self.score,
			self.bc, self.wc, 119-self.ep, 119-self.kp)

	def move(self, m):
		i, j = m
		p, q = self.board[i], self.board[j]
		put = lambda board, i, p: board[:i] + p + board[i+1:]

		board = self.board
		wc, bc, ep, kp = self.wc, self.bc, 0, 0
		score = self.score + self.value(m)

		# Actual move
		board = put(board, j, board[i])
		board = put(board, i, '.')
		# Castling rights
		if i == 21: wc = (False, wc[1])
		if i == 28: wc = (wc[0], False)
		if j == 91: bc = (bc[0], False)
		if j == 98: bc = (False, bc[1])
		# Castling
		if p == 'K':
			wc = (False, False)
			if abs(j-i) == 2:
				kp = (i+j)//2
				board = put(board, 21 if j < i else 28, '.')
				board = put(board, kp, 'R')
		# Special pawn stuff
		if p == 'P':
			if 90 < j < 100:
				board = put(board, j, 'Q')
			if j - i == 20:
				ep = i + 10
			if j - i in (9, 11) and q == '.':
				board = put(board, j-10, '.')

		return Position(board, score, wc, bc, ep, kp).flip()

	def value(self, move):
		i, j = move
		p, q = self.board[i], self.board[j]
		# Actual move
		score = pst[p][j] - pst[p][i]
		# Capture
		if q.islower():
			score += pst[q.upper()][j]
			score += weight[q.upper()]
		# King capture
		if abs(j-self.kp) < 2:
			score += weight['K']
		# Castling
		if p == 'K':
			ri = 21 if j < i else 28
			rj = (i+j)//2
			score += pst['R'][rj] - pst['R'][ri]
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
N = 0

# Optimizations:
#	The entry.gamma <= score < gamma case
#	Faster hashing
#	Qsearch
def bound(pos, gamma, depth):
	""" gamma -- should be within the interval (-MATE_VALUE, MATE_VALUE]
		returns s(pos) <= r < gamma    if s(pos) < gamma
		returns s(pos) >= r >= gamma   if s(pos) >= gamma """
	global N; N += 1
	assert -MATE_VALUE < gamma <= MATE_VALUE

	# Look in the table if we have already searched this position before.
	# We use the table value if it was done with at least as deep a search
	# as ours, and the gamma value is compatible.
	entry = tp.get(pos)
	if entry is not None and entry.depth >= depth and (
			entry.score < gamma and entry.score < entry.gamma or
			gamma <= entry.score and entry.gamma <= entry.score):
		return entry.score

	# Stop searching if we have run out of depth or have won/lost
	if depth == 0 or abs(pos.score) >= MATE_VALUE:
		return pos.score

	# We generate all possible, pseudo legal moves and order them to provoke
	# cuts. At the next level of the tree we are going to minimize the score.
	# This can be shown equal to maximizing the negative score, with a slightly
	# adjusted gamma value.
	best, bmove = -MATE_VALUE, None
	for m in sorted(pos.genMoves(), key=pos.value, reverse=True):
		score = -bound(pos.move(m), 1-gamma, depth-1)
		if score > best:
			best = score
			bmove = m
		if score >= gamma:
			break

	# We save the found move together with the score, so we can retrieve it in
	# the play loop. We also trim the transposition table in FILO order.
	tp[pos] = Entry(depth, best, gamma, bmove)
	if len(tp) > TABLE_SIZE:
		tp.pop()
	return best

def search(pos, maxn=NODES_SEARCHED):
	""" Iterative deepening MTD-bi search """
	global N; N = 0

	# We limit the depth to some constant, so we don't get a stack overflow in
	# the end game.
	for depth in range(1, 99):
		# The inner loop is a binary search on the score of the position.
		# Inv: lower <= score <= upper
		lower, upper = -MATE_VALUE, MATE_VALUE
		while lower < upper:
			#print (lower, upper)
			gamma = (lower+upper+1)//2
			score = bound(pos, gamma, depth)
			if score >= gamma:
				lower = score
			if score < gamma:
				upper = score
		
		#print("Searched %d nodes. Depth %d." % (N, depth))

		# We stop deepening if the global N counter shows we have spent too
		# long, or if we have already won the game.
		if N >= maxn or abs(lower) >= MATE_VALUE:
			#print("Searched %d nodes. Depth was %d.")
			break

	# If the game hasn't finished we can retrieve our move from the
	# transposition table.
	entry = tp.get(pos)
	if entry is not None:
		return entry.move
	return None

############

import sys
if sys.version_info[0] == 2:
	input = raw_input

def parse(c):
	return 11 + (ord(c[1])-ord('0'))*10 + ord(c[0])-ord('a')

def render(i):
	return chr(i%10 + ord('a')-1) + str(i//10 - 1)

def main():
	pos = Position(initial,0,(True,True),(True,True),0,0)
	while True:
		print('\n'.join(pos.board[i:i+10] for i in range(100,0,-10)))

		smove = input("\nYou're move: ")
		move = parse(smove[0:2]), parse(smove[2:4])
		
		pos = pos.move(move)
		print('\n'.join(pos.flip().board[i:i+10] for i in range(100,0,-10)))

		m = search(pos)
		if m is None:
			print("\nGame over")
			break
		print("\nMy move: %s%s" % (render(119-m[0]),render(119-m[1])))
		print("Visited %d nodes." % N)
		pos = pos.move(m)

if __name__ == '__main__':
	main()

	