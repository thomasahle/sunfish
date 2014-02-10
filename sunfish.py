from itertools import count
from collections import Counter, OrderedDict, namedtuple

initial = (
	'##########' +
	'##########' +
	'#rnbqkbnr#' +
	'#pppppppp#' +
	'#........#' +
	'#........#' +
	'#........#' +
	'#........#' +
	'#PPPPPPPP#' +
	'#RNBQKBNR#' +
	'##########' +
	'##########')

directions = {
	'p': (10, 20, 9, 11),
	'n': (-19, -8, 12, 21, 19, 8, -12, -21),
	'b': (-9, 11, 9, -11),
	'r': (-10, 1, 10, -1),
	'q': (-9, 11, 9, -11, -10, 1, 10, -1),
	'k': (-2, 2, -9, 11, 9, -11, -10, 1, 10, -1)
}

weight = {
	'p': 100,   'P': -100,
	'n': 320,   'N': -320,
	'b': 330,   'B': -330,
	'r': 500,   'R': -500,
	'q': 900,   'Q': -900,
	'k': 20000, 'K': -20000,
}
MATE_VALUE = 10000

pst = {
	'p': (
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
	'b': (
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
	'n': (
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
	'r': (
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
	'q': (
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
	'k': (
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

# Jeg gad godt finde en måde at fixe castling på, så jeg kan bruge reverse() i steddet for flip

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
			if p.islower():
				for d in directions[p]:
					for j in count(i+d, d):
						# Stay inside the board
						if board[j] == '#': break
						# No friendly captures
						if board[j].islower(): break
						# Special pawn stuff
						if p == 'p' and d in (9, 11) and board[j] == '.' and j not in (self.ep, self.kp): break
						if p == 'p' and d in (10, 20) and board[j] != '.': break
						if p == 'p' and d == 20 and (i > 40 or board[j-10] != '.'): break
						# Castling
						if p == 'k' and d == -2 and (not self.wc[0] or board[i-3:i] != '...'): break
						if p == 'k' and d == 2 and (not self.wc[1] or board[i+1:i+3] != '..'): break

						yield (i,j)

						# Stop crawlers from sliding
						if p in ('p','n','k'): break
						# No sliding after captures
						if board[j].isupper(): break

	def flip(self):
		# Vertical mirroring and color swapping
		board = ''.join(self.board[i:i+10] for i in range(110,-1,-10)).swapcase()
		ep, kp = (11-self.ep//10)*10+self.ep%10, (11-self.kp//10)*10+self.kp%10
		return Position(board, -self.score, self.bc, self.wc, ep, kp)

	def move(self, m):
		i, j = m
		p, q = self.board[i], self.board[j]
		put = lambda board, i, p: board[:i] + p + board[i+1:]
		# def put(board,i,p):
		# 	print self
		# 	print m
		# 	assert 21 <= i <= 99
		# 	return board[:i] + p + board[i+1:]

		board = self.board
		wc, bc, ep, kp = self.wc, self.bc, 0, 0
		score = self.score + self.value(m)

		# Actual move
		board = put(board, j, board[i])
		board = put(board, i, '.')
		# Castling
		if i == 21: wc = (False, wc[1])
		if i == 28: wc = (wc[0], False)
		if j == 91: bc = (False, bc[1])
		if j == 98: bc = (bc[0], False)
		if p == 'k':
			wc = (False, False)
			if j - i == -2:
				board = put(board, i-1, 'r')
				board = put(board, i-4, '.')
				kp = i - 1
			if j - i == 2:
				board = put(board, i+1, 'r')
				board = put(board, i+3, '.')
				kp = i + 1
		# Special pawn stuff
		if p == 'p':
			if 90 < j < 100:
				board = put(board, j, 'q')
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
		if q.isupper():
			score += pst[q.lower()][j]
			score += weight[q.lower()]
		# King capture
		if abs(j-self.kp) < 2:
			score += weight['k']
		# Castling
		if p == 'k':
			if j - i == -2:
				score += pst['r'][i-1] - pst['r'][i-4]
			if j - i == 2:
				score += pst['r'][i+1] - pst['r'][i+3]
		# Special pawn stuff
		if p == 'p':
			if 80 < j < 90:
				score += weight['q'] - weight['p']
				score += pst['q'][j] - pst['p'][j]
			if j == self.ep:
				score += pst['p'][j-10]
				score += weight['p']
		return score

tp = {}
Entry = namedtuple('Entry', 'depth score gamma move')

N = 0

# Optimizations:
#	The entry.gamma <= score < gamma case
#	Don't research if the table says we lost
#	Qsearch
#	Maybe it was a mistake to stop using orderedtable

def bound(pos, gamma, depth):
	""" gamma -- should be within the interval (-MATE_VALUE, MATE_VALUE]
		returns s(pos) <= r < gamma    if s(pos) < gamma
		returns s(pos) >= r >= gamma   if s(pos) >= gamma """
	global N; N += 1
	assert -MATE_VALUE < gamma <= MATE_VALUE

	# The thing is though, we can't just check for 
	entry = None#tp.get(pos.board)
	if entry is not None and entry.depth >= depth:
		if entry.score < gamma and entry.score < entry.gamma or \
				gamma <= entry.score and entry.gamma <= entry.score:
			return entry.score

	if depth == 0 or abs(pos.score) >= MATE_VALUE:
		return pos.score

	best, bmove = -MATE_VALUE, None
	for m in sorted(pos.genMoves(), key=pos.value, reverse=True):
		score = -bound(pos.move(m), 1-gamma, depth-1)
		if score > best:
			best = score
			bmove = m
		if score >= gamma:
			break

	tp[pos.board] = Entry(depth, best, gamma, bmove)
	return best

def search(pos, maxn=2e4, maxd=99):
	""" Iterative deepening MTD-bi search """
	global N; N = 0

	for depth in range(1, maxd+1):
		lower, upper = -MATE_VALUE, MATE_VALUE
		# Inv: lower <= score <= upper
		while lower < upper:
			gamma = (lower+upper+1)//2
			score = bound(pos, gamma, depth)
			if score >= gamma:
				lower = score
			if score < gamma:
				upper = score
		
		print depth, lower, "%s%s" % (irender(tp[pos.board].move[0]),irender(tp[pos.board].move[1]))
		if N >= maxn or lower >= MATE_VALUE:
			break

	move, score = tp[pos.board].move, lower
	#print ("Visited %d noposdes. Depth %d. Score is %d" % (N,depth,score))
	return move, score

############

import sys
if sys.version_info[0] == 2:
	input = raw_input

def parse(c):
	return 11 + (ord(c[1])-ord('0'))*10 + ord(c[0])-ord('a')

def iparse(c):
	return 101 + (-ord(c[1])+ord('0'))*10 + ord(c[0])-ord('a')

def render(i):
	return chr(i%10 + ord('a')-1) + str(i//10 - 1)

def irender(i):
	return chr(i%10 + ord('a')-1) + str(10 - i//10)

def main():
	pos = Position(initial,0,(True,True),(True,True),0,0)
	while True:
		print('\n'.join(''.join(pos.board[i:i+8]) for i in range(91,11,-10)))
		smove = input("You're move: ")
		move = parse(smove[0:2]), parse(smove[2:4])
		
		pos = pos.move(move)
		print('\n'.join(''.join(pos.flip().board[i:i+8]) for i in range(91,11,-10)))

		# TODO: Test gameover
		i,j = itd(pos)
		print("My move: %s%s" % (irender(i),irender(j)))
		pos = pos.move((i,j))

if __name__ == '__main__':
	main()

	