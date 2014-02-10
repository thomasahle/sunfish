from itertools import count
from collections import Counter, OrderedDict, namedtuple

# We need a testbed for move generation and mate finding

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

# Check ideas:
#	With accents
#	A Position flag
#	Saved in the side of the board
#	Don't use the direct graphical representation

#	Don't hash ep and king-rights. Just check after retrieval

#	Kill-map for ep and castling (can just be passed in the recur? No, then fen problem)
#	Let the move generation take the previous move as an argument

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
		put = lambda board, i, p: board[:i] + p + board[i+1:]

		board = self.board
		wc, bc, ep, kp = self.wc, self.bc, 0, 0
		score = self.score + self.value(m)

		# Castling
		if i == 21: wc = (False, wc[1])
		if i == 28: wc = (wc[0], False)
		if j == 91: bc = (False, bc[1])
		if j == 98: bc = (bc[0], False)
		if board[i] == 'k':
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
		if board[i] == 'p':
			if 90 < j < 100:
				board = put(board, j, 'q')
			if j - i == 20:
				ep = i + 10
			if j - i in (9, 11) and board[j] == '.':
				board = put(board, j-10, '.')
		# Actual move
		board = put(board, j, board[i])
		board = put(board, i, '.')

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

def search(pos, depth, gamma):
	""" returns s(pos) <= r < gamma    if s(pos) < gamma
		returns s(pos) >= r >= gamma   if s(pos) >= gamma """
	global N; N += 1

	# The thing is though, we can't just check for 
	entry = tp.get(pos.board)
	if entry is not None and entry.depth >= depth:
		if entry.score < gamma and entry.score < entry.gamma or \
				gamma <= entry.score and entry.gamma <= entry.score:
			return entry.score

	if depth == 0:
		return pos.score

	best, bmove = -weight['k'], None
	for m in sorted(pos.genMoves(), key=pos.value, reverse=True):
		score = -search(pos.move(m), depth-1, 1-gamma)
		if score > best:
			best = score
			bmove = m
		if best >= gamma:
			break

	tp[pos.board] = Entry(depth, best, gamma, bmove)
	return best

def MTD(pos, depth):
	lower, upper = -weight['k'], weight['k']
	while upper-lower > 5:
		g = (lower+upper)//2
		s = search(pos, depth, g)
		if s >= g:
			lower = s
		if s < g:
			upper = s
	return (lower+upper)//2

def itd(pos):
	global N
	for d in range(1,99):
		s = MTD(pos, d)
		#if N > 2e4: break
		if d == 7: break
	print ("Visited %d noposdes. Depth %d. Score is %d" % (N,d,s))
	N = 0
	return tp[pos.board].move

def parse(c):
	return 11 + (ord(c[1])-ord('0'))*10 + ord(c[0])-ord('a')

def format(i):
	return chr(i%10 + ord('a')-1) + str(i//10 - 1)

def iformat(i):
	return chr(i%10 + ord('a')-1) + str(10 - i//10)

import sys
if sys.version_info[0] == 2:
	input = raw_input

############

start = Position(initial,0,(True,True),(True,True),0,0)

def xboard():
	pos = Position(initial,0)
	while True:
		smove = input()
		if smove == 'quit':
			break
		if smove == 'protover 2':
			print 'feature myname="SmallChess"'
			print 'feature done=1'
			continue
		if len(smove) < 4:
			continue
		i, j = parse(smove[0:2]), parse(smove[2:4])
		if not (21 <= i <= 85 and 21 <= j <= 85):
			continue
		pos = pos.move((i,j))

		i,j = itd(pos)
		print ("move %s%s" % (iformat(i),iformat(j)))
		pos = pos.move((i,j))

def selfplay():
	pos = Position(initial,0)
	#pos = parseFEN('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
	#pos = parseFEN('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1')
	#pos = parseFEN('2k5/4B3/1K6/8/2B5/8/8/5r2 w - - 0 0')
	#pos = parseFEN('5Kbk/6pp/6P1/8/8/8/8/7R w - - 0 0')
	for _ in range(100):
		print('\n'.join(''.join(pos.board[i:i+8]) for i in range(91,11,-10)))
		print

		m = itd(pos)
		if m is None:
			print "Game over"
			break
		else:
			i, j = m
			print ("move %s%s" % (format(i),format(j)))
			pos = pos.move((i,j))
		print('\n'.join(''.join(pos.flip().board[i:i+8]) for i in range(91,11,-10)))
		print

		m = itd(pos)
		if m is None:
			print "Game over"
			break
		else:
			i, j = m
			print ("move %s%s" % (iformat(i),iformat(j)))
			pos = pos.move((i,j))
		

def normal():
	pos = parseFEN('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
	while True:
		print('\n'.join(''.join(pos.board[i:i+8]) for i in range(91,11,-10)))
		smove = input("You're move: ")
		move = parse(smove[0:2]), parse(smove[2:4])
		
		pos = pos.move(move)
		print('\n'.join(''.join(pos.flip().board[i:i+8]) for i in range(91,11,-10)))

		i,j = itd(pos)
		print ("move %s%s" % (iformat(i),iformat(j)))
		pos = pos.move((i,j))

def allperft(path):
	# with open(path) as f:
	# 	for line in f:
	# 		parts = line.split(';')
	# 		board = parseFEN(parts[0])
	# 		print ';'.join([parts[0]]+[str(perft(board,d)) for d in range(1,5)])
	# for d in range(1,5):
	# 	print
	# 	print "Going to depth %d" % d
	# 	with open(path) as f:
	# 		for line in f:
	# 			parts = line.split(';')
	# 			board = parseFEN(parts[0])
	# 			print parts[0]
	# 			if perft(board,d) != int(parts[d][2:].strip()):
	# 				print '======================'
	# 				print "ERROR at depth %d" % d
	# 				print '======================'
	with open(path) as f:
		for line in f:
			parts = line.split(';')
			pos = parseFEN(parts[0].strip())
			print parts[0]
			for d,s in enumerate(parts[1:]):
				res = perft(pos,d+1)
				if res != int(s):
					print '========================================='
					print "ERROR at depth %d. Gave %d rather than %d" % (d+1, res, int(s))
					print '========================================='
					if d+1 == 1:
						print pos
					perft(pos,d+1,True)
					break
				print d+1,
				import sys; sys.stdout.flush()
			print
import re
def parseFEN(fen):
	# Fen uses the opposite color system of us. Maybe we should swap.
	board, color, castling, enpas, hclock, fclock = fen.split()
	board = re.sub('\d', (lambda m: '.'*int(m.group(0))), board)
	board = '#'*21 + board.replace('/','##') + '#'*21
	if color == 'w':
		board = Position(board,0,0,0,0,0).flip().board
		castling = castling.swapcase()
	wc = ('q' in castling, 'k' in castling)
	bc = ('Q' in castling, 'K' in castling)
	ep = parse(enpas) if enpas != '-' else 0
	return Position(board, 0, wc, bc, ep, 0)
def checkKingCapt(pos):
	for m in pos.genMoves():
		#print "%s%s %d" % (format(m[0]),format(m[1]),pos.value(m))
		if pos.value(m) >= MATE_VALUE:
			return True
	return False
def perft(pos, depth, divide=False):
	if depth == 0:
		return 1
	res = 0
	for m in pos.genMoves():
		pos1 = pos.move(m)
		if not checkKingCapt(pos1):
			#div = iformat(m[0])+iformat(m[1]) == "g2g1"
			#if div:
			#	print pos1.board
			sub = perft(pos1, depth-1, False)
			if divide:
				#print pos1.board
				print " "*depth+format(m[0])+format(m[1]), sub
			res += sub
	return res


if __name__ == '__main__':
	allperft('queen.epd')
	#selfplay()
	
	
