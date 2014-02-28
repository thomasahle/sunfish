
import sys
import re

from sunfish import Position, MATE_VALUE, search, parse, render, bound

def parseFEN(fen):
	""" Parses a string in Forsyth-Edwards Notation into a Position """
	board, color, castling, enpas, hclock, fclock = fen.split()
	board = re.sub('\d', (lambda m: '.'*int(m.group(0))), board)
	board = ' '*21 + '  '.join(board.split('/')[::-1]) + ' '*21
	wc = ('Q' in castling, 'K' in castling)
	bc = ('k' in castling, 'q' in castling)
	ep = parse(enpas) if enpas != '-' else 0
	pos = Position(board, 0, wc, bc, ep, 0)
	return pos if color == 'w' else pos.rotate()

###############################################################################
# Playing test
###############################################################################

def selfplay():
	""" Start a game sunfish vs. sunfish """
	pos = parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
	for d in range(200):
		# Always print the board from the same direction
		if d % 2 == 0: pos = pos.rotate()
		print('\n'.join(pos.board[i:i+10] for i in range(100,0,-10)))
		if d % 2 == 0: pos = pos.rotate()

		m, _ = search(pos)
		if m is None:
			print("Game over")
			break
		print("\nmove %s%s" % tuple(map(render,m)))
		pos = pos.move(m)

###############################################################################
# Perft test
###############################################################################

def allperft(path, depth=4):
	for d in range(1, depth+1):
		print("Going to depth %d" % d)
		with open(path) as f:
			for line in f:
				parts = line.split(';')
				print(parts[0])

				pos, score = parseFEN(parts[0]), int(parts[d])
				res = perft(pos, d)
				if res != score:
					print('=========================================')
					print("ERROR at depth %d. Gave %d rather than %d" % (d, res, score))
					print('=========================================')
					if d == 1:
						print(pos)
					perft(pos, d, divide=True)
					return
		print('')

def perft(pos, depth, divide=False):
	if depth == 0:
		return 1
	res = 0
	for m in pos.genMoves():
		pos1 = pos.move(m)
		# Make sure the move was legal
		if not any(pos1.value(m) >= MATE_VALUE for m in pos1.genMoves()):
			sub = perft(pos1, depth-1, False)
			if divide:
				print(" "*depth+render(m[0])+render(m[1]), sub)
			res += sub
	return res

###############################################################################
# Find mate test
###############################################################################

def allmate(path):
	with open(path) as f:
		for line in f:
			line = line.strip()
			print(line)

			pos = parseFEN(line)
			_, score = search(pos, maxn=1e9)
			if score < MATE_VALUE:
				print("Unable to find mate. Only got score = %d" % score)
				break

def quickmate(path, depth):
	""" Similar to allmate, but uses the `bound` function directly to only
	search for moves that will win us the game """
	with open(path) as f:
		for line in f:
			line = line.strip()
			print(line)

			pos = parseFEN(line)
			for d in range(depth, 99):
				score = bound(pos, MATE_VALUE, d)
				if score >= MATE_VALUE:
					break
				print(d, score)
			else:
				print("Unable to find mate. Only got score = %d" % score)
				return

# Python 2 compatability
if sys.version_info[0] == 2:
	input = raw_input

if __name__ == '__main__':
	#allperft('queen.epd')
	quickmate('mate1.epd', 3)
	quickmate('mate2.epd', 5)
	quickmate('mate3.epd', 7)
	#selfplay()
