
# We need a testbed for move generation and mate finding

import sys
import re

from sunfish import Position, MATE_VALUE, search, parse, iparse, render, irender, bound

# Python 2 compatability
if sys.version_info[0] == 2:
	input = raw_input


def parseFEN(fen):
	# Fen uses the opposite color system of us. Maybe we should swap.
	board, color, castling, enpas, hclock, fclock = fen.split()
	board = re.sub('\d', (lambda m: '.'*int(m.group(0))), board)
	board = ' '*21 + board.replace('/','  ') + ' '*21
	wc = ('q' in castling, 'k' in castling)
	bc = ('Q' in castling, 'K' in castling)
	ep = iparse(enpas) if enpas != '-' else 0
	pos = Position(board, 0, wc, bc, ep, 0)
	return pos.flip() if color == 'w' else pos

############################
# Playing test
############################

def xboard():
	pos = parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0')
	while True:
		smove = input()
		if smove == 'quit':
			break
		elif smove == 'protover 2':
			print 'feature myname="Sunfish"'
			print 'feature usermove=1'
			print 'feature done=1'
			continue
		elif smove.startswith('usermove'):
			smove = smove[9:]
			i, j = parse(smove[0:2]), parse(smove[2:4])
			assert 21 <= i <= 89 and 21 <= j <= 98
		else:
			print "Didn't understand command '%s'" % smove
			continue

		pos = pos.move((i,j))
		m = search(pos)
		print("move %s%s" % tuple(map(irender,m)))
		pos = pos.move(m)

def selfplay():
	pos = parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0')
	for d in range(200):
		if d % 2 == 0:
			print('\n'.join(pos.board[i:i+10] for i in range(100,0,-10)))
		else:
			print('\n'.join(pos.flip().board[i:i+10] for i in range(100,0,-10)))

		m = search(pos)
		if m is None:
			print("Game over")
			break
		print ("\nmove %s%s" % tuple(map(render,m)))
		pos = pos.move(m)

############################
# Perft test
############################

def allperft(path, depth=4):
	for d in range(1, depth+1):
		print
		print "Going to depth %d" % d
		with open(path) as f:
			for line in f:
				parts = line.split(';')
				print parts[0]

				pos, score = parseFEN(parts[0]), int(parts[d])
				res = perft(pos, d)
				if res != score:
					print '========================================='
					print "ERROR at depth %d. Gave %d rather than %d" % (d, res, score)
					print '========================================='
					if d == 1:
						print pos
					perft(pos, d, divide=True)
					return

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
				print " "*depth+irender(m[0])+irender(m[1]), sub
			res += sub
	return res

############################
# Find mate test
############################

def allmate(path):
	with open(path) as f:
		for line in f:
			line = line.strip()
			print(line)

			pos = parseFEN(line)
			_, score = search(pos, maxn=1e9)
			if score < MATE_VALUE:
				print "Unable to find mate. Only got score = %d" % score
				break

# This one is about twice as fast
def quickmate(path, depth):
	with open(path) as f:
		for line in f:
			line = line.strip()
			print(line)

			pos = parseFEN(line)
			for d in range(depth, 99):
				score = bound(pos, MATE_VALUE, d)
				if score >= MATE_VALUE:
					break
				print d, score
			else:
				print "Unable to find mate. Only got score = %d" % score
				return

if __name__ == '__main__':
	#allperft('queen.epd')
	#quickmate('mate3.epd', 7)
	xboard()
