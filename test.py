
# We need a testbed for move generation and mate finding

import sys
import re

if sys.version_info[0] == 2:
	input = raw_input

from sunfish import Position, MATE_VALUE, search, parse, iparse, render, irender, bound

def parseFEN(fen):
	# Fen uses the opposite color system of us. Maybe we should swap.
	board, color, castling, enpas, hclock, fclock = fen.split()
	board = re.sub('\d', (lambda m: '.'*int(m.group(0))), board)
	board = '#'*21 + board.replace('/','##') + '#'*21
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
		print ("move %s%s" % (irender(i),irender(j)))
		pos = pos.move((i,j))

def selfplay():
	pos = parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0')
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
			print ("move %s%s" % (render(i),render(j)))
			pos = pos.move((i,j))
		print('\n'.join(''.join(pos.flip().board[i:i+8]) for i in range(91,11,-10)))
		print

		m = itd(pos)
		if m is None:
			print "Game over"
			break
		else:
			i, j = m
			print ("move %s%s" % (irender(i),irender(j)))
			pos = pos.move((i,j))

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
			else:
				print "Unable to find mate. Only got score = %d" % score
				return

if __name__ == '__main__':
	allperft('queen.epd')
	#quickmate('mate1.epd', 3)
