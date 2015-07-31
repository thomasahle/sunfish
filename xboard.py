#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import re
import sys
import sunfish

# Python 2 compatability
if sys.version_info[0] == 2:
	input = raw_input

# Sunfish doesn't know about colors. We hav to.
WHITE, BLACK = range(2)
FEN_INITIAL = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def parseFEN(fen):
	""" Parses a string in Forsyth-Edwards Notation into a Position """
	board, color, castling, enpas, hclock, fclock = fen.split()
	board = re.sub('\d', (lambda m: '.'*int(m.group(0))), board)
	board = ' '*19+'\n ' + '\n '.join(board.split('/')) + ' \n'+' '*19
	wc = ('Q' in castling, 'K' in castling)
	bc = ('k' in castling, 'q' in castling)
	ep = sunfish.parse(enpas) if enpas != '-' else 0
	score = sum(sunfish.pst[p][i] for i,p in enumerate(board) if p.isupper())
	score -= sum(sunfish.pst[p.upper()][i] for i,p in enumerate(board) if p.islower())
	pos = sunfish.Position(board, score, wc, bc, ep, 0)
	return pos if color == 'w' else pos.rotate()

def mrender(color, pos, m):
	# Sunfish always assumes promotion to queen
	p = 'q' if sunfish.A8 <= m[1] <= sunfish.H8 and pos.board[m[0]] == 'P' else ''
	m = m if color == WHITE else (119-m[0], 119-m[1])
	return sunfish.render(m[0]) + sunfish.render(m[1]) + p

def mparse(color, move):
	m = (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
	return m if color == WHITE else (119-m[0], 119-m[1])

def pv(color, pos):
	res = []
	origc = color
	res.append(str(pos.score))
	while True:
		entry = sunfish.tp.get(pos)
		if entry is None:
			break
		if entry.move is None:
			res.append('null')
			break
		move = mrender(color,pos,entry.move)
		if move in res:
			res.append(move)
			res.append('loop')
			break
		res.append(move)
		pos, color = pos.move(entry.move), 1-color
		res.append(str(pos.score if color==origc else -pos.score))
	return ' '.join(res)

def main():
	pos = parseFEN(FEN_INITIAL)
	forced = False
	color = WHITE
	time, otim = 1, 1

	stack = []
	while True:
		if stack:
			smove = stack.pop()
		else: smove = input()

		if smove == 'quit':
			break

		elif smove == 'protover 2':
			print('feature done=0')
			print('feature myname="Sunfish"')
			print('feature usermove=1')
			print('feature setboard=1')
			print('feature ping=1')
			print('feature sigint=0')
			print('feature variants="normal"')
			print('feature done=1')

		elif smove == 'new':
			stack.append('setboard ' + FEN_INITIAL)

		elif smove.startswith('setboard'):
			_, fen = smove.split(' ', 1)
			pos = parseFEN(fen)
			color = WHITE if fen.split()[1] == 'w' else BLACK

		elif smove == 'force':
			forced = True

		elif smove == 'go':
			forced = False

			# Let's follow the clock of our opponent
			nodes = 2e4
			if time > 0 and otim > 0: nodes *= time/otim
			m, s = sunfish.search(pos, maxn=nodes)
			# We don't play well once we have detected our death
			if s <= -sunfish.MATE_VALUE:
				print('resign')
			else:
				print('# %d %+d %d %d %s' % (0, s, 0, sunfish.nodes, pv(color,pos)))
				print('move', mrender(color, pos, m))
				print('score before %d after %+d' % (pos.score, pos.value(m)))
				pos = pos.move(m)
				color = 1-color

		elif smove.startswith('ping'):
			_, N = smove.split()
			print('pong', N)

		elif smove.startswith('usermove'):
			_, smove = smove.split()
			m = mparse(color, smove)
			pos = pos.move(m)
			color = 1-color
			if not forced:
				stack.append('go')

		elif smove.startswith('time'):
			time = int(smove.split()[1])
		
		elif smove.startswith('otim'):
			otim = int(smove.split()[1])

		elif any(smove.startswith(x) for x in ('xboard','post','random','hard','accepted','level')):
			pass

		else:
			print("Error (unkown command):", smove)

if __name__ == '__main__':
	main()
