#!/usr/bin/env pypy -u
from __future__ import print_function
from __future__ import division
import os
import re
import sys
import sunfish
import test
import math

# Python 2 compatability
if sys.version_info[0] == 2:
	input = raw_input

# Sunfish doesn't know about colors. We hav to.
WHITE, BLACK = range(2)

def render(color, pos, m):
	# Sunfish always assumes promotion to queen
	p = 'q' if sunfish.A8 <= m[1] <= sunfish.H8 and pos.board[m[0]] == 'P' else ''
	m = (119-m[0],119-m[1]) if color == BLACK else m
	return sunfish.render(m[0]) + sunfish.render(m[1]) + p
def parse(color, m):
	r = (sunfish.parse(m[0:2]), sunfish.parse(m[2:4]))
	if color == BLACK:
		return (119-r[0], 119-r[1])
	return r
def pv(color, pos):
	res = []
	origc = color
	res.append(str(pos.score))
	while True:
		entry = sunfish.tp.get(pos)
		if entry is None:
			break
		move = render(color,pos,entry.move)
		if move in res:
			res.append(move)
			res.append('loop')
			break
		res.append(move)
		pos, color = pos.move(entry.move), 1-color
		res.append(str(pos.score if color==origc else -pos.score))
	return res

def main():
	pos = test.parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
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
			stack.append('setboard rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

		elif smove.startswith('setboard'):
			_, fen = smove.split(' ', 1)
			pos = test.parseFEN(fen)
			forced = True
			color = WHITE if fen.split()[1] == 'w' else BLACK

		elif smove == 'force':
			forced = True

		elif smove == 'go':
			forced = False

			# Let's follow the clock of our opponent
			nodes = sunfish.NODES_SEARCHED
			if time > 0 and otim > 0: nodes *= time/otim
			m, s = sunfish.search(pos, maxn=nodes)
			# We don't play well once we have detected our death
			if s <= -sunfish.MATE_VALUE:
				print('resign')
			else:
				print('# %d %+d %d %d %s' % (0, s, 0, sunfish.N, ' '.join(pv(color,pos))))
				print('move %s' % render(color, pos, m))
				print('score before %d after %+d' % (pos.score, pos.value(m)))
				pos = pos.move(m)
				color = 1-color

		elif smove.startswith('ping'):
			_, N = smove.split()
			print('pong', N)

		elif smove.startswith('usermove'):
			_, smove = smove.split()
			m = parse(color, smove)
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
