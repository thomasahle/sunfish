#!/usr/bin/env pypy -u
from __future__ import print_function
import os
import re
import sys
import sunfish
import test

# Python 2 compatability
if sys.version_info[0] == 2:
	input = raw_input

# Sunfish doesn't know about colors. We hav to.
WHITE, BLACK = range(2)

if __name__ == '__main__':
	pos = test.parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
	forced = False
	color = WHITE

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

			m, s = sunfish.search(pos)
			# We don't play well once we have detected our death
			if s <= -sunfish.MATE_VALUE:
				print('resign')
			else:
				if color == WHITE:
					move = "%s%s" % (sunfish.render(m[0]), sunfish.render(m[1]))
				else: move = "%s%s" % (sunfish.render(119-m[0]), sunfish.render(119-m[1]))
				# Sunfish always assumes promotion to queen
				if sunfish.A8 <= m[1] <= sunfish.H8 and pos.board[m[0]] == 'P':
					print('move %sq' % move)
				else: print('move %s' % move)
				pos = pos.move(m)
				color = 1-color

		elif smove.startswith('ping'):
			_, N = smove.split()
			print('pong', N)

		elif smove.startswith('usermove'):
			_, smove = smove.split()
			if color == WHITE:
				m = (sunfish.parse(smove[0:2]), sunfish.parse(smove[2:4]))
			else: m = (119-sunfish.parse(smove[0:2]), 119-sunfish.parse(smove[2:4]))
			pos = pos.move(m)
			color = 1-color
			if not forced:
				stack.append('go')

		elif any(smove.startswith(x) for x in ('time', 'otim', 'xboard')):
			pass

		else:
			print("Error (unkown command):", smove)
