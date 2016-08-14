#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import re
import sys
import time

import sunfish

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input

# Disable buffering
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

# Sunfish doesn't know about colors. We hav to.
WHITE, BLACK = range(2)
FEN_INITIAL = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def parseFEN(fen):
    """ Parses a string in Forsyth-Edwards Notation into a Position """
    board, color, castling, enpas, _hclock, _fclock = fen.split()
    board = re.sub(r'\d', (lambda m: '.'*int(m.group(0))), board)
    board = ' '*19+'\n ' + '\n '.join(board.split('/')) + ' \n'+' '*19
    wc = ('Q' in castling, 'K' in castling)
    bc = ('k' in castling, 'q' in castling)
    ep = sunfish.parse(enpas) if enpas != '-' else 0
    score = sum(sunfish.pst[p][i] for i,p in enumerate(board) if p.isupper())
    score -= sum(sunfish.pst[p.upper()][119-i] for i,p in enumerate(board) if p.islower())
    pos = sunfish.Position(board, score, wc, bc, ep, 0)
    return pos if color == 'w' else pos.rotate()

def renderFEN(pos, color, half_move_clock=0, full_move_clock=1):
    if color == BLACK:
        pos = pos.rotate()
    board = '/'.join(re.sub(r'\.+', (lambda m: str(len(m.group(0)))), rank)
                      for rank in pos.board.split())
    color = 'w' if color == WHITE else 'b'
    castling = ('K' if pos.wc[1] else '') + ('Q' if pos.wc[0] else '') \
            + ('k' if pos.bc[0] else '') + ('q' if pos.bc[1] else '') or '-'
    ep = sunfish.render(pos.ep) if pos.ep else '-'
    clock = str(half_move_clock) + ' ' + str(full_move_clock)
    return ' '.join((board, color, castling, ep, clock))

def mrender(color, pos, m):
    # Sunfish always assumes promotion to queen
    p = 'q' if sunfish.A8 <= m[1] <= sunfish.H8 and pos.board[m[0]] == 'P' else ''
    m = m if color == WHITE else (119-m[0], 119-m[1])
    return sunfish.render(m[0]) + sunfish.render(m[1]) + p

def mparse(color, move):
    m = (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
    return m if color == WHITE else (119-m[0], 119-m[1])

def pv(searcher, color, pos, include_scores=True):
    res = []
    seen_pos = set()
    origc = color
    if include_scores:
        res.append(str(pos.score))
    while True:
        move = searcher.tp_move.get(pos)
        if move is None:
            break
        res.append(mrender(color, pos, move))
        pos, color = pos.move(move), 1-color
        if pos in seen_pos:
            res.append('loop')
            break
        seen_pos.add(pos)
        if include_scores:
            res.append(str(pos.score if color==origc else -pos.score))
    return ' '.join(res)

def main():
    pos = parseFEN(FEN_INITIAL)
    searcher = sunfish.Searcher()
    forced = False
    color = WHITE
    our_time, opp_time = 1000, 1000 # time in centi-seconds

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

            moves_remain = 40
            use = our_time/moves_remain
            # Let's follow the clock of our opponent
            if our_time >= 100 and opp_time >= 100:
                use *= our_time/opp_time
            
            start = time.time()
            for _ in searcher._search(pos, secs=use/100):
                # ply score time nodes pv
                ply = searcher.depth
                entry = searcher.tp_score.get((pos, ply, True))
                assert entry is not None
                score = '{}:{}'.format(entry.lower, entry.upper)
                #if score is None: score = '?'
                used = int((time.time() - start)*100 + .5)
                moves = pv(searcher, color, pos, include_scores=False)
                print('#{:>3} {:>13} {:>8} {:>8}  {}'.format(
                    ply, score, used, searcher.nodes, moves))
            m, s = searcher.tp_move.get(pos), entry.lower
            # We don't play well once we have detected our death
            if s <= -sunfish.MATE_VALUE:
                print('resign')
            else:
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
            our_time = int(smove.split()[1])
        
        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        elif any(smove.startswith(x) for x in ('xboard','post','random','hard','accepted','level')):
            pass

        else:
            print("Error (unkown command):", smove)

if __name__ == '__main__':
    main()
