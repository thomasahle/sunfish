#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import re
import sys
import time

import tools
import sunfish

from tools import WHITE, BLACK

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

def main():
    pos = tools.parseFEN(tools.FEN_INITIAL)
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
            stack.append('setboard ' + tools.FEN_INITIAL)

        elif smove.startswith('setboard'):
            _, fen = smove.split(' ', 1)
            pos = tools.parseFEN(fen)
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
                moves = tools.pv(searcher, pos, include_scores=False)
                print('#{:>3} {:>13} {:>8} {:>8}  {}'.format(
                    ply, score, used, searcher.nodes, moves))
            m, s = searcher.tp_move.get(pos), entry.lower
            # We only resign once we are mated.. That's never?
            if s == -sunfish.MATE_UPPER:
                print('resign')
            else:
                print('move', tools.mrender(pos, m))
                print('score before %d after %+d' % (pos.score, pos.value(m)))
                pos = pos.move(m)
                color = 1-color

        elif smove.startswith('ping'):
            _, N = smove.split()
            print('pong', N)

        elif smove.startswith('usermove'):
            _, smove = smove.split()
            m = tools.mparse(color, smove)
            pos = pos.move(m)
            color = 1-color
            if not forced:
                stack.append('go')

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])
        
        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        elif smove.startswith('perft'):
            start = time.time()
            for d in range(1,10):
                res = sum(1 for _ in tools.collect_tree_depth(tools.expand_position(pos), d))
                print('{:>8} {:>8}'.format(res, time.time()-start))

        elif any(smove.startswith(x) for x in ('xboard','post','random','hard','accepted','level')):
            pass

        else:
            print("Error (unkown command):", smove)

if __name__ == '__main__':
    main()
