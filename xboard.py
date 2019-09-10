#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import importlib
import re
import sys
import time

import tools
import sunfish

from tools import WHITE, BLACK

if len(sys.argv) > 1:
    sunfish = importlib.import_module(sys.argv[1])

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
        sys.stderr.write(data)
        sys.stderr.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

from datetime import datetime
now = datetime.now()
path = 'sunfish-' + now.strftime("%d:%m:%Y-%H:%M:%S:%f") + '.log'
sys.stderr = open(path, 'a')

def main():
    pos = tools.parseFEN(tools.FEN_INITIAL)
    searcher = sunfish.Searcher()
    forced = False
    color = WHITE
    our_time, opp_time = 1000, 1000 # time in centi-seconds
    show_thinking = False
    options = {}
    history = []

    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else:
            smove = input()
            print('>>>', smove, file=sys.stderr)
            sys.stderr.flush() # For Python 2.7 support

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
            print('feature option="qs_limit -spin {} -100 1000"'.format(sunfish.QS_LIMIT))
            print('feature option="eval_roughness -spin {} 1 1000"'.format(sunfish.EVAL_ROUGHNESS))
            print('feature option="draw_test -spin {} 0 1"'.format(int(sunfish.DRAW_TEST)))
            print('feature done=1')

        elif smove == 'new':
            stack.append('setboard ' + tools.FEN_INITIAL)
            # Clear out the old searcher, including the tables
            searcher = sunfish.Searcher()
            del history[:]

        elif smove.startswith('setboard'):
            _, fen = smove.split(' ', 1)
            pos = tools.parseFEN(fen)
            color = WHITE if fen.split()[1] == 'w' else BLACK
            del history[:]

        elif smove == 'force':
            forced = True

        elif smove.startswith('option'):
            _, aeqb = smove.split(maxsplit=1)
            if '=' in aeqb:
                name, val = aeqb.split('=')
            else: name, val = aeqb, True
            if name == 'qs_limit':
                sunfish.QS_LIMIT = int(val)
            if name == 'eval_roughness':
                sunfish.EVAL_ROUGHNESS = int(val)
            if name == 'draw_test':
                sunfish.DRAW_TEST = bool(int(val))
            options[name] = val

        elif smove == 'go':
            forced = False

            moves_remain = 40
            use = our_time/moves_remain
            # Let's follow the clock of our opponent
            if our_time >= 100 and opp_time >= 100:
                use *= our_time/opp_time

            start = time.time()
            for ply, move, score in searcher.search(pos, history):
                entry = searcher.tp_score.get((pos, ply, True))
                score = int(round((entry.lower + entry.upper)/2))
                if show_thinking:
                    used = int((time.time() - start)*100 + .5)
                    moves = tools.pv(searcher, pos, include_scores=False)
                    seldepth = 0
                    nps = int(searcher.nodes / (time.time() - start) + .5)
                    print('{:>3} {:>8} {:>8} {:>13} {:>1} {:>4} \t{}'.format(
                        ply, score, used, searcher.nodes, seldepth, nps, moves))
                    print('# Hashfull: {:.3f}%; {} <= score < {}'.format(
                        len(searcher.tp_score)/sunfish.TABLE_SIZE*100, entry.lower, entry.upper))
                # If found mate, just stop
                if entry.lower >= sunfish.MATE_UPPER:
                    break
                if time.time() - start > use/100:
                    break
            # We sometimes make illegal moves when we're losing,
            # so it's safer to just resign.
            if score == -sunfish.MATE_UPPER:
                print('resign')
            else:
                print('move', tools.mrender(pos, move))
            pos = pos.move(move)
            history.append(pos)
            color = 1-color

        elif smove.startswith('ping'):
            _, N = smove.split()
            print('pong', N)

        elif smove.startswith('usermove'):
            _, smove = smove.split()
            m = tools.mparse(color, smove)
            pos = pos.move(m)
            history.append(pos)
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

        elif smove.startswith('post'):
            show_thinking = True

        elif smove.startswith('nopost'):
            show_thinking = False

        elif any(smove.startswith(x) for x in ('xboard','random','hard','accepted','level','easy','st','result','?')):
            print('# Ignoring command {}.'.format(smove))

        elif smove.startswith('reject'):
            _, feature = smove.split()[:2] # split(maxsplit=2) doesnt work in python2.7
            print('# Warning ({} rejected): Might not work as expected.'.format(feature))

        else:
            print('# Warning (unkown command): {}. Treating as move.'.format(smove))
            stack.append('usermove {}'.format(smove))

if __name__ == '__main__':
    main()

