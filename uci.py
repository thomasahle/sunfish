#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import importlib
import re
import sys
import time
import logging

import tools
import sunfish

from tools import WHITE, BLACK
from xboard import Unbuffered

def main():
    logging.basicConfig(filename='sunfish.log', level=logging.DEBUG)
    out = Unbuffered(sys.stdout)
    def output(line):
        print(line, file=out)
        logging.debug(line)
    pos = tools.parseFEN(tools.FEN_INITIAL)
    searcher = sunfish.Searcher()
    color = WHITE
    our_time, opp_time = 1000, 1000 # time in centi-seconds
    show_thinking = True

    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else: smove = input()

        logging.debug(f'>>> {smove} ')

        if smove == 'quit':
            break

        elif smove == 'uci':
            output('id name Sunfish')
            output('id author Thomas Ahle & Contributors')
            output('uciok')

        elif smove == 'isready':
            output('readyok')

        elif smove == 'ucinewgame':
            stack.append('position fen ' + tools.FEN_INITIAL)

        elif smove.startswith('position fen'):
            _, _, fen = smove.split(' ', 2)
            pos = tools.parseFEN(fen)
            color = WHITE if fen.split()[1] == 'w' else BLACK

        elif smove.startswith('position startpos'):
            params = smove.split(' ')
            pos = tools.parseFEN(tools.FEN_INITIAL)
            color = WHITE

            if params[2] == 'moves':
                for move in params[3:]:
                    pos = pos.move(tools.mparse(color, move))
                    color = 1 - color

        elif smove.startswith('go'):
            #  default options
            depth = 1000
            movetime = -1

            _, *params = smove.split(' ')
            for param, val in zip(*2*(iter(params),)):
                if param == 'depth':
                    depth = int(params[i])
                if param == 'movetime':
                    movetime = int(params[i])
                if param == 'wtime':
                    our_time = int(params[i])
                if param == 'btime':
                    opp_time = int(params[i])

            moves_remain = 40

            start = time.time()
            ponder = None
            for _ in searcher._search(pos):
                moves = tools.pv(searcher, pos, include_scores=False)

                if show_thinking:
                    entry = searcher.tp_score.get((pos, searcher.depth, True))
                    score = int(round((entry.lower + entry.upper)/2))
                    usedtime = int((time.time() - start) * 1000)
                    moves_str = moves if len(moves) < 15 else ''
                    output('info depth {} score cp {} time {} nodes {} pv {}'.format(searcher.depth, score, usedtime, searcher.nodes, moves_str))

                if len(moves) > 5:
                    ponder = moves[1]

                if movetime > 0 and (time.time() - start) * 1000 > movetime:
                    break

                if (time.time() - start) * 1000 > our_time/moves_remain:
                    break

                if searcher.depth >= depth:
                    break

            entry = searcher.tp_score.get((pos, searcher.depth, True))
            m, s = searcher.tp_move.get(pos), entry.lower
            # We only resign once we are mated.. That's never?
            if s == -sunfish.MATE_UPPER:
                output('resign')
            else:
                moves = moves.split(' ')
                if len(moves) > 1:
                    output(f'bestmove {moves[0]} ponder {moves[1]}')
                else:
                    output('bestmove ' + moves[0])

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()

