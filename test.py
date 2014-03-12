#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import re
import time
import subprocess
import functools
import os
import signal

import sunfish
import xboard

###############################################################################
# Playing test
###############################################################################

def selfplay():
    """ Start a game sunfish vs. sunfish """
    pos = xboard.parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    for d in range(200):
        # Always print the board from the same direction
        board = pos.board if d % 2 == 0 else pos.rotate().board
        print(' '.join(board))

        m, _ = sunfish.search(pos, maxn=200)
        if m is None:
            print("Game over")
            break
        print("\nmove", xboard.mrender(d%2, pos, m))
        pos = pos.move(m)

###############################################################################
# Test Xboard
###############################################################################

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def testxboard(python='python3'):
    print('Xboard test \'%s\'' % python)
    fish = subprocess.Popen([python, '-u', 'xboard.py'],
       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
       universal_newlines=True)

    def waitFor(regex):
        with timeout(20, '%s was never encountered'%regex):
            while True:
                line = fish.stdout.readline()
                # print("Saw lines", line)
                if re.search(regex, line):
                    return

    try:
        print('xboard', file=fish.stdin)
        print('protover 2', file=fish.stdin)
        waitFor('done\s*=\s*1')

        print('usermove e2e4', file=fish.stdin)
        waitFor('move ')

        print('setboard rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1', file=fish.stdin)
        print('usermove e7e5', file=fish.stdin)
        waitFor('move ')

        print('quit', file=fish.stdin)
        with timeout(5, 'quit did not terminate sunfish'):
            fish.wait()
    finally:
        if fish.poll() is None:
            fish.kill()

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

                pos, score = xboard.parseFEN(parts[0]), int(parts[d])
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
        if not any(pos1.value(m) >= sunfish.MATE_VALUE for m in pos1.genMoves()):
            sub = perft(pos1, depth-1, False)
            if divide:
                print(" "*depth+xboard.mrender(m), sub)
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

            pos = xboard.parseFEN(line)
            _, score = sunfish.search(pos, maxn=1e9)
            if score < sunfish.MATE_VALUE:
                print("Unable to find mate. Only got score = %d" % score)
                break

def quickdraw(path, depth):
    with open(path) as f:
        for line in f:
            line = line.strip()
            print(line)

            pos = xboard.parseFEN(line)
            for d in range(depth, 99):
                s0 = sunfish.bound(pos, 0, d)
                s1 = sunfish.bound(pos, 1, d)
                if s0 >= 0 and s1 < 1:
                    break
                print(d, s0, s1, xboard.pv(0, pos))
            else:
                print("Fail: Unable to find draw!")
                return

def quickmate(path, depth):
    """ Similar to allmate, but uses the `bound` function directly to only
    search for moves that will win us the game """
    with open(path) as f:
        for line in f:
            line = line.strip()
            print(line)

            pos = xboard.parseFEN(line)
            for d in range(depth, 99):
                score = sunfish.bound(pos, sunfish.MATE_VALUE, d)
                if score >= sunfish.MATE_VALUE:
                    break
                print(d, score)
            else:
                print("Unable to find mate. Only got score = %d" % score)
                return

###############################################################################
# Best move test
###############################################################################

def renderSAN(pos, move):
    # TODO: How do we simply make this work for black as well?
    i, j = move
    csrc, cdst = sunfish.render(i), sunfish.render(j)
    # Check
    pos1 = pos.move(move)
    cankill = lambda p: any(p.board[b]=='k' for a,b in p.genMoves())
    check = ''
    if cankill(pos1.rotate()):
        check = '+'
        if all(cankill(pos1.move(move1)) for move1 in pos1.genMoves()):
            check = '#'
    # Castling
    if pos.board[i] == 'K' and csrc == 'e1' and cdst in ('c1','g1'):
        if cdst == 'c1':
            return 'O-O-O' + check
        return 'O-O' + check
    # Pawn moves
    if pos.board[i] == 'P':
        pro = '=Q' if cdst[1] == '8' else ''
        cap = csrc[0] + 'x' if pos.board[j] != '.' or j == pos.ep else ''
        return cap + cdst + pro + check
    # Normal moves
    p = pos.board[i]
    srcs = [a for a,b in pos.genMoves() if pos.board[a] == p and b == j]
    # TODO: We can often get away with just sending the rank or file here.
    src = csrc if len(srcs) > 1 else ''
    cap = 'x' if pos.board[j] != '.' else ''
    return p + src + cap + cdst + check

def parseSAN(pos, color, msan):
    # Normal moves
    normal = re.match('([KQRBN])([a-h])?([1-8])?x?([a-h][1-8])', msan)
    if normal:
        p, fil, rank, dst = normal.groups()
        src = (fil or '[a-h]')+(rank or '[1-8]')
    # Pawn moves
    pawn = re.match('([a-h])?x?([a-h][1-8])', msan)
    if pawn:
        p, (fil, dst) = 'P', pawn.groups()
        src = (fil or '[a-h]')+'[1-8]'
    # Castling
    if msan == "O-O-O":
        p, src, dst = 'K', 'e1|d1', 'c1|f1'
    if msan == "O-O":
        p, src, dst = 'K', 'e1|d1', 'g1|b1'
    # Find possible match
    for i, j in pos.genMoves():
        # TODO: Maybe check for check here?
        csrc, cdst = sunfish.render(i), sunfish.render(j)
        if pos.board[i] == p and re.match(dst, cdst) and re.match(src, csrc):
            return (i, j)

def parseEPD(epd):
    parts = epd.strip('\n ;').replace('"','').split(maxsplit=6)
    fen = ' '.join(parts[:6])
    opts = dict(p.split(maxsplit=1) for p in parts[6].split(';'))
    return fen, opts

def findbest(path, times):
    print('Calibrating search speed...')
    pos = xboard.parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    CAL_NODES = 10000
    start = time.time()
    _ = sunfish.search(pos, CAL_NODES)
    factor = CAL_NODES/(time.time()-start)

    print('Running benchmark with %.1f nodes per second...' % factor)
    print('-'*60)
    totalpoints = 0
    totaltests = 0
    with open(path) as f:
        for k, line in enumerate(f):
            fen, opts = parseEPD(line)
            pos = xboard.parseFEN(fen)
            color = 0 if fen.split()[1] == 'w' else 1
            # am -> avoid move; bm -> best move
            am = parseSAN(pos,color,opts['am']) if 'am' in opts else None
            bm = parseSAN(pos,color,opts['bm']) if 'bm' in opts else None
            points = 0
            print(opts['id'], end=' ', flush=True)
            for t in times:
                move, _ = sunfish.search(pos, factor*t)
                mark = renderSAN(pos, move)
                if am and move != am or bm and move == bm:
                    mark += '(1)'
                    points += 1
                    totalpoints += 1
                else:
                    mark += '(0)'
                print(mark, end=' ', flush=True)
                totaltests + 1
            print(points)
    print('-'*60)
    print('Total Points: %d/%d', totalpoints, totaltests)

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input

if __name__ == '__main__':
    allperft('tests/queen.fen', depth=3)
    quickmate('tests/mate1.fen', 3)
    quickmate('tests/mate2.fen', 5)
    quickmate('tests/mate3.fen', 7)
    testxboard('python')
    testxboard('python3')
    testxboard('pypy')
    # findbest('tests/ccr_one_hour_test.epd', [15, 30, 60, 120])
    # findbest('tests/bratko_kopec_test.epd', [15, 30, 60, 120])
    # quickdraw('tests/stalemate2.fen', 3)
    # selfplay()
