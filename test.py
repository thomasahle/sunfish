#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import re
import time
import subprocess
import signal
import argparse
import importlib
import multiprocessing
import random

import sunfish
import xboard

###############################################################################
# Playing test
###############################################################################

def selfplay(maxn=200):
    """ Start a game sunfish vs. sunfish """
    pos = xboard.parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    for d in range(200):
        # Always print the board from the same direction
        board = pos.board if d % 2 == 0 else pos.rotate().board
        print(' '.join(board))

        m, _ = sunfish.search(pos, maxn)
        if m is None:
            print("Game over")
            break
        print("\nmove", xboard.mrender(d%2, pos, m))
        pos = pos.move(m)

def self_arena(version1, version2, games, maxn):
    pool = multiprocessing.Pool(8)
    instances = [(version1, version2, maxn, random.Random()) for _ in range(games)]
    for r in pool.imap_unordered(play, instances):
        print(r)

def play(version1_version2_maxn_rand):
    ''' returns 1 if fish1 won, 0 for draw and -1 otherwise '''
    version1, version2, maxn, rand = version1_version2_maxn_rand
    fish1 = importlib.import_module(version1)
    fish2 = importlib.import_module(version2)
    pos = xboard.parseFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    old = None
    tdelta = 0
    for d in range(200):
        nodes = maxn
        nodes *= (1+abs(tdelta)/5) if (tdelta<0)==(d%2==0) else 1
        nodes *= .75 + rand.random()/2
        before = time.time()
        m, score = (fish1 if d%2==0 else fish2).search(pos, nodes)
        tdelta += (time.time()-before)*(1 if d%2==0 else -1)
        if m is not None:
            pos = pos.move(m)
            # Test repetition draws
            if d%4==0:
                if pos.board == old:
                    return 0
                old = pos.board
        else:
            assert score < -1000
            return 1 if d%2 == 1 else -1
    print('200 moves reached')
    return 0

###############################################################################
# Test Xboard
###############################################################################

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, _signum, frame):
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
        # print('waiting for', regex)
        with timeout(20, '%s was never encountered'%regex):
            while True:
                line = fish.stdout.readline()
                # print("Saw lines", line)
                if re.search(regex, line):
                    return

    try:
        print('xboard', file=fish.stdin)
        print('protover 2', file=fish.stdin)
        waitFor(r'done\s*=\s*1')

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

def allperft(f, depth=4):
    lines = f.readlines()
    for d in range(1, depth+1):
        print("Going to depth %d" % d)
        for line in lines:
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
    for m in pos.gen_moves():
        pos1 = pos.move(m)
        # Make sure the move was legal
        if not any(pos1.value(m) >= sunfish.MATE_VALUE for m in pos1.gen_moves()):
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

def quickdraw(f, depth):
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

def quickmate(f, min_depth=1, draw=False):
    """ Similar to allmate, but uses the `bound` function directly to only
    search for moves that will win us the game """
    if draw:
        return quickdraw(f, min_depth)

    for line in f:
        line = line.strip()
        print(line)

        pos = xboard.parseFEN(line)
        for d in range(min_depth, 99):
            score = sunfish.bound(pos, sunfish.MATE_VALUE, d)
            if score >= sunfish.MATE_VALUE:
                break
            print('Score at depth {}: {}'.format(d, score))
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
    cankill = lambda p: any(p.board[b]=='k' for a,b in p.gen_moves())
    check = ''
    if cankill(pos1.rotate()):
        check = '+'
        if all(cankill(pos1.move(move1)) for move1 in pos1.gen_moves()):
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
    srcs = [a for a,b in pos.gen_moves() if pos.board[a] == p and b == j]
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
    for i, j in pos.gen_moves():
        # TODO: Maybe check for check here?
        csrc, cdst = sunfish.render(i), sunfish.render(j)
        if pos.board[i] == p and re.match(dst, cdst) and re.match(src, csrc):
            return (i, j)

def parseEPD(epd):
    parts = epd.strip('\n ;').replace('"','').split(maxsplit=6)
    fen = ' '.join(parts[:6])
    if len(parts) == 7:
        opts = dict(p.split(maxsplit=1) for p in parts[6].split(';'))
        return fen, opts
    return fen, {}

def findbest(f, times):
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
    for line in f:
        fen, opts = parseEPD(line)
        pos = xboard.parseFEN(fen)
        color = 0 if fen.split()[1] == 'w' else 1
        # am -> avoid move; bm -> best move
        am = parseSAN(pos,color,opts['am']) if 'am' in opts else None
        bm = parseSAN(pos,color,opts['bm']) if 'bm' in opts else None
        points = 0
        print(opts.get('id','unnamed'), end=' ', flush=True)
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

###############################################################################
# Actions
###############################################################################

def add_action(parser, f):
    class LambdaAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            f(namespace)
    parser.add_argument('_action', nargs='?',
        help=argparse.SUPPRESS, action=LambdaAction)

class PerftAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        allperft(namespace.file, namespace.depth)
class QuickMateAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        quickmate(namespace.file, namespace.depth)

def main():
    parser = argparse.ArgumentParser(
        description='Run various tests for speed and correctness of sunfish.')
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser('perft',
        help='tests for correctness and speed of move generator.')
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('file', type=argparse.FileType('r'),
        help='such as tests/queen.fen.')
    add_action(p, lambda n: allperft(n.file, n.depth))

    p = subparsers.add_parser('quickmate',
        help='uses the `bound` function directly to search for moves that will win us the game.')
    p.add_argument('file', type=argparse.FileType('r'),
        help='such as tests/mate{1,2,3}.fen or tests/stalemate2.fen.')
    p.add_argument('--mindepth', type=int, default=3, metavar='D',
        help='optional minimum number of plies to search for.')
    p.add_argument('--draw', action='store_true',
        help='search for draws rather than mates.')
    add_action(p, lambda n: quickmate(n.file, n.mindepth, n.draw))

    p = subparsers.add_parser('xboard',
        help='starts the xboard.py script and runs a few commands.')
    p.add_argument('--python', type=str, default='python',
        help='what version of python to use, e.g. python3, pypy.')
    add_action(p, lambda n: testxboard(n.python))

    p = subparsers.add_parser('selfplay',
        help='run a simple visual sunfish vs sunfish game.')
    p.add_argument('--nodes', type=int, default=200,
        help='number of nodes to search per move. Default=%(default)s.')
    add_action(p, lambda n: selfplay(n.nodes))

    p = subparsers.add_parser('arena',
        help='run a number of games between two sunfish versions.')
    p.add_argument('fish1', type=str, help='sunfish')
    p.add_argument('fish2', type=str, help='sunfish2')
    p.add_argument('--games', type=int, default=10,
        help='number of games to play. Default=%(default)s.')
    p.add_argument('--nodes', type=int, default=200,
        help='number of nodes to search per move. Default=%(default)s.')
    add_action(p, lambda n: self_arena(n.fish1, n.fish2, n.games, n.nodes))

    p = subparsers.add_parser('findbest',
        help='reports the best moves found at certain positions after certain intervals of time.')
    p.add_argument('file', type=argparse.FileType('r'),
        help='tests/ccr_one_hour_test.epd or tests/bratko_kopec_test.epd.')
    p.add_argument('--times', type=int, nargs='+',
        help='a list of times (in seconds) at which to report the best move. Default is %(default)s.',
        default=[15, 30, 60, 120])
    add_action(p, lambda n: findbest(n.file, n.times))

    _args, unknown = parser.parse_known_args()
    if unknown:
        print('Notice: unused arguments', ' '.join(unknown))
    if len(sys.argv) == 1:
        parser.print_help()

if __name__ == '__main__':
    main()
