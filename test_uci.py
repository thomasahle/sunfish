#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import subprocess
import signal
import argparse
import importlib
import itertools
import random
import warnings
import chess
import chess.engine
import pathlib
import tqdm
import asyncio

import tools


root = pathlib.Path(__file__).parent

class Command:
    @classmethod
    def add_arguments(cls, parser):
        raise NotImplementedError

    @classmethod
    async def run(cls, args):
        raise NotImplementedError

###############################################################################
# Test uci
###############################################################################

async def new_engine(args, debug=False):
    transport, engine = await chess.engine.popen_uci(args.split())
    return engine


###############################################################################
# Perft test
###############################################################################


class Perft(Command):
    name="perft"
    help="tests for correctness and speed of move generator."

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--depth", type=int)
        parser.add_argument("file", type=argparse.FileType("r"), help="such as tests/queen.fen.")

    @classmethod
    async def run(cls, args):
        import gc

        lines = f.readlines()
        for d in range(1, depth + 1):
            if verbose:
                print("Going to depth {}/{}".format(d, depth))
            for line in lines:
                parts = line.split(";")
                if len(parts) <= d:
                    continue
                if verbose:
                    print(parts[0])

                pos, score = tools.parseFEN(parts[0]), int(parts[d])
                res = sum(
                    1 for _ in tools.collect_tree_depth(tools.expand_position(pos), d)
                )
                if res != score:
                    print("=========================================")
                    print("ERROR at depth %d. Gave %d rather than %d" % (d, res, score))
                    print("=========================================")
                    print(tools.renderFEN(pos, 0))
                    for move in pos.gen_moves():
                        split = sum(
                            1
                            for _ in tools.collect_tree_depth(
                                tools.expand_position(pos.move(move)), 1
                            )
                        )
                        print("{}: {}".format(tools.mrender(pos, move), split))
                    return False
            if verbose:
                print("")


class Bench(Command):
    name="bench"
    help="""Run through a fen file, search every position to a certain depth."""

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument("--depth", type=int, default=100, help="Maximum plies at which to find the mate")
        parser.add_argument("--limit", type=int, default=100, help="Maximum positions to analyse")

    @classmethod
    async def run(self, engine, args):
        # Run through
        limit = chess.engine.Limit(depth=args.depth)
        lines = args.file.readlines()
        lines = lines[:args.limit]

        total_nodes = 0
        start = time.time()

        pb = tqdm.tqdm(lines)
        for line in pb:
            board, _ = chess.Board.from_epd(line)
            with await engine.analysis(board, limit, info=chess.engine.INFO_ALL) as analysis:
                async for info in analysis:
                    desc = []
                    if 'nodes' in info and 'time' in info:
                        nps = info['nodes'] / info['time']
                        desc.append(f"knps: {round(nps/1000, 2)}")
                    if 'depth' in info:
                        desc.append(f"depth: {info['depth']}")
                    pb.set_description(', '.join(desc))
            total_nodes += info.get('nodes', 0)

        print(f'Total nodes: {total_nodes}.')
        print(f'Average knps: {round(total_nodes/(time.time() - start)/1000, 2)}.')


###############################################################################
# Find mate test
###############################################################################


class Mates(Command):
    name='mate'
    help = "Find the mates"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument("--depth", type=int, default=100, help="Maximum plies at which to find the mate")
        parser.add_argument("--quick", action="store_true", help="Use mate specific search in the engine, if supported")

    @classmethod
    async def run(cls, args):
        if args.quick:
            # Use "go mate" which allows engine to only look for mates
            limit = chess.engine.Limit(mate=args.depth)
        else:
            limit = chess.engine.Limit(depth=args.depth)
        total = 0
        success = 0
        engine = new_engine(args.args, args.debug)
        try:
            lines = args.file.readlines()
            for line in tqdm.tqdm(lines):
                total += 1
                board, _ = chess.Board.from_epd(line)
                play_result = engine.analyse(board, limit)
                score = play_result['score']
                if score.is_mate() or score.relative.cp > 10000:
                    # TODO: Something about if we found mate, but not as fast as we'd like?
                    success += 1
                    continue
                print('Failed on', line)
                print('Result:', play_result)
        finally:
            engine.close()
        print(f'Succeeded in {success}/{total} cases.')


def quickdraw(f, depth):
    k, n = 0, 0
    for line in f:
        line = line.strip()
        print(line)
        n += 1

        pos = tools.parseFEN(line)
        searcher = sunfish.Searcher()
        for d in range(depth, 10):
            s0 = searcher.bound(pos, 0, d, root=True)
            s1 = searcher.bound(pos, 1, d, root=True)
            if s0 >= 0 and s1 < 1:
                k += 1
                break
            else:
                print("depth {}, s0 {}, s1 {}".format(d, s0, s1))
            # print(d, s0, s1, tools.pv(0, pos))
        else:
            print("Fail: Unable to find draw!")
            # return
        print(tools.pv(searcher, pos, False))
    print("Found {}/{} draws".format(k, n))


###############################################################################
# Best move test
###############################################################################


def findbest(f, times):
    pos = tools.parseFEN(tools.FEN_INITIAL)
    searcher = sunfish.Searcher()

    print("Printing best move after seconds", times)
    print("-" * 60)
    totalpoints = 0
    totaltests = 0
    for line in f:
        fen, opts = tools.parseEPD(line, opt_dict=True)
        if type(opts) != dict or ("am" not in opts and "bm" not in opts):
            print("Line didn't have am/bm in opts", line, opts)
            continue
        pos = tools.parseFEN(fen)
        # am -> avoid move; bm -> best move
        am = tools.parseSAN(pos, opts["am"]) if "am" in opts else None
        bm = tools.parseSAN(pos, opts["bm"]) if "bm" in opts else None
        print("Looking for am/bm", opts.get("am"), opts.get("bm"))
        points = 0
        print(opts.get("id", "unnamed"), end=" ", flush=True)
        for t in times:
            move, _, _ = tools.search(searcher, pos, t)
            mark = tools.renderSAN(pos, move)
            if am and move != am or bm and move == bm:
                mark += "(1)"
                points += 1
            else:
                mark += "(0)"
            print(mark, end=" ", flush=True)
            totaltests += 1
        print(points)
        totalpoints += points
    print("-" * 60)
    print("Total Points: %d/%d", totalpoints, totaltests)


###############################################################################
# Actions
###############################################################################


def add_action(parser, f):
    class LambdaAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            f(namespace)

    parser.add_argument(
        "_action", nargs="?", help=argparse.SUPPRESS, action=LambdaAction
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run various tests for speed and correctness of sunfish."
    )
    parser.add_argument('args', help="Command and arguments to run")
    parser.add_argument('--debug', action='store_true')
    subparsers = parser.add_subparsers()

    for cls in Command.__subclasses__():
        sub = subparsers.add_parser(cls.name, help=cls.help)
        cls.add_arguments(sub)
        sub.set_defaults(func=cls.run)

    args = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    async def run():
        engine = await new_engine(args.args, args.debug)
        try:
            await args.func(engine, args)
        finally:
            await engine.quit()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    start = time.time()
    asyncio.run(run())
    print(f'Took {round(time.time() - start, 2)} seconds.')


if __name__ == "__main__":
    main()
