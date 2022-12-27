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
import collections


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


###############################################################################
# Perft test
###############################################################################


class Perft(Command):
    name = "perft"
    help = "tests for correctness and speed of move generator."

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--depth", type=int)
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/queen.fen."
        )

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
    name = "bench"
    help = """Run through a fen file, search every position to a certain depth."""

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=100,
            help="Maximum plies at which to find the mate",
        )
        parser.add_argument(
            "--limit", type=int, default=100, help="Maximum positions to analyse"
        )

    @classmethod
    async def run(self, engine, args):
        # Run through
        limit = chess.engine.Limit(depth=args.depth)
        lines = args.file.readlines()
        lines = lines[: args.limit]

        total_nodes = 0
        start = time.time()

        pb = tqdm.tqdm(lines)
        for line in pb:
            board, _ = chess.Board.from_epd(line)
            with await engine.analysis(board, limit) as analysis:
                async for info in analysis:
                    pb.set_description(info_to_desc(info))
            total_nodes += info.get("nodes", 0)

        print(f"Total nodes: {total_nodes}.")
        print(f"Average knps: {round(total_nodes/(time.time() - start)/1000, 2)}.")


class Selfplay(Command):
    name = "self-play"
    help = "Play the engine a single game against itself, using increments"

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument("--time", type=int, default=3, help="White time in seconds")
        parser.add_argument(
            "--inc", type=int, default=1, help="Increment time in seconds"
        )

    @classmethod
    async def run(self, engine, args):
        board = chess.Board()
        wtime = btime = int(args.time)
        winc = binc = int(args.inc)
        while not board.is_game_over():
            print(board)
            start = time.time()
            result = await engine.play(
                board,
                chess.engine.Limit(
                    white_clock=wtime,
                    black_clock=btime,
                    white_inc=winc,
                    black_inc=binc,
                ),
            )
            if board.turn == chess.WHITE:
                wtime -= time.time() - start
                if wtime <= 0:
                    print("White lose on time.")
                    break
                wtime += winc
            else:
                btime -= time.time() - start
                if btime <= 0:
                    print("Black lose on time.")
                    break
                btime += binc
            if result.resigned:
                print("Resigned")
                break
            print(
                f"{board.fullmove_number}{'..' if board.turn == chess.BLACK else '.'}",
                board.san(result.move),
                f"wtime={round(wtime,1)}, btime={round(btime,1)}",
                # f"score={result.score}"
            )
            board.push(result.move)


###############################################################################
# Find mate test
###############################################################################


def info_to_desc(info):
    desc = []
    if "nodes" in info and "time" in info:
        # Add 1 to denominator, since time could be rounded to 0
        nps = info["nodes"] / (info["time"] + 1)
        desc.append(f"knps: {round(nps/1000, 2)}")
    if "depth" in info:
        desc.append(f"depth: {info['depth']}")
    return ", ".join(desc)


class Mate(Command):
    name = "mate"
    help = "Find the mates"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=100,
            help="Maximum plies at which to find the mate",
        )
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Use mate specific search in the engine, if supported",
        )

    @classmethod
    async def run(cls, engine, args):
        if args.quick:
            # Use "go mate" which allows engine to only look for mates
            limit = chess.engine.Limit(mate=args.depth)
        else:
            limit = chess.engine.Limit(depth=args.depth)
        total = 0
        success = 0
        lines = args.file.readlines()
        pb = tqdm.tqdm(lines)
        for line in pb:
            total += 1
            board, _ = chess.Board.from_epd(line)
            with await engine.analysis(board, limit) as analysis:
                async for info in analysis:
                    pb.set_description(info_to_desc(info))
                    if not 'score' in info:
                        continue
                    score = info["score"]
                    if score.is_mate() or score.relative.cp > 10000:
                        if args.debug:
                            print("Found it!")
                        success += 1
                        break
                else:
                    print("Failed on", line)
                    print("Result:", info)
        if not args.quiet:
            print(f"Succeeded in {success}/{total} cases.")


class Draw(Command):
    name = "draw"
    help = "Find the draws"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/stalemate2.fen."
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=100,
            help="Maximum plies at which to find the mate",
        )
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Use mate specific search in the engine, if supported",
        )

    @classmethod
    async def run(cls, engine, args):
        if args.quick:
             #This is not currently supported by any engines
            limit = chess.engine.Limit(draw=args.depth)
        else:
            limit = chess.engine.Limit(depth=args.depth)
        total, success = 0, 0
        cnt = collections.Counter()
        pb = tqdm.tqdm(args.file.readlines())
        for line in pb:
            total += 1
            board, _ = chess.Board.from_epd(line)
            with await engine.analysis(board, limit) as analysis:
                async for info in analysis:
                    pb.set_description(info_to_desc(info))
                    if not 'score' in info:
                        continue
                    score = info["score"]
                    # It should be draw here
                    if not score.is_mate() and score.relative.cp == 0:
                        success += 1
                        cnt[info["depth"]] += 1
                        break
                else:
                    if not args.quiet:
                        print("Failed on", line.strip())
                        print("Result:", info)
        print(f"Succeeded in {success}/{total} cases.")
        if not args.quiet:
            print('Depths:')
            for depth, c in cnt.most_common():
                print(f'{depth}: {c}')


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


def main():
    parser = argparse.ArgumentParser(
        description="Run various tests for speed and correctness of sunfish."
    )
    parser.add_argument("args", help="Command and arguments to run")
    parser.add_argument(
        "--debug", action="store_true", help="Write lots of extra stuff"
    )
    parser.add_argument("--quiet", action="store_true", help="Only write pass/fail")
    parser.add_argument(
        "--xboard", action="store_true", help="Use xboard protocol instead of uci"
    )
    subparsers = parser.add_subparsers()
    subparsers.required = True

    for cls in Command.__subclasses__():
        sub = subparsers.add_parser(cls.name, help=cls.help)
        cls.add_arguments(sub)
        sub.set_defaults(func=cls.run)

    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    async def run():
        if args.xboard:
            _, engine = await chess.engine.popen_xboard(args.args.split())
        else:
            _transport, engine = await chess.engine.popen_uci(args.args.split())
        try:
            await args.func(engine, args)
        finally:
            await engine.quit()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    start = time.time()
    asyncio.run(run())
    print(f"Took {round(time.time() - start, 2)} seconds.")


if __name__ == "__main__":
    main()
