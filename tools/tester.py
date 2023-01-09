#!/usr/bin/env python3
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


random.seed(42)


class Command:
    @classmethod
    def add_arguments(cls, parser):
        raise NotImplementedError

    @classmethod
    async def run(cls, engine, args):
        raise NotImplementedError


###############################################################################
# Perft test
###############################################################################

from chess.engine import BaseCommand, UciProtocol


async def uci_perft(engine, depth):
    class UciPerftCommand(BaseCommand[UciProtocol, None]):
        def __init__(self, engine: UciProtocol):
            super().__init__(engine)
            self.moves = []

        def start(self, engine: UciProtocol) -> None:
            engine.send_line(f"go perft {depth}")

        def line_received(self, engine: UciProtocol, line: str) -> None:
            match = re.match("(\w+): (\d+)", line)
            if match:
                move = chess.Move.from_uci(match.group(1))
                cnt = int(match.group(2))
                self.moves.append((move, cnt))

            match = re.match("Nodes searched: (\d+)", line)
            if match:
                self.result.set_result(self.moves)
                self.set_finished()

    return await engine.communicate(UciPerftCommand)


class Perft(Command):
    name = "perft"
    help = "tests for correctness and speed of move generator."

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/queen.fen."
        )
        parser.add_argument("--depth", type=int, default=3)

    @classmethod
    async def run(cls, engine, args):
        lines = args.file.readlines()
        for d in range(1, args.depth + 1):
            if not args.quiet:
                print(f"Going to depth {d}/{args.depth}")

            for line in tqdm.tqdm(lines):
                board, opts = chess.Board.from_epd(line)
                engine._position(board)
                moves = await uci_perft(engine, d)

                cnt = sum(c for m, c in moves)
                opt_cnt = int(opts[f"D{d}"])

                # TODO: Also test that the _number_ of different moves is correct

                if cnt != opt_cnt:
                    print("=========================================")
                    print(f"ERROR at depth {d}. Gave {cnt} rather than {opt_cnt}")
                    print("=========================================")
                    print(board)
                    for m, c in moves:
                        print(m, c)
                    break


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
            "--limit", type=int, default=10000, help="Maximum positions to analyse"
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


###############################################################################
# Self-play
###############################################################################


class SelfPlay(Command):
    name = "self-play"
    help = "make sure the engine can complete a game without crashing."

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--time", type=int, default=4000, help="start time in ms")
        parser.add_argument("--inc", type=int, default=100, help="increment in ms")

    @classmethod
    async def run(cls, engine, args):
        wtime = btime = args.time / 1000
        inc = args.inc / 1000
        board = chess.Board()
        with tqdm.tqdm(total=100) as pbar:
            while not board.is_game_over():
                limit = chess.engine.Limit(white_clock=wtime, black_clock=btime, white_inc=inc, black_inc=inc)

                start = time.time()
                result = await engine.play(board, limit)
                elasped = time.time() - start

                if board.turn == chess.WHITE:
                    wtime -= elasped - inc
                else:
                    btime -= elasped - inc

                board.push(result.move)
                pbar.update(1)
            pbar.update(100 - pbar.n)


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
    if "score" in info:
        #:wprint(dir(info['score']))
        desc.append(f"score: {info['score'].pov(chess.WHITE).cp/100:.1f}")
    return ", ".join(desc)


def add_limit_argument(parser):
    parser.add_argument(
        "--depth",
        dest="limit_depth",
        type=int,
        default=0,
        help="Maximum plies at which to find the move",
    )
    parser.add_argument(
        "--mate-depth",
        dest="limit_mate",
        type=int,
        default=0,
        help="Maximum plies at which to find the mate",
    )
    parser.add_argument(
        "--movetime",
        dest="limit_movetime",
        type=int,
        default=100,
        help="Movetime in ms",
    )


def get_limit(args):
    if args.limit_depth:
        return chess.engine.Limit(depth=args.limit_depth)
    elif args.limit_mate:
        return chess.engine.Limit(mate=args.limit_mate)
    elif args.limit_movetime:
        return chess.engine.Limit(time=args.limit_movetime / 1000)


class Mate(Command):
    name = "mate"
    help = "Find the mates"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10000,
            help="Take only this many lines from the file",
        )
        add_limit_argument(parser)

    @classmethod
    async def run(cls, engine, args):
        limit = get_limit(args)
        total = 0
        success = 0
        lines = args.file.readlines()
        lines = lines[: args.limit]
        pb = tqdm.tqdm(lines)
        for line in pb:
            board, _ = chess.Board.from_epd(line)
            with await engine.analysis(board, limit) as analysis:
                async for info in analysis:
                    pb.set_description(info_to_desc(info))
                    if not "score" in info:
                        continue
                    score = info["score"]
                    if score.is_mate() or score.relative.cp > 10000:
                        if "pv" in info and info["pv"]:
                            b = board.copy()
                            for move in info["pv"]:
                                b.push(move)
                            if not b.is_game_over():
                                if args.debug:
                                    print("Got mate score, but PV is not mate...")
                                continue
                        if args.debug:
                            print("Found it!")
                        success += 1
                        break
                else:
                    if not args.quiet:
                        print("Failed on", line)
                        print("Result:", info)
        print(f"Succeeded in {success}/{len(lines)} cases.")


class Draw(Command):
    name = "draw"
    help = "Find the draws"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/stalemate2.fen."
        )
        add_limit_argument(parser)

    @classmethod
    async def run(cls, engine, args):
        limit = get_limit(args)
        total, success = 0, 0
        cnt = collections.Counter()
        pb = tqdm.tqdm(args.file.readlines())
        for line in pb:
            total += 1
            board, _ = chess.Board.from_epd(line)
            with await engine.analysis(board, limit) as analysis:
                async for info in analysis:
                    pb.set_description(info_to_desc(info))
                    if not "score" in info:
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
                        pass
        print(f"Succeeded in {success}/{total} cases.")
        if not args.quiet:
            print("Depths:")
            for depth, c in cnt.most_common():
                print(f"{depth}: {c}")


###############################################################################
# Best move test
###############################################################################


class Best(Command):
    name = "best"
    help = "Find the best move"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as bratko_kopec_test.epd."
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10000,
            help="Take only this many lines from the file",
        )
        add_limit_argument(parser)

    @classmethod
    async def run(cls, engine, args):
        limit = get_limit(args)
        points, total = 0, 0
        lines = args.file.readlines()
        # random.shuffle(lines)
        lines = lines[: args.limit]
        for line in (pb := tqdm.tqdm(lines)):
            board, opts = chess.Board.from_epd(line)
            if "pv" in opts:
                for move in opts["pv"]:
                    board.push(move)
            # When using a pv I need to supply bm/am using a comment, because
            # otherwise python-chess won't parse it
            if "c0" in opts:
                # The comment format is expected to be like 'am f1f2; bm f2f3'
                for key, val in re.findall("(\w+) (\w+)", opts["c0"]):
                    opts[key] = [chess.Move.from_uci(val)]
            if "am" not in opts and "bm" not in opts:
                if not args.quiet:
                    print("Line didn't have am/bm in opts", line, opts)
                continue
            # am -> avoid move; bm -> best move
            pb.set_description(opts.get("id", ""))
            result = await engine.play(board, limit, info=chess.engine.INFO_SCORE)
            errors = []
            if "bm" in opts:
                total += 1
                if result.move in opts["bm"]:
                    points += 1
                else:
                    errors.append(f'Gave move {result.move} rather than {opts["bm"]}')
            if "am" in opts:
                total += 1
                if result.move not in opts["am"]:
                    points += 1
                else:
                    errors.append(f'Gave move {result.move} which is in {opts["am"]}')
            if not args.quiet and errors:
                print("Failed on", line.strip())
                for er in errors:
                    print(er)
                print("Full result:", result)
                print()
            pb.set_postfix(acc=points / total)
        print(f"Succeeded in {points}/{total} cases.")


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
