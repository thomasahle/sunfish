#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chess.engine
import json
import argparse
import importlib.util
import sys
import os
import random
import time
import sys
import asyncio
import pathlib
import logging
import math
import textwrap


parser = argparse.ArgumentParser()
parser.add_argument("-cmd", nargs="?", help="Command of (UCI) engine to use")
parser.add_argument("-conf", nargs="?", help="Location of engines.json file to use")
parser.add_argument("-name", nargs="?", help="Name of engine to use from conf")
parser.add_argument("-selfplay", action="store_true", help="Play against itself")
parser.add_argument("-debug", action="store_true", help="Enable debugging of engine")
parser.add_argument("-movetime", type=int, default=0, help="Movetime in ms")
parser.add_argument("-nodes", type=int, default=0, help="Maximum nodes")
parser.add_argument(
    "-pvs",
    nargs="?",
    const=3,
    default=0,
    type=int,
    help="Show Principal Variations (when mcts)",
)
parser.add_argument(
    "-fen",
    help="Start from given position",
    default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
)


def python_launcher(argv):
    # Windows can't execute .py files directly (issue #113), so run them
    # through the current Python interpreter.
    if os.name == "nt" and argv and argv[0].endswith(".py"):
        return [sys.executable] + argv
    return argv


async def load_engine_from_cmd(cmd, debug=False):
    _, engine = await chess.engine.popen_uci(python_launcher(cmd.split()))
    if hasattr(engine, "debug"):
        engine.debug(debug)
    return engine


async def load_engine_from_conf(engine_args, name, debug=False):
    args = next(a for a in engine_args if a["name"] == name)
    curdir = str(pathlib.Path(__file__).parent)
    popen_args = {}
    if "workingDirectory" in args:
        popen_args["cwd"] = args["workingDirectory"].replace("$FILE", curdir)
    cmd = args["command"].split()
    if cmd[0] == "$PYTHON":
        cmd[0] = sys.executable
    if args["protocol"] == "uci":
        _, engine = await chess.engine.popen_uci(cmd, **popen_args)
    elif args["protocol"] == "xboard":
        _, engine = await chess.engine.popen_xboard(cmd, **popen_args)
    if hasattr(engine, "debug"):
        engine.debug(debug)
    await engine.configure(
        {opt["name"]: opt["value"] for opt in args.get("options", [])}
    )
    return engine


def get_user_move(board):
    # Get well-formated move
    move = None
    while move is None:
        san_option = random.choice([board.san(m) for m in board.legal_moves])
        uci_option = random.choice([m.uci() for m in board.legal_moves])
        uci = input(f"   Your move (e.g. {san_option} or {uci_option}): ")
        if uci in ("quit", "exit"):
            return None

        for parse in (board.parse_san, chess.Move.from_uci):
            try:
                move = parse(uci)
            except ValueError:
                pass

    # Check legality
    if move not in board.legal_moves:
        print("   Illegal move.")
        return get_user_move(board)

    return move


def get_user_color():
    color = ""
    while color not in ("white", "black"):
        color = input("   Do you want to be white or black? ")
    return chess.WHITE if color == "white" else chess.BLACK


def print_unicode_board(board, perspective=chess.WHITE):
    """Prints the position from a given perspective.

    Both sides use the FILLED glyphs, told apart by foreground color -
    the outline white glyphs read as black pieces on most terminal
    fonts, which made the sides nearly indistinguishable."""
    # Mid-tone squares so BOTH piece colors clear them: pale squares wash
    # out the white pieces, near-black squares swallow the black ones.
    print()
    light, dark = 137, 94                        # deep tan / dark walnut
    hl_light, hl_dark = 143, 101                 # two-tone last-move olive
    label, reset = "\x1b[38;5;245m", "\x1b[0m"
    last = board.move_stack[-1] if board.move_stack else None
    ours, theirs = captured_pieces(board, perspective)
    ranks = list(range(8) if perspective == chess.BLACK else range(7, -1, -1))
    for i, r in enumerate(ranks):
        line = [f"{label}{r + 1}{reset} "]
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            sq = 8 * r + c
            is_light = (r + c) % 2 == 1
            if last and sq in (last.to_square, last.from_square):
                bg = hl_light if is_light else hl_dark
            else:
                bg = light if is_light else dark
            piece = board.piece_at(sq)
            if piece:
                fg = 231 if piece.color == chess.WHITE else 16
                glyph = f"\x1b[38;5;{fg};48;5;{bg};1m {chess.UNICODE_PIECE_SYMBOLS[piece.symbol().lower()]} "
            else:
                glyph = f"\x1b[48;5;{bg}m   "
            line.append(glyph)
        # Captured material flanks the board: what the top player has
        # won beside the top rank, the bottom player's beside the bottom.
        # Each tray holds one side's lost pieces and wears that side's
        # piece color, on the terminal's own background.
        tray = theirs if i == 0 else ours if i == 7 else ""
        if tray:
            lost_by = perspective if i == 0 else not perspective
            tray_fg = 231 if lost_by == chess.WHITE else 16
            tray = f"  \x1b[38;5;{tray_fg};1m{tray}{reset}"
        print("  " + "".join(line) + reset + tray)
    files = "abcdefgh" if perspective == chess.WHITE else "hgfedcba"
    print(f"    {label}" + " ".join(f" {f}" for f in files) + f"{reset}\n")


def captured_pieces(board, perspective):
    """Each player's net captures after pairwise cancellation, as glyph
    strings: (bottom player's tray, top player's tray). A surplus of a
    type for one side means the OTHER side lost those pieces - they sit
    in the first side's tray."""
    order = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
    ours, theirs = [], []
    for pt in order:
        net = len(board.pieces(pt, perspective)) - len(board.pieces(pt, not perspective))
        glyph = chess.UNICODE_PIECE_SYMBOLS[chess.piece_symbol(pt)]
        (ours if net > 0 else theirs).extend(glyph * abs(net))
    return "".join(ours), "".join(theirs)


async def get_engine_move(engine, board, limit, game_id, multipv, debug=False):
    # XBoard engine doesn't support multipv, and there python-chess doesn't support
    # getting the first PV while playing a game.
    if isinstance(engine, chess.engine.XBoardProtocol):
        play_result = await engine.play(board, limit, game=game_id)
        return play_result.move

    multipv = min(multipv, board.legal_moves.count())
    with await engine.analysis(
        board, limit, game=game_id, info=chess.engine.INFO_ALL, multipv=multipv or None
    ) as analysis:

        infos = [None for _ in range(multipv)]
        printed = 0
        async for new_info in analysis:
            # If multipv = 0 it means we don't want them at all,
            # but uci requires MultiPV to be at least 1.
            if multipv and "multipv" in new_info:
                infos[new_info["multipv"] - 1] = new_info

            # Parse optional arguments into a dict
            if debug and "string" in new_info:
                print(new_info["string"])

            if not debug and all(infos) and "score" in analysis.info:
                # The PV wraps to a varying number of rows, so the in-place
                # rewrite erases exactly what the previous update printed.
                print("\u001b[1A\u001b[K" * printed, end="")
                printed = 0

                info = analysis.info
                score = info["score"].relative
                # Pawns with a sign, like every GUI, not raw centipawns.
                shown = (
                    f"{score.score() / 100:+.2f}"
                    if score.score() is not None
                    else f"mate in {abs(score.mate())}"
                    + (" (for them)" if score.mate() < 0 else "")
                )

                def human(n):
                    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else str(n)

                dim, bold, reset = "\x1b[38;5;245m", "\x1b[1m", "\x1b[0m"
                parts = [f"{bold}{shown}{reset}"]
                if "nodes" in info:
                    parts.append(f'{human(info["nodes"])} nodes')
                if "nps" in info:
                    parts.append(f'{human(info["nps"])}/s')
                parts.append(f'{float(info.get("time", 0)):.1f}s')
                print("   " + f"{dim} · {reset}".join(parts))
                printed += 1
                # PV below the stats, wrapped to keep the block's width
                # roughly even (a single long line flickers on rewrite).
                if "pv" in info:
                    for row in textwrap.wrap(board.variation_san(info["pv"]), 40):
                        print(f"   {dim}{row}{reset}")
                        printed += 1

                for info in infos:
                    if "pv" in info:
                        variation = board.variation_san(info["pv"][:10])
                    else:
                        variation = ""

                    if "score" in info:
                        score = info["score"].relative
                        score = (
                            math.tanh(score.score() / 600)
                            if score.score() is not None
                            else score.mate()
                        )
                        key, *val = info.get("string", "").split()
                        if key == "pv_nodes":
                            nodes = int(val[0])
                            rel = nodes / analysis.info["nodes"]
                            score_rel = f"({score:.2f}, {rel*100:.0f}%)"
                        else:
                            score_rel = f"({score:.2f})"
                    else:
                        score_rel = ""

                    # Something about N
                    print(f'{info["multipv"]}: {score_rel} {variation}')
                    printed += 1

        return analysis.info["pv"][0]


async def play(engine, board, selfplay, pvs, time_limit, debug=False):
    if not selfplay:
        user_color = get_user_color()
    else:
        user_color = chess.WHITE

    if not board:
        board = chess.Board()

    game_id = random.random()

    while not board.is_game_over():
        print_unicode_board(board, perspective=user_color)
        if not selfplay and user_color == board.turn:
            move = get_user_move(board)
            if move is None:
                return
        else:
            move = await get_engine_move(
                engine, board, time_limit, game_id, pvs, debug=debug
            )
            print(f"\n   My move: {board.san(move)}")
        board.push(move)

    # Print status
    print_unicode_board(board, perspective=user_color)
    print("Result:", board.result())


async def main():
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    if not args.conf:
        if args.cmd:
            engine = await load_engine_from_cmd(args.cmd, debug=args.debug)
        elif importlib.util.find_spec("sunfish"):
            # Installed wheel (sunfish-play): run the bundled engine.
            engine = await load_engine_from_cmd(
                f"{sys.executable} -m sunfish", debug=args.debug)
        else:
            path = pathlib.Path(__file__).parent / "engines.json"
            if not path.is_file():
                print("Unable to locate engines.json file.")
                return
            conf = json.load(path.open())
    else:
        if args.conf:
            conf = json.load(open(args.conf))
        else:
            path = pathlib.Path(__file__).parent / "engines.json"
            if not path.is_file():
                print("Unable to locate engines.json file.")
                return
            conf = json.load(path.open())
        engine = await load_engine_from_conf(conf, args.name, debug=args.debug)

    if "author" in engine.id:
        print(f"   Playing against {engine.id['name']} by {engine.id['author']}.")
    else:
        print(f"   Playing against {engine.id['name']}.")

    board = chess.Board(args.fen)

    if args.movetime:
        limit = chess.engine.Limit(time=args.movetime / 1000)
    elif args.nodes:
        limit = chess.engine.Limit(nodes=args.nodes)
    else:
        limit = chess.engine.Limit(
            white_clock=30, black_clock=30, white_inc=1, black_inc=1
        )

    # Ctrl+C tears the engine down mid-search: the SIGTERM'd process leaves
    # an EngineTerminatedError on a future nobody will ever await, and the
    # loop's default handler prints it as scary noise after "Goodbye!".
    # Only the UNRETRIEVED copy is filtered - an engine dying mid-game still
    # raises loudly out of the awaited play() call.
    def quiet_teardown(loop, context):
        if not isinstance(context.get("exception"), chess.engine.EngineTerminatedError):
            loop.default_exception_handler(context)
    asyncio.get_running_loop().set_exception_handler(quiet_teardown)

    try:
        await play(
            engine,
            board,
            selfplay=args.selfplay,
            pvs=args.pvs,
            time_limit=limit,
            debug=args.debug,
        )
    finally:
        print("\nGoodbye!")
        try:
            await engine.quit()
        except chess.engine.EngineError:
            pass  # already dead (Ctrl+C killed the process group)


def run():
    # Console entry point (pip install 'sunfish[play]' && sunfish-play).
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
