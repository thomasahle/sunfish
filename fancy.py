import chess.engine
import json
import argparse
import random
import time
import sys
import asyncio
import pathlib
import logging
import math


parser = argparse.ArgumentParser()
parser.add_argument('conf', default='engines.json', nargs='?',
                    help='Location of engines.json file to use')
parser.add_argument('name', default='sunfish', nargs='?', help='Name of engine to use')
parser.add_argument('-selfplay', action='store_true', help='Play against itself')
parser.add_argument('-debug', action='store_true', help='Enable debugging of engine')
parser.add_argument('-movetime', type=int, default=0, help='Movetime in ms')
parser.add_argument('-nodes', type=int, default=0, help='Maximum nodes')
parser.add_argument('-pvs', nargs='?', const=3, default=0, type=int,
                    help='Show Principal Variations (when mcts)')
parser.add_argument('-fen', help='Start from given position',
                    default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

async def load_engine(engine_args, name, debug=False):
    args = next(a for a in engine_args if a['name'] == name)
    curdir = str(pathlib.Path(__file__).parent)
    popen_args = {}
    if 'workingDirectory' in args:
        popen_args['cwd'] = args['workingDirectory'].replace('$FILE', curdir)
    cmd = args['command'].split()
    if cmd[0] == '$PYTHON':
        cmd[0] = sys.executable
    if args['protocol'] == 'uci':
        _, engine = await chess.engine.popen_uci(cmd, **popen_args)
    elif args['protocol'] == 'xboard':
        _, engine = await chess.engine.popen_xboard(cmd, **popen_args)
    if hasattr(engine, 'debug'):
        engine.debug(debug)
    await engine.configure({opt['name']: opt['value'] for opt in args.get('options', [])})
    return engine


def get_user_move(board):
    # Get well-formated move
    move = None
    while move is None:
        san_option = random.choice([board.san(m) for m in board.legal_moves])
        uci_option = random.choice([m.uci() for m in board.legal_moves])
        uci = input(f'Your move (e.g. {san_option} or {uci_option}): ')
        for parse in (board.parse_san, chess.Move.from_uci):
            try:
                move = parse(uci)
            except ValueError:
                pass

    # Check legality
    if move not in board.legal_moves:
        print('Illegal move.')
        return get_user_move(board)

    return move


def get_user_color():
    color = ''
    while color not in ('white', 'black'):
        color = input('Do you want to be white or black? ')
    return chess.WHITE if color == 'white' else chess.BLACK


def print_unicode_board(board, perspective=chess.WHITE):
    """ Prints the position from a given perspective. """
    sc, ec = '\x1b[0;30;107m', '\x1b[0m'
    for r in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line = [f'{sc} {r+1}']
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = '\x1b[48;5;255m' if (r + c) % 2 == 1 else '\x1b[48;5;253m'
            if board.move_stack:
                if board.move_stack[-1].to_square == 8 * r + c:
                    color = '\x1b[48;5;153m'
                elif board.move_stack[-1].from_square == 8 * r + c:
                    color = '\x1b[48;5;153m'
            piece = board.piece_at(8 * r + c)
            line.append(color +
                        (chess.UNICODE_PIECE_SYMBOLS[piece.symbol()] if piece else ' '))
        print(' ' + ' '.join(line) + f' {sc} {ec}')
    if perspective == chess.WHITE:
        print(f' {sc}   a b c d e f g h  {ec}\n')
    else:
        print(f' {sc}   h g f e d c b a  {ec}\n')


async def get_engine_move(engine, board, limit, game_id, multipv, debug=False):
    # XBoard engine doesn't support multipv, and there python-chess doesn't support
    # getting the first PV while playing a game.
    if isinstance(engine, chess.engine.XBoardProtocol):
        play_result = await engine.play(board, limit, game=game_id)
        return play_result.move

    multipv = min(multipv, board.legal_moves.count())
    with await engine.analysis(board, limit, game=game_id,
                               info=chess.engine.INFO_ALL, multipv=multipv or None) as analysis:

        infos = [None for _ in range(multipv)]
        first = True
        async for new_info in analysis:
            # If multipv = 0 it means we don't want them at all,
            # but uci requires MultiPV to be at least 1.
            if multipv and 'multipv' in new_info:
                infos[new_info['multipv'] - 1] = new_info

            # Parse optional arguments into a dict
            if debug and 'string' in new_info:
                print(new_info['string'])

            if not debug and all(infos) and 'score' in analysis.info:
                if not first:
                    #print('\n'*(multipv+1), end='')
                    print(f"\u001b[1A\u001b[K" * (multipv + 1), end='')
                else:
                    first = False

                info = analysis.info
                score = info['score'].relative
                score = f'Score: {score.score()}' \
                        if score.score() is not None else f'Mate in {score.mate()}'
                print(f'{score}, nodes: {info.get("nodes", "N/A")}, nps: {info.get("nps", "N/A")},'
                      f' time: {float(info.get("time", "N/A")):.1f}', end='')
                print()

                for info in infos:
                    variation = board.variation_san(info['pv'][:10])

                    if 'score' in info:
                        score = info['score'].relative
                        score = math.tanh(score.score() / 600) \
                            if score.score() is not None else score.mate()
                        key, *val = info.get('string', '').split()
                        if key == 'pv_nodes':
                            nodes = int(val[0])
                            rel = nodes / analysis.info['nodes']
                            score_rel = f'({score:.2f}, {rel*100:.0f}%)'
                        else:
                            score_rel = f'({score:.2f})'
                    else:
                        score_rel = ''

                    # Something about N
                    print(f'{info["multipv"]}: {score_rel} {variation}')

        return analysis.info['pv'][0]


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
        else:
            move = await get_engine_move(engine, board, time_limit, game_id, pvs, debug=debug)
            print(f' My move: {board.san(move)}')
        board.push(move)

    # Print status
    print_unicode_board(board, perspective=user_color)
    print('Result:', board.result())


async def main():
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    if not args.conf:
        path = pathlib.Path(__file__).parent / 'engines.json'
        if not path.is_file():
            print('Unable to locate engines.json file.')
            return
        conf = json.load(path.open())
    else:
        conf = json.load(open(args.conf))

    engine = await load_engine(conf, args.name, debug=args.debug)
    if 'author' in engine.id:
        print(f"Playing against {engine.id['name']} by {engine.id['author']}.")
    else:
        print(f"Playing against {engine.id['name']}.")

    board = chess.Board(args.fen)

    if args.movetime:
        limit = chess.engine.Limit(time=args.movetime / 1000)
    elif args.nodes:
        limit = chess.engine.Limit(nodes=args.nodes)
    else:
        limit = chess.engine.Limit(white_clock=30, black_clock=30, remaining_moves=30)

    try:
        await play(engine, board, selfplay=args.selfplay, pvs=args.pvs, time_limit=limit, debug=args.debug)
    finally:
        print('\nGoodbye!')
        await engine.quit()


asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
