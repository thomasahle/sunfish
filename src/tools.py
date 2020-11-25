import itertools
import re
import time
import sys

import sunfish

################################################################################
# This module contains functions used by test.py and xboard.py.
# Nothing from here is imported into sunfish.py which is entirely self-sufficient
################################################################################

# Sunfish doesn't have to know about colors, but for more advanced things, such
# as xboard support, we have to.
WHITE, BLACK = range(2)

FEN_INITIAL = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def search(searcher, pos, secs, history=()):
    """ This used to be in the Searcher class """
    start = time.time()
    for depth, move, score in searcher.search(pos, history):
        if time.time() - start > secs:
            break
    return move, score, depth


################################################################################
# Parse and Render moves
################################################################################

def gen_legal_moves(pos):
    ''' pos.gen_moves(), but without those that leaves us in check.
        Also the position after moving is included. '''
    for move in pos.gen_moves():
        pos1 = pos.move(move)
        if not can_kill_king(pos1):
            yield move, pos1

def can_kill_king(pos):
    # If we just checked for opponent moves capturing the king, we would miss
    # captures in case of illegal castling.
    return any(pos.value(m) >= sunfish.MATE_LOWER for m in pos.gen_moves())

def mrender(pos, m):
    # Sunfish always assumes promotion to queen
    p = 'q' if sunfish.A8 <= m[1] <= sunfish.H8 and pos.board[m[0]] == 'P' else ''
    m = m if get_color(pos) == WHITE else (119-m[0], 119-m[1])
    return sunfish.render(m[0]) + sunfish.render(m[1]) + p

def mparse(color, move):
    m = (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
    return m if color == WHITE else (119-m[0], 119-m[1])

def renderSAN(pos, move):
    ''' Assumes board is rotated to position of current player '''
    i, j = move
    csrc, cdst = sunfish.render(i), sunfish.render(j)
    # Rotate flor black
    if get_color(pos) == BLACK:
        csrc, cdst = sunfish.render(119-i), sunfish.render(119-j)
    # Check
    pos1 = pos.move(move)
    cankill = lambda p: any(p.board[b]=='k' for a,b in p.gen_moves())
    check = ''
    if cankill(pos1.rotate()):
        check = '+'
        if all(cankill(pos1.move(move1)) for move1 in pos1.gen_moves()):
            check = '#'
    # Castling
    if pos.board[i] == 'K' and abs(i-j) == 2:
        if get_color(pos) == WHITE and j > i or get_color(pos) == BLACK and j < i:
            return 'O-O' + check
        else:
            return 'O-O-O' + check
    # Pawn moves
    if pos.board[i] == 'P':
        pro = '=Q' if sunfish.A8 <= j <= sunfish.H8 else ''
        cap = csrc[0] + 'x' if pos.board[j] != '.' or j == pos.ep else ''
        return cap + cdst + pro + check
    # Figure out what files and ranks we need to include
    srcs = [a for (a,b),_ in gen_legal_moves(pos) if pos.board[a] == pos.board[i] and b == j]
    srcs_file = [a for a in srcs if (a - sunfish.A1) % 10 == (i - sunfish.A1) % 10]
    srcs_rank = [a for a in srcs if (a - sunfish.A1) // 10 == (i - sunfish.A1) // 10]
    assert srcs, 'No moves compatible with {}'.format(move)
    if len(srcs) == 1: src = ''
    elif len(srcs_file) == 1: src = csrc[0]
    elif len(srcs_rank) == 1: src = csrc[1]
    else: src = csrc
    # Normal moves
    p = pos.board[i]
    cap = 'x' if pos.board[j] != '.' else ''
    return p + src + cap + cdst + check

def parseSAN(pos, msan):
    ''' Assumes board is rotated to position of current player '''
    # Normal moves
    normal = re.match('([KQRBN])([a-h])?([1-8])?x?([a-h][1-8])', msan)
    if normal:
        p, fil, rank, dst = normal.groups()
        src = (fil or '[a-h]')+(rank or '[1-8]')
    # Pawn moves
    pawn = re.match('([a-h])?x?([a-h][1-8])', msan)
    if pawn:
        assert not re.search('[RBN]$', msan), 'Sunfish only supports queen promotion in {}'.format(msan)
        p, (fil, dst) = 'P', pawn.groups()
        src = (fil or '[a-h]')+'[1-8]'
    # Castling
    if re.match(msan, "O-O-O[+#]?"):
        p, src, dst = 'K', 'e[18]', 'c[18]'
    if re.match(msan, "O-O[+#]?"):
        p, src, dst = 'K', 'e[18]', 'g[18]'
    # Find possible match
    assert 'p' in vars(), 'No piece to move with {}'.format(msan)
    for (i, j), _ in gen_legal_moves(pos):
        if get_color(pos) == WHITE:
            csrc, cdst = sunfish.render(i), sunfish.render(j)
        else: csrc, cdst = sunfish.render(119-i), sunfish.render(119-j)
        if pos.board[i] == p and re.match(dst,cdst) and re.match(src,csrc):
            return (i, j)
    assert False, 'Couldn\'t find legal move matching {}. Had {}'.format(msan, {
        'p':p, 'src':src, 'dst': dst, 'mvs':list(gen_legal_moves(pos))})

def readPGN(file):
    """ Yields a number of [(pos, move), ...] lists. """
    def _parse_single_pgn(lines):
        # Remove comments and numbers.
        parts = re.sub('{.*?}', '', ' '.join(lines)).split()
        msans = [part for part in parts if not part[0].isdigit()]
        pos = parseFEN(FEN_INITIAL)
        for msan in msans:
            try:
                move = parseSAN(pos, msan)
            except AssertionError:
                print('PGN was:', ' '.join(lines))
                raise
            yield pos, move
            pos = pos.move(move)

    # TODO: Currently assumes all games start at the initial position.
    current_game = []
    for line in file:
        if line.startswith('['):
            if current_game:
                yield ' '.join(current_game), list(_parse_single_pgn(current_game))
            del current_game[:]
        else:
            current_game.append(line.strip())


################################################################################
# Parse and Render positions
################################################################################

def get_color(pos):
    ''' A slightly hacky way to to get the color from a sunfish position '''
    return BLACK if pos.board.startswith('\n') else WHITE

def parseFEN(fen):
    """ Parses a string in Forsyth-Edwards Notation into a Position """
    board, color, castling, enpas, _hclock, _fclock = fen.split()
    board = re.sub(r'\d', (lambda m: '.'*int(m.group(0))), board)
    board = list(21*' ' + '  '.join(board.split('/')) + 21*' ')
    board[9::10] = ['\n']*12
    #if color == 'w': board[::10] = ['\n']*12
    #if color == 'b': board[9::10] = ['\n']*12
    board = ''.join(board)
    wc = ('Q' in castling, 'K' in castling)
    bc = ('k' in castling, 'q' in castling)
    ep = sunfish.parse(enpas) if enpas != '-' else 0
    score = sum(sunfish.pst[p][i] for i,p in enumerate(board) if p.isupper())
    score -= sum(sunfish.pst[p.upper()][119-i] for i,p in enumerate(board) if p.islower())
    pos = sunfish.Position(board, score, wc, bc, ep, 0)
    return pos if color == 'w' else pos.rotate()

def renderFEN(pos, half_move_clock=0, full_move_clock=1):
    color = 'wb'[get_color(pos)]
    if get_color(pos) == BLACK:
        pos = pos.rotate()
    board = '/'.join(pos.board.split())
    board = re.sub(r'\.+', (lambda m: str(len(m.group(0)))), board)
    castling = ''.join(itertools.compress('KQkq', pos.wc[::-1]+pos.bc)) or '-'
    ep = sunfish.render(pos.ep) if not pos.board[pos.ep].isspace() else '-'
    clock = '{} {}'.format(half_move_clock, full_move_clock)
    return ' '.join((board, color, castling, ep, clock))

def parseEPD(epd, opt_dict=False):
    epd = epd.strip('\n ;').replace('"','')
    parts = epd.split(maxsplit=6)
    opt_part = ''
    if len(parts) >= 6 and parts[4].isdigit() and parts[5].isdigit():
        fen = ' '.join(parts[:6])
        opt_part = ' '.join(parts[6:])
    else:
        # Sometimes fen doesn't include half move clocks
        fen = ' '.join(parts[:4]) + ' 0 1'
        opt_part = ' '.join(parts[4:])
    # EPD operations may either be <opcode> or (<opcode> <operand>)
    opts = opt_part.split(';')
    if opt_dict:
        opts = dict(p.split(maxsplit=1) for p in opts)
    return fen, opts

################################################################################
# Pretty print
################################################################################

def pv(searcher, pos, include_scores=True, include_loop=False):
    res = []
    seen_pos = set()
    color = get_color(pos)
    origc = color
    if include_scores:
        res.append(str(pos.score))
    while True:
        move = searcher.tp_move.get(pos)
        # The tp may have illegal moves, given lower depths don't detect king killing
        if move is None or can_kill_king(pos.move(move)):
            break
        res.append(mrender(pos, move))
        pos, color = pos.move(move), 1-color
        if pos in seen_pos:
            if include_loop:
                res.append('loop')
            break
        seen_pos.add(pos)
        if include_scores:
            res.append(str(pos.score if color==origc else -pos.score))
    return ' '.join(res)

################################################################################
# Bulk move generation
################################################################################

def expand_position(pos):
    ''' Yields a tree of generators [p, [p, [...], ...], ...] rooted at pos '''
    yield pos
    for _, pos1 in gen_legal_moves(pos):
        yield expand_position(pos1)

def collect_tree_depth(tree, depth):
    ''' Yields positions exactly at depth '''
    root = next(tree)
    if depth == 0:
        yield root
    else:
        for subtree in tree:
            for pos in collect_tree_depth(subtree, depth-1):
                yield pos

def flatten_tree(tree, depth):
    ''' Yields positions exactly at less than depth '''
    if depth == 0:
        return
    yield next(tree)
    for subtree in tree:
        for pos in flatten_tree(subtree, depth-1):
            yield pos

################################################################################
# Non chess related tools
################################################################################

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

