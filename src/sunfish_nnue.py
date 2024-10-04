#!/usr/bin/env pypy3

import sys, time, pickle
from itertools import count
from collections import namedtuple
import numpy as np
from functools import partial
from contextlib import contextmanager

print = partial(print, flush=True)

version = 'sunfish nnue'

###############################################################################
# A small neural network to evaluate positions
###############################################################################

L0, L1, L2 = 10, 10, 10
model = pickle.load(open(sys.argv[1], "br"))
# pos_emb, comb, piece_val, comb_col layers0-1
nn = [np.frombuffer(ar, dtype=np.int8) / 127.0 for ar in model["ars"]]
layer1, layer2 = nn[4].reshape(L2, 2 * L1 - 2), nn[5].reshape(1, L2)
# Pad the position embedding to fit with our 10x12 board
pad = np.pad(nn[0].reshape(8, 8, 6)[::-1], ((2, 2), (1, 1), (0, 0))).reshape(120, 6)
# Combine piece table and pos table into one piece-square table
pst = np.einsum("sd,odp->pso", pad, nn[1].reshape(L0, 6, 6))
pst = np.einsum("psd,odc->cpso", pst, nn[3].reshape(L0, L0, 2))
pst = dict(zip("PNBRQKpnbrqk", pst.reshape(12, 120, L0)))
pst["."] = [[0]*L0] * 120

# for i, p in enumerate("PNBRQKpnbrqk"):
#     table = pst[p][:, 0].reshape(12,10)[2:10,1:9]
#     print(p, table.mean().round(2))
#     print(table.round(2))

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
MATE = 100000
# Since move ordering uses the lower-case version, we need to include the
# mate score in it, since otherwise we wouldn't find checks in QS search.
pst['K'][:, 0] += MATE//2
pst['k'][:, 0] -= MATE//2
MATE_LOWER = MATE // 2
MATE_UPPER = MATE * 3//2


#def manual_wf(board):
#    wf = 0
#    for i, p in enumerate(board):
#        col = p.isupper()
#        ptyp = "PNBRQK".find(p.upper())
#        if p.isalpha():
#            mrank, fil = divmod(i - A1, 10)
#            sq = -mrank*8 + fil
#            sq_emb = nn[0].reshape(64, 6)[sq]
#            comb = nn[1].reshape(L1, 6, 6)[:, :, ptyp]
#            emb = comb @ sq_emb
#            col_comb = nn[3].reshape(L0, L0, 2)[:, :, 1-int(col)]
#            emb = col_comb @ emb
#            wf += emb
#    return wf

def features(board):
    wf = sum(pst[p][i] for i, p in enumerate(board) if p.isalpha())
    #assert np.allclose(wf, manual_wf(board))
    bf = sum(pst[p.swapcase()][119 - i] for i, p in enumerate(board) if p.isalpha())
    return wf, bf

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N + N, N + W, N + E),
    "N": (N + N + E,
        E + N + E,
        E + S + E,
        S + S + E,
        S + S + W,
        W + S + W,
        W + N + W,
        N + N + W,
    ),
    "B": (N + E, S + E, S + W, N + W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N + E, S + E, S + W, N + W),
    "K": (N, E, S, W, N + E, S + E, S + W, N + W),
}

# Constants for tuning search
EVAL_ROUGHNESS = 13

# minifier-hide start
opt_ranges = dict(
    EVAL_ROUGHNESS = (0, 50),
)
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################

Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score wf bf wc bc ep kp")):
    # The state of a chess game
    # board -- a 120 char representation of the board
    # score -- the board evaluation
    # turn
    # wf -- our features
    # bf -- opponent features
    # wc -- the castling rights, [west/queen side, east/king side]
    # bc -- the opponent castling rights, [west/king side, east/queen side]
    # ep - the en passant square
    # kp - the king passant square

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        # If the pawn moves forward, it has to not hit anybody
                        if d in (N, N + N) and q != ".":
                            break
                        # If the pawn moves forward twice, it has to be on the first row
                        # and it has to not jump over anybody
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."):
                            break
                        # If the pawn captures, it has to either be a piece, an
                        # enpassant square, or a moving king.
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                            # and j != self.ep and abs(j - self.kp) >= 2
                        ):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q.islower():
                        break
                    # Castling, by sliding the rook next to the king. This way we don't
                    # need to worry about jumping over pieces while castling.
                    # We don't need to check for being a root, since if the piece starts
                    # at A1 and castling queen side is still allowed, it must be a rook.
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    @contextmanager
    def rotate(self, nullmove=False):
        # Rotates the board, preserving enpassant.
        # A nullmove is nearly a rotate, but it always clear enpassant.
        pos = Position(
            self.board[::-1].swapcase(),
            0, self.bf, self.wf, self.bc, self.wc,
            0 if nullmove or not self.ep else 119 - self.ep,
            0 if nullmove or not self.kp else 119 - self.kp,
        )
        return pos._replace(score=pos.compute_value())

    def put(self, i, p, stack=None):
        q = self.board[i]
        # TODO: I could update a zobrist hash here as well...
        # Then we are really becoming a real chess program...
        self.board[i] = p
        self.wf += pst[p][i] - pst[q][i]
        self.bf += pst[p.swapcase()][119 - i] - pst[q.swapcase()][119 - i]
        if stack:
            self.stack.append((i, q))

    @contextmanager
    def move(self, move):
        i, j, pr = move
        p, q = self.board[i], self.board[j]
        # We make this stack to keep track of what we change
        stack = []

        old_ep, old_kp, old_wc, old_bc = self.ep, self.kp, self.ec, self.bc
        self.ep, self.kp = 0, 0

        # Actual move
        self.put(j, p, stack)
        self.put(i, ".", stack)

        # Castling rights, we move the rook or capture the opponent's
        if i == A1: self.wc=(False, self.wc[1])
        if i == H1: self.wc=(self.wc[0], False)
        if j == A8: self.bc=(self.bc[0], False)
        if j == H8: self.bc=(False, self.bc[1])

        # Capture the moving king. Actually we get an extra free king. Same thing.
        if abs(j - self.kp) < 2:
            self.put(self.board.find('k'), ' ')

        # Castling
        if p == "K":
            self.wc=(False, False)
            if abs(j - i) == 2:
                self.kp=(i + j) // 2
                self.put(A1 if j < i else H1, ".", stack)
                self.put((i + j) // 2, "R", stack)

        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                self.put(j, pr, stack)
            if j - i == 2 * N:
                self.ep = i + N
            if j == self.ep:
                self.put(j + S, ".", stack)

        # Should this also be a context manager then?
        self.rotate()
        yield self
        self.rotate()

        # Now unmove by putting the pieces back
        for i, q in self.stack[::-1]:
            self.put(i, q)

        # And restore the fields
        self.ep, self.kp, self.ec, self.bc = old_ep, old_kp, old_wc, old_bc

    def is_capture(self, move):
        # The original sunfish just checked that the evaluation of a move
        # was larger than a certain constant. However the current NN version
        # can have too much fluctuation in the evals, which can lead QS-search
        # to last forever (until python stackoverflows.) Thus we need to either
        # dampen the eval function, or like here, reduce QS search to captures
        # only. Well, captures plus promotions.
        return self.board[move.j] != "." or abs(move.j - self.kp) < 2 or move.prom

    def compute_value(self):
        #relu6 = lambda x: np.minimum(np.maximum(x, 0), 6)
        # TODO: We can maybe speed this up using a fixed `out` array,
        # as well as using .dot istead of @.
        act = np.tanh
        wf, bf = self.wf, self.bf
        # Pytorch matrices are in the shape (out_features, in_features)
        #hidden = layer1 @ act(np.concatenate([wf[1:], bf[1:]]))
        hidden = (layer1[:,:9] @ act(wf[1:])) + (layer1[:,9:] @ act(bf[1:]))
        score = layer2 @ act(hidden)
        #if verbose:
        #    print(f"Score: {score + model['scale'] * (wf[0] - bf[0])}")
        #    print(f"from model: {score}, pieces: {wf[0]-bf[0]}")
        #    print(f"{wf=}")
        #    print(f"{bf=}")
        return int((score + model["scale"] * (wf[0] - bf[0])) * 360)

    def hash(self):
        # return self.board
        # return self.score
        # return hash(self.board)
        return hash((self.board, self.wc, self.bc, self.ep, self.kp))
        # return (self.wf + self.bf).sum()
        # return self._replace(wf=0, bf=0)


class MutablePosition(namedtuple("Position", "board score wf bf wc bc ep kp")):
    # The state of a chess game
    # board -- a 120 char representation of the board
    # score -- the board evaluation
    # wf -- our features
    # bf -- opponent features
    # wc -- the castling rights, [west/queen side, east/king side]
    # bc -- the opponent castling rights, [west/king side, east/queen side]
    # ep - the en passant square
    # kp - the king passant square

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        # If the pawn moves forward, it has to not hit anybody
                        if d in (N, N + N) and q != ".":
                            break
                        # If the pawn moves forward twice, it has to be on the first row
                        # and it has to not jump over anybody
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."):
                            break
                        # If the pawn captures, it has to either be a piece, an
                        # enpassant square, or a moving king.
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                            # and j != self.ep and abs(j - self.kp) >= 2
                        ):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q.islower():
                        break
                    # Castling, by sliding the rook next to the king. This way we don't
                    # need to worry about jumping over pieces while castling.
                    # We don't need to check for being a root, since if the piece starts
                    # at A1 and castling queen side is still allowed, it must be a rook.
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        # Rotates the board, preserving enpassant.
        # A nullmove is nearly a rotate, but it always clear enpassant.
        pos = Position(
            self.board[::-1].swapcase(),
            0, self.bf, self.wf, self.bc, self.wc,
            0 if nullmove or not self.ep else 119 - self.ep,
            0 if nullmove or not self.kp else 119 - self.kp,
        )
        return pos._replace(score=pos.compute_value())

    def move(self, move):
        put = lambda pos, i, p: pos._replace(
            # f-strings are a bit faster in python, but the same in pypy
            board=pos.board[:i] + p + pos.board[i + 1 :],
            wf=pos.wf + pst[p][i] - pst[pos.board[i]][i],
            bf=pos.bf + pst[p.swapcase()][119 - i] - pst[pos.board[i].swapcase()][119 - i],
        )

        i, j, pr = move
        p, q = self.board[i], self.board[j]
        # Copy variables and reset ep and kp
        pos = self._replace(ep=0, kp=0)
        # Actual move
        pos = put(pos, j, p)
        pos = put(pos, i, ".")

        # Would something like this be easier?
        # if i in pos.castl:
        #     pos = pos._replace(castl=pos.castle - {i})
        # if j in pos.castl:
        #     pos = pos._replace(castl=pos.castle - {j})

        # Castling rights, we move the rook or capture the opponent's
        if i == A1: pos = pos._replace(wc=(False, pos.wc[1]))
        if i == H1: pos = pos._replace(wc=(pos.wc[0], False))
        if j == A8: pos = pos._replace(bc=(pos.bc[0], False))
        if j == H8: pos = pos._replace(bc=(False, pos.bc[1]))
        # Capture the moving king. Actually we get an extra free king. Same thing.
        if abs(j - self.kp) < 2:
            pos = put(pos, self.kp, "K")
            # If using king-nnue, we might have to do some stuff here as well...
            # Or maybe it doesn't matter whether scoring is correct when the king
            # is dead anyway.
            # Actually: Storing the king position would allow us to do this in a less
            # hacky way.
        # Castling
        if p == "K":
            pos = pos._replace(wc=(False, False))
            if abs(j - i) == 2:
                pos = pos._replace(kp=(i + j) // 2)
                pos = put(pos, A1 if j < i else H1, ".")
                pos = put(pos, (i + j) // 2, "R")
            # If we used "king" NNUE we could recompute features here
            # wf, bf = features(pos.board)
            # Actually we'd only have to recompute wf...
            # We'd also have to add king-position to Position so the right
            # tables can be used at every normal move.
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                pos = put(pos, j, pr)
            if j - i == 2 * N:
                pos = pos._replace(ep=i + N)
            if j == self.ep:
                pos = put(pos, j + S, ".")

        # wf, bf = features(pos.board)
        # assert np.allclose(pos.wf, wf)
        # assert np.allclose(pos.bf, bf)

        return pos.rotate()

    def is_capture(self, move):
        # The original sunfish just checked that the evaluation of a move
        # was larger than a certain constant. However the current NN version
        # can have too much fluctuation in the evals, which can lead QS-search
        # to last forever (until python stackoverflows.) Thus we need to either
        # dampen the eval function, or like here, reduce QS search to captures
        # only. Well, captures plus promotions.
        return self.board[move.j] != "." or abs(move.j - self.kp) < 2 or move.prom

    def compute_value(self):
        #relu6 = lambda x: np.minimum(np.maximum(x, 0), 6)
        # TODO: We can maybe speed this up using a fixed `out` array,
        # as well as using .dot istead of @.
        act = np.tanh
        wf, bf = self.wf, self.bf
        # Pytorch matrices are in the shape (out_features, in_features)
        #hidden = layer1 @ act(np.concatenate([wf[1:], bf[1:]]))
        hidden = (layer1[:,:9] @ act(wf[1:])) + (layer1[:,9:] @ act(bf[1:]))
        score = layer2 @ act(hidden)
        #if verbose:
        #    print(f"Score: {score + model['scale'] * (wf[0] - bf[0])}")
        #    print(f"from model: {score}, pieces: {wf[0]-bf[0]}")
        #    print(f"{wf=}")
        #    print(f"{bf=}")
        return int((score + model["scale"] * (wf[0] - bf[0])) * 360)

    def hash(self):
        # return self.board
        # return self.score
        # return hash(self.board)
        return hash((self.board, self.wc, self.bc, self.ep, self.kp))
        # return (self.wf + self.bf).sum()
        # return self._replace(wf=0, bf=0)


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        # returns r where
        #    s(pos) <= r < gamma    if gamma > s(pos)
        #    gamma <= r <= s(pos)   if gamma <= s(pos)
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        # I think this line also makes sure we never fail low on king-capture
        # replies, which might hide them and lead to illegal moves.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        # We need to include depth and root, since otherwise the function wouldn't
        # be consistent. By consistent I mean that if the function is called twice
        # with the same parameters, it will always fail in the same direction (hi / low).
        # It might return different soft values though, exactly because the tp tables
        # have changed.
        entry = self.tp_score.get(
            (pos.hash(), depth, root), Entry(-MATE_UPPER, MATE_UPPER)
        )
        if entry.lower >= gamma:
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # We detect 3-fold captures by comparing against previously
        # _actually played_ positions.
        # Note that we need to do this before we look in the table, as the
        # position may have been previously reached with a different score.
        # This is what prevents a search instability.
        # Actually, this is not true, since other positions will be affected by
        # the new values for all the drawn positions.
        # This is why I've decided to just clear tp_score every time history changes.
        if not root and pos.hash() in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            if depth > 2 and not root and any(c in pos.board for c in "NBRQ"):
                yield None, -self.bound(pos.rotate(nullmove=True), 1-gamma, depth-3, False)
            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score
            # Then killer move. We search it twice, but the tp will fix things for us.
            # Note, we don't have to check for legality, since we've already done it
            # before. Also note that in QS the killer must be a capture, otherwise we
            # will be non deterministic.
            def mvv_lva(move):
                # Recall mvv_lva gives the _negative_ score
                if abs(move.j - pos.kp) < 2: return -MATE
                i, j = move.i, move.j
                p, q = pos.board[i], pos.board[j]
                p2 = move.prom or p
                score = pst[q][j][0] - (pst[p2][j][0] - pst[p][i][0])
                pp, qq, pp2 = p.swapcase(), q.swapcase(), p2.swapcase()
                score -= pst[qq][119-j][0] - (pst[pp2][119-j][0] - pst[pp][119-i][0])
                #pp, qq = p.swapcase(), q.swapcase()
                #score = pst[q][j][0] - (pst[p][j][0] - pst[p][i][0])
                #score -= pst[qq][119-j][0] - (pst[pp][119-j][0] - pst[pp][119-i][0])
                return score

            killer = self.tp_move.get(pos.hash())
            if killer and (depth > 0 or pos.is_capture(killer)):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, False)

            # Then all the other moves
            # moves = [(move, pos.move(move)) for move in pos.gen_moves()]
            # moves.sort(key=lambda move_pos: pst[pos.board[move_pos[0].i][move

            # Sort by the score after moving. Since that's from the perspective of our
            # opponent, smaller score means the move is better for us.
            # print(f'Searching at {depth=}')
            # TODO: Maybe try MMT/LVA sorting here. Could be cheaper and work better since
            # the current evaluation based method doesn't take into account that e.g. capturing
            # with the queen shouldn't usually be our first option...
            # It could be fun to train a network too, that scores all the from/too target
            # squares, say, and uses that to sort...
            #for move, pos1 in sorted(moves, key=lambda move_pos: move_pos[1].score):
            for move in sorted(pos.gen_moves(), key=mvv_lva):
                # TODO: We seem to have some issues with our QS search, which eventually
                # leads to very large jumps in search time. (Maybe we get the classical
                # "Queen plunders everything" case?) Hence Improving this might solve some
                # of our timeout issues. It could also be that using a more simple ordering
                # would speed up the move generation?
                # See https://home.hccnet.nl/h.g.muller/mvv.html for inspiration
                # If depth is 0 we only try moves with high intrinsic score (captures and
                # promotions). Otherwise we do all moves.
                #if depth > 0 or -pos1.score-pos.score >= QS_LIMIT:
                if depth > 0 or pos.is_capture(move):
                #print(mvv_lva(move)*360)
                #if -mvv_lva(move)*360 >= 30  - depth * 10:
                #if depth > 0 or (QS_TYPE == QS_CAPTURE and pos.is_capture(move)) or (QS_TYPE != QS_CAPTURE and -mvv_lva(move) >= QS_LIMIT/360):
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos.hash()] = move
                break

        # Stalemate checking
        if depth > 0 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            # Hopefully this is already in the TT because of null-move
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Table part 2
        self.tp_score[pos.hash(), depth, root] = (
            Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        )

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0
        pos = history[-1]
        self.history = {pos.hash() for pos in history}
        # Clearing table due to new history. This is because having a new "seen"
        # position alters the score of all other positions, as there may now be
        # a path that leads to a repetition.
        self.tp_score.clear()
        # We save the gamma function between depths, so we can start from the most
        # interesting position at the next level
        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 1000):
            #yield depth, None, 0, "cp"
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays
            # better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(pos.hash())
                gamma = (lower + upper + 1) // 2


###############################################################################
# UCI interface
###############################################################################


def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)


wf, bf = features(initial)
hist = [Position(initial, 0, wf, bf, (True, True), (True, True), 0, 0)]
searcher = Searcher()


# minifier-hide start
if '--profile' in sys.argv:
    import cProfile
    def go_depth_5():
        for depth, _, _, _ in searcher.search(hist):
            if depth == 5:
                break
    cProfile.run('go_depth_5()')
else:
    import tools.uci
    tools.uci.run(sys.modules[__name__], hist[-1])
sys.exit()
# minifier-hide end


while True:
    args = input().split()
    if args[0] == "uci":
        print(f"id name {version}")
        print("uciok")

    elif args[0] == "isready":
        print("readyok")

    elif args[0] == "quit":
        break

    elif args[:2] == ["position", "startpos"]:
        del hist[1:]
        for ply, move in enumerate(args[3:]):
            i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
            if ply % 2 == 1:
                i, j = 119 - i, 119 - j
            hist.append(hist[-1].move(Move(i, j, prom)))

    elif args[0] == "go":
        wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
        if len(hist) % 2 == 0:
            wtime, winc = btime, binc
        think = min(wtime / 40 + winc, wtime / 2 - 1)

        start = time.time()
        move_str = None
        for depth, gamma, score, move in Searcher().search(hist):
            # The only way we can be sure to have the real move in tp_move,
            # is if we have just failed high.
            if score >= gamma:
                i, j = move.i, move.j
                if len(hist) % 2 == 0:
                    i, j = 119 - i, 119 - j
                move_str = render(i) + render(j) + move.prom.lower()
                print(f"info depth {depth} score cp {score} pv {move_str}")
            if move_str and time.time() - start > think * 0.8:
                break

        print("bestmove", move_str or '(none)')
