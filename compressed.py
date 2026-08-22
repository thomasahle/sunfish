import time
from itertools import count
from collections import namedtuple
version = "sunfish 2026"
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20
K_MID, K_END = pst["K"], tuple(piece["K"] + 70
   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"
    "         \n"
    " rnbqkbnr\n"
    " pppppppp\n"
    " ........\n"
    " ........\n"
    " ........\n"
    " ........\n"
    " PPPPPPPP\n"
    " RNBQKBNR\n"
    "         \n"
    "         \n"
)
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}
MATE_LOWER = piece["K"] - 13 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]
QS = 40
QS_A = 140
LMR = 75
EVAL_ROUGHNESS = 15
NULL_MARGIN = -200
DELAY = 200
TABLE_SIZE = 10**6
Move = namedtuple("Move", "i j prom")
class Position(namedtuple("Position", "board score wc bc ep kp")):
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if p not in "PNBRQK":
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    if q in " \nPNBRQK": break
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if d in (N + W, N + E) and q == "." and j != self.ep and abs(j - self.kp) > 1: break
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    yield Move(i, j, "")
                    if p in "PNK" or q in "pnbrqk": break
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]: yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]: yield Move(j + W, j + E, "")
    def rotate(self, nullmove=False):
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )
    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        board = put(board, j, board[i])
        board = put(board, i, ".")
        wc = (wc[0] and i != A1, wc[1] and i != H1)
        bc = (bc[0] and j != H8, bc[1] and j != A8)
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        if p == "P":
            if A8 <= j <= H8:  board = put(board, j, prom)
            if j - i == 2 * N: ep = i + N
            if j == self.ep:   board = put(board, j + S, ".")
        return Position(board, score, wc, bc, ep, kp).rotate()
    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        score = pst[p][j] - pst[p][i]
        if q in "pnbrqk": score += pst[q.upper()][119 - j]
        if abs(j - self.kp) < 2: score += pst["K"][119 - j]
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        if p == "P":
            if A8 <= j <= H8: score += pst[prom][j] - pst["P"][j]
            if j == self.ep: score += pst["P"][119 - (j + S)]
        return score
    def king_capture(self):
        return next((m for m in self.gen_moves()
                     if self.board[m.j] == "k" or abs(m.j - self.kp) < 2), None)
class Stop(Exception): pass
Entry = namedtuple("Entry", "lower upper")
class Searcher:
    def __init__(self):
        self.tp_score, self.tp_move, self.history = {}, {}, set()
        self.nodes, self.deadline, self.soft = 0, 1 << 63, 1 << 63
    def bound(self, pos, gamma, depth, root=False):
        self.nodes += 1
        if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop
        depth = max(depth, 0)
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma: return entry.lower
            if entry.upper < gamma: return entry.upper
            if depth > 0 and pos in self.history: return 0
        killer = self.tp_move.get(pos)
        def moves():
            if 2 < depth < 6 and guard:
                yield None, None
            if depth == 0:
                yield None, None
            if killer and ((val := pos.value(killer)) >= QS or depth) and (val >= MATE_LOWER or depth > 3
                    or pos.score + val + max(depth - 1, 0) * QS_A >= gamma):
                yield val, killer
            yield from sorted(((v, m) for m in pos.gen_moves() if (v := pos.value(m)) >= QS or depth), reverse=True)
        calm = abs(pos.score) < 750 and any(c in pos.board for c in "RBNQ")
        guard = not root and calm
        t = pos.score + NULL_MARGIN
        nmr = (calm and depth >= 6 and
               -self.bound(pos.rotate(nullmove=True), 1 - t, depth - 7) >= t)
        best, live = -MATE_UPPER, False
        for val, move in moves():
            if move is None and depth == 0:
                score = pos.score
            elif move is None:
                if (cap := pos.score + EVAL_ROUGHNESS) >= gamma:
                    score = min(cap, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))
                    if score >= gamma and (proof := pos.king_capture()):
                        move, score, live = proof, MATE_UPPER, True
                else:
                    score = cap
            else:
                if val >= MATE_LOWER:
                    score = MATE_UPPER
                    live = True
                else:
                    cap = MATE_UPPER if depth > 3 else pos.score + val + max(depth - 1, 0) * QS_A
                    if cap < gamma: best = max(best, cap); break
                    move_depth = depth - 1 - (guard and depth >= 6 and val < LMR) - int(nmr)
                    score = min(cap, -self.bound(pos.move(move), 1 - gamma, move_depth))
                    live |= score > -MATE_UPPER
            best = max(best, score)
            if best >= gamma:
                if move is not None and depth:
                    self.tp_move[pos] = move
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break
        if depth and not live and all(
                pos.move(m).king_capture() for m in pos.gen_moves()):
            mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
            best = mate if pos.rotate(nullmove=True).king_capture() else 0
        if not root:
            self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        if len(self.tp_score) > TABLE_SIZE:
            del self.tp_score[next(iter(self.tp_score))]
        return best
    def search(self, history):
        self.nodes, self.history = 0, set(history)
        self.tp_score.clear()
        pos = self.root = history[-1]
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
        gamma = 0
        for depth in range(1, 1000):
            lower, upper = 1 - MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma: lower = score
                if score < gamma: upper = score
                yield depth, gamma, score, self.tp_move.get(pos)
                gamma = (lower + upper + 1) // 2
            if time.time() > self.soft: return
def parse(c): return A1 + ord(c[0]) - ord("a") - 10 * (int(c[1]) - 1)
def render(i): return chr((i - A1) % 10 + ord("a")) + str(1 - (i - A1) // 10)
hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
def main():
    searcher = Searcher()
    while True:
        args = input().split()
        if args[0] == "uci":
            print("id name", version)
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
            times = dict(zip(args[1::2], map(int, args[2::2])))
            side = "wb"[len(hist) % 2 == 0]
            wtime, winc = times.get(side + "time", 60000), times.get(side + "inc", 0)
            budget = wtime / 40 + winc - DELAY
            soft = max(min(budget, wtime / 4 - DELAY), 100) / 1000
            think = max(min(5 * budget, wtime / 2 - DELAY), 200) / 1000
            start = time.time()
            searcher.deadline, searcher.soft = start + think, start + soft
            best, cand, d0 = None, None, 1
            try:
                for depth, gamma, score, move in searcher.search(hist):
                    if depth > d0:
                        best, d0 = cand or best, depth
                    if score >= gamma:
                        if move is None: print("info depth", depth, "score cp", score); break
                        i, j = move.i, move.j
                        if len(hist) % 2 == 0: i, j = 119 - i, 119 - j
                        cand = render(i) + render(j) + move.prom.lower()
                        print("info depth", depth, "score cp", score, "pv", cand)
            except Stop:
                cand = best or cand
            print("bestmove", cand or best or '(none)')
if __name__ == "__main__":
    main()
