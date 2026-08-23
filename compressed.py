Move = namedtuple("Move", "i j prom")
class Position(namedtuple("Position", "board score wc bc ep kp")):
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if p not in "PNBRQK": continue
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
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        p, q, board, wc, bc, ep, kp = self.board[i], self.board[j], self.board, self.wc, self.bc, 0, 0
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
        return next((m for m in self.gen_moves() if self.board[m.j] == "k" or abs(m.j - self.kp) < 2), None)
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
        if pos.score <= -MATE_LOWER: return -MATE_UPPER
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma: return entry.lower
            if entry.upper < gamma: return entry.upper
            if depth > 0 and pos in self.history: return 0
        killer = self.tp_move.get(pos)
        def moves():
            if 2 < depth < 6 and guard: yield None, None
            if depth == 0: yield None, None
            if killer and ((val := pos.value(killer)) >= QS or depth) and (val >= MATE_LOWER or depth > 3
                    or pos.score + val + max(depth - 1, 0) * QS_A >= gamma): yield val, killer
            yield from sorted(((v, m) for m in pos.gen_moves() if (v := pos.value(m)) >= QS or depth), reverse=True)
        calm = abs(pos.score) < 750 and any(c in pos.board for c in "RBNQ")
        guard = not root and calm
        t = pos.score + NULL_MARGIN
        nmr = calm and depth >= 6 and -self.bound(pos.rotate(nullmove=True), 1 - t, depth - 7) >= t
        best, live = -MATE_UPPER, False
        for val, move in moves():
            if move is None and depth == 0: score = pos.score
            elif move is None:
                if (cap := pos.score + EVAL_ROUGHNESS) >= gamma:
                    score = min(cap, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))
                    if score >= gamma and (proof := pos.king_capture()): move, score, live = proof, MATE_UPPER, True
                else: score = cap
            else:
                if val >= MATE_LOWER: score, live = MATE_UPPER, True
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
                    if len(self.tp_move) > TABLE_SIZE: del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break
        if depth and not live and all(pos.move(m).king_capture() for m in pos.gen_moves()):
            mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
            best = mate if pos.rotate(nullmove=True).king_capture() else 0
        if not root: self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        if len(self.tp_score) > TABLE_SIZE: del self.tp_score[next(iter(self.tp_score))]
        return best
    def search(self, history):
        self.nodes, self.history, self.tp_score = 0, set(history), {}
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
