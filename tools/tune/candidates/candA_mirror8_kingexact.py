#!/bin/sh
""":"
# Polyglot header: run with pypy3 when available, else python3 (issue #102).
# No -u needed: all UCI output is flushed explicitly (tools/uci.py).
for cmd in pypy3 python3; do
   command -v "$cmd" > /dev/null && exec "$cmd" "$0" "$@"
done
echo "Error: sunfish requires pypy3 or python3" >&2
exit 1
":"""

import os
import time
from itertools import count
from collections import namedtuple

__version__ = "2026-packed"
version = "sunfish " + __version__

###############################################################################
# Packed big-integer NNUE residual
###############################################################################
# The evaluation is
#     score = pst(pos)  +  clip(nn(pos), -CLAMP, CLAMP)
# where pst() is classic sunfish's exact incremental piece-square score (so
# `value(move)` stays exact for move ordering, the QS gate and futility) and
# nn() is a 768 -> N -> 1 net whose whole accumulator and whole head live in
# ONE Python int.  See packed/pnet.py for the lane layout and why the head
# needs no per-lane multiply.

piece = {"P": 100, "N": 272, "B": 293, "R": 484, "Q": 842, "K": 60000}
_v=0
for _c in "$iV_4Ri&GGi3(+a}XL*`5Vfp=1>#MDgpf-o:QG&lyW*uiPF;Nc@X;n*nAMCY*:/,Rw1%lcP>x=pfr{P2,uJCOF|1E7XSs+My:Dl*Y|F69;c$k(&YG*EvY~R&O@e;WnBp|2":
 _d=ord(_c)-35;_v=_v*90+_d-(_d>4)-(_d>56)
pst = {}
for _k in "PNBRQ":
 _t = [_v // 38 ** _i % 38 * 8 - 128 + piece[_k] for _i in range(32)]
 _v //= 38 ** 32
 _r = [_t[_i * 4:_i * 4 + 4] for _i in range(8)]
 pst[_k] = tuple([0] * 20 + sum(([0] + _q + _q[::-1] + [0] for _q in _r), []) + [0] * 20)
_v=0
for _c in "}:^o$X09O<a:NI?xy{|2q2QBnLicNSzFJlTdzAL.Eaku^M.48d&Uv{ezO4E+I<zWi6P+&ls&5q":
 _d=ord(_c)-35;_v=_v*90+_d-(_d>4)-(_d>56)
for _k in "K":
 _t = [_v // 183 ** _i % 183 * 1 - 99 + piece[_k] for _i in range(64)]
 _v //= 183 ** 64
 pst[_k] = tuple([0] * 20 + sum(([0] + _t[_i * 8:_i * 8 + 8] + [0] for _i in range(8)), []) + [0] * 20)
K_MID, K_END = pst["K"], tuple(piece["K"] + 70
   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))

###############################################################################

# With xz compression this whole section takes 652 bytes.
# That's pretty good given we have 64*6 = 384 values.
# Though probably we could do better...
# For one thing, they could easily all fit into int8.
# MATE values derive from the classic piece values (K=60000, Q=929);
# the tables themselves ride in the net file (see the loader above).
MATE_LOWER = 60000 - 13 * 929
MATE_UPPER = 60000 + 10 * 929

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
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
# The margin must cover the largest army a kingless side can face: nine
# queens (8 promotions + original) plus 2R+2B+2N with piece-square
# bonuses sums to 11749, which the old 10-queen margin (9290) missed by
# 2459 - a kingless position could evade the king-gone check below.
# 13 queens covers it with slack (formal/Sunfish/EvalBounds.lean proves
# both the leak and the repair).
# Every static evaluation must stay within [-MATE_UPPER, MATE_UPPER]: the
# transposition table's fresh entries assume it (formal/Sunfish/Tricks.lean,
# `Bounded`). The tables above guarantee it; keep it true if you change them.

# Constants for tuning search
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15
# Probes per depth before the MTD driver gives up and commits what it has.
# The stable engine provably needs <= 15; this engine may not, so the bound
# is enforced rather than assumed.
PROBE_CAP = 40
# Late move reduction: reduce quiet moves whose static value is below this,
# once past the first few in the sorted list. 0 disables (classic parity).
LMR = 60
# Max entries kept in each transposition table, roughly 1GB per million.
# Python dicts keep insertion order, so we cheaply evict the oldest entry
# when full (see issue #95).
TABLE_SIZE = 10**6

# minifier-hide start
opt_ranges = dict(
    QS = (0, 300),
    QS_A = (0, 300),
    EVAL_ROUGHNESS = (0, 50),
    TABLE_SIZE = (10**4, 10**8),
)
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation: ps + the clipped net residual
    ps -- the piece-square part of the score alone, kept exactly incremental
          so that value(move) below stays an exact delta of it
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    acc -- the packed NNUE accumulator (one big int, 2N + 2*nb lanes)
    pf -- perspective flag: which of the two lane blocks is the mover's
    kb -- combined king-bucket index B*bucket(white) + bucket(black), in
          ABSOLUTE colours (0 for plain B == 1 nets)

    score/ps/acc/pf/kb are all functions of the other fields, so
    identity -- what the transposition table, the killer table and the
    repetition set key on -- deliberately ignores them.  Keeping the
    accumulator out of __hash__ also keeps hashing off the big int, which
    would otherwise cost more than the evaluation it feeds.
    """

    def __hash__(self):
        return hash((self.board, self.wc, self.bc, self.ep, self.kp))

    def __eq__(self, o):
        return (self.board == o.board and self.ep == o.ep and self.kp == o.kp
                and self.wc == o.wc and self.bc == o.bc)

    def __ne__(self, o):
        return not self.__eq__(o)

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        # NB: `in <literal-str>` is ~30% faster than the equivalent .isupper() /
        # .isspace() / .islower() method calls in CPython; this matters because
        # these checks run millions of times per search.
        for i, p in enumerate(self.board):
            if p not in "PNBRQK":
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q in " \nPNBRQK":
                        break
                    # Pawn move, double move and capture
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                            #and j != self.ep and abs(j - self.kp) >= 2
                        ):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q in "pnbrqk":
                        break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove.
        The accumulator is unchanged; only which block is "ours" flips, and
        that flips the sign of the net residual exactly as it flips ps."""
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # Castling rights, we move the rook or capture the opponent's
        wc = (wc[0] and i != A1, wc[1] and i != H1)
        bc = (bc[0] and j != H8, bc[1] and j != A8)
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                r = A1 if j < i else H1
                board = put(board, r, ".")
                board = put(board, kp, "R")
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        # We rotate the returned position, so it's ready for the next player
        return Position(board[::-1].swapcase(), -score, bc, wc,
                        119 - ep if ep else 0, 119 - kp if kp else 0)

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q in "pnbrqk":
            score += pst[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        # Castling
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        # Special pawn stuff
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
        return score

    def king_capture(self):
        """The move that takes the opponent king, if any - i.e. the proof
        that this position was reached by an illegal move. Same test as
        gen_moves/value: the target is the king, or within one of the
        king-passant square (kp == 0 is safe: targets are >= A8 > 1).
        Serves double duty: found from a position it is the sentinel
        witness the search substitutes for a virtual cutoff; found from
        the null-rotation it says the side to move is in check."""
        return next((m for m in self.gen_moves()
                     if self.board[m.j] == "k" or abs(m.j - self.kp) < 2), None)


###############################################################################
# Search logic
###############################################################################

# Raised inside the search when the wall-clock deadline passes
class Stop(Exception): pass


# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score, self.tp_move, self.history = {}, {}, set()
        self.nodes, self.deadline = 0, 1 << 63
        # minifier-hide start
        self.node_cap = 1 << 62          # testing only; see bound()
        # minifier-hide end

    def bound(self, pos, gamma, depth, root=False):
        """ Let s* be the score of the sub-tree from pos at this depth, as
            a function of (pos, depth) alone. This includes null moves and
            QS pruning, and global parameters like self.history that don't
            change during search. (Things that change, like tp_move or gamma,
            are not allowed to change the sub-tree and value of s*.)

            It is assumed 1 - MATE_UPPER < gamma <= MATE_UPPER.

            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound)

            Note, bound() is not guaranteed to be deterministic: stored values
            in self.tp_score may be used to return a bound that is not the best
            possible, but it is guaranteed to be valid according to the rules above.

            On top of the bound, three exact promises:
            - our own king already captured: r = -MATE_UPPER.
            - if depth >= 1:
                - if the opponent king capturable: r = MATE_UPPER
                  (note this is stronger than just gamma <= r <= s*.)
                - if mate/stalemate returns the exact -MATE_LOWER / 0.
            - if gamma <= r, tp_move[pos] will hold a legal move achieving r.
            """

        self.nodes += 1
        # minifier-hide start
        # Node budget enforced INSIDE the search, at the same granularity as
        # the deadline. Checking a node cap only between completed depths
        # rewards whichever engine prunes LESS: its last iteration is bigger,
        # so it sails further past the cap. Measured at a 20000 cap, classic
        # (no LMR) reached 34742 nodes -- 1.74x -- against 26336 for the same
        # engine with LMR, a ~30% free advantage worth ~38 Elo. Testing-only:
        # the 4k rules mandate no node command, so the artifact carries none
        # of this.
        if self.nodes % 2048 == 0 and self.nodes > self.node_cap: raise Stop
        # minifier-hide end
        # Enforce the time budget inside the search: iteration boundaries can
        # be seconds apart on slow hardware, this is checked every ~2k nodes.
        if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        # This reads `ps`, the piece-square part, NOT the evaluated score:
        # the sentinel means "the king is literally gone", and a net residual
        # of up to CLAMP could otherwise lift a kingless position back over
        # the threshold and hide the capture. On ps the test is bit-for-bit
        # classic's.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # Driver probes (the search root, and IID below) are UNSTORED: they
        # skip the table in both directions, the repetition-0 and the null
        # option, and store nothing - so every entry in the table describes
        # ONE value function, determined by (pos, depth) alone, and the key
        # needs no flag. (The root's own entry was provably dead weight: the
        # driver picks each gamma strictly inside its bracket, which is the
        # same two numbers the entry held.) At the root 'entry' stays
        # unbound - its only other reader is the store below, also skipped.
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma: return entry.lower
            if entry.upper < gamma: return entry.upper

            # Let's not repeat positions. We don't chat
            # - at the root (a driver probe) since it is in history, but not a draw.
            # - at depth=0, since it would be expensive and break "futility pruning".
            if depth > 0 and pos in self.history: return 0


        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # Look for the strongest move from earlier searches of this position.
            # See https://chessprogramming.org/Killer_Move for details.
            # We read this "killer move" before null-move in case it would get
            # evicted from the table or replaced with something else worse.
            killer = self.tp_move.get(pos)

            # First try not moving at all, i.e. the null move.
            # See https://chessprogramming.org/Null_Move for details.
            # The idea is that "doing nothing" is a lower bound on the score
            # of the position, but we have to be careful with zugzwang, where
            # passing is better than any move - the piece test guards that
            # (K+P endings). No null at root, so we can always return a move.
            #
            # THE SCORE IS NOT CAPPED, and that is a real difference from
            # classic, which clamps it to min(pos.score + EVAL_ROUGHNESS, ...).
            # We never have capped it: git log -S finds no such line anywhere
            # in this file's history. The comment that stood here until
            # 2026-08-13 asserted the cap was present ("both halves stay") and
            # cited a 900-game test of removing it - it had been adapted from
            # classic and described a decision that was never implemented
            # here, so it is deleted rather than corrected.
            #
            # Uncapped is looser in two ways, not one: `score >= gamma` fires
            # more often AND the yielded score becomes this node's value, so
            # an over-optimistic pass propagates into the tt and into the MTD
            # bisection, which then trusts it. Whether the cap earns its bytes
            # against a PST eval is UNDER MEASUREMENT (2026-08-13 round-robin;
            # an older 300-game test on the NNUE eval was flat, which by the
            # (feature, eval) rule does not settle it here). Until that lands,
            # this comment describes the code as written.
            if not root and depth > 2 and abs(pos.score) < 500 and any(c in pos.board for c in "RBNQ"):
                score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)
                # A fail high is a virtual claim, and needs verification
                # before it may cut: if the king is capturable the capture is
                # substituted (the node must report the exact MATE_UPPER)
                proof = score >= gamma and (self.tp_move.get(pos) or pos.king_capture())
                if proof and pos.value(proof) >= MATE_LOWER:
                    yield proof, MATE_UPPER
                # a remaining mate-band claim is vacuous (if passing wins the
                # king, capturing it is a real move too). Otherwise one probe
                # at the band boundary is decisive both ways
                # (boundary_window_decisive): fail-low means the pass really
                # wins in the band - vetoed by omission - and fail-high
                # certifies the value sub-band, letting the cutoff stand
                # with no chess assumption (the premise it replaced is false
                # in real chess: 8/6p1/6R1/k7/2K5/8/8/8 w).
                elif score < gamma or self.bound(pos.rotate(nullmove=True),
                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:
                    yield None, score

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else. (Note depth at root is always > 0.)
            if depth == 0:
                yield None, pos.score

            # Back to killer moves: This heuristic is so good, that if there
            # is no registered move, it's worth it to run a shallow search to find one.
            # See https://chessprogramming.org/Internal_Iterative_Deepening for detais.
            # This is known as Internal Iterative Deepening (IID). The probe
            # runs as a driver probe (root=True): no null cutoff that would
            # end it without storing a move, no repetition truncation, and
            # no table entry under deviant semantics.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, root=True)
                killer = self.tp_move.get(pos)

            # We only generate moves with an intrinsic score above some treshold
            # that decreases with depth. This is a generalization of Quiescent Search,
            # See https://chessprogramming.org/Quiescence_Search for details.
            val_lower = QS - depth * QS_A

            # Now finally play the killer move. But note that we have to respect
            # the QS lower bound, otherwise we would get search instability.
            # We will search it again in the main loop below, but the tp will
            # make this mostly free.
            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Then all the other moves
            # Quiescent search: only moves above the val-limit are admitted -
            # filtering BEFORE the sort skips sorting the sub-threshold tail
            # (most of the list at QS nodes), and is literally the model's
            # movesAbove form (formal/Sunfish/Stalemate.lean).
            # NOTE the iteration order is a soundness contract, not a
            # heuristic: the futility break below discards the rest of the
            # list, which is only valid when iteration descends in static
            # value. A history-credit order key tried here scrambled that
            # order and paid -449 Elo (ledger 5f5f34d); made sound, the
            # history table measured a 1.01 node ratio -- worthless.
            for cnt, (val, move) in enumerate(sorted(((v, m) for m in pos.gen_moves()
                                     if (v:=pos.value(m)) >= val_lower), reverse=True)):
                # If the new score is less than gamma, the opponent will for sure just
                # stand pat, since ""pos.score + val < gamma === -(pos.score + val) >= 1-gamma""
                # This is known as futility pruning.
                if depth <= 1 and pos.score + val < gamma:
                    # Need special case for MATE, since it would normally be caught
                    # before standing pat. A sub-mate futility yield estimates
                    # the child's stand-pat without searching it, so it is
                    # value evidence only, never legality evidence: it goes
                    # out as a virtual (None) yield - it can never cut (its
                    # score is below gamma by construction), and it must not
                    # set 'live' and mask the terminality correction below.
                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)
                    # We can also break, since we have ordered the moves by value,
                    # so it can't get any better than this.
                    break

                # LATE MOVE REDUCTION. The move list is sorted by static
                # value, so a quiet move arriving late is one the ordering
                # already judged unpromising: search it a ply shallower and
                # only pay full depth if it surprises us. This is the first
                # deliberate break of one-value-per-key -- the reduction
                # depends on cnt, which depends on ordering, which depends on
                # mutable table state, so the same position can now be
                # searched to different depths on different visits. The MTD
                # guards in search() exist for exactly this.
                # A null-window driver makes the verification re-search cheap:
                # the reduced search only needs to be trusted when it FAILS
                # LOW (score < gamma), and a fail-high is re-run at full depth.
                red = LMR and depth > 2 and cnt > 2 and val < LMR
                score = -self.bound(pos.move(move), 1 - gamma, depth - 2 if red else depth - 1)
                if red and score >= gamma:
                    score = -self.bound(pos.move(move), 1 - gamma, depth - 1)
                yield move, score

        # Run through the moves, shortcutting when score >= gamma.
        # live is True if we saw a legal (not null, score > -MATE_UPPER) move
        best, live = -MATE_UPPER, False
        for move, score in moves():
            best = max(best, score)
            live |= move is not None and score > -MATE_UPPER
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None and depth:
                    self.tp_move[pos] = move
                    # Never evict the current search root: its killer is the
                    # answer go_loop plays, and once the table churns more
                    # than TABLE_SIZE stores in one deep probe, FIFO would
                    # age it out MID-SEARCH - a later deep fail-low probe
                    # then stores whatever capture sorts first and a timeout
                    # plays it (three -5ish queen/piece giveaways in 145
                    # production games).
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break

        # If we didn't see any legal moves, it might just be that we failed
        # high on a null move and stopped searching, but it could also be that
        # we genuinely re in checkmate or stalemate. There's no way to know but
        # to check.
        if depth and not live and all(
                pos.move(m).king_capture() for m in pos.gen_moves()):
            # We can't move, but is it a checkmate or stalemate?
            best = -MATE_LOWER if pos.rotate(nullmove=True).king_capture() else 0

        # Table part 2. Every search decision is gamma-independent, so all
        # bounds target one value function determined by the key and stored
        # entries can never contradict each other (lower > upper). A change
        # that lets gamma (or any key-external state) select between
        # incomparable evaluations of a move breaks this - that is a bug,
        # not a configuration; see formal/README.md.
        if not root:
            self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        if len(self.tp_score) > TABLE_SIZE:
            del self.tp_score[next(iter(self.tp_score))]

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes, self.history = 0, set(history)
        self.tp_score.clear()
        # Table choice is fixed for the whole search (and tp_score is
        # cleared above), so every bound targets one value function.
        pos = self.root = history[-1]

        # Classic's K_END is a centralization gradient, and classic keys it
        # on queens-off. Both directions every search: table state must
        # never outlive the condition.
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
        # The carried score was accumulated under the OTHER table.
        pos = self.root = from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp)

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            lower, upper, probes = 1 - MATE_UPPER, MATE_UPPER, 0
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                # INSTABILITY GUARDS. This engine deliberately breaks
                # one-value-per-key (reductions, history, gamma-dependent
                # cutoffs), and MTD-bi has no real window in which to absorb
                # a contradiction: it only ever probes null windows and
                # bisects on the answers, assuming they bracket ONE function.
                # Two probes can now disagree (">= 100" at gamma=100, then
                # "< 50" at gamma=50), so:
                #  (a) tighten MONOTONICALLY -- max/min rather than plain
                #      assignment -- so a contradictory probe can never widen
                #      the bracket and spin the loop forever;
                #  (b) stop if the bracket CROSSES, committing the last
                #      self-consistent value rather than bisecting nonsense;
                #  (c) cap the probes per depth. We used to PROVE <= 15; that
                #      proof does not survive instability, so the invariant
                #      becomes a runtime check that trips loudly (dev builds)
                #      instead of silently ceasing to hold.
                if score >= gamma: lower = max(lower, score)
                else: upper = min(upper, score)
                probes += 1
                yield depth, gamma, score, self.tp_move.get(pos)
                if lower > upper or probes > PROBE_CAP:
                    # minifier-hide start
                    if probes > PROBE_CAP:
                        print("info string MTD-GUARD probe cap hit: depth", depth,
                              "lower", lower, "upper", upper, flush=True)
                    if lower > upper:
                        print("info string MTD-GUARD bracket crossed: depth", depth,
                              "lower", lower, "upper", upper, flush=True)
                    # minifier-hide end
                    break
                gamma = (lower + upper + 1) // 2


###############################################################################
# UCI User interface
###############################################################################

# parse/render/hist live at module level: tools/uci.py (and the tests)
# reach them as engine-module attributes, and main() uses hist before its
# own body would define it.
def parse(c): return A1 + ord(c[0]) - ord("a") - 10 * (int(c[1]) - 1)
def render(i): return chr((i - A1) % 10 + ord("a")) + str(1 - (i - A1) // 10)

def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position from scratch; `board` is in the mover's orientation."""
    score = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
                for i, p in enumerate(board) if p.isalpha())
    return Position(board, score, wc, bc, ep, kp)


hist = [from_board(initial)]


def main():
    # minifier-hide start
    # Development checkout: use the full-featured UCI interface in
    # sunfish_ui/ (pondering, Hash option, spec-complete go parsing,
    # FEN, node limits). An installed or packed sunfish has no
    # sunfish_ui/ and falls through to the built-in loop below, which is
    # all the 4k rules require.
    try:
        import sys
        import inspect
        # NOTE this puts THIS FILE'S GRANDPARENT at the front of sys.path,
        # ahead of PYTHONPATH and the repo, so any stray sunfish_ui/ sitting
        # there wins the import. A stale copy did exactly that once and
        # silently turned 425 fixed-node games into movetime games -- every
        # one a time forfeit -- because it predated `go nodes`. Hence the two
        # rules below: say which driver resolved, and refuse to run one that
        # is missing a capability rather than degrading into the builtin loop.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import sunfish_ui.uci as _drv
    except ImportError:
        _drv = None
    if _drv is not None:
        _p = inspect.signature(_drv.go_loop).parameters
        _nodes, _fen = "max_nodes" in _p, hasattr(_drv, "from_fen")
        # A capability check catches a MISSING feature; a stale copy that
        # merely predates a FIX passes every capability test while behaving
        # differently. The version stamp is the only thing that catches that,
        # and it is the cheapest insurance against tonight's worst failure
        # class -- three separate incidents from one shadowed driver.
        _ver = getattr(_drv, "DRIVER_VERSION", 0)
        print("info string driver", _drv.__file__, "v%d" % _ver,
              "nodes" if _nodes else "NO-NODES", "fen" if _fen else "NO-FEN",
              flush=True)
        REQUIRED_DRIVER = 2      # raise with DRIVER_VERSION, same commit
        if _ver < REQUIRED_DRIVER:
            raise SystemExit(
                "sunfish_ui driver at %s is version %d, need >= %d. This is a "
                "STALE copy shadowing the repo one: sys.path puts this file's "
                "grandparent first, so a scratch copy wins the import and "
                "silently behaves like an older engine (it voided 425 games "
                "once). Delete it or refresh it from the repo."
                % (_drv.__file__, _ver, REQUIRED_DRIVER))
        if not (_nodes and _fen):
            raise SystemExit(
                "sunfish_ui driver at %s lacks required capabilities "
                "(max_nodes=%s from_fen=%s). Refusing to fall back to the "
                "builtin loop: that fallback is what made the last failure "
                "silent (movetime-only, then startpos-only against an EPD "
                "book). Remove the stale copy or fix the driver."
                % (_drv.__file__, _nodes, _fen))
        return _drv.run(sys.modules[__name__], hist[-1])
    # minifier-hide end

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
            # The times may come in any order and combination, e.g. "go wtime 100 btime 100"
            times = dict(zip(args[1::2], map(int, args[2::2])))
            side = "wb"[len(hist) % 2 == 0]
            wtime, winc = times.get(side + "time", 60000), times.get(side + "inc", 0)
            # increment-aware budget; see sunfish_ui/uci.py for the audit
            # numbers and the safety argument
            think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)
            # minifier-hide start
            # SUDDEN DEATH needs a flatter divisor. With winc == 0, /12 spends
            # 7% of the whole budget on one early move (12.8s of 180s on ply 9
            # in lichess.org/EAThUL0P) and the game is lost on time at move 73
            # without a single move overrunning: below 2s the wtime/2 - 1000
            # cap goes negative, the budget collapses to the 0.05s floor, and
            # ~200ms/move of unavoidable lag drains the rest.
            # /40 is what classic uses and classic does not flag, so this is a
            # constant with production evidence rather than a fit to one game.
            # Movecount-aware divisors were simulated and are WORSE: a
            # shrinking "moves remaining" divisor spends MORE per move as the
            # game lengthens, which is backwards for sudden death.
            # TCEC is 1800+3, so winc is always non-zero there and this line
            # is dead code in the artifact -- which is why it is hidden, and
            # why the artifact stays byte-for-byte unchanged. The increment
            # case is identical to the line above by construction.
            think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000)
            # minifier-hide end
            # A GUI-supplied movetime is a HARD limit that the GUI itself
            # enforces, so spending all of it forfeits: the node counter is
            # only checked every 2048 nodes, so the search returns at
            # movetime + epsilon and the GUI has already called the flag.
            # Keep 5% back (min 30ms) as polling slack. Measured the hard
            # way: 425 local fixed-node games, every single one a forfeit.
            think = times.get("movetime", think) / 1000
            if "movetime" in times: think -= max(think * .05, .03)

            start = time.time()
            # Hard in-search deadline: iteration boundaries can be seconds
            # apart at deep depths, so the soft 0.8*think break alone can
            # overrun arbitrarily and forfeit on time.  max(.05) keeps a
            # degenerate clock from stopping before any move exists.
            searcher.deadline = start + max(think, .05)
            # best is COMMITTED only when a depth completes (its bracket
            # converged): a mid-depth fail-high can come from a deep
            # fail-low dive probe at an absurd gamma and is only a
            # candidate (classic's Qxc6 giveaway class).
            best, cand, d0 = None, None, 1
            # minifier-hide start
            # "go nodes N": equal-effort matches. Testing-only -- the 4k
            # rules mandate no such command, so the artifact does not carry
            # it. Without this a fixed-node match silently becomes a
            # movetime match and every game ends in a forfeit.
            max_nodes = times.get("nodes", 0)
            searcher.node_cap = max_nodes or 1 << 62
            # minifier-hide end
            try:
                for depth, gamma, score, move in searcher.search(hist):
                    if depth > d0:
                        best, d0 = cand or best, depth
                    # minifier-hide start
                    if max_nodes and searcher.nodes >= max_nodes and (best or cand):
                        break
                    # minifier-hide end
                    if score >= gamma:
                        i, j = move.i, move.j
                        if len(hist) % 2 == 0:
                            i, j = 119 - i, 119 - j
                        cand = render(i) + render(j) + move.prom.lower()
                        print("info depth", depth, "score cp", score, "pv", cand)
                    if (best or cand) and time.time() - start > think * 0.8:
                        break
            except Stop:
                pass

            print("bestmove", best or cand or '(none)')


if __name__ == "__main__":
    main()
