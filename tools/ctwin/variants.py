"""Python reference for the tp_move replacement-policy battery.

The C twin's EVICT_POLICY / EVICT_SCAN_K / KILLER_COUNT knobs need a
Python reference to difftest against.  sunfish.py only contains policy 0
with one killer, so the variants live here: VariantSearcher transcribes
sunfish.Searcher.bound VERBATIM except that every tp_move access goes
through a policy object, and the policy objects define the battery
semantics (documented below, implemented bit-for-bit in sunfish.c).

DRIFT GUARD: the transcription is only a reference while it matches the
real bound().  The SHA256 of sunfish.Searcher.bound/search source is
pinned here and asserted on import -- if sunfish.py changes, this module
refuses to run rather than measure a stale reference (never hide errors).
On top of the static guard, difftest proves the transcription live:
USE_VARIANT=1 with default knobs must be byte-identical to the real
Searcher (and to the C twin) before any variant cell counts.

Battery semantics (shared contract with sunfish.c):
- Killer lists: the KILLER_COUNT most recent DISTINCT fail-high moves,
  most recent first, each with its last-store depth.  Single-move readers
  (null proof, driver yield) take the most recent; the killer search
  phase tries all k in order, each gated by the QS threshold.  k-deepest
  is a noted follow-up, not implemented.
- EVICT_POLICY 0: master root-guarded FIFO insert-then-evict (>).
- EVICT_POLICY 1: unguarded evict-BEFORE-insert (>=): may evict the root,
  and may evict the very key being stored (fresh re-insert at the tail).
- EVICT_POLICY 2: insert-then-evict (>); scan the first
  min(EVICT_SCAN_K, len) FIFO entries, evict the shallowest last-store
  depth, ties to the earliest scanned.  No root guard.
- EVICT_POLICY 3: fixed hash-slot table, TABLE_SIZE buckets x two tiers
  (deep slot: replace if new depth >= stored depth; else always-replace
  slot).  Bucketing uses the C twin's content hash, reproduced here
  bit-for-bit, so collisions -- which are observable through search
  behavior -- agree across languages.  Exact-position compare on read.

Frozen-guide battery (design: Thomas Ahle) -- same shared contract:
- Two generations of the move table.  tp_old is the FROZEN GUIDE, a
  value-bearing policy held constant for one epoch; tp_move stays the
  MUTABLE CURRENT table and may only affect ordering and the returned
  move.  A mutable table that could move a reduction, an admission or a
  searched depth would change the value at a tp_score key under it.
- THE EPOCH RULE: a tp_score interval is valid for exactly one guide, so
  every promotion clears tp_score in the same breath.
  GUIDE_MODE 1 promotes after each COMPLETED ID bracket (a partial
    iteration is never promoted: a mid-depth stop keeps the previous
    completed guide).  GUIDE_MODE 2 freezes one guide per search() call
    -- the previous root search's completed table -- which is exactly
    the existing tp_score lifetime and costs no epoch churn.
- SCORE_EPOCH 1 is the isolated control: clear tp_score per ID iteration
  and change nothing else.
- TWO_KILLERS searches the guide as a second killer when it is distinct
  from the current killer; KILLER_DEDUP skips already-searched killers
  in the sorted list, AFTER the futility test so the futility yield
  master would emit still happens (then it is exactly max(x, x, ...) =
  max(x, ...)).
- GUIDE_IIR replaces the recursive IID probe with a one-ply reduction on
  guideless nodes.  Root is never reduced and NOMINAL depth still keys
  tp_score, sets val_lower, scores mates, classifies terminals and gates
  every eligibility test; only the real-child recursion is shortened.
- GUIDE_INJECT admits the guide at positive depth regardless of
  val_lower (A_G = A union {G(p)}).  GUIDE_PV gives the guide the full
  child depth and takes a ply off the alternatives.
- Guide lookups are delayed to after the null phases and gated at
  depth > 3, so a null cutoff and every shallow node pay nothing.
"""
import hashlib
import inspect
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sunfish as S
from sunfish import Entry, MATE_LOWER, MATE_UPPER, Searcher, Stop

_PINNED = {
    # post-#192 master: fuel-oracle null (classic null bounded 2<depth<6,
    # fuel probe from depth 6 at pos.score + NULL_MARGIN, real moves at d-1)
    "bound": "fa52e0701c8e3f1f405915ed0a8079bc21ebd65d66ebfb02c1ac74b993dfe528",
    "search": "ffae8dfd56348310dd38a86126a2291b7111870b8bc66bced4b2d724ed1ee721",
}
for _name, _want in _PINNED.items():
    _got = hashlib.sha256(inspect.getsource(getattr(Searcher, _name)).encode()).hexdigest()
    if _got != _want:
        raise RuntimeError(
            "variants.py drift guard: sunfish.Searcher.%s changed "
            "(sha256 %s != pinned %s). Re-transcribe VariantSearcher.bound, "
            "re-run the full difftest gate, then re-pin." % (_name, _got, _want))

# Battery knobs (set by pyref.py; defaults = master semantics).
EVICT_POLICY = 0
EVICT_SCAN_K = 4
KILLER_COUNT = 1
SCORE_EPOCH = 0
GUIDE_MODE = 0
GUIDE_MIN_DEPTH = 3
GUIDE_COPY = 0
TWO_KILLERS = 0
KILLER_DEDUP = 0
GUIDE_IIR = 0
GUIDE_INJECT = 0
GUIDE_PV = 0

_M64 = (1 << 64) - 1


def _mix64(x):
    x &= _M64
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & _M64
    x ^= x >> 29
    x = (x * 0xC4CEB9FE1A85EC53) & _M64
    x ^= x >> 32
    return x


def pos_hash(pos):
    """Bit-for-bit reproduction of sunfish.c pos_seal (little-endian)."""
    ws = struct.unpack("<15Q", pos.board.encode("ascii"))
    h = 0
    for k, w in enumerate(ws):
        h = ((h * 0x100000001B3) & _M64) ^ _mix64(w + k)
    flags = int(pos.wc[0]) | int(pos.wc[1]) << 1 | int(pos.bc[0]) << 2 | int(pos.bc[1]) << 3
    h = ((h * 0x100000001B3) & _M64) ^ _mix64(((pos.score & 0xFFFFFFFF) << 8) | flags)
    h = ((h * 0x100000001B3) & _M64) ^ _mix64(((pos.ep & 0xFFFFFFFF) << 32) | (pos.kp & 0xFFFFFFFF))
    return h


def _push(lst, move, depth):
    """Most-recent-first dedupe push, capped at KILLER_COUNT."""
    return [(depth, move)] + [(d0, m0) for d0, m0 in lst if m0 != move][:KILLER_COUNT - 1]


class DictKillers:
    """tp_move for policies 0-2: insertion-ordered dict pos -> killer list."""

    def __init__(self, searcher):
        self.d = {}
        self.searcher = searcher

    def get(self, pos, default=None):          # single-move readers
        e = self.d.get(pos)
        return e[0][1] if e else default

    def get_all(self, pos):
        e = self.d.get(pos)
        return [m for _, m in e] if e else []

    def store(self, pos, move, depth):
        if EVICT_POLICY == 1 and len(self.d) >= S.TABLE_SIZE:
            # Unguarded evict-before-insert; read the old list only AFTER
            # the eviction (the evicted key may be pos itself).
            del self.d[next(iter(self.d))]
        self.d[pos] = _push(self.d.get(pos) or [], move, depth)
        if EVICT_POLICY == 0 and len(self.d) > S.TABLE_SIZE:
            del self.d[next(k for k in self.d if k != self.searcher.root)]
        elif EVICT_POLICY == 2 and len(self.d) > S.TABLE_SIZE:
            it = iter(self.d.items())
            scan = [next(it) for _ in range(min(EVICT_SCAN_K, len(self.d)))]
            victim = min(scan, key=lambda kv: kv[1][0][0])[0]   # shallowest, ties earliest
            del self.d[victim]


class SlotKillers:
    """tp_move for policy 3: TABLE_SIZE buckets x (deep, always) slots."""

    def __init__(self, searcher):
        self.n = S.TABLE_SIZE if S.TABLE_SIZE > 0 else 1
        self.slots = [None] * (2 * self.n)

    def _base(self, pos):
        return 2 * (pos_hash(pos) % self.n)

    def get(self, pos, default=None):
        a = self.get_all(pos)
        return a[0] if a else default

    def get_all(self, pos):
        b = self._base(pos)
        for t in (b, b + 1):
            e = self.slots[t]
            if e is not None and e[0] == pos:
                return [m for _, m in e[1]]
        return []

    def store(self, pos, move, depth):
        b = self._base(pos)
        for t in (b, b + 1):
            e = self.slots[t]
            if e is not None and e[0] == pos:                  # in-place update
                self.slots[t] = (pos, _push(e[1], move, depth))
                return
        deep = self.slots[b]
        t = b if (deep is None or depth >= deep[1][0][0]) else b + 1
        self.slots[t] = (pos, [(depth, move)])


def make_killers(searcher):
    return SlotKillers(searcher) if EVICT_POLICY == 3 else DictKillers(searcher)


def copy_killers(src):
    """dict(tp_move): same keys, same insertion order, frozen lists."""
    out = DictKillers(src.searcher)
    out.d = dict(src.d)
    return out


class VariantSearcher(Searcher):
    """sunfish.Searcher with tp_move behind a policy object.

    bound() below is a transcription of sunfish.Searcher.bound (drift-
    guarded above); the ONLY differences are the killer-table accesses:
      killer = tp_move.get(pos)          -> killers = tp_move.get_all(pos)
      if killer and value >= val_lower   -> for killer in killers: ...
      tp_move[pos] = move (+ eviction)   -> tp_move.store(pos, move, depth)
    search() is inherited: its tp_move.get(pos) yield-read hits the policy
    object's dict-compatible .get()."""

    def __init__(self):
        super().__init__()
        self.tp_move = make_killers(self)
        self.tp_old = make_killers(self)

    def promote(self):
        """The guide changes only here, and tp_score is emptied in the
        same breath, so no interval outlives its guide.  GUIDE_COPY takes
        the guide as a SNAPSHOT instead of emptying the current table:
        the killer lists are rebuilt, never mutated in place, so a
        shallow dict copy is genuinely frozen."""
        if GUIDE_COPY:
            self.tp_old = copy_killers(self.tp_move)
        else:
            self.tp_old, self.tp_move = self.tp_move, make_killers(self)
        self.tp_score.clear()

    def search(self, history):
        if GUIDE_MODE and EVICT_POLICY == 3:
            # Policy 3 replaces tp_move with a slot table, so a promotion
            # would freeze an empty guide.  Refuse, never measure a no-op.
            raise RuntimeError("GUIDE_MODE is incompatible with EVICT_POLICY 3")
        self.nodes, self.history = 0, set(history)
        self.tp_score.clear()
        if GUIDE_MODE == 2: self.promote()
        pos = self.root = history[-1]
        S.pst["K"] = S.K_MID if "Q" in pos.board and "q" in pos.board else S.K_END
        gamma = 0
        for depth in range(1, 1000):
            lower, upper = 1 - MATE_UPPER, MATE_UPPER
            while lower < upper - S.EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma: lower = score
                if score < gamma: upper = score
                yield depth, gamma, score, self.tp_move.get(pos)
                gamma = (lower + upper + 1) // 2
            # Bracket COMPLETED: only now may the guide be promoted.  A
            # Stop inside the bracket never reaches this line, so the
            # previous completed guide stays in place.
            if GUIDE_MODE == 1: self.promote()
            elif SCORE_EPOCH: self.tp_score.clear()

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

        def moves():
            killers = self.tp_move.get_all(pos)

            if not root and 2 < depth < 6 and abs(pos.score) < 500 and any(c in pos.board for c in "RBNQ"):
                score = min(pos.score + S.EVAL_ROUGHNESS,
                    -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))
                proof = score >= gamma and (self.tp_move.get(pos) or pos.king_capture())
                yield (proof, MATE_UPPER) if proof and pos.value(proof) >= MATE_LOWER else (None, score)

            # Fuel oracle (master since #192): a fuel decision, never a
            # score candidate; real moves below recurse to d - 1.
            d = depth
            if depth >= 6 and abs(pos.score) < 500 and any(c in pos.board for c in "RBNQ"):
                target = pos.score + S.NULL_MARGIN
                if -self.bound(pos.rotate(nullmove=True), 1 - target, depth - 3) >= target:
                    d = depth - 1

            if depth == 0:
                yield None, pos.score

            # Frozen guide, read only AFTER the null phases: a null
            # cutoff and every node at depth <= 3 pay no lookup.
            guide = self.tp_old.get(pos) if (GUIDE_MODE and depth > GUIDE_MIN_DEPTH) else None

            # Either the recursive IID probe or the one-ply IIR that
            # replaces it.  The reduction reads the FROZEN table only, so
            # the searched depth is a function of (pos, depth, epoch).
            red = 0
            if GUIDE_IIR:
                red = int(not root and depth > 3 and guide is None)
            elif not killers and depth > 3:
                self.bound(pos, gamma, depth - 3, root=True)
                killers = self.tp_move.get_all(pos)

            val_lower = S.QS - depth * S.QS_A
            # Child depth for every real move.  NOMINAL depth keeps the
            # table key, val_lower, mate distance, terminal classification
            # and every eligibility test above.
            cd = gd = d - 1 - red
            if GUIDE_PV and guide is not None and not root and depth > 3:
                gd, cd = d - 1, d - 2

            tried = []
            for killer in killers:
                if pos.value(killer) >= val_lower:
                    tried.append(killer)
                    yield killer, -self.bound(pos.move(killer), 1 - gamma, cd)
            if (guide is not None and (TWO_KILLERS or GUIDE_INJECT or GUIDE_PV)
                    and guide not in tried
                    and (pos.value(guide) >= val_lower or (GUIDE_INJECT and depth > 0))):
                tried.append(guide)
                yield guide, -self.bound(pos.move(guide), 1 - gamma, gd)

            for val, move in sorted(((v, m) for m in pos.gen_moves() if (v := pos.value(m)) >= val_lower), reverse=True):
                if depth <= 1 and pos.score + val < gamma:
                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)
                    break
                # Dedup AFTER the futility test, so the futility yield
                # master would emit for this move still happens.
                if KILLER_DEDUP and move in tried:
                    continue

                yield move, -self.bound(pos.move(move), 1 - gamma, cd)

        best, live = -MATE_UPPER, False
        for move, score in moves():
            best = max(best, score)
            live |= move is not None and score > -MATE_UPPER
            if best >= gamma:
                if move is not None and depth:
                    self.tp_move.store(pos, move, depth)
                break

        if depth and not live and all(
                pos.move(m).king_capture() for m in pos.gen_moves()):
            mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * S.EVAL_ROUGHNESS)
            best = mate if pos.rotate(nullmove=True).king_capture() else 0

        if not root:
            self.tp_score[pos, depth] = Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        if len(self.tp_score) > S.TABLE_SIZE:
            del self.tp_score[next(iter(self.tp_score))]

        return best
