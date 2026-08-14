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
    # Intrinsic-LMR candidate: forcing-only qsearch; every positive-depth
    # move is admitted and negative-valued moves consume an extra ply.
    "bound": "593997395e10a8ad99bb66176379fc3103af760a92f7e3558f93a33b60ef45a4",
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

            if not killers and depth > 3:
                self.bound(pos, gamma, depth - 3, root=True)
                killers = self.tp_move.get_all(pos)

            val_lower = S.QS if not depth else -MATE_UPPER

            for killer in killers:
                if (val := pos.value(killer)) >= val_lower:
                    yield killer, -self.bound(pos.move(killer), 1 - gamma, d - 1 - (val < 0))

            values = ((v, m) for m in pos.gen_moves()
                if (v := pos.value(m)) >= val_lower)
            for val, move in sorted(values, reverse=True):
                if depth == 0 and pos.score + val < gamma:
                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, d - 1 - (val < 0))

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
