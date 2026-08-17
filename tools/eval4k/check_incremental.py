"""TWO WITNESSES for an incremental evaluation, because one is not enough.

  1. `value(move)` is an EXACT delta of the carried `score`;
  2. the carried score of a child equals a FROM-SCRATCH rebuild of it.

Check 1 alone certifies nothing. It compares `value()` against `move()`, and
those are the two halves of the same idea written by the same hand -- when the
`pdbl` accumulator invented a phantom doubled pawn on every straight push, both
halves invented it identically and check 1 passed on 17,932 moves. Check 2 is
the independent witness, and it is the one that caught it.

Any extra Position field the arm carries is compared too, so an accumulator
that drifts is caught even while the score happens to agree.

usage: check_incremental.py ENGINE.py [GAMES] [PLIES]
"""
import importlib.util, random, sys
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec)
    sys.modules[n] = m; spec.loader.exec_module(m); return m
ENGINE = sys.argv[1]
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
PLIES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
m = load(ENGINE, "arm")
random.seed(20260817)
bad_delta = bad_rebuild = checked = 0
for game in range(GAMES):
    pos = m.from_board(m.initial)
    for ply in range(PLIES):
        ms = pos.gen_moves()
        legal = [mv for mv in ms if not pos.move(mv).k()]
        if not legal: break
        mv = random.choice(legal)
        child = pos.move(mv)
        checked += 1
        if child.score != -(pos.score + pos.value(mv)):
            bad_delta += 1
            if bad_delta <= 3:
                print("DELTA MISMATCH", mv, "carried", child.score,
                      "expected", -(pos.score + pos.value(mv)))
        fresh = m.from_board(child.board)
        # compare EVERY derived field, not just the score: an accumulator can
        # drift while the score it feeds happens to agree.
        drift = [f for f in child._fields
                 if f not in ("board", "wc", "bc", "ep", "kp", "r")
                 and getattr(fresh, f) != getattr(child, f)]
        if drift:
            bad_rebuild += 1
            if bad_rebuild <= 3:
                print("REBUILD MISMATCH", mv, "fields", drift,
                      "carried", [getattr(child, f) for f in drift],
                      "fresh", [getattr(fresh, f) for f in drift])
        pos = child
print("%s: moves checked %d   delta mismatches %d   rebuild mismatches %d"
      % (ENGINE, checked, bad_delta, bad_rebuild))
sys.exit(1 if (bad_delta or bad_rebuild) else 0)
