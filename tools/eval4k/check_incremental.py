"""The invariant that decides whether pdbl is a term or a bug:
`value(move)` must be an EXACT delta of `score`, at every move of a random
walk, and the carried score must equal a from-scratch rebuild of the child."""
import importlib.util, random, sys
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec)
    sys.modules[n] = m; spec.loader.exec_module(m); return m
m = load("bin/e_pdbl.py", "pd")
random.seed(20260817)
bad_delta = bad_rebuild = checked = 0
for game in range(300):
    pos = m.from_board(m.initial)
    for ply in range(60):
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
        if fresh.score != child.score or fresh.w != child.w:
            bad_rebuild += 1
            if bad_rebuild <= 3:
                print("REBUILD MISMATCH", mv, "carried", child.score, child.w,
                      "fresh", fresh.score, fresh.w)
        pos = child
print("moves checked %d   delta mismatches %d   rebuild mismatches %d"
      % (checked, bad_delta, bad_rebuild))
sys.exit(1 if (bad_delta or bad_rebuild) else 0)
