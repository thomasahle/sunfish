"""The gate a mate suite is NOT: does the engine ALWAYS produce a LEGAL move?

LMP passed its mate gate 5-vs-5 on the very build that emitted an illegal
move, because "are mates found" and "is a move always produced" are different
questions. This asks the second one, and checks legality too.

Positions in CHECK are the dangerous class: this engine generates pseudo-legal
moves and has no notion of check, so the single legal reply can sort to the
tail where count-triggered pruning discards it.
"""
import subprocess, sys, chess, random

ENGINE = sys.argv[1]
MOVETIME = int(sys.argv[2]) if len(sys.argv) > 2 else 300
random.seed(20260813)


def positions(n_forced=40, n_check=30, n_plain=30):
    """Three classes. FORCED (in check, <=2 legal replies) is the dangerous one:
    the reproduction was in check with EXACTLY ONE legal move, which sorts to
    the tail where a count-triggered rule discards it. An earlier version of
    this gate sampled in-check positions indiscriminately and PASSED a build
    that demonstrably emits illegal moves -- most in-check positions have
    several escapes, so the pathological case almost never came up."""
    forced, checks, plains = [], [], []
    tries = 0
    while (len(forced) < n_forced or len(checks) < n_check or len(plains) < n_plain) and tries < 40000:
        tries += 1
        b = chess.Board()
        for _ in range(random.randint(4, 70)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(random.choice(ms))
        if b.is_game_over():
            continue
        line = " ".join(m.uci() for m in b.move_stack)
        n_legal = b.legal_moves.count()
        if b.is_check() and n_legal <= 2 and len(forced) < n_forced:
            forced.append((line, b))
        elif b.is_check() and len(checks) < n_check:
            checks.append((line, b))
        elif not b.is_check() and len(plains) < n_plain:
            plains.append((line, b))
    return forced, checks, plains


def ask(line):
    p = subprocess.Popen([sys.executable, ENGINE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"uci\nisready\nposition startpos moves {line}\ngo movetime {MOVETIME}\n")
    p.stdin.flush()
    mv = None
    while True:
        o = p.stdout.readline()
        if not o:
            break
        if o.startswith("bestmove"):
            mv = o.split()[1] if len(o.split()) > 1 else None
            break
    p.kill()
    return mv


forced, checks, plains = positions()
fails = []
for label, group in (("FORCED", forced), ("IN CHECK", checks), ("quiet", plains)):
    none_ct = illegal_ct = 0
    for line, board in group:
        mv = ask(line)
        if mv in (None, "(none)", "0000"):
            none_ct += 1
            fails.append((label, board.fen(), mv, "NO MOVE"))
            continue
        try:
            m = chess.Move.from_uci(mv)
        except Exception:
            illegal_ct += 1
            fails.append((label, board.fen(), mv, "UNPARSEABLE"))
            continue
        if m not in board.legal_moves:
            illegal_ct += 1
            fails.append((label, board.fen(), mv, "ILLEGAL"))
    print(f"{label:9} n={len(group):3}  no-move={none_ct}  illegal={illegal_ct}")

print()
if fails:
    print(f"GATE FAILED: {len(fails)} bad answers")
    for lab, fen, mv, why in fails[:8]:
        print(f"  [{lab}] {why}: bestmove {mv}   {fen}")
else:
    print("GATE PASSED: every position produced a legal move")
