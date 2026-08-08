#!/usr/bin/env python3
"""Validate terminal_bench.epd ground-truth labels against python-chess
and (when available) Stockfish.

Terminal classes (mate-now*, stalemate-now*, witness-*) are certified by
python-chess directly - is_checkmate()/is_stalemate() IS the ground
truth there. Parent classes are checked two ways: a mating/stalemating
move must exist by direct enumeration, and Stockfish (depth 18) must
agree the side to move has a forced win / at least a draw. Run after
regenerating or extending the bench; a violation means a label is
wrong, not that an engine is weak.

Usage: tools/validate_terminal_bench.py [path-to-stockfish]
"""
import sys
from pathlib import Path

import chess
import chess.engine

BENCH = Path(__file__).parent / "test_files" / "terminal_bench.epd"


def main():
    sf_path = sys.argv[1] if len(sys.argv) > 1 else "stockfish"
    try:
        eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    except FileNotFoundError:
        eng = None
        print("stockfish not found: terminal classes only")
    bad = 0
    for line in BENCH.read_text().splitlines():
        fen_part, rest = line.split("; class ", 1)
        cls = rest.split(";")[0].strip()
        b = chess.Board(" ".join(fen_part.split()[:4]) + " 0 1")
        ok = True
        if cls in ("mate-now", "mate-now-corner", "witness-standpat-mate"):
            ok = b.is_checkmate()
        elif cls in ("stalemate-now", "stalemate-now-ahead", "witness-null-stalemate"):
            ok = b.is_stalemate()
        elif cls in ("parent-of-mate", "parent-of-stalemate"):
            want_mate = cls == "parent-of-mate"
            found = False
            for m in b.legal_moves:
                b.push(m)
                if (b.is_checkmate() if want_mate else b.is_stalemate()):
                    found = True
                b.pop()
                if found:
                    break
            ok = found
            if ok and eng is not None:
                sc = eng.analyse(b, chess.engine.Limit(depth=18))["score"].pov(b.turn)
                if want_mate:
                    ok = sc.is_mate() and sc.mate() > 0
                else:
                    ok = sc.is_mate() and sc.mate() > 0 or not sc.is_mate() and sc.score() >= -10
        if not ok:
            bad += 1
            print(f"LABEL VIOLATION [{cls}]: {fen_part.strip()}")
    if eng is not None:
        eng.quit()
    print("all labels valid" if bad == 0 else f"{bad} violations")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
