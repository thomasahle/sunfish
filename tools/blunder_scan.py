#!/usr/bin/env python3
"""Scan a lichess bot's recent games for blunders, with local Stockfish.

    tools/blunder_scan.py sunfish-engine --games 20
    tools/blunder_scan.py sunfish-nnue-engine --games 100 --threshold 200

For every move OUR bot played, Stockfish evaluates the position before
and after (same fixed depth, scores from the mover's side); a drop
beyond the threshold is a blunder. Output is one line per blunder with
the clickable ply anchor (lichess.org/<id>/black#<ply>), the eval
swing, and the better move Stockfish wanted - newest games first, so
this doubles as the production audit's first pass. Games where the
opponent flagged mid-blunder still count: the eval is about the move,
not the result.

Rate limits: lichess soft-blocks bursts (429, then 404s). One export
call fetches ALL requested games; only Stockfish runs locally after
that, so the scan makes exactly one API request.
"""
import argparse
import io
import shutil
import subprocess
import sys
import urllib.request

import chess
import chess.engine
import chess.pgn

MATE_CP = 10_000


def fetch_games(user, n):
    url = (f"https://lichess.org/api/games/user/{user}"
           f"?max={n}&moves=true&clocks=false&evals=false&perfType="
           "bullet,blitz,rapid,classical")
    req = urllib.request.Request(url, headers={"Accept": "application/x-chess-pgn"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode()
    games, stream = [], io.StringIO(text)
    while (g := chess.pgn.read_game(stream)) is not None:
        games.append(g)
    return games


def cp(score, color):
    """Centipawns from `color`'s view, mates folded to +/-MATE_CP."""
    s = score.pov(color)
    return s.score() if s.score() is not None else (MATE_CP if s.mate() > 0 else -MATE_CP)


def scan_game(game, engine, me, depth, threshold):
    hdr = game.headers
    my_color = chess.WHITE if hdr.get("White", "").lower() == me else chess.BLACK
    board = game.board()
    limit = chess.engine.Limit(depth=depth)
    found = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if board.turn == my_color and len(list(board.legal_moves)) > 1:
            info = engine.analyse(board, limit)
            best_cp = cp(info["score"], my_color)
            best_move = info.get("pv", [None])[0]
            if move != best_move:
                board.push(move)
                after = cp(engine.analyse(board, limit)["score"], my_color)
                board.pop()
                loss = best_cp - after
                # Only positions that were still holdable: being lost
                # already and staying lost is not the signal we hunt.
                if loss >= threshold and best_cp > -threshold:
                    found.append((ply, board.san(move),
                                  best_cp, after,
                                  board.san(best_move) if best_move else "?"))
        board.push(move)
    return my_color, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user", help="lichess username of our bot")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--threshold", type=int, default=300,
                    help="centipawn loss that counts as a blunder")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--engine", default=shutil.which("stockfish"),
                    help="path to a UCI engine for the oracle")
    args = ap.parse_args()
    if not args.engine:
        sys.exit("no stockfish on PATH (brew install stockfish)")

    me = args.user.lower()
    games = fetch_games(args.user, args.games)
    print(f"{len(games)} games fetched for {args.user}; "
          f"oracle depth {args.depth}, threshold {args.threshold}cp\n")

    engine = chess.engine.SimpleEngine.popen_uci([args.engine])
    engine.configure({"Threads": 2, "Hash": 256})
    total_blunders, games_with = 0, 0
    try:
        for game in games:
            hdr = game.headers
            color, found = scan_game(game, engine, me, args.depth, args.threshold)
            side = "white" if color == chess.WHITE else "black"
            opp = hdr.get("Black" if color == chess.WHITE else "White", "?")
            gid = hdr.get("Site", "").rsplit("/", 1)[-1]
            res = hdr.get("Result", "?")
            if found:
                games_with += 1
                total_blunders += len(found)
                print(f"{gid} vs {opp} ({res}, {hdr.get('TimeControl', '?')}):")
                for ply, san, best, after, better in found:
                    moveno = (ply + 1) // 2
                    dots = "." if color == chess.WHITE else "..."
                    print(f"  {moveno}{dots}{san}  {best/100:+.2f} -> {after/100:+.2f}"
                          f"  (SF wanted {better})"
                          f"  https://lichess.org/{gid}/{side}#{ply}")
    finally:
        engine.quit()
    print(f"\n{total_blunders} blunders (>= {args.threshold}cp) "
          f"in {games_with}/{len(games)} games")


if __name__ == "__main__":
    main()
