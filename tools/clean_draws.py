import chess, chess.pgn, chess.engine
engine = chess.engine.SimpleEngine.popen_uci('stockfish')
do = open('3fold_do.pgn', 'w')
dont = open('3fold_dont.pgn', 'w')
with open('3fold.pgn') as file:
    for i, game in enumerate(iter(lambda:chess.pgn.read_game(file), None)):
        print(i)
        res = engine.analyse(game.end().parent.board(), limit=chess.engine.Limit(time=.01))
        if res.pv[0] != game.end().move:
            # We skip situations that re too ambigious
            if res.score.is_mate() or abs(res.score.relative.cp) > 200:
                print(game, file=dont, end='\n\n', flush=True)
        else:
            # TODO: Some skipping here as well
            print(game, file=do, end='\n\n', flush=True)
engine.quit()
