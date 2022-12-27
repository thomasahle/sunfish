# Code from sunfish_nnue_color removed from compressed version.


def render_move(move, white_pov):
    if move is None:
        return '0000'
    i, j = move.i, move.j
    if not white_pov:
        i, j = 119 - i, 119 - j
    return render(i) + render(j) + move.prom.lower()

def main():
    global debug
    wf, bf = features(initial)
    pos0 = Position(initial, 0, wf, bf, (True, True), (True, True), 0, 0)
    pos0 = pos0._replace(score=pos0.compute_value(verbose=debug))
    hist = [pos0]
    searcher = Searcher()
    while True:
        args = input().split()
        if args[0] == "uci":
            print("id name Sunfish NNUE")
            print(f"option name EVAL_ROUGHNESS type spin default {EVAL_ROUGHNESS} min 1 max 100")
            print(f"option name QS_LIMIT type spin default {QS_LIMIT} min 0 max 2000")
            print(f"option name QS_TYPE type spin default {QS_TYPE} min 0 max 2000")
            print("uciok")

        elif args[0] == "isready":
            print("readyok")

        elif args[0] == "debug":
            debug = args[1] == 'on'

        elif args[0] == "ucinewgame":
            hist = [pos0]

        # case ["setoption", "name", uci_key, "value", uci_value]:
        elif args[0] == "setoption":
            _, uci_key, _, uci_value = args[1:]
            globals()[uci_key] = int(uci_value)

        # FEN support is just for testing. Remove before TCEC
        # case ["position", "fen", *fen]:
        elif args[:2] == ["position", "fen"]:
            fen = args[2:]
            board, color, castling, enpas, _hclock, _fclock = fen
            board = re.sub(r"\d", (lambda m: "." * int(m.group(0))), board)
            board = list(21 * " " + "  ".join(board.split("/")) + 21 * " ")
            board[9::10] = ["\n"] * 12
            board = "".join(board)
            wc = ("Q" in castling, "K" in castling)
            bc = ("k" in castling, "q" in castling)
            ep = parse(enpas) if enpas != "-" else 0
            wf, bf = features(board)
            pos = Position(board, 0, wf, bf, wc, bc, ep, 0)
            pos = pos._replace(score=pos.compute_value())
            if color == "w":
                hist = [pos]
            else:
                hist = [pos, pos.rotate()]
            if debug:
                print(hist[-1].board)
                print(hist[-1].compute_value(verbose=True))

        #case ["position", "startpos", *moves]:
        elif args[:2] == ["position", "startpos"]:
            moves = args[2:]
            hist = [pos0]
            for i, move in enumerate(moves[1:]):
                a, b, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
                if i % 2 == 1:
                    a, b = 119 - a, 119 - b
                hist.append(hist[-1].move(Move(a, b, prom)))
            if debug:
                print(hist[-1].board)
                print(hist[-1].compute_value())

        #case ["quit"]:
        elif args[0] == "quit":
            break

        # case ["go", *args]:
        elif args[0] == "go":
            # case ['movetime', movetime]:
            #case []:
            if len(args) == 1:
                think = 24 * 3600
            elif args[1] == "movetime":
                movetime = args[2]
                think = int(movetime) / 1000
            # case ['wtime', wtime, 'btime', btime, 'winc', winc, 'binc', binc]:
            elif args[1] == "wtime":
                _, wtime, _, btime, _, winc, _, binc = args[1:]
                wtime, btime, winc, binc = int(wtime), int(btime), int(winc), int(binc)
                # we always consider ourselves white, but uci doesn't
                if len(hist) % 2 == 0:
                    wtime, winc = btime, binc
                think = wtime / 1000 / 40 + winc / 1000
                if think > wtime:
                    think = wtime/2
                # let's go fast for the first moves
                if len(hist) < 3:
                    think = min(think, 1)
            #case ['depth', max_depth]:
            elif args[1] == 'depth':
                max_depth = args[2]
                think = -1
                max_depth = int(max_depth)
            #case ['mate', max_depth]:
            elif args[1] == 'mate':
                max_depth = args[2]
                for i in range(int(max_depth)):
                    searcher = searcher() # need to clear stuff
                    score = searcher.bound(hist[-1], mate_lower, i+1, root=true)
                    move = searcher.tp_move.get(hist[-1].hash())
                    move_str = render_move(move, white_pov=len(hist)%2==1)
                    print("info", "score cp", score, "pv", move_str)
                    if score >= mate_lower:
                        break
                print("bestmove", move_str, "score cp", score)
                continue
            if debug:
                print(f"i want to think for {think} seconds.")
            start = time.time()
            try:
                for depth, move, score, is_lower in searcher.search(hist):
                    if think < 0 and depth == max_depth and is_lower is none:
                        break
                    if move is none:
                        continue
                    move_str = render_move(move, white_pov=len(hist)%2==1)
                    elapsed = time.time() - start
                    print(
                        "info depth",
                        depth,
                        "score cp",
                        score,
                        "" if is_lower is none else ("lowerbound" if is_lower else "upperbound"),
                        "time",
                        int(1000 * elapsed),
                        "nodes",
                        searcher.nodes,
                        "pv",
                        move_str,
                    )
                    if think > 0 and time.time() - start > think * 2 / 3:
                        break
            except keyboardinterrupt:
                continue
            if debug:
                print(f"stopped thinking after {round(elapsed,3)} seconds")
            print("bestmove", move_str, 'score cp', score)


if __name__ == "__main__":
    main()
