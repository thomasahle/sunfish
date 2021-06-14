![Sunfish logo](https://raw.github.com/thomasahle/sunfish/master/logo/sunfish_large.png)

## Introduction
Sunfish is a simple, but strong chess engine, written in Python, mostly for teaching purposes. Without tables and its simple interface, it takes up just 111 lines of code! (see [`compressed.py`](https://github.com/thomasahle/sunfish/blob/master/compressed.py)) Yet [it plays at ratings above 2000 at Lichess](https://lichess.org/@/sunfish-engine).

Because Sunfish is small and strives to be simple, the code provides a great platform for experimenting. People have used it for testing parallel search algorithms, experimenting with evaluation functions, and developing deep learning chess programs. Fork it today and see what you can do!

## Screenshot

    My move: g8f6
    
      8 ♖ ♘ ♗ ♕ ♔ ♗ · ♖
      7 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
      6 · · · · · ♘ · ·
      5 · · · · · · · ·
      4 · · · · ♟ · · ·
      3 · · · · · · · ·
      2 ♟ ♟ ♟ ♟ · ♟ ♟ ♟
      1 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        a b c d e f g h


    Your move:

# Run it!

Sunfish is self contained in the `sunfish.py` file from the repository. I recommend running it with `pypy` or `pypy3` for optimal performance.

If you want a more "fancy" terminal experience, run `fancy.py` instead. (Note this requires the [python-chess](https://github.com/niklasf/python-chess/) package.)

It is also possible to run Sunfish with a graphical interface, such as [PyChess](http://pychess.org), [Arena](http://www.playwitharena.com) or your chess interface of choice. Sunfish' can communicate through the [XBoard](http://www.gnu.org/software/xboard/)/CECP protocol by the command `pypy -u xboard.py`. Ruxy Sylwyka has [a note on making it all work on Windows](http://www.talkchess.com/forum/viewtopic.php?topic_view=threads&p=560462).

You can [play sunfish now on Lichess](https://lichess.org/@/sunfish-engine) (requires log in) or play against [Recursing's Rust port](https://github.com/Recursing/sunfish_rs),
also [on Lichess](https://lichess.org/@/sunfish_rs), which is about 100 ELO stronger.

# Features

1. Built around the simple, but deadly efficient MTD-bi search algorithm.
2. Filled with classic as well as modern 'chess engine tricks' for simpler and faster code.
3. Easily adaptive evaluation function through Piece Square Tables.
4. Uses standard Python collections and data structures for clarity and efficiency.

# Limitations

Sunfish supports castling, en passant, and promotion. It doesn't however do minor promotions to rooks, knights or bishops - all input must be done in simple 'two coordinate' notation, as shown in the screenshot.

There are many ways in which you may try to make Sunfish stronger. First you could change from a board representation to a mutable array and add a fast way to enumerate pieces. Then you could implement dedicated capture generation, check detection and check evasions. You could also move everything to bitboards, implement parts of the code in C or experiment with parallel search!

The other way to make Sunfish stronger is to give it more knowledge of chess. The current evaluation function only uses piece square tables - it doesn't even distinguish between midgame and endgame. You can also experiment with more pruning - currently only null move is done - and extensions - currently none are used. Finally Sunfish might benefit from a more advanced move ordering, MVV/LVA and SEE perhaps?

An easy way to get a strong Sunfish is to run with with the [PyPy Just-In-Time intepreter](https://pypy.org/). In particular the python2.7 version of pypy gives a 250 ELO boost compared to the cpython (2 or 3) intepreters at fast time controls:

    Rank Name                    Elo     +/-   Games   Score   Draws
       1 pypy2.7 (7.1)           166      38     300   72.2%   19.7%
       2 pypy3.6 (7.1)            47      35     300   56.7%   21.3%
       3 python3.7               -97      36     300   36.3%   20.7%
       4 python2.7              -109      35     300   34.8%   24.3%


# Why Sunfish?

The name Sunfish actually refers to the [Pygmy Sunfish](http://en.wikipedia.org/wiki/Pygmy_sunfish), which is among the very few fish to start with the letters 'Py'. The use of a fish is in the spirit of great engines such as Stockfish, Zappa and Rybka.

In terms of Heritage, Sunfish borrows much more from [Micro-Max by Geert Muller](http://home.hccnet.nl/h.g.muller/max-src2.html) and [PyChess](http://pychess.org).

# License

[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)
